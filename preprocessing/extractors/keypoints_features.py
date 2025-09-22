"""
MediaPipe keypoint extraction utilities for Filipino sign language recognition.

This module provides:
- MediaPipe Holistic model management
- Keypoint extraction from RGB frames (pose, hands, face)
- Gap interpolation for missing keypoints
- Normalized coordinate output in [0,1] range

Keypoint structure (156 dimensions):
- Pose landmarks: 25 points × 2 coordinates = 50 dims
- Left hand: 21 points × 2 coordinates = 42 dims  
- Right hand: 21 points × 2 coordinates = 42 dims
- Face mesh: 11 points × 2 coordinates = 22 dims
Total: 156 dimensions

Input: RGB frame (H, W, 3)
Output: vec156 [156] float32, mask78 [78] bool
"""

# Standard library imports
from dataclasses import dataclass  # Data structure definitions

# Numerical computing
import numpy as np  # Numerical arrays and mathematical operations

# Computer vision framework
import mediapipe as mp  # Google's MediaPipe for keypoint detection (pose, hands, face)

# ----------------------------
# Keypoint Configuration Constants
# ----------------------------
# These constants define which specific keypoints are extracted from MediaPipe's full set

# Upper body pose keypoints (25 points from MediaPipe's 33-point pose model)
# Includes: face landmarks (0-10), shoulders/arms (11-16), torso (17-22), hips (23-24)
POSE_UPPER_25 = list(range(0, 11)) + list(range(11, 17)) + list(range(17, 23)) + [23, 24]

# Hand keypoints (21 points per hand from MediaPipe's hand model)
# Includes: wrist (0), thumb (1-4), index (5-8), middle (9-12), ring (13-16), pinky (17-20)
N_HAND = 21

# Key facial landmarks (11 points from MediaPipe's 468-point face mesh)
# Selected for sign language: key facial features and expressions
FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]


# ----------------------------
# MediaPipe Solution References
# ----------------------------
# Store references to MediaPipe solutions for keypoint detection
mp_hands = mp.solutions.hands          # Hand landmark detection (21 points per hand)
mp_face_mesh = mp.solutions.face_mesh  # Face mesh detection (468 points, we use 11)
mp_drawing = mp.solutions.drawing_utils # Visualization utilities (unused in processing)
mp_pose = mp.solutions.pose            # Body pose detection (33 points, we use upper 25)


# ----------------------------
# MediaPipe Model Container
# ----------------------------

@dataclass
class MPModels:
    """Container for initialized MediaPipe models.
    
    This dataclass holds references to the MediaPipe models needed for processing:
    - seg: Selfie segmentation model for background removal
    - hol: Holistic model for combined pose, hand, and face detection
    
    Attributes:
        seg: Selfie segmentation model instance
        hol: Holistic model instance
    """
    seg: any  # Selfie segmentation model for person/background separation
    hol: any  # Holistic model for unified pose, hands, and face detection


# ----------------------------
# Model Initialization and Cleanup
# ----------------------------

def create_models(seg_model=1, detection_conf=0.5, tracking_conf=0.5):
    """Initialize MediaPipe models for comprehensive keypoint extraction.
    
    Creates and configures MediaPipe models for:
    1. Person segmentation (background removal)
    2. Holistic keypoint detection (pose + hands + face)
    
    Args:
        seg_model: Selfie segmentation model selection (0=general, 1=landscape)
        detection_conf: Minimum detection confidence threshold (0.0-1.0)
        tracking_conf: Minimum tracking confidence threshold (0.0-1.0)
        
    Returns:
        MPModels: Container with initialized and configured models
    """
    # Initialize person segmentation model for background removal
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=seg_model)
    
    # Initialize holistic model for unified pose, hands, and face detection
    hol = mp.solutions.holistic.Holistic(
        model_complexity=1,           # Balance between accuracy and speed (0=lite, 1=full, 2=heavy)
        smooth_landmarks=True,        # Apply temporal smoothing to reduce jitter
        refine_face_landmarks=True,   # Enable face mesh refinement for better facial keypoints
        min_detection_confidence=detection_conf,  # Minimum confidence for initial detection
        min_tracking_confidence=tracking_conf,    # Minimum confidence for tracking across frames
    )
    
    return MPModels(seg=seg, hol=hol)


def close_models(models: MPModels):
    """Close MediaPipe models to free resources and prevent memory leaks.
    
    Args:
        models: MPModels container with initialized models
    """
    models.seg.close()  # Close segmentation model
    models.hol.close()  # Close holistic model


# ----------------------------
# Coordinate Conversion Utilities
# ----------------------------

def xy_from_landmark(lm, w, h):
    """Convert MediaPipe landmark to normalized coordinates.
    
    MediaPipe landmarks are already normalized to [0, 1] range, but this function
    ensures they stay within valid bounds and provides a consistent interface.
    
    Args:
        lm: MediaPipe landmark object with .x and .y attributes
        w: Image width (unused, kept for compatibility with legacy code)
        h: Image height (unused, kept for compatibility with legacy code)
        
    Returns:
        Tuple of (x, y) coordinates clamped to [0, 1] range
    """
    x = float(lm.x)  # Extract x coordinate (already normalized by MediaPipe)
    y = float(lm.y)  # Extract y coordinate (already normalized by MediaPipe)
    # Clamp coordinates to valid [0, 1] range to handle any edge cases
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def _lerp(a, b, t):
    """Linear interpolation between two values.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0.0 = a, 1.0 = b)
        
    Returns:
        Interpolated value between a and b
    """
    return a + (b - a) * t


# ----------------------------
# Gap Interpolation for Temporal Consistency
# ----------------------------

def interpolate_gaps(X, mask, max_gap=5):
    """Interpolate missing keypoints in temporal sequences using linear interpolation.
    
    This function fills gaps in keypoint sequences to improve temporal consistency.
    It only interpolates gaps within the specified maximum length to avoid
    unrealistic interpolation across long missing sequences.
    
    The algorithm:
    1. For each keypoint, find all valid (detected) frames
    2. Identify gaps between valid detections
    3. If gap length ≤ max_gap, linearly interpolate x,y coordinates
    4. Update visibility mask to mark interpolated points as valid
    
    Args:
        X: Keypoint coordinates [T, 156] as float32 - flattened x,y coords for 78 keypoints
        mask: Keypoint visibility mask [T, 78] as bool - True if keypoint is visible/confident
        max_gap: Maximum gap length to interpolate (frames)
        
    Returns:
        Tuple of (X_filled, mask_filled) with interpolated values
    """
    T = X.shape[0]  # Number of time steps (frames)
    K = mask.shape[1]  # Number of keypoints (78)
    X_out = X.copy()  # Copy input coordinates
    mask_out = mask.copy()  # Copy input visibility mask

    # Process each keypoint independently
    for k in range(K):
        # Calculate coordinate indices for this keypoint (x, y)
        xi = 2 * k      # X coordinate index in flattened array
        yi = 2 * k + 1  # Y coordinate index in flattened array
        
        # Find all frames where this keypoint is visible
        valid_idxs = np.where(mask[:, k])[0]
        if len(valid_idxs) == 0:
            continue  # Skip if keypoint is never visible

        # Process gaps between consecutive valid detections
        prev = valid_idxs[0]  # Start with first valid detection
        for vi in valid_idxs[1:]:  # Iterate through remaining valid detections
            if vi == prev + 1:  # No gap - consecutive frames
                prev = vi
                continue
            
            # Found a gap - calculate gap boundaries and length
            gap_start = prev + 1
            gap_end = vi - 1
            gap_len = gap_end - gap_start + 1
            
            # Only interpolate if gap is within acceptable length
            if 1 <= gap_len <= max_gap:
                # Get start and end coordinates for interpolation
                x0, y0 = X_out[prev, xi], X_out[prev, yi]  # Start point
                x1, y1 = X_out[vi, xi], X_out[vi, yi]      # End point
                
                # Linearly interpolate coordinates for each frame in the gap
                for t_idx, t_rel in enumerate(range(gap_start, gap_end + 1), start=1):
                    t = t_idx / (gap_len + 1)  # Interpolation factor [0, 1]
                    X_out[t_rel, xi] = _lerp(x0, x1, t)  # Interpolate x coordinate
                    X_out[t_rel, yi] = _lerp(y0, y1, t)  # Interpolate y coordinate
                    mask_out[t_rel, k] = True  # Mark as valid (interpolated)
            
            prev = vi  # Move to next valid detection

    return X_out, mask_out


# ----------------------------
# Main Keypoint Extraction Function
# ----------------------------

def extract_keypoints_from_frame(img_rgb, models: MPModels, conf_thresh=0.5):
    """Extract comprehensive keypoints from a single RGB frame.
    
    This is the main keypoint extraction function that processes an RGB frame through
    MediaPipe Holistic to extract pose, hand, and face landmarks. It returns a
    structured 156-dimensional vector with normalized coordinates and visibility flags.
    
    Keypoint structure (156 dimensions total):
    - Pose landmarks: 25 points × 2 coordinates = 50 dims (upper body)
    - Left hand: 21 points × 2 coordinates = 42 dims
    - Right hand: 21 points × 2 coordinates = 42 dims  
    - Face mesh: 11 points × 2 coordinates = 22 dims (key facial features)
    
    Args:
        img_rgb: RGB frame (H, W, 3) in [0, 255] pixel range
        models: Initialized MediaPipe models (MPModels container)
        conf_thresh: Confidence threshold for keypoint detection (0.0-1.0)
        
    Returns:
        vec156: Keypoint coordinates [156] as float32 in [0,1] normalized range
        mask78: Visibility mask [78] as bool - True if keypoint is visible/confident
    """
    H, W, _ = img_rgb.shape  # Get image dimensions
    res = models.hol.process(img_rgb)  # Process frame through MediaPipe Holistic
    coords = []      # Accumulate flattened x,y coordinates
    vis_mask = []    # Accumulate visibility flags for each keypoint

    # ---- POSE LANDMARKS (25 points) ----
    # Extract upper body pose keypoints for sign language recognition
    pose_present = res.pose_landmarks is not None
    pose_points = [(0.0, 0.0)] * len(POSE_UPPER_25)  # Initialize with default coordinates
    pose_mask = [False] * len(POSE_UPPER_25)          # Initialize visibility as False
    
    if pose_present:
        lms = res.pose_landmarks.landmark  # Get all pose landmarks
        for i, idx in enumerate(POSE_UPPER_25):  # Process only upper body keypoints
            lm = lms[idx]  # Get specific landmark
            # Convert to normalized coordinates [0, 1]
            x, y = xy_from_landmark(lm, W, H)
            pose_points[i] = (x, y)
            # Check visibility confidence (pose landmarks have visibility attribute)
            pose_mask[i] = (getattr(lm, "visibility", 0.0) or 0.0) >= conf_thresh

    # ---- HAND LANDMARKS (21 points per hand) ----
    # Extract hand keypoints for both left and right hands
    def hand_block(hand_lms):
        """Process hand landmarks and return coordinates and visibility mask."""
        pts = [(0.0, 0.0)] * N_HAND  # Initialize with default coordinates
        m = [False] * N_HAND         # Initialize visibility as False
        
        if hand_lms is not None:  # Hand detected
            for i, lm in enumerate(hand_lms.landmark[:N_HAND]):  # Process all 21 hand points
                x, y = xy_from_landmark(lm, W, H)  # Convert to normalized coordinates
                pts[i] = (x, y)
                # Hand landmarks don't have visibility attribute, so check coordinate bounds
                m[i] = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)
        return pts, m

    # Process both hands using the same function
    left_pts, left_mask = hand_block(res.left_hand_landmarks)   # Left hand (21 points)
    right_pts, right_mask = hand_block(res.right_hand_landmarks) # Right hand (21 points)

    # ---- FACE LANDMARKS (11 key points) ----
    # Extract key facial landmarks from the 468-point face mesh
    face_points = [(0.0, 0.0)] * len(FACEMESH_11)  # Initialize with default coordinates
    face_mask = [False] * len(FACEMESH_11)          # Initialize visibility as False
    
    if res.face_landmarks is not None:  # Face detected
        fl = res.face_landmarks.landmark  # Get all face mesh landmarks
        for i, idx in enumerate(FACEMESH_11):  # Process only key facial points
            lm = fl[idx]  # Get specific facial landmark
            x, y = xy_from_landmark(lm, W, H)  # Convert to normalized coordinates
            face_points[i] = (x, y)
            # Face landmarks don't have visibility attribute, so check coordinate bounds
            face_mask[i] = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)

    # ---- CONCATENATE ALL KEYPOINTS ----
    # Combine all keypoint blocks in the specified order to create 156D vector
    # Order: pose (50D) + left_hand (42D) + right_hand (42D) + face (22D) = 156D
    for block_pts, block_mask in [
        (pose_points, pose_mask),    # 25 points × 2 coords = 50 dimensions
        (left_pts, left_mask),       # 21 points × 2 coords = 42 dimensions
        (right_pts, right_mask),     # 21 points × 2 coords = 42 dimensions
        (face_points, face_mask),    # 11 points × 2 coords = 22 dimensions
    ]:
        # Flatten (x, y) coordinates into the coordinate vector
        for (x, y) in block_pts:
            coords.extend([x, y])  # Add x and y to flattened coordinate list
        # Add visibility flags to the mask vector
        vis_mask.extend(block_mask)

    # Convert to numpy arrays with appropriate data types
    return np.array(coords, dtype=np.float32), np.array(vis_mask, dtype=bool)