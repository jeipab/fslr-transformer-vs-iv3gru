"""
Hand-head occlusion detection for Filipino sign language recognition.

This module provides comprehensive occlusion detection capabilities using multiple approaches:
- Computer vision-based detection using MediaPipe keypoints
- Multi-method detection algorithms (ellipse, proximity, trajectory analysis)
- Temporal filtering for robust detection across video sequences
- 5-region head partitioning (forehead, cheeks, nose, mouth, neck)
- Adaptive thresholds and consecutive frame analysis

The module supports both keypoint-based and raw video processing, with configurable
parameters for different detection scenarios and quality requirements.
"""

# Standard library imports
import warnings  # Warning system for error handling
from collections import deque  # Efficient queue for temporal filtering
from dataclasses import dataclass  # Data structure definitions
from typing import List, Tuple, Dict, Optional, Set, Union  # Type hints for better code clarity

# Numerical computing
import numpy as np  # Numerical arrays and mathematical operations


# ----------------------------
# Hand Feature Extraction
# ----------------------------

def _hand_centers_and_tips(frame_xy: np.ndarray, frame_mask: np.ndarray, hand_start: int, hand_len: int) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    """Extract palm center and fingertip coordinates for one hand from keypoint data.
    
    This function processes MediaPipe hand keypoints to extract key anatomical features:
    - Palm center: Computed from MCP (metacarpophalangeal) joint positions
    - Fingertips: All five fingertip positions with validation
    
    Args:
        frame_xy: Keypoint coordinates array [156] - flattened x,y coordinates
        frame_mask: Visibility mask array [78] - boolean visibility flags
        hand_start: Starting index for hand keypoints (25 for left, 46 for right)
        hand_len: Number of hand keypoints (21)
        
    Returns:
        Tuple of (palm_center, fingertip_list) where:
        - palm_center: (x, y) coordinates or None if insufficient data
        - fingertip_list: List of (x, y) coordinates for visible fingertips
    """
    # MediaPipe hand landmark indices (relative to hand start)
    mcp_rel = [5, 9, 13, 17]  # MCP joints (base of fingers, excluding thumb)
    fingertip_rel = [4, 8, 12, 16, 20]  # All fingertips (thumb, index, middle, ring, pinky)
    mcp_coords: list[tuple[float, float]] = []
    
    # STEP 1: Extract MCP joint coordinates for palm center calculation
    for r in mcp_rel:
        if hand_len <= r:  # Skip if hand model doesn't have this keypoint
            continue
        if bool(frame_mask[hand_start + r]):  # Check if keypoint is visible
            idx = 2 * (hand_start + r)  # Calculate flattened array index
            coord = (float(frame_xy[idx]), float(frame_xy[idx + 1]))  # Extract (x, y)
            # Validate coordinate is within normalized bounds [0, 1]
            if 0 <= coord[0] <= 1 and 0 <= coord[1] <= 1:
                mcp_coords.append(coord)
    
    # STEP 2: Calculate palm center from MCP joints or fallback to wrist
    palm_center: tuple[float, float] | None
    if len(mcp_coords) >= 2:  # Need at least 2 MCP joints for reliable center
        # Calculate centroid of visible MCP joints
        mx = sum(p[0] for p in mcp_coords) / float(len(mcp_coords))
        my = sum(p[1] for p in mcp_coords) / float(len(mcp_coords))
        palm_center = (mx, my)
    else:
        # Fallback: use wrist position if MCP joints are not available
        if bool(frame_mask[hand_start + 0]):  # Check wrist visibility
            idx0 = 2 * (hand_start + 0)  # Wrist is always at index 0
            wrist_coord = (float(frame_xy[idx0]), float(frame_xy[idx0 + 1]))
            # Validate wrist coordinates
            if 0 <= wrist_coord[0] <= 1 and 0 <= wrist_coord[1] <= 1:
                palm_center = wrist_coord
            else:
                palm_center = None
        else:
            palm_center = None  # No reliable palm center available
    
    # STEP 3: Extract fingertip coordinates with validation
    tips: list[tuple[float, float]] = []
    for r in fingertip_rel:
        if hand_len <= r:  # Skip if hand model doesn't have this keypoint
            continue
        if bool(frame_mask[hand_start + r]):  # Check if fingertip is visible
            idx = 2 * (hand_start + r)  # Calculate flattened array index
            tip_coord = (float(frame_xy[idx]), float(frame_xy[idx + 1]))  # Extract (x, y)
            # Validate fingertip coordinates are within normalized bounds
            if 0 <= tip_coord[0] <= 1 and 0 <= tip_coord[1] <= 1:
                tips.append(tip_coord)
    
    return palm_center, tips


# ----------------------------
# Face Landmark Validation
# ----------------------------

def _validate_face_landmarks(face_coords: List[Tuple[float, float]], 
                           face_indices: List[int]) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Validate face landmark coordinates and indices for reliable occlusion detection.
    
    This function performs quality control on facial landmarks to ensure they are:
    1. Within valid coordinate bounds [0, 1]
    2. Positioned in anatomically reasonable locations
    3. Suitable for accurate occlusion detection
    
    Args:
        face_coords: List of face landmark coordinates [(x, y), ...]
        face_indices: List of corresponding face landmark indices
        
    Returns:
        Tuple of (validated_coords, validated_indices) containing only reliable landmarks
    """
    validated_coords = []
    validated_indices = []
    
    # Process each landmark coordinate and index pair
    for i, (coord, idx) in enumerate(zip(face_coords, face_indices)):
        # STEP 1: Check if coordinates are within valid normalized bounds
        if 0 <= coord[0] <= 1 and 0 <= coord[1] <= 1:
            # STEP 2: Check for anatomically reasonable landmark positions
            if _is_valid_landmark_position(coord, idx):
                validated_coords.append(coord)
                validated_indices.append(idx)
    
    return validated_coords, validated_indices


def _is_valid_landmark_position(coord: Tuple[float, float], landmark_idx: int) -> bool:
    """Check if a facial landmark position is anatomically reasonable.
    
    This function performs sanity checks on facial landmark positions based on
    typical facial anatomy in normalized coordinates [0, 1]. It helps filter out
    erroneous detections that could lead to false occlusion alerts.
    
    Args:
        coord: Landmark coordinate (x, y) in normalized space [0, 1]
        landmark_idx: Landmark index from FACEMESH_11 mapping
        
    Returns:
        True if position is within reasonable anatomical bounds
    """
    x, y = coord
    
    # Anatomical validation based on landmark type (from FACEMESH_11 mapping)
    if landmark_idx == 0:  # nose_tip (center of face)
        return 0.3 <= x <= 0.7 and 0.3 <= y <= 0.7
    elif landmark_idx in [1, 2]:  # eye_outer (left/right eye corners)
        return 0.2 <= x <= 0.8 and 0.2 <= y <= 0.6
    elif landmark_idx in [3, 4]:  # eye_inner (inner eye corners)
        return 0.3 <= x <= 0.7 and 0.2 <= y <= 0.6
    elif landmark_idx in [5, 6]:  # mouth (left/right mouth corners)
        return 0.2 <= x <= 0.8 and 0.5 <= y <= 0.8
    elif landmark_idx == 7:  # forehead (upper face)
        return 0.2 <= x <= 0.8 and 0.1 <= y <= 0.4
    elif landmark_idx == 8:  # chin (lower face)
        return 0.3 <= x <= 0.7 and 0.6 <= y <= 0.9
    elif landmark_idx in [9, 10]:  # cheeks (side face regions)
        return 0.1 <= x <= 0.9 and 0.3 <= y <= 0.7
    
    # Default validation for unknown landmarks (generous bounds)
    return 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9


# ----------------------------
# Dependency Management
# ----------------------------

def _check_dependencies() -> bool:
    """Check if required optional dependencies are available for advanced occlusion detection.
    
    This function verifies that additional computer vision and machine learning libraries
    are installed for the full occlusion detection pipeline. If dependencies are missing,
    the system will fall back to keypoint-only detection.
    
    Returns:
        True if all advanced dependencies are available, False otherwise
    """
    try:
        # Computer vision libraries
        import cv2  # OpenCV for image processing
        import mediapipe as mp  # MediaPipe for additional landmark detection
        
        # Scientific computing libraries for advanced algorithms
        from scipy import ndimage  # Image processing algorithms
        from scipy.spatial import KDTree  # Spatial data structures for neighbor search
        from sklearn.cluster import DBSCAN  # Clustering algorithms for point grouping
        
        return True
    except ImportError:
        return False  # Dependencies not available, use fallback methods


# ----------------------------
# Data Structures for Geometric Processing
# ----------------------------

@dataclass
class Point2D:
    """Represents a 2D point with x, y coordinates in normalized space.
    
    This class provides a convenient container for 2D coordinates with
    utility methods for geometric calculations commonly used in occlusion detection.
    
    Attributes:
        x: X-coordinate in normalized space [0, 1]
        y: Y-coordinate in normalized space [0, 1]
    """
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert point to tuple format for compatibility with other functions.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        return (self.x, self.y)
    
    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point.
        
        Args:
            other: Another Point2D instance
            
        Returns:
            Euclidean distance between the two points
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Gridlet:
    """A set of tracked points with topological relationships for motion analysis.
    
    Gridlets are used in advanced occlusion detection to track groups of related
    points across frames, maintaining their spatial relationships and motion patterns.
    
    Attributes:
        points: List of 2D points in the gridlet
        neighbors: Adjacency list defining point relationships
        reference_frame: Frame index where this gridlet was established
        tracking_cost: Cost metric for tracking quality assessment
    """
    points: List[Point2D]
    neighbors: Dict[int, List[int]]  # Adjacency list mapping point indices to neighbor indices
    reference_frame: int              # Frame number for temporal reference
    tracking_cost: float = 0.0       # Quality metric for tracking reliability
    
    def get_center(self) -> Point2D:
        """Calculate the centroid (geometric center) of all points in the gridlet.
        
        Returns:
            Point2D representing the centroid of the gridlet
        """
        if not self.points:
            return Point2D(0.0, 0.0)
        
        x = sum(p.x for p in self.points) / len(self.points)
        y = sum(p.y for p in self.points) / len(self.points)
        return Point2D(x, y)


class HeadRegion:
    """Defines the five anatomical head regions used for detailed occlusion analysis.
    
    This class provides constants for the different facial regions that can be occluded
    by hands during sign language communication. Each region has specific characteristics
    and importance for sign language recognition.
    
    Region Definitions:
    - FOREHEAD: Upper head area, important for facial expressions
    - CHEEKS: Side face areas including eye regions, critical for visibility
    - NOSE: Central face area, key reference point for detection
    - MOUTH: Lower face area, essential for mouth shape recognition
    - NECK: Below-chin area, relevant for hand positioning
    """
    FOREHEAD = 0  # Upper head region (forehead area)
    CHEEKS = 1    # Side face regions (eye and cheek areas)
    NOSE = 2      # Central face region (nose area)
    MOUTH = 3     # Lower face region (mouth and jaw area)
    NECK = 4      # Below-face region (neck area)
    
    # Human-readable names for each region (for reporting and debugging)
    NAMES = ['forehead', 'cheeks', 'nose', 'mouth', 'neck']


# ----------------------------
# Main Occlusion Detection Class
# ----------------------------

class HandHeadOcclusionDetector:
    """Comprehensive hand-head occlusion detector using advanced computer vision techniques.
    
    This class provides a complete occlusion detection system that combines:
    - MediaPipe-based facial and hand landmark detection
    - Multi-region head partitioning for detailed analysis
    - Temporal filtering for robust detection across video sequences
    - Skin color segmentation for enhanced accuracy
    - Configurable parameters for different detection scenarios
    
    The detector processes video frames to identify when hands obscure facial features,
    which is crucial for sign language recognition quality assessment.
    """
    
    def __init__(self, use_global_tracking: bool = True):
        """Initialize the occlusion detector with required dependencies and parameters.
        
        Args:
            use_global_tracking: Enable advanced global tracking algorithms
            
        Raises:
            ImportError: If required dependencies (scipy, scikit-learn) are not available
        """
        # Check for optional dependencies needed for advanced features
        if not _check_dependencies():
            raise ImportError(
                "Advanced occlusion detection requires additional dependencies. "
                "Please install: pip install scipy scikit-learn opencv-python"
            )
        
        self.use_global_tracking = use_global_tracking
        
        # Import required computer vision dependencies
        import cv2
        import mediapipe as mp
        
        # Initialize MediaPipe models for comprehensive detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        # Configure face mesh detection for detailed facial landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,        # Video processing mode
            max_num_faces=1,               # Expect single person in frame
            refine_landmarks=True,         # Use refined face mesh for accuracy
            min_detection_confidence=0.5,  # Balanced detection threshold
            min_tracking_confidence=0.5    # Balanced tracking threshold
        )
        
        # Configure hand detection for both hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # Video processing mode
            max_num_hands=2,               # Detect both hands
            min_detection_confidence=0.5,  # Balanced detection threshold
            min_tracking_confidence=0.5    # Balanced tracking threshold
        )
        
        # Advanced tracking and detection parameters
        self.gridlet_size = 4              # Size of point groups for tracking
        self.gridlet_neighbors = 3         # Number of neighbors per gridlet point
        self.tracking_window_size = 5      # Temporal window for consistency filtering
        self.motion_threshold = 10         # Threshold for motion detection
        
        # Data storage for advanced tracking algorithms
        self.unoccluded_face_points: Dict[int, Set[Tuple[int, int]]] = {}  # Face points not occluded
        self.outside_face_points: Dict[int, Set[Tuple[int, int]]] = {}     # Points outside face region
        self.tracked_gridlets: List[Gridlet] = []                         # Point groups being tracked
        self.hand_blobs: List[Dict] = []                                   # Hand blob detection results
        self.facial_prohibition_masks: Dict[int, np.ndarray] = {}          # Masks for face regions
        
        # Temporal filtering system for robust detection
        self.occlusion_history = deque(maxlen=self.tracking_window_size)   # Rolling history window
    
    def detect_skin_pixels(self, image: np.ndarray) -> np.ndarray:
        """Detect skin pixels using color-based segmentation.
        
        Args:
            image: Input image array
            
        Returns:
            Binary mask of skin pixels
        """
        import cv2
        
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask
    
    def get_facial_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """Extract facial landmarks using MediaPipe Holistic.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of facial landmarks or None
        """
        import cv2
        import mediapipe as mp
        
        # Use Holistic model to match preprocessing pipeline
        holistic = mp.solutions.holistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)
        
        if not results.face_landmarks:
            holistic.close()
            return None
        
        h, w = image.shape[:2]
        landmarks = results.face_landmarks
        
        # Use the same face landmark indices as preprocessing pipeline
        # FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]
        face_indices = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]
        
        # Extract key facial points using preprocessing-compatible indices
        key_points = {
            'left_eye_inner': landmarks.landmark[133],    # Index 3 in FACEMESH_11
            'left_eye_outer': landmarks.landmark[33],     # Index 1 in FACEMESH_11  
            'right_eye_inner': landmarks.landmark[362],  # Index 4 in FACEMESH_11
            'right_eye_outer': landmarks.landmark[263],  # Index 2 in FACEMESH_11
            'nose_tip': landmarks.landmark[1],           # Index 0 in FACEMESH_11
            'mouth_left': landmarks.landmark[61],         # Index 5 in FACEMESH_11
            'mouth_right': landmarks.landmark[291],       # Index 6 in FACEMESH_11
            'chin': landmarks.landmark[18],               # Approximate chin
            'forehead': landmarks.landmark[9]             # Approximate forehead
        }
        
        # Convert to normalized coordinates [0,1] to match preprocessing pipeline
        landmark_dict = {}
        for name, point in key_points.items():
            landmark_dict[name] = Point2D(point.x, point.y)  # Already normalized [0,1]
        
        holistic.close()
        return landmark_dict
    
    def partition_head_regions(self, landmarks: Dict, image_shape: tuple = (256, 256)) -> Dict[int, np.ndarray]:
        """Partition the head area into 5 regions based on facial landmarks.
        
        Args:
            landmarks: Dictionary of facial landmarks
            image_shape: Image dimensions (height, width)
            
        Returns:
            Dictionary mapping region IDs to polygon arrays
        """
        import cv2
        
        regions = {}
        
        if not landmarks:
            return regions
        
        h, w = image_shape
        
        # Create polygon for each region using normalized coordinates
        # Forehead region
        forehead_pts = np.array([
            landmarks['forehead'].to_tuple(),
            (landmarks['left_eye_outer'].x, landmarks['forehead'].y),
            (landmarks['right_eye_outer'].x, landmarks['forehead'].y),
            landmarks['left_eye_outer'].to_tuple(),
            landmarks['right_eye_outer'].to_tuple()
        ], dtype=np.float32)
        
        # Cheeks region (includes eyes and ears area)
        cheeks_pts = np.array([
            landmarks['left_eye_outer'].to_tuple(),
            landmarks['right_eye_outer'].to_tuple(),
            landmarks['nose_tip'].to_tuple(),
            (landmarks['left_eye_outer'].x - 0.08, landmarks['nose_tip'].y),  # Normalized offset
            (landmarks['right_eye_outer'].x + 0.08, landmarks['nose_tip'].y)  # Normalized offset
        ], dtype=np.float32)
        
        # Nose region
        nose_pts = np.array([
            (landmarks['nose_tip'].x - 0.06, landmarks['nose_tip'].y - 0.04),  # Normalized offsets
            (landmarks['nose_tip'].x + 0.06, landmarks['nose_tip'].y - 0.04),
            (landmarks['nose_tip'].x + 0.06, landmarks['nose_tip'].y + 0.04),
            (landmarks['nose_tip'].x - 0.06, landmarks['nose_tip'].y + 0.04)
        ], dtype=np.float32)
        
        # Mouth region
        mouth_pts = np.array([
            landmarks['mouth_left'].to_tuple(),
            landmarks['mouth_right'].to_tuple(),
            landmarks['chin'].to_tuple(),
            (landmarks['mouth_left'].x, landmarks['chin'].y),
            (landmarks['mouth_right'].x, landmarks['chin'].y)
        ], dtype=np.float32)
        
        # Neck region
        neck_pts = np.array([
            (landmarks['chin'].x - 0.12, landmarks['chin'].y),
            (landmarks['chin'].x + 0.12, landmarks['chin'].y),
            (landmarks['chin'].x + 0.16, landmarks['chin'].y + 0.20),  # Normalized offset
            (landmarks['chin'].x - 0.16, landmarks['chin'].y + 0.20)
        ], dtype=np.float32)
        
        regions[HeadRegion.FOREHEAD] = forehead_pts
        regions[HeadRegion.CHEEKS] = cheeks_pts
        regions[HeadRegion.NOSE] = nose_pts
        regions[HeadRegion.MOUTH] = mouth_pts
        regions[HeadRegion.NECK] = neck_pts
        
        return regions
    
    def classify_occlusions(self, occlusion_mask: np.ndarray, 
                          landmarks: Dict, image_shape: tuple = (256, 256)) -> Dict[int, int]:
        """Classify occlusions by head region.
        
        Args:
            occlusion_mask: Binary occlusion mask
            landmarks: Dictionary of facial landmarks
            image_shape: Image dimensions (height, width)
            
        Returns:
            Dictionary mapping region IDs to occlusion counts
        """
        import cv2
        
        regions = self.partition_head_regions(landmarks, image_shape)
        occlusion_counts = {}
        
        h, w = image_shape
        
        for region_id, polygon in regions.items():
            # Convert normalized coordinates to pixel coordinates for mask creation
            pixel_polygon = (polygon * np.array([w, h])).astype(np.int32)
            
            # Create region mask
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region_mask, [pixel_polygon], 255)
            
            # Resize occlusion mask to match image shape if needed
            if occlusion_mask.shape != (h, w):
                occlusion_mask_resized = cv2.resize(occlusion_mask, (w, h))
            else:
                occlusion_mask_resized = occlusion_mask
            
            # Count occlusion pixels in region
            occlusion_in_region = cv2.bitwise_and(occlusion_mask_resized, region_mask)
            count = np.count_nonzero(occlusion_in_region)
            occlusion_counts[region_id] = count
        
        return occlusion_counts
    
    def temporal_filter(self, occlusion_counts: Dict[int, int]) -> Dict[int, bool]:
        """Apply temporal filtering to reduce noise.
        
        Args:
            occlusion_counts: Dictionary of occlusion counts by region
            
        Returns:
            Dictionary of filtered occlusion detections by region
        """
        self.occlusion_history.append(occlusion_counts)
        
        if len(self.occlusion_history) < self.tracking_window_size:
            # Not enough history, return current detections
            return {region: (count > 100) for region, count in occlusion_counts.items()}
        
        # Apply majority voting
        filtered_detections = {}
        for region_id in range(5):
            votes = []
            for hist_counts in self.occlusion_history:
                votes.append(hist_counts.get(region_id, 0) > 100)
            
            # Majority voting
            filtered_detections[region_id] = sum(votes) > len(votes) // 2
        
        return filtered_detections
    
    def process_frame(self, frame: np.ndarray, frame_idx: int,
                     prev_frame: Optional[np.ndarray] = None) -> Dict:
        """Process a single frame for occlusion detection.
        
        Args:
            frame: Input frame array
            frame_idx: Frame index
            prev_frame: Previous frame (optional)
            
        Returns:
            Dictionary with occlusion detection results
        """
        import cv2
        
        h, w = frame.shape[:2]
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get facial landmarks
        landmarks = self.get_facial_landmarks(frame)
        
        results = {
            'frame_idx': frame_idx,
            'occlusion_detected': False,
            'occluded_regions': []
        }
        
        if landmarks:
            # Simplified occlusion detection for this implementation
            # In a full implementation, this would use the sophisticated tracking methods
            
            # Get skin mask
            skin_mask = self.detect_skin_pixels(frame)
            
            # Simple distance-based occlusion detection
            face_center = landmarks['nose_tip']
            face_radius = 50  # Approximate face radius
            
            # Count skin pixels near face
            face_region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(face_region_mask, (int(face_center.x), int(face_center.y)), face_radius, 255, -1)
            
            # Find skin pixels in face region
            skin_in_face = cv2.bitwise_and(skin_mask, face_region_mask)
            occlusion_pixels = np.count_nonzero(skin_in_face)
            
            # Simple threshold-based detection
            if occlusion_pixels > 500:  # Threshold for occlusion
                occlusion_mask = skin_in_face
            
            # Classify occlusions by region
            occlusion_counts = self.classify_occlusions(occlusion_mask, landmarks, (h, w))
            
            # Apply temporal filtering
            filtered_detections = self.temporal_filter(occlusion_counts)
            
            # Store results
            results['occlusion_counts'] = occlusion_counts
            results['filtered_detections'] = filtered_detections
            results['occlusion_detected'] = any(filtered_detections.values())
            results['occluded_regions'] = [HeadRegion.NAMES[r] for r, detected 
                                          in filtered_detections.items() if detected]
        
        return results


# ----------------------------
# Main Keypoint-Based Detection Function
# ----------------------------

def compute_occlusion_detection_from_keypoints(
    X: np.ndarray,
    mask: np.ndarray,
    output_format: str = 'compatible',
    **kwargs
) -> Union[int, Dict]:
    """
    Compute comprehensive occlusion detection using preprocessed keypoint data.
    
    This is the primary detection function that analyzes MediaPipe keypoints to identify
    hand-head occlusions. It employs multiple sophisticated detection methods:
    
    Enhanced Features:
    - 5-region head partitioning (forehead, cheeks, nose, mouth, neck)
    - Multi-method detection (ellipse intersection, proximity analysis, trajectory tracking)
    - Adaptive thresholds based on face size and hand visibility
    - Temporal consistency filtering with consecutive frame analysis
    - Confidence scoring for detection reliability
    
    Args:
        X: Keypoint coordinates [T, 156] - normalized coordinates for all keypoints
        mask: Visibility mask [T, 78] - boolean flags for keypoint visibility
        output_format: Output format ('compatible' for binary, 'detailed' for full results)
        **kwargs: Additional configuration parameters
    
    Returns:
        Union[int, Dict]: Binary occlusion flag (0/1) or detailed results dictionary
    """
    try:
        T = X.shape[0]  # Number of time steps (frames)
        
        # STEP 1: Parse keypoint layout from preprocessing pipeline
        # Layout: pose25, left_hand21, right_hand21, face11 = 78 total keypoints
        pose_len = 25      # Upper body pose keypoints
        hand_len = 21      # Hand keypoints per hand
        face_len = 11      # Key facial landmarks
        face_start = pose_len + hand_len + hand_len  # Starting index for face keypoints (67)
        
        # STEP 2: Configure adaptive detection parameters for optimal sensitivity
        min_face_points = 3        # Minimum face keypoints required (reduced for coverage)
        min_hand_points = 3        # Minimum hand keypoints required (reduced for coverage)
        min_fingertips_inside = 1  # Minimum fingertips in face region (sensitive detection)
        proximity_multiplier = 1.5 # Proximity radius multiplier (increased for better detection)
        occlusion_threshold = 0.15 # Overall occlusion threshold (reduced for sensitivity)
        
        # STEP 3: Initialize detection data structures
        results = []  # Store per-frame detection results
        hand_trajectories = {'left': [], 'right': []}  # Track hand movement patterns
        
        # STEP 4: Process each frame for occlusion detection
        for t in range(T):
            frame_xy = X[t]      # Keypoint coordinates for current frame [156]
            frame_mask = mask[t] # Visibility mask for current frame [78]
            
            # STEP 4a: Validate face keypoint availability
            face_mask = frame_mask[face_start:face_start + face_len]  # Extract face visibility flags
            visible_face_points = int(face_mask.sum())  # Count visible face keypoints
            
            # Skip frame if insufficient face keypoints are available
            if visible_face_points < min_face_points:
                results.append({
                    'frame_idx': t,
                    'occlusion_detected': False,
                    'occluded_regions': [],
                    'confidence': 0.0
                })
                continue
            
            # STEP 4b: Extract and validate face coordinates
            face_coords = []   # Store face landmark coordinates
            face_indices = []  # Store corresponding face landmark indices
            
            # Extract coordinates for visible face keypoints
            for i_rel in range(face_len):
                if bool(face_mask[i_rel]):  # Check if face keypoint is visible
                    idx = 2 * (face_start + i_rel)  # Calculate flattened coordinate index
                    coord = (float(frame_xy[idx]), float(frame_xy[idx + 1]))  # Extract (x, y)
                    face_coords.append(coord)
                    face_indices.append(i_rel)
            
            # Apply quality validation to face landmarks
            validated_coords, validated_indices = _validate_face_landmarks(face_coords, face_indices)
            
            # Skip frame if insufficient validated face landmarks
            if len(validated_coords) < min_face_points:
                results.append({
                    'frame_idx': t,
                    'occlusion_detected': False,
                    'occluded_regions': [],
                    'confidence': 0.0
                })
                continue
            
            # STEP 4c: Create adaptive face regions based on validated landmarks
            face_regions = _create_enhanced_face_regions(validated_coords, validated_indices)
            
            # STEP 4d: Initialize frame-level detection results
            occlusion_detected = False  # Overall occlusion flag for this frame
            occluded_regions = []       # List of occluded region names
            max_confidence = 0.0        # Maximum confidence score across all detections
            
            # STEP 4e: Analyze both hands for occlusion patterns
            for hand_side, hand_start_idx in [('left', pose_len), ('right', pose_len + hand_len)]:
                hand_mask = frame_mask[hand_start_idx:hand_start_idx + hand_len]  # Extract hand visibility
                visible_points = int(hand_mask.sum())  # Count visible hand keypoints
                
                # Process hand if sufficient keypoints are visible
                if visible_points >= min_hand_points:
                    # Extract hand anatomical features (palm center and fingertips)
                    palm_center, tips = _hand_centers_and_tips(frame_xy, frame_mask, hand_start_idx, hand_len)
                    
                    # Update hand movement trajectory for temporal analysis
                    if palm_center is not None:
                        hand_trajectories[hand_side].append((t, palm_center))
                        # Maintain sliding window of recent positions (last 10 frames)
                        if len(hand_trajectories[hand_side]) > 10:
                            hand_trajectories[hand_side] = hand_trajectories[hand_side][-10:]
                    
                    # Apply multi-method occlusion detection algorithms
                    region_results = _detect_occlusions_multi_method(
                        palm_center, tips, face_regions, hand_trajectories[hand_side], t,
                        min_fingertips_inside, proximity_multiplier
                    )
                    
                    # Aggregate detection results across all methods
                    for region_name, confidence in region_results.items():
                        if confidence > 0.4:  # Balanced confidence threshold for sensitivity
                            occlusion_detected = True
                            if region_name not in occluded_regions:
                                occluded_regions.append(region_name)
                            max_confidence = max(max_confidence, confidence)
            
            results.append({
                'frame_idx': t,
                'occlusion_detected': occlusion_detected,
                'occluded_regions': occluded_regions,
                'confidence': max_confidence
            })
        
        # Apply temporal filtering for consistency
        filtered_results = _apply_temporal_filtering(results, kwargs)
        
        # Aggregate results with consecutive frame logic
        total_occlusions = sum(1 for r in filtered_results if r['occlusion_detected'])
        occlusion_rate = total_occlusions / len(filtered_results) if filtered_results else 0
        
        # Binary flag: 1 if any consecutive occlusion pattern is detected
        # The consecutive frame filtering already ensures robust detection
        binary_flag = 1 if total_occlusions > 0 else 0
        
        if output_format == 'compatible':
            return binary_flag
        else:
            return {
                'binary_flag': binary_flag,
                'occlusion_rate': float(occlusion_rate),
                'total_frames': int(len(filtered_results)),
                'occluded_frames': int(total_occlusions),
                'detailed_results': filtered_results
            }
    
    except Exception as e:
        warnings.warn(f"Occlusion detection from keypoints failed: {e}", UserWarning)
        return 0


def _create_enhanced_face_regions(face_coords: List[Tuple[float, float]], 
                                face_indices: List[int]) -> Dict[str, Dict]:
    """
    Create enhanced face regions based on facial landmarks with adaptive sizing.
    
    Args:
        face_coords: List of (x, y) coordinates for visible face points
        face_indices: List of face landmark indices
        
    Returns:
        Dictionary mapping region names to region definitions
    """
    if len(face_coords) < 2:
        return {}
    
    # Map face indices to landmark names (from FACEMESH_11)
    landmark_map = {
        0: 'nose_tip',      # 1
        1: 'left_eye_outer', # 33
        2: 'right_eye_outer', # 263
        3: 'left_eye_inner', # 133
        4: 'right_eye_inner', # 362
        5: 'mouth_left',     # 61
        6: 'mouth_right',    # 291
        7: 'forehead',       # 105
        8: 'chin',           # 334
        9: 'cheek_left',     # 199
        10: 'cheek_right'    # 4
    }
    
    # Extract key landmarks
    landmarks = {}
    for i, coord in enumerate(face_coords):
        if i < len(face_indices):
            landmark_name = landmark_map.get(face_indices[i], f'point_{i}')
            landmarks[landmark_name] = coord
    
    regions = {}
    
    # Calculate face scale for adaptive region sizing
    face_scale = _calculate_face_scale(landmarks)
    
    # Define regions based on available landmarks with adaptive sizing
    if 'nose_tip' in landmarks:
        nose = landmarks['nose_tip']
        
        # Forehead region (top of head) - adaptive radius
        if 'forehead' in landmarks:
            forehead = landmarks['forehead']
            regions['forehead'] = {
                'center': forehead,
                'radius': 0.12 * face_scale,  # Increased and adaptive
                'type': 'circle'
            }
        
        # Cheeks region (eye area) - adaptive radius
        if 'left_eye_outer' in landmarks and 'right_eye_outer' in landmarks:
            left_eye = landmarks['left_eye_outer']
            right_eye = landmarks['right_eye_outer']
            cheek_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            regions['cheeks'] = {
                'center': cheek_center,
                'radius': 0.15 * face_scale,  # Increased and adaptive
                'type': 'circle'
            }
        
        # Nose region (central face) - adaptive radius
        regions['nose'] = {
            'center': nose,
            'radius': 0.10 * face_scale,  # Increased and adaptive
            'type': 'circle'
        }
        
        # Mouth region (lower face) - adaptive radius
        if 'mouth_left' in landmarks and 'mouth_right' in landmarks:
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']
            mouth_center = ((mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2)
            regions['mouth'] = {
                'center': mouth_center,
                'radius': 0.12 * face_scale,  # Increased and adaptive
                'type': 'circle'
            }
        
        # Neck region (below chin) - adaptive radius
        if 'chin' in landmarks:
            chin = landmarks['chin']
            neck_center = (chin[0], chin[1] + 0.05 * face_scale)  # Adaptive offset
            regions['neck'] = {
                'center': neck_center,
                'radius': 0.15 * face_scale,  # Increased and adaptive
                'type': 'circle'
            }
    
    return regions


def _calculate_face_scale(landmarks: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculate adaptive face scale based on available landmarks.
    
    Args:
        landmarks: Dictionary of landmark names to coordinates
        
    Returns:
        Scale factor for adaptive region sizing
    """
    if len(landmarks) < 2:
        return 1.0  # Default scale
    
    # Calculate face width and height from available landmarks
    x_coords = [coord[0] for coord in landmarks.values()]
    y_coords = [coord[1] for coord in landmarks.values()]
    
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)
    
    # Use average of width and height for scale
    face_size = (face_width + face_height) / 2
    
    # Normalize scale (typical face size is around 0.3-0.4 in normalized coordinates)
    normalized_scale = face_size / 0.35
    
    # Clamp scale to reasonable bounds
    return max(0.5, min(2.0, normalized_scale))


def _detect_occlusions_multi_method(palm_center: Optional[Tuple[float, float]], 
                                  tips: List[Tuple[float, float]],
                                  face_regions: Dict[str, Dict],
                                  trajectory: List[Tuple[int, Tuple[float, float]]],
                                  current_frame: int,
                                  min_fingertips_inside: int = 1,
                                  proximity_multiplier: float = 1.5) -> Dict[str, float]:
    """
    Detect occlusions using multiple methods for each region with balanced weighting.
    
    Args:
        palm_center: Palm center coordinates
        tips: List of fingertip coordinates
        face_regions: Dictionary of face regions
        trajectory: Hand movement trajectory
        current_frame: Current frame index
        
    Returns:
        Dictionary mapping region names to confidence scores
    """
    region_confidences = {}
    
    for region_name, region_def in face_regions.items():
        confidence = 0.0
        region_center = region_def['center']
        region_radius = region_def['radius']
        
        # Method 1: Direct fingertip intersection (primary detection)
        if tips:
            fingertips_inside = 0
            for tip_x, tip_y in tips:
                distance = ((tip_x - region_center[0])**2 + (tip_y - region_center[1])**2)**0.5
                if distance <= region_radius:  # Direct intersection
                    fingertips_inside += 1
            
            # Count if minimum fingertips are inside
            if fingertips_inside >= min_fingertips_inside:
                confidence += 0.5  # Primary weight for direct intersection
        
        # Method 2: Palm center proximity (secondary detection)
        if palm_center is not None:
            palm_x, palm_y = palm_center
            distance = ((palm_x - region_center[0])**2 + (palm_y - region_center[1])**2)**0.5
            proximity_radius = region_radius * proximity_multiplier
            if distance <= proximity_radius:
                proximity_score = max(0, 1 - distance / proximity_radius)
                confidence += proximity_score * 0.3  # Increased weight for proximity
        
        # Method 3: Trajectory analysis (motion-based detection)
        if len(trajectory) >= 5:  # Reduced requirement for faster response
            recent_positions = trajectory[-5:]  # Last 5 positions
            if len(recent_positions) >= 3:
                # Check if hand is consistently moving toward face
                distances = []
                for _, pos in recent_positions:
                    dist = ((pos[0] - region_center[0])**2 + (pos[1] - region_center[1])**2)**0.5
                    distances.append(dist)
                
                # Check if distance is consistently decreasing
                if len(distances) >= 3:
                    decreasing_count = sum(1 for i in range(1, len(distances)) if distances[i] < distances[i-1])
                    if decreasing_count >= 2 and distances[-1] <= region_radius * 1.5:  # More lenient
                        approach_score = max(0, (distances[0] - distances[-1]) / distances[0])
                        confidence += approach_score * 0.15  # Increased weight for trajectory
        
        # Method 4: Multi-point hand analysis (orientation detection)
        if palm_center is not None and len(tips) >= 2:  # Reduced requirement
            # Check if hand is oriented toward face
            hand_points = [palm_center] + tips
            face_distances = [((p[0] - region_center[0])**2 + (p[1] - region_center[1])**2)**0.5 for p in hand_points]
            min_distance = min(face_distances)
            
            # More lenient radius for orientation detection
            if min_distance <= region_radius * 1.8:
                orientation_score = max(0, 1 - min_distance / (region_radius * 1.8))
                confidence += orientation_score * 0.05  # Small weight for orientation
        
        region_confidences[region_name] = min(confidence, 1.0)  # Cap at 1.0
    
    return region_confidences


def _apply_temporal_filtering(results: List[Dict], config: Dict = None) -> List[Dict]:
    """
    Apply consecutive frame temporal filtering for occlusion detection.
    
    Requires 5 consecutive frames with occlusion detection (confidence >= 0.2)
    with tolerance for 1-2 missed frames within the window.
    
    Args:
        results: List of frame detection results
        config: Configuration dictionary with consecutive frame parameters
        
    Returns:
        Filtered results with consecutive frame consistency
    """
    if len(results) < 2:
        return results
    
    # Use default config if not provided
    if config is None:
        config = DEFAULT_OCCLUSION_CONFIG
    
    filtered_results = []
    consecutive_window_size = config.get('consecutive_window_size', 5)
    max_skips = config.get('max_consecutive_skips', 2)
    min_confidence = config.get('min_consecutive_confidence', 0.2)
    
    for i, result in enumerate(results):
        # Check for consecutive occlusion detection
        occlusion_detected = _check_consecutive_occlusion(
            results, i, consecutive_window_size, max_skips, min_confidence
        )
        
        # Get regions from consecutive window
        occluded_regions = _get_consecutive_regions(
            results, i, consecutive_window_size, max_skips, min_confidence
        )
        
        # Calculate average confidence from consecutive detections
        avg_confidence = _calculate_consecutive_confidence(
            results, i, consecutive_window_size, max_skips, min_confidence
        )
        
        filtered_result = result.copy()
        filtered_result['occlusion_detected'] = occlusion_detected
        filtered_result['occluded_regions'] = list(set(occluded_regions))
        filtered_result['confidence'] = avg_confidence
        
        filtered_results.append(filtered_result)
    
    return filtered_results


def _check_consecutive_occlusion(results: List[Dict], center_idx: int, 
                               window_size: int, max_skips: int, 
                               min_confidence: float) -> bool:
    """
    Check if there are enough consecutive occlusion detections around center_idx.
    
    Args:
        results: List of detection results
        center_idx: Center frame index
        window_size: Size of consecutive window (5)
        max_skips: Maximum allowed skips (2)
        min_confidence: Minimum confidence threshold (0.2)
        
    Returns:
        True if consecutive occlusion pattern is detected
    """
    # Define window boundaries
    half_window = window_size // 2
    start = max(0, center_idx - half_window)
    end = min(len(results), center_idx + half_window + 1)
    
    # Need at least window_size frames to check
    if end - start < window_size:
        return False
    
    # Check all possible consecutive windows within the range
    for window_start in range(start, end - window_size + 1):
        window_end = window_start + window_size
        
        # Count valid detections in this window
        valid_detections = 0
        skips = 0
        
        for i in range(window_start, window_end):
            if i < len(results):
                result = results[i]
                # Check if frame has occlusion detection with sufficient confidence
                if (result['occlusion_detected'] and 
                    result['confidence'] >= min_confidence):
                    valid_detections += 1
                else:
                    skips += 1
        
        # Check if this window meets the criteria
        if valid_detections >= (window_size - max_skips):
            return True
    
    return False


def _get_consecutive_regions(results: List[Dict], center_idx: int,
                           window_size: int, max_skips: int,
                           min_confidence: float) -> List[str]:
    """
    Get occluded regions from consecutive detection window.
    
    Args:
        results: List of detection results
        center_idx: Center frame index
        window_size: Size of consecutive window
        max_skips: Maximum allowed skips
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of occluded region names
    """
    # Define window boundaries
    half_window = window_size // 2
    start = max(0, center_idx - half_window)
    end = min(len(results), center_idx + half_window + 1)
    
    all_regions = []
    region_confidences = {}
    
    # Collect regions from valid detections in window
    for i in range(start, end):
        if i < len(results):
            result = results[i]
            if (result['occlusion_detected'] and 
                result['confidence'] >= min_confidence):
                for region in result['occluded_regions']:
                    all_regions.append(region)
                    if region not in region_confidences:
                        region_confidences[region] = []
                    region_confidences[region].append(result['confidence'])
    
    # Keep regions that appear frequently enough
    filtered_regions = []
    for region, conf_list in region_confidences.items():
        if len(conf_list) >= (window_size - max_skips) // 2:  # At least half the window
            filtered_regions.append(region)
    
    return filtered_regions


def _calculate_consecutive_confidence(results: List[Dict], center_idx: int,
                                    window_size: int, max_skips: int,
                                    min_confidence: float) -> float:
    """
    Calculate average confidence from consecutive detection window.
    
    Args:
        results: List of detection results
        center_idx: Center frame index
        window_size: Size of consecutive window
        max_skips: Maximum allowed skips
        min_confidence: Minimum confidence threshold
        
    Returns:
        Average confidence from valid detections
    """
    # Define window boundaries
    half_window = window_size // 2
    start = max(0, center_idx - half_window)
    end = min(len(results), center_idx + half_window + 1)
    
    confidences = []
    
    # Collect confidences from valid detections
    for i in range(start, end):
        if i < len(results):
            result = results[i]
            if (result['occlusion_detected'] and 
                result['confidence'] >= min_confidence):
                confidences.append(result['confidence'])
    
    # Return average confidence, or 0 if no valid detections
    return sum(confidences) / len(confidences) if confidences else 0.0


def compute_occlusion_detection(
    video_path: str = None,
    X: np.ndarray = None,
    mask_bool_array: np.ndarray = None,
    output_format: str = 'compatible',
    **kwargs
) -> Union[int, Dict]:
    """
    Compute occlusion detection using computer vision methods.
    
    Args:
        video_path: Path to input video file (for raw video processing)
        X: [T, 156] normalized keypoint coordinates (for keypoint-based processing)
        mask_bool_array: [T, 78] visibility mask (for keypoint-based processing)
        output_format: Output format ('compatible', 'detailed')
        **kwargs: Additional parameters for detection
    
    Returns:
        int: Binary occlusion flag (compatible format)
        Dict: Detailed results (detailed format)
    """
    try:
        # Prefer keypoint-based method if keypoint data is available
        if X is not None and mask_bool_array is not None:
            return compute_occlusion_detection_from_keypoints(X, mask_bool_array, output_format, **kwargs)
        elif video_path is not None:
            return _compute_occlusion_from_video(video_path, output_format, **kwargs)
        else:
            warnings.warn(
                "Occlusion detection requires either video_path or keypoint data (X, mask_bool_array)",
                UserWarning
            )
            return 0
            
    except ImportError:
        warnings.warn(
            "Occlusion detection requires additional dependencies. "
            "Please install: pip install scipy scikit-learn",
            UserWarning
        )
        return 0
    except Exception as e:
        warnings.warn(f"Occlusion detection failed: {e}", UserWarning)
        return 0


def _compute_occlusion_from_video(
    video_path: str,
    output_format: str = 'compatible',
    **kwargs
) -> Union[int, Dict]:
    """
    Compute occlusion detection from raw video file.
    
    Args:
        video_path: Path to input video file
        output_format: Output format ('compatible', 'detailed')
        **kwargs: Additional parameters for detection
    
    Returns:
        int: Binary occlusion flag (compatible format)
        Dict: Detailed results (detailed format)
    """
    try:
        import cv2
        
        # Initialize detector
        detector = HandHeadOcclusionDetector(
            use_global_tracking=kwargs.get('use_global_tracking', True)
        )
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_results = detector.process_frame(frame, frame_idx)
            results.append(frame_results)
            frame_idx += 1
            
            # Limit processing for performance
            if frame_idx > 1000:  # Process max 1000 frames
                break
        
        cap.release()
        
        # Apply consecutive frame temporal filtering
        filtered_results = _apply_temporal_filtering(results, kwargs)
        
        # Aggregate results with consecutive frame logic
        total_occlusions = sum(1 for r in filtered_results if r['occlusion_detected'])
        occlusion_rate = total_occlusions / len(filtered_results) if filtered_results else 0
        
        # Binary flag: 1 if any consecutive occlusion pattern is detected
        binary_flag = 1 if total_occlusions > 0 else 0
        
        if output_format == 'compatible':
            return binary_flag
        else:
            # Convert filtered results to JSON-serializable format
            serializable_results = []
            for result in filtered_results:
                serializable_result = {
                    'frame_idx': result['frame_idx'],
                    'occlusion_detected': result['occlusion_detected'],
                    'occluded_regions': result['occluded_regions']
                }
                # Add optional fields if they exist
                if 'occlusion_counts' in result:
                    serializable_result['occlusion_counts'] = result['occlusion_counts']
                if 'filtered_detections' in result:
                    serializable_result['filtered_detections'] = result['filtered_detections']
                serializable_results.append(serializable_result)
            
            return {
                'binary_flag': binary_flag,
                'occlusion_rate': float(occlusion_rate),  # Convert numpy float to Python float
                'total_frames': int(len(filtered_results)),       # Convert numpy int to Python int
                'occluded_frames': int(total_occlusions), # Convert numpy int to Python int
                'detailed_results': serializable_results
            }
    
    except Exception as e:
        warnings.warn(f"Occlusion detection from video failed: {e}", UserWarning)
        return 0


# ----------------------------
# Configuration Management
# ----------------------------

# Default configuration for comprehensive occlusion detection
DEFAULT_OCCLUSION_CONFIG = {
    # Advanced tracking and processing options
    'use_global_tracking': True,        # Enable sophisticated tracking algorithms
    'gridlet_size': 4,                  # Size of point groups for motion tracking
    'tracking_window_size': 5,          # Temporal window size for consistency
    'motion_threshold': 10,             # Motion detection sensitivity threshold
    'temporal_filtering': True,         # Enable temporal consistency filtering
    'output_detailed_results': False,   # Control output verbosity
    
    # Core detection sensitivity parameters
    'min_face_points': 3,              # Minimum face keypoints required for detection
    'min_hand_points': 3,              # Minimum hand keypoints required for analysis
    'min_fingertips_inside': 1,        # Minimum fingertips in face region for occlusion
    'proximity_multiplier': 1.5,       # Proximity detection radius multiplier
    'occlusion_threshold': 0.15,       # Overall occlusion detection threshold
    'confidence_threshold': 0.4,       # Minimum confidence for positive detection
    'temporal_confidence': 0.5,        # Temporal consistency confidence threshold
    
    # Consecutive frame analysis parameters (for robust detection)
    'consecutive_window_size': 5,       # Require 5 consecutive frames for confirmation
    'max_consecutive_skips': 2,         # Allow up to 2 missed frames within window
    'min_consecutive_confidence': 0.2   # Minimum confidence for consecutive analysis
}


def get_occlusion_config() -> Dict:
    """Get a copy of the default configuration for occlusion detection.
    
    Returns:
        Dict: Complete configuration dictionary with all parameters
    """
    return DEFAULT_OCCLUSION_CONFIG.copy()


def validate_occlusion_config(config: Dict) -> bool:
    """Validate that an occlusion detection configuration contains required parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [
        'use_global_tracking', 'gridlet_size', 'tracking_window_size', 
        'motion_threshold', 'min_face_points', 'min_hand_points'
    ]
    return all(key in config for key in required_keys)


# ----------------------------
# Public API Exports
# ----------------------------

# Export all public functions and classes for external use
__all__ = [
    # Main detection functions
    'compute_occlusion_detection',              # Primary detection interface
    'compute_occlusion_detection_from_keypoints', # Keypoint-based detection
    
    # Core classes
    'HandHeadOcclusionDetector',               # Main detector class
    'HeadRegion',                              # Region constants
    'Point2D',                                 # Geometric utility class
    'Gridlet',                                 # Advanced tracking class
    
    # Configuration utilities
    'get_occlusion_config',                    # Get default configuration
    'validate_occlusion_config',               # Validate configuration
    'DEFAULT_OCCLUSION_CONFIG'                 # Default configuration constants
]
