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

from dataclasses import dataclass
import numpy as np
import mediapipe as mp

# ----------------------------
# Keypoint Configuration
# ----------------------------
POSE_UPPER_25 = list(range(0, 11)) + list(range(11, 17)) + list(range(17, 23)) + [23, 24]
N_HAND = 21
FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]


# ----------------------------
# MediaPipe Model Management
# ----------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


@dataclass
class MPModels:
    """Container for MediaPipe models."""
    seg: any  # Selfie segmentation model
    hol: any  # Holistic model


def create_models(seg_model=1, detection_conf=0.5, tracking_conf=0.5):
    """Initialize MediaPipe models for keypoint extraction.
    
    Args:
        seg_model: Selfie segmentation model selection (0 or 1)
        detection_conf: Minimum detection confidence threshold
        tracking_conf: Minimum tracking confidence threshold
        
    Returns:
        MPModels: Container with initialized models
    """
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=seg_model)
    hol = mp.solutions.holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    )
    return MPModels(seg=seg, hol=hol)


def close_models(models: MPModels):
    """Close MediaPipe models to free resources."""
    models.seg.close()
    models.hol.close()


def xy_from_landmark(lm, w, h):
    """Convert MediaPipe landmark to normalized coordinates.
    
    Args:
        lm: MediaPipe landmark object
        w: Image width (unused, kept for compatibility)
        h: Image height (unused, kept for compatibility)
        
    Returns:
        Tuple of (x, y) coordinates clamped to [0, 1]
    """
    x = float(lm.x)
    y = float(lm.y)
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def _lerp(a, b, t):
    """Linear interpolation between two values."""
    return a + (b - a) * t


def interpolate_gaps(X, mask, max_gap=5):
    """Interpolate missing keypoints in temporal sequences.
    
    Fills gaps in keypoint sequences using linear interpolation.
    Only interpolates gaps within the specified maximum length.
    
    Args:
        X: Keypoint coordinates [T, 156] as float32
        mask: Keypoint visibility mask [T, 78] as bool
        max_gap: Maximum gap length to interpolate
        
    Returns:
        Tuple of (X_filled, mask_filled) with interpolated values
    """
    T = X.shape[0]
    K = mask.shape[1]
    X_out = X.copy()
    mask_out = mask.copy()

    for k in range(K):
        xi = 2 * k
        yi = 2 * k + 1
        valid_idxs = np.where(mask[:, k])[0]
        if len(valid_idxs) == 0:
            continue

        prev = valid_idxs[0]
        for vi in valid_idxs[1:]:
            if vi == prev + 1:
                prev = vi
                continue
            gap_start = prev + 1
            gap_end = vi - 1
            gap_len = gap_end - gap_start + 1
            if 1 <= gap_len <= max_gap:
                x0, y0 = X_out[prev, xi], X_out[prev, yi]
                x1, y1 = X_out[vi, xi], X_out[vi, yi]
                for t_idx, t_rel in enumerate(range(gap_start, gap_end + 1), start=1):
                    t = t_idx / (gap_len + 1)
                    X_out[t_rel, xi] = _lerp(x0, x1, t)
                    X_out[t_rel, yi] = _lerp(y0, y1, t)
                    mask_out[t_rel, k] = True
            prev = vi

    return X_out, mask_out


def extract_keypoints_from_frame(img_rgb, models: MPModels, conf_thresh=0.5):
    """Extract keypoints from a single RGB frame.
    
    Extracts pose, hand, and face landmarks using MediaPipe Holistic.
    Returns normalized coordinates and visibility mask.
    
    Args:
        img_rgb: RGB frame (H, W, 3)
        models: Initialized MediaPipe models
        conf_thresh: Confidence threshold for keypoint detection
        
    Returns:
        vec156: Keypoint coordinates [156] as float32 in [0,1]
        mask78: Visibility mask [78] as bool
    """
    H, W, _ = img_rgb.shape
    res = models.hol.process(img_rgb)
    coords = []
    vis_mask = []

    # ---- POSE (25) ----
    pose_present = res.pose_landmarks is not None
    pose_points = [ (0.0, 0.0) ] * len(POSE_UPPER_25)
    pose_mask = [False] * len(POSE_UPPER_25)
    if pose_present:
        lms = res.pose_landmarks.landmark
        for i, idx in enumerate(POSE_UPPER_25):
            lm = lms[idx]
            # Clamp to [0,1] like hands/face for consistency
            x, y = xy_from_landmark(lm, W, H)
            pose_points[i] = (x, y)
            pose_mask[i] = (getattr(lm, "visibility", 0.0) or 0.0) >= conf_thresh

    # ---- HANDS (21 + 21) ----
    def hand_block(hand_lms):
        pts = [ (0.0, 0.0) ] * N_HAND
        m = [False] * N_HAND
        if hand_lms is not None:
            for i, lm in enumerate(hand_lms.landmark[:N_HAND]):
                x, y = xy_from_landmark(lm, W, H)
                pts[i] = (x, y)
                m[i] = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)
        return pts, m

    left_pts, left_mask = hand_block(res.left_hand_landmarks)
    right_pts, right_mask = hand_block(res.right_hand_landmarks)

    # ---- FACE (11) ----
    face_points = [ (0.0, 0.0) ] * len(FACEMESH_11)
    face_mask = [False] * len(FACEMESH_11)
    if res.face_landmarks is not None:
        fl = res.face_landmarks.landmark
        for i, idx in enumerate(FACEMESH_11):
            lm = fl[idx]
            x, y = xy_from_landmark(lm, W, H)
            face_points[i] = (x, y)
            face_mask[i] = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)

    # Concatenate in the requested order
    for block_pts, block_mask in [
        (pose_points, pose_mask),
        (left_pts, left_mask),
        (right_pts, right_mask),
        (face_points, face_mask),
    ]:
        for (x, y) in block_pts:
            coords.extend([x, y])
        vis_mask.extend(block_mask)

    return np.array(coords, dtype=np.float32), np.array(vis_mask, dtype=bool)