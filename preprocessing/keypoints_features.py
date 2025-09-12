"""
Keypoint preprocessing utilities using MediaPipe Holistic.

What this provides:
- Landmark index sets and constants used to build the 156-D vector.
- Model lifecycle helpers (`create_models`, `close_models`).
- Per-frame extraction of 78 keypoints → `X` (156 floats) and `mask` (78 bools).
- Small-gap interpolation to fill missing keypoints in time.

Input/Output overview:
- Input image is RGB (H, W, 3), float/uint8 in standard OpenCV format.
- `extract_keypoints_from_frame` returns:
  - `vec156`: np.float32 shape (156,) with values in [0,1].
  - `mask78`: np.bool_ shape (78,) visibility per keypoint.
"""

from dataclasses import dataclass
import numpy as np
import mediapipe as mp

# ----------------------------
# Config: landmark index sets
# ----------------------------
POSE_UPPER_25 = list(range(0, 11)) + list(range(11, 17)) + list(range(17, 23)) + [23, 24]
N_HAND = 21
FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]


# ----------------------------
# MediaPipe wrappers
# ----------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


@dataclass
class MPModels:
    seg: any
    hol: any


def create_models(seg_model=1, detection_conf=0.5, tracking_conf=0.5):
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
    models.seg.close()
    models.hol.close()


def xy_from_landmark(lm, w, h):
    """Convert a MediaPipe landmark to normalized (x, y) in [0, 1]."""
    x = float(lm.x)
    y = float(lm.y)
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def _lerp(a, b, t):
    return a + (b - a) * t


def interpolate_gaps(X, mask, max_gap=5):
    """Linearly interpolate short missing spans in a keypoint sequence.

    Args:
        X: [T,156] float32 keypoint coordinates (x1, y1, x2, y2, ...).
        mask: [T,78] bool visibility mask.
        max_gap: Interpolate only gaps with length in [1, max_gap].

    Returns:
        Tuple (X_filled, mask_filled) with the same shapes.
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
    """Extract 78 keypoints and a visibility mask from one RGB frame.

    Keypoint order: `pose25,left_hand21,right_hand21,face11`.

    Args:
        img_rgb: RGB frame (H, W, 3).
        models: MPModels with initialized holistic and segmentation models.
        conf_thresh: Minimum visibility (pose) to mark keypoint as present.

    Returns:
        vec156: np.float32 (156,) normalized coordinates in [0,1].
        mask78: np.bool_ (78,) True where the keypoint is considered visible.
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