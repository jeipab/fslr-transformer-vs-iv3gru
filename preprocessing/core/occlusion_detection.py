"""
Occlusion detection utilities.

Provides functions to compute a clip-level occlusion flag.

Two complementary methods are available:
- Keypoints-based hand–head occlusion (preferred): approximate the head with a
  robust face ellipse using face keypoints and detect whether either hand (palm
  center or fingertips) enters this ellipse. Aggregates per-frame events into a
  clip-level flag.
- Visibility-based fallback: use overall keypoint visibility rate with
  proportion and run-length thresholds (legacy).
"""

from __future__ import annotations

import numpy as np

# Note: we avoid importing heavy vision libs here; rely only on keypoint geometry.


def _point_in_ellipse(px: float, py: float, cx: float, cy: float, ax: float, by: float) -> bool:
    """Check if point (px, py) lies inside or on the ellipse centered at (cx, cy)
    with semi-axes (ax, by). Coordinates are normalized to [0,1].
    """
    if ax <= 0.0 or by <= 0.0:
        return False
    dx = (px - cx) / ax
    dy = (py - cy) / by
    return (dx * dx + dy * dy) <= 1.0


def _compute_face_ellipse(frame_xy: np.ndarray, frame_mask: np.ndarray, face_start: int, face_len: int, face_min_points: int, face_scale: float) -> tuple[float, float, float, float] | None:
    """Compute a robust face ellipse (cx, cy, ax, by) from visible face keypoints.

    Returns None if not enough points are visible or geometry is degenerate.
    """
    face_mask = frame_mask[face_start : face_start + face_len]
    if int(face_mask.sum()) < int(face_min_points):
        return None
    # Collect visible face points
    coords = []
    for i_rel in range(face_len):
        if not bool(face_mask[i_rel]):
            continue
        idx = 2 * (face_start + i_rel)
        coords.append((float(frame_xy[idx]), float(frame_xy[idx + 1])))
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))
    # Center and semi-axes from bounding box, scaled
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    ax = max((x_max - x_min) * 0.5 * float(face_scale), 1e-3)
    by = max((y_max - y_min) * 0.5 * float(face_scale), 1e-3)
    return (cx, cy, ax, by)


def _hand_centers_and_tips(frame_xy: np.ndarray, frame_mask: np.ndarray, hand_start: int, hand_len: int) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    """Return (palm_center, fingertip_points) for one hand.

    Uses MediaPipe Hands indexing (21 landmarks). Palm center is the mean of MCP
    joints (5, 9, 13, 17) when available, else the wrist (0) if visible.
    Fingertips are indices [4, 8, 12, 16, 20] when visible.
    """
    # Fingertips relative indices in a 21-point hand
    fingertip_rel = [4, 8, 12, 16, 20]
    mcp_rel = [5, 9, 13, 17]
    # Palm center
    mcp_coords = []
    for r in mcp_rel:
        if hand_len <= r:
            continue
        if bool(frame_mask[hand_start + r]):
            idx = 2 * (hand_start + r)
            mcp_coords.append((float(frame_xy[idx]), float(frame_xy[idx + 1])))
    palm_center: tuple[float, float] | None
    if len(mcp_coords) >= 2:
        mx = sum(p[0] for p in mcp_coords) / float(len(mcp_coords))
        my = sum(p[1] for p in mcp_coords) / float(len(mcp_coords))
        palm_center = (mx, my)
    else:
        # fallback to wrist if visible
        if bool(frame_mask[hand_start + 0]):
            idx0 = 2 * (hand_start + 0)
            palm_center = (float(frame_xy[idx0]), float(frame_xy[idx0 + 1]))
        else:
            palm_center = None
    # Fingertips
    tips: list[tuple[float, float]] = []
    for r in fingertip_rel:
        if hand_len <= r:
            continue
        if bool(frame_mask[hand_start + r]):
            idx = 2 * (hand_start + r)
            tips.append((float(frame_xy[idx]), float(frame_xy[idx + 1])))
    return palm_center, tips


def compute_hand_head_occlusion_frames(
    X: np.ndarray,
    mask: np.ndarray,
    *,
    pose_len: int = 25,
    hand_len: int = 21,
    face_len: int = 11,
    face_min_points: int = 5,
    face_scale: float = 1.3,
    min_hand_points: int = 4,
    min_fingertips_inside: int = 1,
    near_face_multiplier: float = 1.2,
) -> np.ndarray:
    """Detect per-frame hand–head occlusion events from keypoints.

    Args:
        X: [T, 156] normalized coordinates.
        mask: [T, 78] visibility mask (True = visible).
        pose_len, hand_len, face_len: block sizes in the keypoint layout order
            pose25, left_hand21, right_hand21, face11.
        face_min_points: required visible face points to construct ellipse.
        face_scale: inflate ellipse axes to cover forehead/cheeks.
        min_hand_points: min visible points to consider a hand present.
        min_fingertips_inside: fingertips within ellipse to count as occlusion.
        near_face_multiplier: additional proximity condition for palm center.

    Returns:
        np.ndarray [T] of booleans: True if frame is occluded (hand over head).
    """
    if X is None or mask is None:
        return np.zeros((0,), dtype=bool)
    if X.ndim != 2 or X.shape[1] != 156 or mask.ndim != 2 or mask.shape[1] != 78:
        return np.zeros((0,), dtype=bool)
    T = X.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=bool)

    occ = np.zeros((T,), dtype=bool)
    left_start = pose_len
    right_start = pose_len + hand_len
    face_start = pose_len + hand_len + hand_len

    for t in range(T):
        frame_xy = X[t]
        frame_mask = mask[t]

        face_ellipse = _compute_face_ellipse(
            frame_xy, frame_mask, face_start=face_start, face_len=face_len,
            face_min_points=face_min_points, face_scale=face_scale,
        )
        if face_ellipse is None:
            # Cannot decide via geometry this frame
            continue
        cx, cy, ax, by = face_ellipse

        # Left hand
        l_visible = int(frame_mask[left_start : left_start + hand_len].sum())
        r_visible = int(frame_mask[right_start : right_start + hand_len].sum())

        def hand_inside(hand_start: int, visible_count: int) -> bool:
            if visible_count < int(min_hand_points):
                return False
            palm_center, tips = _hand_centers_and_tips(frame_xy, frame_mask, hand_start, hand_len)
            inside_tip = sum(1 for (x, y) in tips if _point_in_ellipse(x, y, cx, cy, ax, by))
            if inside_tip >= int(min_fingertips_inside):
                return True
            if palm_center is not None:
                px, py = palm_center
                # proximity: within a slightly larger ellipse
                if _point_in_ellipse(px, py, cx, cy, ax * float(near_face_multiplier), by * float(near_face_multiplier)):
                    return True
            return False

        if hand_inside(left_start, l_visible) or hand_inside(right_start, r_visible):
            occ[t] = True

    return occ


def _compute_occlusion_from_mask(
    mask_bool_array: np.ndarray,
    visibility_threshold: float = 0.6,
    frame_prop_threshold: float = 0.4,
    min_consecutive_occ_frames: int = 15,
) -> int:
    """Compute a binary occlusion flag from per-frame keypoint visibility mask.

    Args:
        mask_bool_array: np.ndarray [T, 78] of booleans (True means keypoint visible).
        visibility_threshold: A frame is considered occluded if visible_kp/78 < threshold.
        frame_prop_threshold: Mark clip as occluded if proportion of occluded frames exceeds this.
        min_consecutive_occ_frames: Or if there exists a consecutive run of occluded frames >= this value.

    Returns:
        int: 1 if occluded, else 0.
    """
    try:
        if mask_bool_array is None:
            return 0
        if mask_bool_array.ndim != 2 or mask_bool_array.shape[1] != 78:
            return 0
        T = mask_bool_array.shape[0]
        if T == 0:
            return 0
        visible_frac = mask_bool_array.sum(axis=1) / 78.0  # [T]
        occ_frames = (visible_frac < float(visibility_threshold))  # [T] bool
        prop = float(occ_frames.mean())
        if prop >= float(frame_prop_threshold):
            return 1
        # longest consecutive run
        max_run = 0
        current = 0
        for v in occ_frames:
            if bool(v):
                current += 1
                if current > max_run:
                    max_run = current
            else:
                current = 0
        if max_run >= int(min_consecutive_occ_frames):
            return 1
        return 0
    except Exception:
        return 0


def compute_occlusion_flag_from_keypoints(
    X: np.ndarray,
    mask_bool_array: np.ndarray,
    *,
    frame_prop_threshold: float = 0.4,
    min_consecutive_occ_frames: int = 15,
    visibility_fallback_threshold: float = 0.6,
) -> int:
    """Compute a clip-level occlusion flag using keypoints geometry with fallback.

    1) Detect per-frame hand–head occlusions using `compute_hand_head_occlusion_frames`.
    2) Aggregate by proportion and longest-run thresholds.
    3) If geometry provides no positive frames and/or face not reliably detected,
       fallback to visibility-based method using `visibility_fallback_threshold`.
    """
    try:
        occ_frames = compute_hand_head_occlusion_frames(X, mask_bool_array)
        if occ_frames.size == 0:
            # Fallback
            return _compute_occlusion_from_mask(
                mask_bool_array,
                visibility_threshold=visibility_fallback_threshold,
                frame_prop_threshold=frame_prop_threshold,
                min_consecutive_occ_frames=min_consecutive_occ_frames,
            )
        # Aggregate
        prop = float(occ_frames.mean()) if occ_frames.size else 0.0
        if prop >= float(frame_prop_threshold):
            return 1
        # longest consecutive run
        max_run = 0
        current = 0
        for v in occ_frames:
            if bool(v):
                current += 1
                if current > max_run:
                    max_run = current
            else:
                current = 0
        if max_run >= int(min_consecutive_occ_frames):
            return 1
        # If no geometry occlusion detected at all, still fallback to visibility signal
        if not bool(occ_frames.any()):
            return _compute_occlusion_from_mask(
                mask_bool_array,
                visibility_threshold=visibility_fallback_threshold,
                frame_prop_threshold=frame_prop_threshold,
                min_consecutive_occ_frames=min_consecutive_occ_frames,
            )
        return 0
    except Exception:
        # Conservative fallback
        return _compute_occlusion_from_mask(
            mask_bool_array,
            visibility_threshold=visibility_fallback_threshold,
            frame_prop_threshold=frame_prop_threshold,
            min_consecutive_occ_frames=min_consecutive_occ_frames,
        )


