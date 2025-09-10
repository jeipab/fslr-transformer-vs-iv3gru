"""
Occlusion detection utilities.

Provides functions to compute an occlusion flag from a per-frame keypoint
visibility mask.
"""

from __future__ import annotations

import numpy as np


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


