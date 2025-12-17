"""
Core preprocessing functionality for video processing.

This module contains the main preprocessing pipeline and occlusion detection utilities.
"""

from .preprocess import process_video, process_videos_multiprocess
from .occlusion_detection import (
    compute_occlusion_detection_from_keypoints,
    compute_occlusion_detection,
    get_occlusion_config,
    validate_occlusion_config,
    DEFAULT_OCCLUSION_CONFIG
)

# Alias for backward compatibility
compute_occlusion_flag_from_keypoints = compute_occlusion_detection_from_keypoints

__all__ = [
    'process_video',
    'process_videos_multiprocess',
    'compute_occlusion_flag_from_keypoints',
    'compute_occlusion_detection',
    'get_occlusion_config',
    'validate_occlusion_config',
    'DEFAULT_OCCLUSION_CONFIG'
]

