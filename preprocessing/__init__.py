"""
Preprocessing module for Filipino Sign Language Recognition.

This module provides preprocessing functionality for converting raw sign language
videos into training-ready features and keypoints for both Transformer and InceptionV3+GRU models.

Key Components:
- Video preprocessing with MediaPipe keypoint extraction
- InceptionV3 feature extraction for CNN-based models
- Multi-process parallelization for faster processing
- Occlusion detection and validation utilities
- File management and dataset validation tools

Available Processors:
- Single video preprocessing with configurable outputs
- Multi-process batch processing with 30-50x speedup
- Feature extraction utilities for both keypoints and CNN features

Usage:
    from preprocessing import preprocess_video, validate_dataset
    
    # Process single video
    python -m preprocessing.core.preprocess input.mp4 output_dir --write-keypoints --write-iv3-features
    
    # Multi-process batch processing
    python -m preprocessing.core.multi_preprocess input_dir output_dir --write-keypoints --write-iv3-features
    
    # Validate processed dataset
    python -m preprocessing.utils.validate_npz processed_dir
"""

# Core preprocessing functionality
# Note: preprocess.py doesn't have a main function, it's designed to be run as a script
# from .core.preprocess import main as preprocess_main  # This doesn't exist

# Conditional imports to avoid errors when dependencies are missing
try:
    from .core.multi_preprocess import main as multi_preprocess_main
    MULTI_PREPROCESS_AVAILABLE = True
except ImportError:
    MULTI_PREPROCESS_AVAILABLE = False
    multi_preprocess_main = None

# Feature extractors
try:
    from .extractors.iv3_features import extract_iv3_features
    IV3_FEATURES_AVAILABLE = True
except ImportError:
    IV3_FEATURES_AVAILABLE = False
    extract_iv3_features = None

try:
    from .extractors.keypoints_features import (
        extract_keypoints_from_frame,
        interpolate_gaps,
        create_models,
        close_models
    )
    KEYPOINTS_FEATURES_AVAILABLE = True
except ImportError:
    KEYPOINTS_FEATURES_AVAILABLE = False
    extract_keypoints_from_frame = None
    interpolate_gaps = None
    create_models = None
    close_models = None

# Utilities
try:
    from .utils.validate_npz import validate_dataset
    VALIDATE_NPZ_AVAILABLE = True
except ImportError:
    VALIDATE_NPZ_AVAILABLE = False
    validate_dataset = None

try:
    from .utils.rename_clips import rename_clips
    RENAME_CLIPS_AVAILABLE = True
except ImportError:
    RENAME_CLIPS_AVAILABLE = False
    rename_clips = None

# Occlusion detection
try:
    from .core.occlusion_detection import (
        compute_occlusion_flag_from_keypoints,
        compute_occlusion_detection,
        get_occlusion_config,
        validate_occlusion_config,
        DEFAULT_OCCLUSION_CONFIG
    )
    OCCLUSION_DETECTION_AVAILABLE = True
except ImportError:
    OCCLUSION_DETECTION_AVAILABLE = False
    compute_occlusion_flag_from_keypoints = None
    compute_occlusion_detection = None
    get_occlusion_config = None
    validate_occlusion_config = None
    DEFAULT_OCCLUSION_CONFIG = None

# Build __all__ list dynamically based on what's available
__all__ = []

if MULTI_PREPROCESS_AVAILABLE:
    __all__.append('multi_preprocess_main')

if IV3_FEATURES_AVAILABLE:
    __all__.append('extract_iv3_features')

if KEYPOINTS_FEATURES_AVAILABLE:
    __all__.extend(['extract_keypoints_from_frame', 'interpolate_gaps', 'create_models', 'close_models'])

if VALIDATE_NPZ_AVAILABLE:
    __all__.append('validate_dataset')

if RENAME_CLIPS_AVAILABLE:
    __all__.append('rename_clips')

if OCCLUSION_DETECTION_AVAILABLE:
    __all__.extend([
        'compute_occlusion_flag_from_keypoints',
        'compute_occlusion_detection',
        'get_occlusion_config',
        'validate_occlusion_config',
        'DEFAULT_OCCLUSION_CONFIG'
    ])
