"""
Preprocessing module for Filipino Sign Language Recognition.

Provides preprocessing functionality for converting raw sign language videos
into training-ready features and keypoints.
"""

# Core preprocessing functionality
try:
    from .core.preprocess import process_video, process_videos_multiprocess
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
    process_video = None
    process_videos_multiprocess = None

# Feature extractors
try:
    from .extractors.iv3_features import extract_iv3_features, BatchedInceptionV3Processor
    IV3_FEATURES_AVAILABLE = True
except ImportError:
    IV3_FEATURES_AVAILABLE = False
    extract_iv3_features = None
    BatchedInceptionV3Processor = None

try:
    from .extractors.keypoints_features import (
        extract_keypoints_from_frame,
        interpolate_gaps,
        smooth_keypoints_ema,
        validate_and_clean_keypoints,
        create_models,
        close_models
    )
    KEYPOINTS_FEATURES_AVAILABLE = True
except ImportError:
    KEYPOINTS_FEATURES_AVAILABLE = False
    extract_keypoints_from_frame = None
    interpolate_gaps = None
    smooth_keypoints_ema = None
    validate_and_clean_keypoints = None
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

try:
    from .utils.resize_360p import VideoResizer
    RESIZE_360P_AVAILABLE = True
except ImportError:
    RESIZE_360P_AVAILABLE = False
    VideoResizer = None

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

if PREPROCESS_AVAILABLE:
    __all__.extend(['process_video', 'process_videos_multiprocess'])

if IV3_FEATURES_AVAILABLE:
    __all__.extend(['extract_iv3_features', 'BatchedInceptionV3Processor'])

if KEYPOINTS_FEATURES_AVAILABLE:
    __all__.extend(['extract_keypoints_from_frame', 'interpolate_gaps', 'smooth_keypoints_ema', 
                    'validate_and_clean_keypoints', 'create_models', 'close_models'])

if VALIDATE_NPZ_AVAILABLE:
    __all__.append('validate_dataset')

if RENAME_CLIPS_AVAILABLE:
    __all__.append('rename_clips')

if RESIZE_360P_AVAILABLE:
    __all__.append('VideoResizer')

if OCCLUSION_DETECTION_AVAILABLE:
    __all__.extend([
        'compute_occlusion_flag_from_keypoints',
        'compute_occlusion_detection',
        'get_occlusion_config',
        'validate_occlusion_config',
        'DEFAULT_OCCLUSION_CONFIG'
    ])
