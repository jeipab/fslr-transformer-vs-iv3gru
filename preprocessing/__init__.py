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
from .core.preprocess import main as preprocess_main
from .core.multi_preprocess import main as multi_preprocess_main

# Feature extractors
from .extractors.iv3_features import extract_iv3_features
from .extractors.keypoints_features import (
    extract_keypoints_from_frame,
    interpolate_gaps,
    create_models,
    close_models
)

# Utilities
from .utils.validate_npz import validate_dataset
from .utils.rename_clips import rename_clips

# Occlusion detection
from .core.occlusion_detection import compute_occlusion_flag_from_keypoints

__all__ = [
    # Core processors
    'preprocess_main',
    'multi_preprocess_main',
    
    # Feature extractors
    'extract_iv3_features',
    'extract_keypoints_from_frame',
    'interpolate_gaps',
    'create_models',
    'close_models',
    
    # Utilities
    'validate_dataset',
    'rename_clips',
    
    # Occlusion detection
    'compute_occlusion_flag_from_keypoints'
]
