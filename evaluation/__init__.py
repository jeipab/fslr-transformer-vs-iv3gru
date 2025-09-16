"""
Evaluation module for Filipino Sign Language Recognition.

This module provides evaluation functionality for trained sign language
recognition models, including prediction on new videos and validation of model performance.

Key Components:
- Model prediction on video files with configurable preprocessing
- Comprehensive validation with accuracy metrics and confusion matrices
- Support for both Transformer and InceptionV3+GRU models
- Detailed performance analysis and reporting

Available Tools:
- Single video prediction with gloss and category classification
- Batch validation with comprehensive metrics
- Performance analysis with occlusion detection
- Results formatting and visualization

Usage:
    from evaluation import predict_video, validate_model
    
    # Predict on single video
    python -m evaluation.prediction.predict input.mp4 --model transformer --checkpoint model.pt
    
    # Validate model performance
    python -m evaluation.validation.validate --model transformer --checkpoint model.pt --data-dir data/
"""

# Prediction functionality
from .prediction.predict import main as predict_main, ModelPredictor

# Validation functionality  
from .validation.validate import main as validate_main

__all__ = [
    'predict_main',
    'ModelPredictor',
    'validate_main'
]
