"""
Prediction module for Filipino Sign Language Recognition.

This module provides prediction functionality for trained sign language
recognition models, including the ModelPredictor class for making
predictions on new videos and NPZ files.

Key Components:
- ModelPredictor: Unified predictor for both Transformer and IV3-GRU models
- Support for both NPZ files (preprocessed data) and video files (raw videos)
- Automatic feature extraction for video files
- Comprehensive prediction results with confidence scores

Usage:
    from evaluation.prediction import ModelPredictor
    
    # Initialize predictor
    predictor = ModelPredictor('transformer', 'path/to/checkpoint.pt')
    
    # Make prediction from NPZ file
    results = predictor.predict_from_npz('data.npz')
    
    # Make prediction from video file
    results = predictor.predict_from_video('video.mp4')
    
    # Clean up resources
    predictor.cleanup()
"""

from .predict import ModelPredictor

__all__ = [
    'ModelPredictor'
]
