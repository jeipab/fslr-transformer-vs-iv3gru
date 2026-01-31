"""
Evaluation module for Filipino Sign Language Recognition.

Provides prediction and validation functionality for trained sign language
recognition models.
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
