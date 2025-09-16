"""
Training module for Filipino Sign Language Recognition.

This module provides training functionality for sign language recognition models,
including multi-task learning (gloss and category classification) with configurable loss weights.

Key Components:
- FSLDataset: PyTorch Dataset for sign language sequences
- FSLFeatureFileDataset: Dataset for precomputed visual features
- Training utilities and evaluation functions
- Support for both Transformer and InceptionV3+GRU models

Usage:
    from training import FSLDataset, evaluate
    python -m training.train --model transformer --epochs 30
"""

# Conditional imports to avoid errors when dependencies are missing
try:
    from .utils import FSLDataset, evaluate
    TRAINING_UTILS_AVAILABLE = True
except ImportError:
    TRAINING_UTILS_AVAILABLE = False
    FSLDataset = None
    evaluate = None

try:
    from .train import (
        FSLFeatureFileDataset,
        train_model
        # Note: train.py doesn't have a main function, it's designed to be run as a script
    )
    TRAINING_TRAIN_AVAILABLE = True
except ImportError:
    TRAINING_TRAIN_AVAILABLE = False
    FSLFeatureFileDataset = None
    train_model = None

# Build __all__ list dynamically based on what's available
__all__ = []

if TRAINING_UTILS_AVAILABLE:
    __all__.extend(['FSLDataset', 'evaluate'])

if TRAINING_TRAIN_AVAILABLE:
    __all__.extend(['FSLFeatureFileDataset', 'train_model'])
