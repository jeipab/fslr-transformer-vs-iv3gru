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

from .utils import FSLDataset, evaluate
from .train import (
    FSLFeatureFileDataset,
    train_model,
    main
)

__all__ = [
    'FSLDataset',
    'FSLFeatureFileDataset', 
    'evaluate',
    'train_model',
    'main'
]
