"""
Models module for Filipino Sign Language Recognition.

This module provides neural network architectures for sign language recognition,
including both Transformer-based and CNN+RNN approaches for processing sign language sequences.

Available Models:
- SignTransformer: Transformer encoder for keypoint sequences [B, T, 156]
- InceptionV3GRU: CNN+GRU model for visual features [B, T, 2048] or raw frames

Key Features:
- Multi-task learning (gloss and category classification)
- Configurable architecture parameters
- Support for both precomputed features and raw input processing
- Positional encoding and attention mechanisms for temporal modeling

Usage:
    from models import SignTransformer, InceptionV3GRU
    
    # Transformer for keypoints
    transformer = SignTransformer(num_gloss=105, num_cat=10)
    
    # CNN+GRU for visual features
    iv3_gru = InceptionV3GRU(num_gloss=105, num_cat=10)
"""

from .transformer import SignTransformer, PositionalEncoding
from .iv3_gru import InceptionV3GRU

__all__ = [
    'SignTransformer',
    'PositionalEncoding', 
    'InceptionV3GRU'
]
