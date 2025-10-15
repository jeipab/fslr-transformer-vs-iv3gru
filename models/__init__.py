"""
Models module for Filipino Sign Language Recognition.

This module provides neural network architectures for sign language recognition,
including Transformer-based and RNN approaches for processing sign language sequences.

Available Models:
- SignTransformer: Transformer encoder for keypoint sequences [B, T, 156]
- MediaPipeGRU: Lightweight GRU model for keypoint sequences [B, T, 156] (mobile-friendly)
- InceptionV3GRU: CNN+GRU model for visual features [B, T, 2048] or raw frames (offline baseline)

Model Comparison:
┌──────────────────┬─────────────────┬──────────────────┬──────────────────┐
│ Model            │ Input           │ Mobile-Ready     │ Use Case         │
├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
│ SignTransformer  │ Keypoints (156) │ ✅ YES (~1-2 MB)  │ Primary mobile   │
│ MediaPipeGRU     │ Keypoints (156) │ ✅ YES (~500 KB)  │ Baseline mobile  │
│ InceptionV3GRU   │ Features (2048) │ ❌ NO (~25 MB)    │ Offline baseline │
└──────────────────┴─────────────────┴──────────────────┴──────────────────┘

Key Features:
- Multi-task learning (gloss and category classification)
- Configurable architecture parameters
- Support for both precomputed features and raw input processing
- Positional encoding and attention mechanisms for temporal modeling

Usage:
    from models import SignTransformer, MediaPipeGRU, InceptionV3GRU
    
    # Transformer for keypoints (primary model)
    transformer = SignTransformer(num_gloss=105, num_cat=10)
    
    # Lightweight GRU for keypoints (mobile baseline)
    mp_gru = MediaPipeGRU(num_gloss=105, num_cat=10)
    
    # CNN+GRU for visual features (offline baseline)
    iv3_gru = InceptionV3GRU(num_gloss=105, num_cat=10)
"""

from .transformer import SignTransformer, PositionalEncoding
from .mediapipe_gru import MediaPipeGRU
from .iv3_gru import InceptionV3GRU

__all__ = [
    'SignTransformer',
    'MediaPipeGRU',
    'PositionalEncoding', 
    'InceptionV3GRU'
]
