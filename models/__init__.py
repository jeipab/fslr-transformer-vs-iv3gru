"""
Models module for Filipino Sign Language Recognition.

This module provides neural network architectures for sign language recognition,
including Transformer-based and RNN approaches for processing sign language sequences.

Available Models:

Classification Models (Isolated Sign Recognition):
- SignTransformer: Transformer encoder for keypoint sequences [B, T, 178]
- MediaPipeGRU: Lightweight GRU model for keypoint sequences [B, T, 178] (mobile-friendly)
- InceptionV3GRU: CNN+GRU model for visual features [B, T, 2048] or raw frames (offline baseline)

CTC Models (Continuous Sign Language Recognition):
- SignTransformerCtc: Transformer with CTC for sequence-to-sequence recognition
- MediaPipeGRUCtc: Lightweight GRU with CTC for continuous recognition
- InceptionV3GRUCtc: CNN+GRU with CTC for continuous recognition (offline baseline)

Model Comparison:
┌─────────────────────┬─────────────────┬──────────────────┬─────────────────────┐
│ Model               │ Input           │ Mobile-Ready     │ Use Case            │
├─────────────────────┼─────────────────┼──────────────────┼─────────────────────┤
│ SignTransformer     │ Keypoints (178) │ ✅ YES (~1-2 MB)  │ Isolated signs      │
│ SignTransformerCtc  │ Keypoints (178) │ ✅ YES (~1-2 MB)  │ Continuous signs    │
│ MediaPipeGRU        │ Keypoints (178) │ ✅ YES (~500 KB)  │ Isolated baseline   │
│ MediaPipeGRUCtc     │ Keypoints (178) │ ✅ YES (~500 KB)  │ Continuous baseline │
│ InceptionV3GRU      │ Features (2048) │ ❌ NO (~25 MB)    │ Offline baseline    │
│ InceptionV3GRUCtc   │ Features (2048) │ ❌ NO (~25 MB)    │ Continuous offline  │
└─────────────────────┴─────────────────┴──────────────────┴─────────────────────┘

Key Features:
- Multi-task learning for classification models (gloss and category)
- CTC-based sequence-to-sequence learning for continuous recognition
- Configurable architecture parameters
- Support for both precomputed features and raw input processing
- Positional encoding and attention mechanisms for temporal modeling

Usage:
    from models import SignTransformer, SignTransformerCtc, MediaPipeGRU, MediaPipeGRUCtc, InceptionV3GRU, InceptionV3GRUCtc
    
    # Classification models (isolated signs)
    transformer = SignTransformer(num_gloss=105, num_cat=10)
    mp_gru = MediaPipeGRU(num_gloss=105, num_cat=10)
    iv3_gru = InceptionV3GRU(num_gloss=105, num_cat=10)
    
    # CTC models (continuous recognition)
    transformer_ctc = SignTransformerCtc(num_ctc_classes=106)
    mp_gru_ctc = MediaPipeGRUCtc(num_ctc_classes=106)
    iv3_gru_ctc = InceptionV3GRUCtc(num_ctc_classes=106)
"""

from .transformer import SignTransformer, SignTransformerCtc, PositionalEncoding
from .mediapipe_gru import MediaPipeGRU, MediaPipeGRUCtc
from .iv3_gru import InceptionV3GRU, InceptionV3GRUCtc

__all__ = [
    'SignTransformer',
    'SignTransformerCtc',
    'MediaPipeGRU',
    'MediaPipeGRUCtc',
    'PositionalEncoding', 
    'InceptionV3GRU',
    'InceptionV3GRUCtc'
]
