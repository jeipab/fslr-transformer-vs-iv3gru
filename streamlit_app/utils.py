"""Utility functions for the Streamlit app."""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import numpy as np


def pad_or_trim(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or trim sequence to target_length along time axis.

    Args:
        sequence: Array shaped [T, D].
        target_length: Desired temporal length T.

    Returns:
        Array shaped [target_length, D], float32.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence [T, D], got shape {sequence.shape}")

    time_steps, feature_dim = sequence.shape
    sequence = sequence.astype(np.float32)

    if time_steps == target_length:
        return sequence
    if time_steps > target_length:
        return sequence[:target_length]

    output = np.zeros((target_length, feature_dim), dtype=np.float32)
    output[:time_steps] = sequence
    return output


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def simulate_predictions(
    random_state: np.random.RandomState,
    num_gloss_classes: int,
    num_category_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated logits for gloss and category heads."""
    gloss_logits = random_state.randn(num_gloss_classes).astype(np.float32)
    cat_logits = random_state.randn(num_category_classes).astype(np.float32)
    return gloss_logits, cat_logits


def topk_from_logits(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and probabilities from logits vector."""
    k = max(1, min(k, logits.shape[-1]))
    probs = softmax(logits)
    topk_idx = np.argpartition(-probs, kth=k - 1)[:k]
    topk_sorted = topk_idx[np.argsort(-probs[topk_idx])]
    return topk_sorted, probs[topk_sorted]


def detect_file_type(uploaded_file) -> str:
    """Detect if uploaded file is NPZ or video based on extension."""
    if uploaded_file is None:
        return 'unknown'
    
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.npz':
        return 'npz'
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
        return 'video'
    else:
        return 'unknown'


class TempUploadedFile:
    """Temporary uploaded file object to handle file content reuse."""
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def read(self):
        return self.content
    def getvalue(self):
        return self.content
