"""Utility functions for the Streamlit app."""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import numpy as np


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


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


# Simulation functions removed - using real model predictions only


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


def check_npz_compatibility(npz_data: Dict[str, np.ndarray]) -> Dict[str, bool]:
    """
    Check if NPZ data is compatible with different model architectures.
    
    Args:
        npz_data: Dictionary containing NPZ file contents
        
    Returns:
        Dictionary with compatibility flags for each model architecture
    """
    compatibility = {
        'transformer': False,
        'iv3_gru': False,
        'both': False
    }
    
    # Try to get model_type from metadata first
    model_type = None
    if 'meta' in npz_data:
        try:
            meta = npz_data['meta']
            if isinstance(meta, str):
                meta_dict = json.loads(meta)
            else:
                meta_dict = json.loads(str(meta))
            model_type = meta_dict.get('model_type')
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            pass
    
    # Use model_type as the final authority for compatibility
    if model_type:
        if model_type == 'T':
            compatibility['transformer'] = True
        elif model_type == 'I':
            compatibility['iv3_gru'] = True
        elif model_type == 'B':
            compatibility['transformer'] = True
            compatibility['iv3_gru'] = True
            compatibility['both'] = True
    else:
        # Fallback to legacy compatibility checking for old files without model_type
        # Check for transformer compatibility (needs 156-D keypoints)
        has_keypoints = 'X' in npz_data
        if has_keypoints:
            X = npz_data['X']
            if X.ndim == 2 and X.shape[1] == 156:
                compatibility['transformer'] = True
        
        # Check for iv3_gru compatibility (needs 2048-D features)
        has_iv3_features = 'X2048' in npz_data
        if has_iv3_features:
            X2048 = npz_data['X2048']
            if X2048.ndim == 2 and X2048.shape[1] == 2048:
                compatibility['iv3_gru'] = True
        
        # Check if both are compatible
        if compatibility['transformer'] and compatibility['iv3_gru']:
            compatibility['both'] = True
    
    return compatibility


def extract_occlusion_flag(npz_data: Dict[str, np.ndarray]) -> int:
    """
    Extract occlusion flag from NPZ metadata.
    
    Args:
        npz_data: Dictionary containing NPZ file contents
        
    Returns:
        Integer occlusion flag: 0 = not occluded, 1 = occluded, -1 = unknown
    """
    try:
        if 'meta' not in npz_data:
            return -1
            
        meta = npz_data['meta']
        
        # Convert metadata to string format for consistent parsing
        if isinstance(meta, str):
            meta_str = meta
        else:
            meta_str = str(meta)
        
        # Parse JSON metadata
        meta_dict = json.loads(meta_str)
        
        # Extract occlusion flag if present
        if 'occluded_flag' in meta_dict:
            return int(meta_dict['occluded_flag'])
        
        # No occlusion flag found
        return -1
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return -1


def interpret_occlusion_flag(occlusion_flag: int) -> str:
    """
    Interpret occlusion flag as human-readable string.
    
    Args:
        occlusion_flag: Integer occlusion flag (0, 1, or -1)
        
    Returns:
        String interpretation: "No", "Yes", or "Unknown"
    """
    if occlusion_flag == 0:
        return "No"
    elif occlusion_flag == 1:
        return "Yes"
    else:
        return "Unknown"


def create_npz_bytes(npz_data: Dict[str, np.ndarray]) -> bytes:
    """
    Create NPZ file bytes from dictionary data.
    
    Args:
        npz_data: Dictionary containing arrays to save
        
    Returns:
        Bytes of the NPZ file
    """
    import io
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **npz_data)
    npz_buffer.seek(0)
    return npz_buffer.getvalue()


class TempUploadedFile:
    """Temporary uploaded file object to handle file content reuse."""
    def __init__(self, name, data, type=None, size=None):
        self.name = name
        self.content = data
        self.data = data
        self.type = type
        self.size = size
    
    def read(self):
        return self.content
    
    def getvalue(self):
        return self.content
    
    def seek(self, position):
        # For compatibility with file-like objects
        pass
