"""Utility functions for the Streamlit app."""

import base64
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
    elif file_extension in ['.mp4', '.mov', '.webm']:
        return 'video'
    else:
        return 'unknown'


def is_continuous_sequence(npz_data: Dict[str, np.ndarray]) -> bool:
    """
    Detect if NPZ file contains a continuous sequence vs isolated sign.
    
    Continuous sequences have metadata with 'num_segments' or 'strategy' fields.
    
    Args:
        npz_data: Dictionary containing NPZ file contents
        
    Returns:
        True if continuous sequence, False if isolated sign
    """
    if 'meta' in npz_data:
        try:
            meta = npz_data['meta']
            # Parse metadata
            if isinstance(meta, str):
                meta_dict = json.loads(meta)
            elif isinstance(meta, np.ndarray):
                meta_dict = json.loads(str(meta.item()))
            else:
                meta_dict = json.loads(str(meta))
            
            # Check for continuous-specific fields
            return 'num_segments' in meta_dict or 'strategy' in meta_dict
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            pass
    
    return False


def extract_continuous_metadata(npz_data: Dict[str, np.ndarray]) -> Optional[Dict]:
    """
    Extract continuous sequence metadata from NPZ file.
    
    Args:
        npz_data: Dictionary containing NPZ file contents
        
    Returns:
        Metadata dictionary if continuous, None if isolated
    """
    if not is_continuous_sequence(npz_data):
        return None
    
    try:
        meta = npz_data['meta']
        if isinstance(meta, str):
            meta_dict = json.loads(meta)
        elif isinstance(meta, np.ndarray):
            meta_dict = json.loads(str(meta.item()))
        else:
            meta_dict = json.loads(str(meta))
        
        return meta_dict
    except:
        return None


def check_npz_compatibility(npz_data: Dict[str, np.ndarray], model_configs: Dict = None) -> Dict[str, bool]:
    """
    Check if NPZ data is compatible with different model architectures.
    
    Args:
        npz_data: Dictionary containing NPZ file contents
        model_configs: Optional model configurations for dynamic checking
        
    Returns:
        Dictionary with compatibility flags for each model architecture
    """
    # Import config functions
    from ..core.config import (
        get_model_input_dim, get_model_supports_keypoints, 
        get_model_supports_features, MODEL_CONFIG
    )
    
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
        # Enhanced compatibility checking based on model capabilities and available data
        # Check what data is available
        has_keypoints = 'X' in npz_data
        has_features = 'X2048' in npz_data
        
        # Validate data shapes if present
        keypoints_valid = False
        features_valid = False
        
        if has_keypoints:
            X = npz_data['X']
            keypoints_valid = X.ndim == 2 and X.shape[1] == 156
        
        if has_features:
            X2048 = npz_data['X2048']
            features_valid = X2048.ndim == 2 and X2048.shape[1] == 2048
        
        # Check compatibility for each model based on their capabilities
        for model_name in ['transformer', 'iv3_gru']:
            model_config = MODEL_CONFIG.get(model_name, {})
            
            # Get model capabilities
            supports_keypoints = get_model_supports_keypoints(model_name)
            supports_features = get_model_supports_features(model_name)
            input_dim = get_model_input_dim(model_name)
            
            # Check compatibility based on model capabilities and available data
            if model_name == 'transformer':
                # Transformer can use either keypoints or features depending on its configuration
                if supports_keypoints and keypoints_valid:
                    compatibility['transformer'] = True
                elif supports_features and features_valid:
                    compatibility['transformer'] = True
                elif input_dim is not None:
                    # If we know the specific input dimension, check for exact match
                    if input_dim == 156 and keypoints_valid:
                        compatibility['transformer'] = True
                    elif input_dim == 2048 and features_valid:
                        compatibility['transformer'] = True
            
            elif model_name == 'iv3_gru':
                # IV3-GRU always needs 2048-D features
                if supports_features and features_valid:
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


def encode_file_to_base64(file_data: bytes, mime_type: str = "video/mp4") -> str:
    """
    Encode file data to Base64 data URI for WebSocket delivery.
    
    This ensures media files pass through WebSocket connection rather than
    HTTP requests, avoiding session affinity issues in load-balanced deployments.
    
    Args:
        file_data: Raw bytes of the file
        mime_type: MIME type of the file (e.g., 'video/mp4', 'video/webm')
        
    Returns:
        Base64-encoded data URI string
    """
    encoded = base64.b64encode(file_data).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"


def decode_base64_file(data_uri: str) -> Tuple[bytes, str]:
    """
    Decode Base64 data URI back to file bytes.
    
    Args:
        data_uri: Base64-encoded data URI string
        
    Returns:
        Tuple of (file_bytes, mime_type)
    """
    # Extract MIME type and Base64 data
    if ',' in data_uri and data_uri.startswith('data:'):
        header, encoded = data_uri.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        file_bytes = base64.b64decode(encoded)
        return file_bytes, mime_type
    else:
        raise ValueError("Invalid data URI format")


def get_mime_type_from_extension(filename: str) -> str:
    """
    Get MIME type from file extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        MIME type string
    """
    ext = Path(filename).suffix.lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.npz': 'application/octet-stream',
        '.avi': 'video/x-msvideo',
    }
    return mime_types.get(ext, 'application/octet-stream')
