"""
Configuration settings for the Streamlit application.

Centralizes all configuration including page settings, model paths,
file processing parameters, UI styling, and default values.
"""

from pathlib import Path
from typing import Dict, Any

# Page configuration
PAGE_CONFIG = {
    'page_title': 'PANSINAYAN',
    'page_icon': '🤟',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Model configuration
MODEL_CONFIG = {
    'transformer': {
        'enabled': True,
        'checkpoint_path': 'trained_models/cmb/transformer/cmb_combined_2204/SignTransformer_best.pt',
        'model_type': 'transformer',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'display_name': 'SignTransformer',
        'input_dim': None,  # Will be auto-detected from checkpoint (2204 for combined model)
        'supports_keypoints': True,  # Can use 156-D keypoints
        'supports_features': True    # Can use 2048-D features
    },
    'iv3_gru': {
        'enabled': True,
        'checkpoint_path': 'trained_models/cmb/iv3_gru/cmb_improved/InceptionV3GRU_best.pt',
        'model_type': 'iv3_gru',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'display_name': 'InceptionV3+GRU',
        'input_dim': 2048,  # Always uses 2048-D features
        'supports_keypoints': False,  # Cannot use 156-D keypoints
        'supports_features': True     # Can use 2048-D features
    }
}

# File processing configuration
PROCESSING_CONFIG = {
    'video': {
        'target_fps': 30,
        'out_size': 256,
        'conf_thresh': 0.5,
        'max_gap': 5,
        'write_keypoints': True,
        'write_iv3_features': True,
        'occ_detailed': False
    },
    'npz': {
        'sequence_length': 150,
        'keypoint_dim': 156,
        'feature_dim': 2048
    },
    'file_limits': {
        'max_files': 10,
        'max_file_size_mb': 100
    }
}

# UI configuration
UI_CONFIG = {
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd'
    },
    'sizes': {
        'header_font': '2.5rem',
        'section_font': '1.5rem',
        'chart_height': 600
    },
    'layout': {
        'sidebar_width': 300,
        'main_content_padding': '1rem'
    }
}

# Dummy data for testing
DUMMY_DATA = {
    'iv3_gru': {
        'gloss_prediction': 4,
        'category_prediction': 0,
        'gloss_probability': 0.882,
        'category_probability': 0.774,
        'gloss_top5': [(4, 0.882), (18, 0.074), (17, 0.013), (85, 0.007), (6, 0.006)],
        'category_top3': [(0, 0.774), (8, 0.160), (1, 0.061)]
    }
}

# Workflow stages
WORKFLOW_STAGES = ['upload', 'preprocessing', 'predictions', 'validation']

# Supported file types
SUPPORTED_FILE_TYPES = {
    'video': ['.mp4', '.mov', '.webm'],
    'preprocessed': ['.npz']
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIG.get(model_name, {})

def get_processing_config(process_type: str) -> Dict[str, Any]:
    """Get configuration for a specific processing type."""
    return PROCESSING_CONFIG.get(process_type, {})

def get_ui_config(category: str) -> Dict[str, Any]:
    """Get UI configuration for a specific category."""
    return UI_CONFIG.get(category, {})

def is_model_enabled(model_name: str) -> bool:
    """Check if a model is enabled."""
    return MODEL_CONFIG.get(model_name, {}).get('enabled', False)

def get_checkpoint_path(model_name: str) -> str:
    """Get the checkpoint path for a model."""
    return MODEL_CONFIG.get(model_name, {}).get('checkpoint_path', '')

def update_model_input_dim(model_name: str, input_dim: int) -> None:
    """Update the input dimension for a model after auto-detection."""
    if model_name in MODEL_CONFIG:
        MODEL_CONFIG[model_name]['input_dim'] = input_dim

def get_model_input_dim(model_name: str) -> int:
    """Get the input dimension for a model."""
    return MODEL_CONFIG.get(model_name, {}).get('input_dim', None)

def get_model_supports_keypoints(model_name: str) -> bool:
    """Check if a model supports 156-D keypoint inputs."""
    return MODEL_CONFIG.get(model_name, {}).get('supports_keypoints', False)

def get_model_supports_features(model_name: str) -> bool:
    """Check if a model supports 2048-D feature inputs."""
    return MODEL_CONFIG.get(model_name, {}).get('supports_features', False)
