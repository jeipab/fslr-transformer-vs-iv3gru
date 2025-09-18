"""
Configuration settings for the Streamlit application.

This module centralizes all configuration settings including:
- Page configuration
- Model settings and paths
- File processing parameters
- UI styling and layout settings
- Default values and constants
"""

from pathlib import Path
from typing import Dict, Any

# ===== PAGE CONFIGURATION =====
PAGE_CONFIG = {
    'page_title': 'FSLR Demo',
    'page_icon': None,
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# ===== MODEL CONFIGURATION =====
MODEL_CONFIG = {
    'transformer': {
        'enabled': True,
        'checkpoint_path': 'shared/trained_by_nov/vast.ai/loss-weights_0.5-0.5/transformer_100_epochs/SignTransformer_best.pt',
        'model_type': 'transformer',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'display_name': 'SignTransformer'
    },
    'iv3_gru': {
        'enabled': True,
        'checkpoint_path': 'shared/trained_by_nov/vast.ai/loss-weights_0.5-0.5/iv3gru_100_epochs/InceptionV3GRU_best.pt',
        'model_type': 'iv3_gru',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'display_name': 'InceptionV3+GRU'
    }
}

# ===== FILE PROCESSING CONFIGURATION =====
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
        'sequence_length': 150,  # Default sequence length for padding/trimming
        'keypoint_dim': 156,     # 78 keypoints × 2D coords
        'feature_dim': 2048      # InceptionV3 feature dimension
    },
    'file_limits': {
        'max_files': 10,
        'max_file_size_mb': 100
    }
}

# ===== UI CONFIGURATION =====
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

# ===== DUMMY DATA FOR TESTING =====
DUMMY_DATA = {
    'iv3_gru': {
        'gloss_prediction': 4,  # HOW ARE YOU
        'category_prediction': 0,  # GREETING
        'gloss_probability': 0.882,
        'category_probability': 0.774,
        'gloss_top5': [(4, 0.882), (18, 0.074), (17, 0.013), (85, 0.007), (6, 0.006)],
        'category_top3': [(0, 0.774), (8, 0.160), (1, 0.061)]
    }
}

# ===== WORKFLOW STAGES =====
WORKFLOW_STAGES = ['upload', 'preprocessing', 'predictions']

# ===== FILE TYPES =====
SUPPORTED_FILE_TYPES = {
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'],
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
