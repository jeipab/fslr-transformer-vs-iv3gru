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

# CTC configuration for continuous sign language recognition
CTC_CONFIG = {
    'num_gloss_classes': 105,
    'blank_token_id': 105,  # The ID for the CTC blank token (num_gloss_classes)
    'num_ctc_classes': 106,  # num_gloss_classes + 1 for blank token
    'beam_width': 10,  # For beam search decoding
    'window_size': 60,  # Frames per window for sliding window inference
    'window_stride': 15  # Stride between windows for sliding window inference
}

# CTC configuration for subset models (e.g., GREETINGS-only)
CTC_CONFIG_SUBSET = {
    'num_gloss_classes': 10,
    'blank_token_id': 10,  # The ID for the CTC blank token (num_gloss_classes)
    'num_ctc_classes': 11,  # num_gloss_classes + 1 for blank token
    'beam_width': 10,  # For beam search decoding
    'window_size': 60,  # Frames per window for sliding window inference
    'window_stride': 15  # Stride between windows for sliding window inference
}

# Model configuration
MODEL_CONFIG = {
    'transformer': {
        'enabled': True,
        'checkpoint_path': 'trained_models/transformer/greetings_classification/SignTransformer_best.pt',
        'model_type': 'transformer',
        'num_gloss_classes': 10,  # greetings_classification model has 10 gloss classes
        'num_category_classes': 1,  # greetings_classification model has 1 category class
        'display_name': 'SignTransformer',
        'input_dim': 178,  # MediaPipe keypoints (89 keypoints × 2 coordinates)
        'supports_keypoints': True,  
        'supports_features': True,
        'training_mode': 'classification'  # 'classification' or 'ctc'
    },
    'transformer_ctc': {
        'enabled': True,  # Enabled for continuous sign recognition
        'checkpoint_path': 'trained_models/transformer/greetings_ctc_mobile/SignTransformerCtc_best.pt',
        'model_type': 'transformer_ctc',
        'num_gloss_classes': 10,  # greetings_ctc model has 10 gloss classes
        'num_ctc_classes': 11,  # greetings_ctc model has 11 CTC classes (10 gloss + 1 blank)
        'num_category_classes': 1,  # greetings_ctc model has 1 category class (dual-task)
        'max_len': 1000,  # Support longer continuous sequences
        'display_name': 'SignTransformer-CTC',
        'input_dim': 178,  # MediaPipe keypoints (89 keypoints × 2 coordinates)
        'supports_keypoints': True,
        'supports_features': True,
        'training_mode': 'ctc'
    },
    'iv3_gru': {
        'enabled': True,
        'checkpoint_path': 'trained_models/iv3_gru/greetings_classification/InceptionV3GRU_best.pt',
        'model_type': 'iv3_gru',
        'num_gloss_classes': 10,  # greetings_classification model has 10 gloss classes
        'num_category_classes': 1,  # greetings_classification model has 1 category class
        'display_name': 'InceptionV3+GRU',
        'input_dim': 2048, 
        'supports_keypoints': False,  
        'supports_features': True,
        'training_mode': 'classification'
    },
    'iv3_gru_ctc': {
        'enabled': True,  # Enabled for continuous sign recognition
        'checkpoint_path': 'trained_models/iv3_gru/greetings_ctc_v2/InceptionV3GRUCtc_best.pt',
        'model_type': 'iv3_gru_ctc',
        'num_gloss_classes': 10,  # greetings_ctc model has 10 gloss classes
        'num_ctc_classes': 11,  # greetings_ctc model has 11 CTC classes (10 gloss + 1 blank)
        'num_category_classes': 1,  # greetings_ctc model has 1 category class (dual-task)
        'blank_token_id': 10,  # greetings_ctc model uses blank_id=10 (subset training)
        'display_name': 'InceptionV3+GRU-CTC',
        'input_dim': 2048,
        'supports_keypoints': False,
        'supports_features': True,
        'training_mode': 'ctc',
        'ctc_config': 'subset'  # Use subset CTC config for proper blank_id
    },
    'mediapipe_gru': {
        'enabled': True,
        'checkpoint_path': 'trained_models/mediapipe_gru/greetings_classification/MediaPipeGRU_best.pt',
        'model_type': 'mediapipe_gru',
        'num_gloss_classes': 10,  # greetings_classification model has 10 gloss classes
        'num_category_classes': 1,  # greetings_classification model has 1 category class
        'display_name': 'MediaPipe-GRU',
        'input_dim': 178,  # MediaPipe keypoints
        'supports_keypoints': True,
        'supports_features': False,
        'training_mode': 'classification'
    },
    'mediapipe_gru_ctc': {
        'enabled': True,  # Enabled for continuous sign recognition
        'checkpoint_path': 'trained_models/mediapipe_gru/greetings_ctc_mobile/MediaPipeGRUCtc_best.pt',
        'model_type': 'mediapipe_gru_ctc',
        'num_gloss_classes': 10,  # greetings_ctc model has 10 gloss classes
        'num_ctc_classes': 11,  # greetings_ctc model has 11 CTC classes (10 gloss + 1 blank)
        'num_category_classes': 1,  # greetings_ctc model has 1 category class (dual-task)
        'blank_token_id': 10,  # greetings_ctc model uses blank_id=10 (subset training)
        'display_name': 'MediaPipe-GRU CTC',
        'input_dim': 178,  # MediaPipe keypoints
        'supports_keypoints': True,
        'supports_features': False,
        'training_mode': 'ctc',
        'ctc_config': 'subset'  # Use subset CTC config for proper blank_id
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
        'keypoint_dim': 178,
        'feature_dim': 2048
    },
    'file_limits': {
        'max_files': 10,
        'max_file_size_mb': 100
    }
}

# Upload configuration
UPLOAD_CONFIG = {
    'use_base64_preview': False,  # Enable Base64 encoding for video previews (recommended for mobile/load-balanced deployments)
    'base64_size_threshold_mb': 50,  # Max file size for Base64 encoding (larger files use standard preview)
    'enable_mobile_camera': True,  # Enable camera capture on mobile devices
    'show_upload_feedback': True,  # Show visual feedback during uploads
    'enable_enhanced_sync': True,  # Enable enhanced JavaScript sync for camera uploads
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

# Recognition modes
RECOGNITION_MODES = {
    'isolated': 'Isolated Sign Recognition',
    'continuous': 'Continuous Sign Recognition'
}

# Workflow stages
WORKFLOW_STAGES = ['upload', 'preprocessing', 'predictions', 'validation']

# Continuous generation configuration
CONTINUOUS_GENERATION_CONFIG = {
    'default_strategy': 1,
    'default_sequences_per_signer': 10,
    'default_min_glosses': 3,
    'default_max_glosses': 6,
    'default_seed': 42,
    'strategies': {
        1: 'Same Category',
        2: 'Different Categories'
    }
}

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

def get_upload_config(key: str = None) -> Any:
    """Get upload configuration settings.
    
    Args:
        key: Optional specific config key to retrieve
        
    Returns:
        Full config dict if key is None, otherwise the specific value
    """
    if key is None:
        return UPLOAD_CONFIG
    return UPLOAD_CONFIG.get(key, None)

def update_upload_config(key: str, value: Any) -> None:
    """Update upload configuration setting.
    
    Args:
        key: Configuration key to update
        value: New value
    """
    if key in UPLOAD_CONFIG:
        UPLOAD_CONFIG[key] = value

def get_ctc_config(key: str = None) -> Any:
    """Get CTC configuration settings.
    
    Args:
        key: Optional specific config key to retrieve
        
    Returns:
        Full config dict if key is None, otherwise the specific value
    """
    if key is None:
        return CTC_CONFIG
    return CTC_CONFIG.get(key, None)

def is_ctc_model(model_name: str) -> bool:
    """Check if a model is configured for CTC training mode.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model uses CTC training mode, False otherwise
    """
    model_config = MODEL_CONFIG.get(model_name, {})
    return model_config.get('training_mode') == 'ctc'

def get_models_by_mode(mode: str) -> list:
    """Get list of models compatible with recognition mode.
    
    Args:
        mode: Recognition mode ('isolated' or 'continuous')
        
    Returns:
        List of model names compatible with the mode
    """
    models = []
    for model_name, config in MODEL_CONFIG.items():
        if not config.get('enabled', False):
            continue
        
        training_mode = config.get('training_mode', 'classification')
        if mode == 'isolated' and training_mode == 'classification':
            models.append(model_name)
        elif mode == 'continuous' and training_mode == 'ctc':
            models.append(model_name)
    
    return models

def get_continuous_config(key: str = None):
    """Get continuous generation configuration.
    
    Args:
        key: Optional specific config key to retrieve
        
    Returns:
        Full config dict if key is None, otherwise the specific value
    """
    if key is None:
        return CONTINUOUS_GENERATION_CONFIG
    return CONTINUOUS_GENERATION_CONFIG.get(key, None)
