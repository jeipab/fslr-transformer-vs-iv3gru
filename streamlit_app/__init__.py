"""
Streamlit App for Filipino Sign Language Recognition.

Web interface for sign language recognition with video upload, preprocessing,
and model prediction capabilities.
"""

# Core application
from .core.main import main
from .core.config import (
    PAGE_CONFIG,
    MODEL_CONFIG,
    PROCESSING_CONFIG,
    UI_CONFIG,
    DUMMY_DATA,
    WORKFLOW_STAGES,
    SUPPORTED_FILE_TYPES
)

# Managers
from .manager.upload_manager import (
    initialize_upload_session_state,
    render_upload_stage,
    remove_file_from_stage
)
from .manager.preprocessing_manager import render_preprocessing_stage
from .manager.prediction_manager import (
    render_predictions_stage,
    cleanup_on_app_exit
)

# Components
from .components.components import (
    set_page,
    render_sidebar,
    render_main_header,
    render_file_upload,
    render_predictions_section
)
from .components.data_processing import process_video_file
from .components.utils import (
    detect_file_type,
    format_file_size,
    pad_or_trim,
    check_npz_compatibility,
    create_npz_bytes,
    extract_occlusion_flag,
    interpret_occlusion_flag
)
from .components.visualization import (
    render_sequence_overview,
    render_animated_keypoints,
    render_feature_charts,
    render_topk_table
)

__all__ = [
    # Core
    'main',
    'PAGE_CONFIG',
    'MODEL_CONFIG',
    'PROCESSING_CONFIG',
    'UI_CONFIG',
    'DUMMY_DATA',
    'WORKFLOW_STAGES',
    'SUPPORTED_FILE_TYPES',
    
    # Managers
    'initialize_upload_session_state',
    'render_upload_stage',
    'remove_file_from_stage',
    'render_preprocessing_stage',
    'render_predictions_stage',
    'cleanup_on_app_exit',
    
    # Components
    'set_page',
    'render_sidebar',
    'render_main_header',
    'render_file_upload',
    'render_predictions_section',
    'process_video_file',
    'detect_file_type',
    'format_file_size',
    'pad_or_trim',
    'check_npz_compatibility',
    'create_npz_bytes',
    'extract_occlusion_flag',
    'interpret_occlusion_flag',
    'render_sequence_overview',
    'render_animated_keypoints',
    'render_feature_charts',
    'render_topk_table'
]
