"""
Streamlit App for Filipino Sign Language Recognition.

This module provides a web interface for sign language recognition,
including video upload, preprocessing, and model prediction capabilities.

Key Components:
- Interactive web interface for video upload and processing
- Real-time preprocessing of video files to keypoints and features
- Model prediction with visualization of results
- Batch processing capabilities for multiple files

Available Modules:
- Core: Main application entry point and configuration
- Managers: Workflow stage managers (upload, preprocessing, prediction)
- Components: UI components, utilities, and visualization tools

Usage:
    streamlit run streamlit_app/core/main.py
    
    # Or from command line
    python -m streamlit_app.core.main
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
