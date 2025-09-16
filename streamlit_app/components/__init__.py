"""
Components module for UI components and utilities.

This module contains reusable UI components, data processing utilities,
and visualization tools for the Streamlit application.

Key Components:
- UI Components: Page setup, sidebar, headers, and form elements
- Data Processing: Video/NPZ file processing utilities
- Utilities: General helper functions for file handling and data manipulation
- Visualization: Charts, keypoint visualization, and prediction displays
"""

from .components import (
    set_page,
    render_sidebar,
    render_main_header,
    render_file_upload,
    render_predictions_section
)
from .data_processing import process_video_file
from .utils import (
    detect_file_type,
    format_file_size,
    pad_or_trim,
    check_npz_compatibility,
    create_npz_bytes,
    extract_occlusion_flag,
    interpret_occlusion_flag
)
from .visualization import (
    render_sequence_overview,
    render_animated_keypoints,
    render_feature_charts,
    render_topk_table
)

__all__ = [
    # UI Components
    'set_page',
    'render_sidebar',
    'render_main_header',
    'render_file_upload',
    'render_predictions_section',
    
    # Data Processing
    'process_video_file',
    
    # Utilities
    'detect_file_type',
    'format_file_size',
    'pad_or_trim',
    'check_npz_compatibility',
    'create_npz_bytes',
    'extract_occlusion_flag',
    'interpret_occlusion_flag',
    
    # Visualization
    'render_sequence_overview',
    'render_animated_keypoints',
    'render_feature_charts',
    'render_topk_table'
]
