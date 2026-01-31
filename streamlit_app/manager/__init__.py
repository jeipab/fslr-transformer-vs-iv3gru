"""
Manager module for workflow stage management.

Managers for upload, preprocessing, prediction, and validation stages.
"""

from .upload_manager import (
    initialize_upload_session_state,
    render_upload_stage,
    remove_file_from_stage
)
from .preprocessing_manager import render_preprocessing_stage
from .prediction_manager import (
    render_predictions_stage,
    cleanup_on_app_exit
)
from .validation_manager import (
    run_validation_from_folder,
    cleanup_temp_files
)

__all__ = [
    'initialize_upload_session_state',
    'render_upload_stage',
    'remove_file_from_stage',
    'render_preprocessing_stage',
    'render_predictions_stage',
    'cleanup_on_app_exit',
    'run_validation_from_folder',
    'cleanup_temp_files'
]
