"""
Core module for the Streamlit application.

This module contains the main application entry point and core configuration.
"""

from .main import main
from .config import (
    PAGE_CONFIG,
    MODEL_CONFIG,
    PROCESSING_CONFIG,
    UI_CONFIG,
    DUMMY_DATA,
    WORKFLOW_STAGES,
    SUPPORTED_FILE_TYPES,
    get_model_config,
    get_processing_config,
    get_ui_config,
    is_model_enabled,
    get_checkpoint_path
)

__all__ = [
    'main',
    'PAGE_CONFIG',
    'MODEL_CONFIG',
    'PROCESSING_CONFIG',
    'UI_CONFIG',
    'DUMMY_DATA',
    'WORKFLOW_STAGES',
    'SUPPORTED_FILE_TYPES',
    'get_model_config',
    'get_processing_config',
    'get_ui_config',
    'is_model_enabled',
    'get_checkpoint_path'
]
