"""Data module for Filipino Sign Language Recognition.

Provides label mapping, data splitting, and dataset organization utilities.
"""

# Label mapping functionality
from .labels.label_mapping import (
    load_label_mappings,
    format_prediction_results,
    print_prediction_summary
)

# Data splitting functionality
from .splitting.data_split import main as data_split_main

__all__ = [
    # Label mapping
    'load_label_mappings',
    'format_prediction_results', 
    'print_prediction_summary',
    
    # Data splitting
    'data_split_main'
]
