"""
Data module for Filipino Sign Language Recognition.

This module provides data management functionality for sign language recognition,
including label mapping, data splitting, and dataset organization utilities.

Key Components:
- Label mapping utilities for converting class IDs to human-readable labels
- Data splitting tools for creating train/validation datasets
- Dataset organization and file management utilities
- Reference data and mapping tables

Available Tools:
- Label mapping: Convert prediction results to readable format
- Data splitting: Organize preprocessed data into train/val splits
- ID assignment: Map text labels to numeric IDs for training
- Dataset validation and organization

Usage:
    from data import load_label_mappings, format_prediction_results
    
    # Load label mappings
    gloss_mapping, category_mapping = load_label_mappings()
    
    # Format prediction results
    formatted_results = format_prediction_results(predictions, gloss_mapping, category_mapping)
    
    # Split dataset
    python -m data.splitting.data_split --processed-root data/processed --labels labels.csv
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
