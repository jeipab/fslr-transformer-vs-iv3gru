"""
Label mapping utility for Sign Language Recognition predictions.

This module provides functions to map class IDs to human-readable labels
using the labels reference CSV file.
"""

import pandas as pd
import os
from pathlib import Path

def load_label_mappings():
    """
    Load gloss and category mappings from the labels reference CSV.
    
    Returns:
        tuple: (gloss_mapping, category_mapping) dictionaries
    """
    # Path to the labels reference CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "splitting" / "labels_reference.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels reference file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Create gloss mapping: {gloss_id: gloss_label}
    gloss_mapping = dict(zip(df['gloss_id'], df['label']))
    
    # Create category mapping: {cat_id: category_name}
    category_mapping = dict(zip(df['cat_id'], df['category']))
    
    return gloss_mapping, category_mapping

def format_prediction_results(results, gloss_mapping=None, category_mapping=None):
    """
    Format prediction results with compact human-readable labels.
    
    Args:
        results (dict): Raw prediction results from model
        gloss_mapping (dict): Optional gloss ID to label mapping
        category_mapping (dict): Optional category ID to label mapping
    
    Returns:
        dict: Compact formatted results with embedded IDs in labels
    """
    if gloss_mapping is None or category_mapping is None:
        gloss_mapping, category_mapping = load_label_mappings()
    
    # Create compact format
    formatted = {}
    
    # Main predictions with embedded IDs
    gloss_id = results['gloss_prediction']
    cat_id = results['category_prediction']
    
    formatted['gloss_prediction'] = f"{gloss_mapping.get(gloss_id, f'Unknown')} ({gloss_id})"
    formatted['category_prediction'] = f"{category_mapping.get(cat_id, f'Unknown')} ({cat_id})"
    
    # Keep probabilities
    formatted['gloss_probability'] = results['gloss_probability']
    formatted['category_probability'] = results['category_probability']
    
    # Compact top predictions
    formatted['gloss_top5'] = [
        [f"{gloss_mapping.get(gloss_id, f'Unknown')} ({gloss_id})", prob]
        for gloss_id, prob in results['gloss_top5']
    ]
    
    formatted['category_top3'] = [
        [f"{category_mapping.get(cat_id, f'Unknown')} ({cat_id})", prob]
        for cat_id, prob in results['category_top3']
    ]
    
    # Add any additional fields (like frames_extracted for videos)
    for key, value in results.items():
        if key not in ['gloss_prediction', 'category_prediction', 'gloss_probability', 
                      'category_probability', 'gloss_top5', 'category_top3']:
            formatted[key] = value
    
    return formatted

def print_prediction_summary(results, gloss_mapping=None, category_mapping=None):
    """
    Print a formatted summary of prediction results.
    
    Args:
        results (dict): Raw prediction results from model
        gloss_mapping (dict): Optional gloss ID to label mapping
        category_mapping (dict): Optional category ID to label mapping
    """
    if gloss_mapping is None or category_mapping is None:
        gloss_mapping, category_mapping = load_label_mappings()
    
    formatted = format_prediction_results(results, gloss_mapping, category_mapping)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Gloss: {formatted['gloss_prediction']} (confidence: {formatted['gloss_probability']:.3f})")
    print(f"Category: {formatted['category_prediction']} (confidence: {formatted['category_probability']:.3f})")
    
    print(f"\nTop 5 Gloss Predictions:")
    for i, (gloss_label_with_id, prob) in enumerate(formatted['gloss_top5'], 1):
        print(f"  {i}. {gloss_label_with_id}: {prob:.3f}")
    
    print(f"\nTop 3 Category Predictions:")
    for i, (cat_label_with_id, prob) in enumerate(formatted['category_top3'], 1):
        print(f"  {i}. {cat_label_with_id}: {prob:.3f}")
    
    if 'frames_extracted' in formatted:
        print(f"\nFrames extracted: {formatted['frames_extracted']}")

def get_all_labels():
    """
    Get all available gloss and category labels.
    
    Returns:
        tuple: (gloss_labels, category_labels) lists
    """
    gloss_mapping, category_mapping = load_label_mappings()
    
    gloss_labels = sorted(gloss_mapping.items())
    category_labels = sorted(category_mapping.items())
    
    return gloss_labels, category_labels

if __name__ == "__main__":
    # Test the mapping
    try:
        gloss_mapping, category_mapping = load_label_mappings()
        print("✓ Label mappings loaded successfully")
        print(f"  - {len(gloss_mapping)} gloss labels")
        print(f"  - {len(category_mapping)} category labels")
        
        # Show some examples
        print("\nSample gloss labels:")
        for i in range(5):
            print(f"  {i}: {gloss_mapping[i]}")
        
        print("\nSample category labels:")
        for i in range(5):
            print(f"  {i}: {category_mapping[i]}")
            
    except Exception as e:
        print(f"Error loading mappings: {e}")
