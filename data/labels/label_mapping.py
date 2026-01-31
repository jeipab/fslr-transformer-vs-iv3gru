"""Label mapping utility for Filipino sign language recognition."""

import pandas as pd
import os
from pathlib import Path

def load_label_mappings():
    """Load gloss and category mappings from labels reference CSV."""
    csv_path = Path(__file__).parent.parent / "labels_reference.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels reference file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    gloss_mapping = dict(zip(df['gloss_id'], df['label']))
    category_mapping = dict(zip(df['cat_id'], df['category']))
    
    return gloss_mapping, category_mapping

def format_prediction_results(results, gloss_mapping=None, category_mapping=None):
    """Format prediction results with human-readable labels."""
    if gloss_mapping is None or category_mapping is None:
        gloss_mapping, category_mapping = load_label_mappings()
    
    formatted = {}
    gloss_id = results['gloss_prediction']
    cat_id = results['category_prediction']
    
    formatted['gloss_prediction'] = f"{gloss_mapping.get(gloss_id, f'Unknown')} ({gloss_id})"
    formatted['category_prediction'] = f"{category_mapping.get(cat_id, f'Unknown')} ({cat_id})"
    formatted['gloss_probability'] = results['gloss_probability']
    formatted['category_probability'] = results['category_probability']
    
    formatted['gloss_top5'] = [
        [f"{gloss_mapping.get(gloss_id, f'Unknown')} ({gloss_id})", prob]
        for gloss_id, prob in results['gloss_top5']
    ]
    
    formatted['category_top3'] = [
        [f"{category_mapping.get(cat_id, f'Unknown')} ({cat_id})", prob]
        for cat_id, prob in results['category_top3']
    ]
    
    for key, value in results.items():
        if key not in ['gloss_prediction', 'category_prediction', 'gloss_probability', 
                      'category_probability', 'gloss_top5', 'category_top3']:
            formatted[key] = value
    
    return formatted

def print_prediction_summary(results, gloss_mapping=None, category_mapping=None):
    """Print formatted summary of prediction results."""
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
    """Get all available gloss and category labels."""
    gloss_mapping, category_mapping = load_label_mappings()
    
    gloss_labels = sorted(gloss_mapping.items())
    category_labels = sorted(category_mapping.items())
    
    return gloss_labels, category_labels


def get_num_gloss_classes():
    """Get the number of gloss classes."""
    gloss_mapping, _ = load_label_mappings()
    return len(gloss_mapping)


def get_ctc_config():
    """Get CTC configuration parameters."""
    num_gloss = get_num_gloss_classes()
    return {
        'num_gloss': num_gloss,
        'num_ctc_classes': num_gloss + 1,
        'blank_id': num_gloss
    }


def validate_ctc_config(num_gloss, num_ctc_classes, blank_id):
    """Validate CTC configuration for consistency."""
    expected_full = get_num_gloss_classes()
    
    if num_ctc_classes != num_gloss + 1:
        raise ValueError(f"num_ctc_classes should be {num_gloss + 1}, got {num_ctc_classes}")
    
    if blank_id != num_gloss:
        raise ValueError(f"blank_id should be {num_gloss}, got {blank_id}")
    
    if num_gloss < expected_full:
        print(f"Warning: Training on subset: {num_gloss}/{expected_full} glosses")
    elif num_gloss > expected_full:
        raise ValueError(f"num_gloss={num_gloss} exceeds available labels ({expected_full})")
    
    return True

if __name__ == "__main__":
    try:
        gloss_mapping, category_mapping = load_label_mappings()
        print(f"Loaded {len(gloss_mapping)} glosses, {len(category_mapping)} categories")
        
        config = get_ctc_config()
        validate_ctc_config(config['num_gloss'], config['num_ctc_classes'], config['blank_id'])
        print(f"CTC config: {config}")
            
    except Exception as e:
        print(f"Error: {e}")
