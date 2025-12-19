"""
Create confusion matrix CSV files from CTC validation results.

This script generates 4 CSV files:
- Recognition_Transformer - Confusion Matrix.csv (gloss-level)
- Recognition_IV3-GRU - Confusion Matrix.csv (gloss-level)
- Classification_Transformer - Confusion Matrix.csv (category-level)
- Classification_IV3-GRU - Confusion Matrix.csv (category-level)

Usage:
    python metrics\matrix\create_matrix.py

Input files (must exist):
    metrics\extract\shared_inputs\ctc_validation_results_transformer.json
    metrics\extract\shared_inputs\ctc_validation_results_iv3gru.json

Output files (saved to metrics\matrix):
    Recognition_Transformer - Confusion Matrix.csv
    Recognition_IV3-GRU - Confusion Matrix.csv
    Classification_Transformer - Confusion Matrix.csv
    Classification_IV3-GRU - Confusion Matrix.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


def load_label_mappings():
    """Load gloss and category mappings from labels reference CSV."""
    labels_ref_path = Path(__file__).parent.parent.parent / "data" / "labels_reference.csv"
    df = pd.read_csv(labels_ref_path)
    
    # Create gloss mapping: {gloss_id: gloss_label}
    gloss_mapping = dict(zip(df['gloss_id'], df['label']))
    
    # Create category mapping: {cat_id: category_name}
    category_mapping = dict(zip(df['cat_id'], df['category']))
    
    # Create gloss_id -> cat_id mapping
    gloss_to_cat = dict(zip(df['gloss_id'], df['cat_id']))
    
    return gloss_mapping, category_mapping, gloss_to_cat


def extract_sequences(json_data):
    """Extract ground truth and predicted sequences from JSON data."""
    predictions = json_data.get('predictions', [])
    
    # Collect all gloss sequences (for recognition)
    gloss_gt_all = []
    gloss_pred_all = []
    
    # Collect all category sequences (for classification)
    cat_gt_all = []
    cat_pred_all = []
    
    for pred in predictions:
        gt_gloss = pred.get('ground_truth_sequence', [])
        pred_gloss = pred.get('predicted_sequence', [])
        
        # For recognition: use gloss sequences directly
        gloss_gt_all.extend(gt_gloss)
        gloss_pred_all.extend(pred_gloss)
        
        # For classification: convert gloss IDs to category IDs
        # Ground truth categories: map from gloss IDs
        gt_categories = [gloss_to_cat.get(gid, -1) for gid in gt_gloss]
        cat_gt_all.extend(gt_categories)
        
        # Predicted categories: use predicted_categories if available, otherwise map from predicted_sequence
        if 'predicted_categories' in pred and pred['predicted_categories']:
            pred_categories = pred['predicted_categories']
        else:
            pred_categories = [gloss_to_cat.get(gid, -1) for gid in pred_gloss]
        cat_pred_all.extend(pred_categories)
    
    return (gloss_gt_all, gloss_pred_all), (cat_gt_all, cat_pred_all)


def create_confusion_matrix_csv(y_true, y_pred, label_mapping, output_path, matrix_type, include_id=False):
    """
    Create a confusion matrix CSV file.
    
    Args:
        y_true: List of true labels (IDs)
        y_pred: List of predicted labels (IDs)
        label_mapping: Dictionary mapping IDs to label names
        output_path: Path to save the CSV file
        matrix_type: 'Recognition' or 'Classification'
        include_id: If True, format labels as "LABEL (ID)", else just "LABEL"
    """
    # Use all labels from the mapping (sorted by ID) to ensure consistent ordering
    all_labels = sorted([lid for lid in label_mapping.keys()])
    
    # Get label names (with or without ID)
    if include_id:
        label_names = [f"{label_mapping[lid]} ({lid})" for lid in all_labels]
    else:
        label_names = [label_mapping[lid] for lid in all_labels]
    
    # Create confusion matrix (includes all labels, even if not in data)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Create DataFrame with labels as both index and columns
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    
    # Save to CSV
    cm_df.to_csv(output_path)
    print(f"Created {matrix_type} confusion matrix: {output_path}")


def main():
    """Main function to generate all confusion matrix CSV files."""
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.resolve()
    input_dir = project_root / "metrics" / "extract" / "shared_inputs"
    output_dir = script_dir
    
    
    # Load label mappings (needs to be global for extract_sequences)
    global gloss_to_cat
    try:
        gloss_mapping, category_mapping, gloss_to_cat = load_label_mappings()
        print(f"Loaded {len(gloss_mapping)} gloss labels and {len(category_mapping)} categories")
    except Exception as e:
        print(f"Error loading label mappings: {e}")
        return
    
    # Process each model
    models = [
        ("transformer", "Transformer"),
        ("iv3gru", "IV3-GRU")
    ]
    
    for model_key, model_name in models:
        json_path = input_dir / f"ctc_validation_results_{model_key}.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping {model_name}")
            continue
        
        # Load JSON data
        print(f"Processing {model_name}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            continue
        
        # Extract sequences
        try:
            (gloss_gt, gloss_pred), (cat_gt, cat_pred) = extract_sequences(json_data)
            print(f"  Extracted {len(gloss_gt)} gloss pairs and {len(cat_gt)} category pairs")
        except Exception as e:
            print(f"Error extracting sequences: {e}")
            continue
        
        # Filter out invalid labels (-1)
        gloss_gt_clean = [(gt, pred) for gt, pred in zip(gloss_gt, gloss_pred) 
                         if gt >= 0 and pred >= 0 and gt in gloss_mapping and pred in gloss_mapping]
        gloss_gt_filtered, gloss_pred_filtered = zip(*gloss_gt_clean) if gloss_gt_clean else ([], [])
        
        cat_gt_clean = [(gt, pred) for gt, pred in zip(cat_gt, cat_pred) 
                       if gt >= 0 and pred >= 0 and gt in category_mapping and pred in category_mapping]
        cat_gt_filtered, cat_pred_filtered = zip(*cat_gt_clean) if cat_gt_clean else ([], [])
        
        # Create Recognition confusion matrix (gloss-level)
        recognition_output = output_dir / f"Recognition_{model_name} - Confusion Matrix.csv"
        if gloss_gt_filtered:
            create_confusion_matrix_csv(
                list(gloss_gt_filtered), 
                list(gloss_pred_filtered),
                gloss_mapping,
                recognition_output,
                f"Recognition {model_name}",
                include_id=True  # Recognition matrices include ID
            )
        
        # Create Classification confusion matrix (category-level)
        classification_output = output_dir / f"Classification_{model_name} - Confusion Matrix.csv"
        if cat_gt_filtered:
            create_confusion_matrix_csv(
                list(cat_gt_filtered),
                list(cat_pred_filtered),
                category_mapping,
                classification_output,
                f"Classification {model_name}",
                include_id=False  # Classification matrices don't include ID
            )
    
    print("All confusion matrices created successfully!")


if __name__ == "__main__":
    main()

