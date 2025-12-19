"""
Analyze confusion errors from CTC validation results.

This script generates 4 CSV files analyzing confusions:
- error_analysis_recognition_transformer.csv (gloss-level)
- error_analysis_recognition_iv3gru.csv (gloss-level)
- error_analysis_classification_transformer.csv (category-level)
- error_analysis_classification_iv3gru.csv (category-level)

Usage:
    python metrics/confusion/error_analysis.py

Input files (must exist):
    metrics/extract/shared_inputs/ctc_validation_results_transformer.json
    metrics/extract/shared_inputs/ctc_validation_results_iv3gru.json

Output files (saved to metrics\confusion):
    error_analysis_recognition_transformer.csv
    error_analysis_recognition_iv3gru.csv
    error_analysis_classification_transformer.csv
    error_analysis_classification_iv3gru.csv

Columns:
    - Actual Gloss/Category Label: The ground truth label
    - Total Ground Truth: Total occurrences of this label in ground truth
    - Predicted Gloss/Category Label: What it was confused as
    - Number of Confusions: Count of this specific confusion
    - Percentage of Confusions: (Number of Confusions / Total Ground Truth) * 100
    - Interpretation: Low (<=5.0%), Moderate (>5.0% and <=10.0%), High (>10.0%)
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


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
    """
    Extract ground truth and predicted sequences from JSON data.
    
    Returns pairs of (gt, pred) for both gloss and category levels.
    This matches the logic used in create_matrix.py.
    """
    predictions = json_data.get('predictions', [])
    
    # Collect all gloss pairs (for recognition)
    gloss_pairs = []
    
    # Collect all category pairs (for classification)
    cat_pairs = []
    
    for pred in predictions:
        # For recognition: pair gloss sequences element-wise
        if 'ground_truth_sequence' in pred and 'predicted_sequence' in pred:
            gt_seq = pred.get('ground_truth_sequence', [])
            pr_seq = pred.get('predicted_sequence', [])
            # Use zip to pair elements at the same index position
            for gt_id, pr_id in zip(gt_seq, pr_seq):
                gloss_pairs.append((int(gt_id), int(pr_id)))
        
        # For classification: pair category sequences element-wise
        gt_cats = pred.get('ground_truth_categories', [])
        pr_cats = pred.get('predicted_categories', [])
        
        if gt_cats and pr_cats:
            # Use zip to pair elements at the same index position
            for gc, pc in zip(gt_cats, pr_cats):
                cat_pairs.append((int(gc), int(pc)))
        else:
            # Fallback: derive categories from gloss sequences if not available
            if 'ground_truth_sequence' in pred and 'predicted_sequence' in pred:
                gt_seq = pred.get('ground_truth_sequence', [])
                pr_seq = pred.get('predicted_sequence', [])
                for gt_gid, pr_gid in zip(gt_seq, pr_seq):
                    gt_cat = gloss_to_cat.get(int(gt_gid), -1)
                    pr_cat = gloss_to_cat.get(int(pr_gid), -1)
                    if gt_cat >= 0 and pr_cat >= 0:
                        cat_pairs.append((gt_cat, pr_cat))
    
    return gloss_pairs, cat_pairs


def calculate_confusion_analysis(pairs: List[Tuple[int, int]], label_mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Calculate confusion analysis from (gt, pred) pairs.
    
    Args:
        pairs: List of (ground_truth_id, predicted_id) tuples
        label_mapping: Dictionary mapping IDs to label names
        
    Returns:
        DataFrame with confusion analysis
    """
    # Count total occurrences of each GT label
    gt_counts = defaultdict(int)
    
    # Count confusions: (gt_label, pred_label) -> count
    confusion_counts = defaultdict(int)
    
    # Process all pairs
    for gt_id, pred_id in pairs:
        # Filter out invalid IDs
        if gt_id < 0 or pred_id < 0 or gt_id not in label_mapping or pred_id not in label_mapping:
            continue
        
        gt_label = label_mapping[gt_id]
        pred_label = label_mapping[pred_id]
        
        # Count total GT occurrences
        gt_counts[gt_label] += 1
        
        # Count confusions (only when GT != Pred)
        if gt_id != pred_id:
            confusion_counts[(gt_label, pred_label)] += 1
    
    # Build results list
    results = []
    
    # For each confusion pair, calculate metrics
    for (gt_label, pred_label), confusion_count in confusion_counts.items():
        total_gt = gt_counts[gt_label]
        if total_gt > 0:
            percentage = (confusion_count / total_gt) * 100
            
            # Determine interpretation
            if percentage <= 5.0:
                interpretation = "Low Confusion"
            elif 5.0 < percentage <= 10.0:
                interpretation = "Moderate Confusion"
            else:  # percentage > 10.0
                interpretation = "High Confusion"
            
            results.append({
                'Actual Gloss/Category Label': gt_label,
                'Total Ground Truth': total_gt,
                'Predicted Gloss/Category Label': pred_label,
                'Number of Confusions': confusion_count,
                'Percentage of Confusions': round(percentage, 2),
                'Interpretation': interpretation
            })
    
    # Sort by Actual Label, then by Percentage (descending)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['Actual Gloss/Category Label', 'Percentage of Confusions'], 
                           ascending=[True, False])
    
    return df


def main():
    """Main function to generate all error analysis CSV files."""
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
        print(f"\nProcessing {model_name}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            continue
        
        # Extract sequences
        try:
            gloss_pairs, cat_pairs = extract_sequences(json_data)
            print(f"  Extracted {len(gloss_pairs)} gloss pairs")
            print(f"  Extracted {len(cat_pairs)} category pairs")
        except Exception as e:
            print(f"Error extracting sequences: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Filter out invalid labels and ensure labels are in mapping
        gloss_pairs_filtered = [(gt, pred) for gt, pred in gloss_pairs
                              if gt >= 0 and pred >= 0 and gt in gloss_mapping and pred in gloss_mapping]
        cat_pairs_filtered = [(gt, pred) for gt, pred in cat_pairs
                            if gt >= 0 and pred >= 0 and gt in category_mapping and pred in category_mapping]
        
        print(f"  Valid gloss pairs: {len(gloss_pairs_filtered)}")
        print(f"  Valid category pairs: {len(cat_pairs_filtered)}")
        
        # Create Recognition error analysis (gloss-level)
        if gloss_pairs_filtered:
            print(f"  Analyzing recognition confusions...")
            rec_df = calculate_confusion_analysis(gloss_pairs_filtered, gloss_mapping)
            rec_output = output_dir / f"error_analysis_recognition_{model_key}.csv"
            rec_df.to_csv(rec_output, index=False)
            print(f"  Saved {len(rec_df)} confusion entries to {rec_output.name}")
            if len(rec_df) > 0:
                print(f"    Sample: {rec_df.head(3).to_string(index=False)}")
        
        # Create Classification error analysis (category-level)
        if cat_pairs_filtered:
            print(f"  Analyzing classification confusions...")
            cls_df = calculate_confusion_analysis(cat_pairs_filtered, category_mapping)
            cls_output = output_dir / f"error_analysis_classification_{model_key}.csv"
            cls_df.to_csv(cls_output, index=False)
            print(f"  Saved {len(cls_df)} confusion entries to {cls_output.name}")
            if len(cls_df) > 0:
                print(f"    Sample: {cls_df.head(3).to_string(index=False)}")
    
    print("\n" + "=" * 60)
    print("All error analysis files created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

