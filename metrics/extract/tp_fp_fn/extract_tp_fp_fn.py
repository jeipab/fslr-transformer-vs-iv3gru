"""
Extract TP, FP, FN counts per Ground Truth Gloss and Category from CTC validation results.
Includes breakdown by occluded, non-occluded, and combined.

Usage:
    python metrics/extract/tp_fp_fn/extract_tp_fp_fn.py

Output:
    Creates two CSV files:
    - gloss_tp_fp_fn_counts.csv - TP, FP, FN counts for each of the 105 glosses
    - category_tp_fp_fn_counts.csv - TP, FP, FN counts for each of the 10 categories
    Both include breakdown by occluded, non-occluded, and combined for both models.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
EXTRACT_DIR = SCRIPT_DIR.parent
SHARED_INPUTS_DIR = EXTRACT_DIR / "shared_inputs"
IV3GRU_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_iv3gru.json"
TRANSFORMER_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_transformer.json"
LABELS_CSV = EXTRACT_DIR.parent.parent / "data" / "labels_reference.csv"
GLOSS_OUTPUT_CSV = SCRIPT_DIR / "gloss_tp_fp_fn_counts.csv"
CATEGORY_OUTPUT_CSV = SCRIPT_DIR / "category_tp_fp_fn_counts.csv"

# Category mapping (0-9)
CATEGORY_NAMES = {
    0: "GREETING",
    1: "SURVIVAL",
    2: "NUMBER",
    3: "CALENDAR",
    4: "DAYS",
    5: "FAMILY",
    6: "RELATIONSHIPS",
    7: "COLOR",
    8: "FOOD",
    9: "DRINK"
}


def load_label_mapping():
    """Load gloss ID to label mapping."""
    df = pd.read_csv(LABELS_CSV)
    return dict(zip(df['gloss_id'], df['label']))


def extract_gloss_counts(json_path, occlusion_filter=None):
    """
    Extract TP, FP, FN counts per gloss from validation results.
    
    Args:
        json_path: Path to validation results JSON file
        occlusion_filter: 0 for non-occluded, 1 for occluded, None for combined
        
    Returns:
        Dictionary mapping gloss_id to {'tp': count, 'fp': count, 'fn': count, 'total_gt': count}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters for all 105 glosses (0-104)
    gloss_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})
    
    for prediction in data['predictions']:
        gt_sequence = prediction.get('ground_truth_sequence', [])
        pred_sequence = prediction.get('predicted_sequence', [])
        gt_occluded = prediction.get('ground_truth_occluded', [])
        matched_pairs = prediction.get('matched_pairs', [])
        unmatched_pred = prediction.get('unmatched_predictions', [])
        
        # Count total ground truth items per gloss
        for gt_idx, gt_gloss in enumerate(gt_sequence):
            if occlusion_filter is None or (gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter):
                gloss_counts[gt_gloss]['total_gt'] += 1
        
        # Create a mapping from pred_idx to gt_idx for matched pairs
        pred_to_gt = {}
        for pair in matched_pairs:
            pred_idx = pair.get('pred_idx')
            gt_idx = pair.get('gt_idx')
            if pred_idx is not None and gt_idx is not None:
                pred_to_gt[pred_idx] = gt_idx
        
        # Track which ground truth items have been matched
        matched_gt_indices = set()
        
        # Process matched pairs
        for pred_idx, gt_idx in pred_to_gt.items():
            if occlusion_filter is None or (gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter):
                matched_gt_indices.add(gt_idx)
                if pred_idx < len(pred_sequence) and gt_idx < len(gt_sequence):
                    pred_gloss = pred_sequence[pred_idx]
                    gt_gloss = gt_sequence[gt_idx]
                    if pred_gloss == gt_gloss:
                        gloss_counts[gt_gloss]['tp'] += 1
                    else:
                        # FP for predicted gloss, FN for ground truth gloss
                        gloss_counts[pred_gloss]['fp'] += 1
                        gloss_counts[gt_gloss]['fn'] += 1
        
        # Process unmatched predictions (FP)
        for pred_idx in unmatched_pred:
            if pred_idx < len(pred_sequence):
                pred_gloss = pred_sequence[pred_idx]
                gloss_counts[pred_gloss]['fp'] += 1
        
        # Process unmatched ground truth items (FN)
        for gt_idx, gt_gloss in enumerate(gt_sequence):
            if occlusion_filter is None or (gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter):
                if gt_idx not in matched_gt_indices:
                    gloss_counts[gt_gloss]['fn'] += 1
    
    return gloss_counts


def extract_category_counts(json_path, occlusion_filter=None):
    """
    Extract TP, FP, FN counts per category from validation results.
    
    Args:
        json_path: Path to validation results JSON file
        occlusion_filter: 0 for non-occluded, 1 for occluded, None for combined
        
    Returns:
        Dictionary mapping category_id to {'tp': count, 'fp': count, 'fn': count, 'total_gt': count}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters for all 10 categories (0-9)
    category_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})
    
    for prediction in data['predictions']:
        gt_categories = prediction.get('ground_truth_categories', [])
        pred_categories = prediction.get('predicted_categories', [])
        gt_occluded = prediction.get('ground_truth_occluded', [])
        matched_pairs = prediction.get('matched_pairs', [])
        unmatched_pred = prediction.get('unmatched_predictions', [])
        
        # Count total ground truth items per category
        for gt_idx, gt_cat in enumerate(gt_categories):
            if occlusion_filter is None or (gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter):
                category_counts[gt_cat]['total_gt'] += 1
        
        # Create a mapping from pred_idx to gt_idx for matched pairs
        pred_to_gt = {}
        for pair in matched_pairs:
            pred_idx = pair.get('pred_idx')
            gt_idx = pair.get('gt_idx')
            if pred_idx is not None and gt_idx is not None:
                pred_to_gt[pred_idx] = gt_idx
        
        # Track which ground truth items have been matched
        matched_gt_indices = set()
        
        # Process matched pairs
        for pred_idx, gt_idx in pred_to_gt.items():
            if occlusion_filter is None or (gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter):
                matched_gt_indices.add(gt_idx)
                if pred_idx < len(pred_categories) and gt_idx < len(gt_categories):
                    pred_cat = pred_categories[pred_idx]
                    gt_cat = gt_categories[gt_idx]
                    if pred_cat == gt_cat:
                        category_counts[gt_cat]['tp'] += 1
                    else:
                        # FP for predicted category, FN for ground truth category
                        category_counts[pred_cat]['fp'] += 1
                        category_counts[gt_cat]['fn'] += 1
        
        # Process unmatched predictions (FP)
        for pred_idx in unmatched_pred:
            if pred_idx < len(pred_categories):
                pred_cat = pred_categories[pred_idx]
                category_counts[pred_cat]['fp'] += 1
        
        # Process unmatched ground truth items (FN)
        for gt_idx, gt_cat in enumerate(gt_categories):
            if occlusion_filter is None or (gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter):
                if gt_idx not in matched_gt_indices:
                    category_counts[gt_cat]['fn'] += 1
    
    return category_counts


def main():
    """Main function to extract and save gloss and category counts."""
    print("Loading label mapping...")
    label_mapping = load_label_mapping()
    
    print("\nExtracting gloss counts from iv3gru results...")
    iv3gru_gloss_occ = extract_gloss_counts(IV3GRU_JSON, occlusion_filter=1)
    iv3gru_gloss_nonocc = extract_gloss_counts(IV3GRU_JSON, occlusion_filter=0)
    iv3gru_gloss_combined = extract_gloss_counts(IV3GRU_JSON, occlusion_filter=None)
    
    print("Extracting gloss counts from transformer results...")
    transformer_gloss_occ = extract_gloss_counts(TRANSFORMER_JSON, occlusion_filter=1)
    transformer_gloss_nonocc = extract_gloss_counts(TRANSFORMER_JSON, occlusion_filter=0)
    transformer_gloss_combined = extract_gloss_counts(TRANSFORMER_JSON, occlusion_filter=None)
    
    print("\nExtracting category counts from iv3gru results...")
    iv3gru_cat_occ = extract_category_counts(IV3GRU_JSON, occlusion_filter=1)
    iv3gru_cat_nonocc = extract_category_counts(IV3GRU_JSON, occlusion_filter=0)
    iv3gru_cat_combined = extract_category_counts(IV3GRU_JSON, occlusion_filter=None)
    
    print("Extracting category counts from transformer results...")
    transformer_cat_occ = extract_category_counts(TRANSFORMER_JSON, occlusion_filter=1)
    transformer_cat_nonocc = extract_category_counts(TRANSFORMER_JSON, occlusion_filter=0)
    transformer_cat_combined = extract_category_counts(TRANSFORMER_JSON, occlusion_filter=None)
    
    # Create DataFrame for gloss counts
    print("\nCreating gloss counts DataFrame...")
    gloss_results = []
    for gloss_id in range(105):
        gloss_label = label_mapping.get(gloss_id, f"Unknown_{gloss_id}")
        gloss_results.append({
            'gloss_id': gloss_id,
            'gloss_label': gloss_label,
            'iv3gru_occluded_tp': iv3gru_gloss_occ[gloss_id]['tp'],
            'iv3gru_occluded_fp': iv3gru_gloss_occ[gloss_id]['fp'],
            'iv3gru_occluded_fn': iv3gru_gloss_occ[gloss_id]['fn'],
            'iv3gru_occluded_count': iv3gru_gloss_occ[gloss_id]['total_gt'],
            'iv3gru_nonoccluded_tp': iv3gru_gloss_nonocc[gloss_id]['tp'],
            'iv3gru_nonoccluded_fp': iv3gru_gloss_nonocc[gloss_id]['fp'],
            'iv3gru_nonoccluded_fn': iv3gru_gloss_nonocc[gloss_id]['fn'],
            'iv3gru_nonoccluded_count': iv3gru_gloss_nonocc[gloss_id]['total_gt'],
            'iv3gru_combined_tp': iv3gru_gloss_combined[gloss_id]['tp'],
            'iv3gru_combined_fp': iv3gru_gloss_combined[gloss_id]['fp'],
            'iv3gru_combined_fn': iv3gru_gloss_combined[gloss_id]['fn'],
            'iv3gru_combined_count': iv3gru_gloss_combined[gloss_id]['total_gt'],
            'transformer_occluded_tp': transformer_gloss_occ[gloss_id]['tp'],
            'transformer_occluded_fp': transformer_gloss_occ[gloss_id]['fp'],
            'transformer_occluded_fn': transformer_gloss_occ[gloss_id]['fn'],
            'transformer_occluded_count': transformer_gloss_occ[gloss_id]['total_gt'],
            'transformer_nonoccluded_tp': transformer_gloss_nonocc[gloss_id]['tp'],
            'transformer_nonoccluded_fp': transformer_gloss_nonocc[gloss_id]['fp'],
            'transformer_nonoccluded_fn': transformer_gloss_nonocc[gloss_id]['fn'],
            'transformer_nonoccluded_count': transformer_gloss_nonocc[gloss_id]['total_gt'],
            'transformer_combined_tp': transformer_gloss_combined[gloss_id]['tp'],
            'transformer_combined_fp': transformer_gloss_combined[gloss_id]['fp'],
            'transformer_combined_fn': transformer_gloss_combined[gloss_id]['fn'],
            'transformer_combined_count': transformer_gloss_combined[gloss_id]['total_gt'],
        })
    
    gloss_df = pd.DataFrame(gloss_results)
    gloss_df = gloss_df.sort_values('gloss_id')
    
    # Create DataFrame for category counts
    print("Creating category counts DataFrame...")
    category_results = []
    for cat_id in range(10):
        cat_label = CATEGORY_NAMES[cat_id]
        category_results.append({
            'category_id': cat_id,
            'category_label': cat_label,
            'iv3gru_occluded_tp': iv3gru_cat_occ[cat_id]['tp'],
            'iv3gru_occluded_fp': iv3gru_cat_occ[cat_id]['fp'],
            'iv3gru_occluded_fn': iv3gru_cat_occ[cat_id]['fn'],
            'iv3gru_occluded_count': iv3gru_cat_occ[cat_id]['total_gt'],
            'iv3gru_nonoccluded_tp': iv3gru_cat_nonocc[cat_id]['tp'],
            'iv3gru_nonoccluded_fp': iv3gru_cat_nonocc[cat_id]['fp'],
            'iv3gru_nonoccluded_fn': iv3gru_cat_nonocc[cat_id]['fn'],
            'iv3gru_nonoccluded_count': iv3gru_cat_nonocc[cat_id]['total_gt'],
            'iv3gru_combined_tp': iv3gru_cat_combined[cat_id]['tp'],
            'iv3gru_combined_fp': iv3gru_cat_combined[cat_id]['fp'],
            'iv3gru_combined_fn': iv3gru_cat_combined[cat_id]['fn'],
            'iv3gru_combined_count': iv3gru_cat_combined[cat_id]['total_gt'],
            'transformer_occluded_tp': transformer_cat_occ[cat_id]['tp'],
            'transformer_occluded_fp': transformer_cat_occ[cat_id]['fp'],
            'transformer_occluded_fn': transformer_cat_occ[cat_id]['fn'],
            'transformer_occluded_count': transformer_cat_occ[cat_id]['total_gt'],
            'transformer_nonoccluded_tp': transformer_cat_nonocc[cat_id]['tp'],
            'transformer_nonoccluded_fp': transformer_cat_nonocc[cat_id]['fp'],
            'transformer_nonoccluded_fn': transformer_cat_nonocc[cat_id]['fn'],
            'transformer_nonoccluded_count': transformer_cat_nonocc[cat_id]['total_gt'],
            'transformer_combined_tp': transformer_cat_combined[cat_id]['tp'],
            'transformer_combined_fp': transformer_cat_combined[cat_id]['fp'],
            'transformer_combined_fn': transformer_cat_combined[cat_id]['fn'],
            'transformer_combined_count': transformer_cat_combined[cat_id]['total_gt'],
        })
    
    category_df = pd.DataFrame(category_results)
    category_df = category_df.sort_values('category_id')
    
    # Save to CSV
    gloss_df.to_csv(GLOSS_OUTPUT_CSV, index=False)
    category_df.to_csv(CATEGORY_OUTPUT_CSV, index=False)
    
    print(f"\nResults saved:")
    print(f"  Gloss counts: {GLOSS_OUTPUT_CSV}")
    print(f"  Category counts: {CATEGORY_OUTPUT_CSV}")
    print(f"\nTotal glosses: {len(gloss_df)}")
    print(f"Total categories: {len(category_df)}")
    print(f"\nSample gloss output:")
    print(gloss_df.head(5).to_string(index=False))
    print(f"\nSample category output:")
    print(category_df.to_string(index=False))


if __name__ == "__main__":
    main()
