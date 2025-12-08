"""
Extract classification precision, recall, and f1-score per category
from CTC validation results JSON files.

Usage:
    python metrics/extract/extract_classification.py

Output:
    - Prints precision, recall, and f1-score values per category for both 
      Transformer and IV3-GRU models for both occluded and non-occluded data.
    - Writes two CSV files:
      * classification_metrics_transformer.csv - Transformer metrics with TP, FP, FN
      * classification_metrics_iv3gru.csv - IV3-GRU metrics with TP, FP, FN
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
IV3GRU_JSON = SCRIPT_DIR / "ctc_validation_results_iv3gru.json"
TRANSFORMER_JSON = SCRIPT_DIR / "ctc_validation_results_transformer.json"

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


def extract_category_metrics(json_path, occlusion_filter):
    """
    Extract precision, recall, and f1-score per category for occluded or non-occluded data.
    
    Args:
        json_path: Path to validation results JSON file
        occlusion_filter: 0 for non-occluded, 1 for occluded
        
    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total_gt'
        mapping category_id to metric value or count
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters for all 10 categories
    category_tp = defaultdict(int)
    category_fp = defaultdict(int)
    category_fn = defaultdict(int)
    category_total_gt = defaultdict(int)
    
    for prediction in data['predictions']:
        # Get ground truth and predicted categories
        gt_categories = prediction.get('ground_truth_categories', [])
        pred_categories = prediction.get('predicted_categories', [])
        gt_occluded = prediction.get('ground_truth_occluded', [])
        
        # Count total ground truth items per category for this occlusion type
        for gt_idx, gt_cat in enumerate(gt_categories):
            if gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter:
                category_total_gt[gt_cat] += 1
        
        # Get matched pairs (these are for glosses, but categories align)
        matched_pairs = prediction.get('matched_pairs', [])
        unmatched_pred = prediction.get('unmatched_predictions', [])
        
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
            if gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter:
                matched_gt_indices.add(gt_idx)
                if pred_idx < len(pred_categories) and gt_idx < len(gt_categories):
                    pred_cat = pred_categories[pred_idx]
                    gt_cat = gt_categories[gt_idx]
                    if pred_cat == gt_cat:
                        category_tp[gt_cat] += 1
                    else:
                        # FP for predicted category, FN for ground truth category
                        category_fp[pred_cat] += 1
                        category_fn[gt_cat] += 1
        
        # Process unmatched predictions (FP)
        for pred_idx in unmatched_pred:
            if pred_idx < len(pred_categories):
                pred_cat = pred_categories[pred_idx]
                category_fp[pred_cat] += 1
        
        # Process unmatched ground truth items (FN)
        for gt_idx, gt_cat in enumerate(gt_categories):
            if gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter:
                if gt_idx not in matched_gt_indices:
                    category_fn[gt_cat] += 1
    
    # Calculate precision, recall, and f1-score for each category
    category_precision = {}
    category_recall = {}
    category_f1 = {}
    category_tp_dict = {}
    category_fp_dict = {}
    category_fn_dict = {}
    category_total_gt_dict = {}
    
    for cat_id in range(10):
        tp = category_tp[cat_id]
        fp = category_fp[cat_id]
        fn = category_fn[cat_id]
        total_gt = category_total_gt[cat_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        category_precision[cat_id] = precision
        category_recall[cat_id] = recall
        category_f1[cat_id] = f1
        category_tp_dict[cat_id] = tp
        category_fp_dict[cat_id] = fp
        category_fn_dict[cat_id] = fn
        category_total_gt_dict[cat_id] = total_gt
    
    return {
        'precision': category_precision,
        'recall': category_recall,
        'f1': category_f1,
        'tp': category_tp_dict,
        'fp': category_fp_dict,
        'fn': category_fn_dict,
        'total_gt': category_total_gt_dict
    }


def write_model_csv(model_metrics_nonocc, model_metrics_occ, model_name, output_path):
    """Write CSV output for a single model with its metrics and counts."""
    headers = [
        'Category ID',
        'Category Label',
        'Occlusion',
        'Total Ground Truth',
        'Total TP',
        'Total FP',
        'Total FN',
        'Precision',
        'Recall',
        'F1-score'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write non-occluded data
        for cat_id in range(10):
            cat_label = CATEGORY_NAMES[cat_id]
            
            # Get counts
            total_gt = model_metrics_nonocc['total_gt'].get(cat_id, 0)
            tp = model_metrics_nonocc['tp'].get(cat_id, 0)
            fp = model_metrics_nonocc['fp'].get(cat_id, 0)
            fn = model_metrics_nonocc['fn'].get(cat_id, 0)
            
            # Metrics
            prec = model_metrics_nonocc['precision'].get(cat_id, 0.0)
            recall = model_metrics_nonocc['recall'].get(cat_id, 0.0)
            f1 = model_metrics_nonocc['f1'].get(cat_id, 0.0)
            
            writer.writerow([
                cat_id,
                cat_label,
                'nonoccluded',
                total_gt,
                tp,
                fp,
                fn,
                f'{prec:.6f}',
                f'{recall:.6f}',
                f'{f1:.6f}'
            ])
        
        # Write occluded data
        for cat_id in range(10):
            cat_label = CATEGORY_NAMES[cat_id]
            
            # Get counts
            total_gt = model_metrics_occ['total_gt'].get(cat_id, 0)
            tp = model_metrics_occ['tp'].get(cat_id, 0)
            fp = model_metrics_occ['fp'].get(cat_id, 0)
            fn = model_metrics_occ['fn'].get(cat_id, 0)
            
            # Metrics
            prec = model_metrics_occ['precision'].get(cat_id, 0.0)
            recall = model_metrics_occ['recall'].get(cat_id, 0.0)
            f1 = model_metrics_occ['f1'].get(cat_id, 0.0)
            
            writer.writerow([
                cat_id,
                cat_label,
                'occluded',
                total_gt,
                tp,
                fp,
                fn,
                f'{prec:.6f}',
                f'{recall:.6f}',
                f'{f1:.6f}'
            ])


def print_metrics_table(transformer_metrics, iv3gru_metrics, metric_name, occlusion_type):
    """Print a formatted table for a specific metric."""
    print(f"\n{metric_name.upper()} Values per Category ({occlusion_type})")
    print("=" * 100)
    print(f"\n{'Category':<20} {'Transformer':<15} {'IV3-GRU':<15} {'Difference':<15}")
    print("-" * 100)
    
    for cat_id in range(10):
        cat_name = CATEGORY_NAMES[cat_id]
        trans_val = transformer_metrics.get(cat_id, 0.0)
        gru_val = iv3gru_metrics.get(cat_id, 0.0)
        diff = trans_val - gru_val
        
        print(f"{cat_name:<20} {trans_val:<15.6f} {gru_val:<15.6f} {diff:<15.6f}")
    
    # Summary statistics
    if transformer_metrics and iv3gru_metrics:
        trans_values = [transformer_metrics[i] for i in range(10)]
        gru_values = [iv3gru_metrics[i] for i in range(10)]
        
        trans_mean = sum(trans_values) / len(trans_values)
        gru_mean = sum(gru_values) / len(gru_values)
        mean_diff = trans_mean - gru_mean
        
        print(f"\nMean Transformer {metric_name.capitalize()}: {trans_mean:.6f}")
        print(f"Mean IV3-GRU {metric_name.capitalize()}: {gru_mean:.6f}")
        print(f"Mean Difference: {mean_diff:.6f}")


def main():
    """Main function to extract and print precision, recall, and f1-score values."""
    print("=" * 100)
    print("Classification Metrics Extraction (Precision, Recall, F1-Score)")
    print("=" * 100)
    
    # Process non-occluded data
    print("\n" + "=" * 100)
    print("NON-OCCLUDED DATA")
    print("=" * 100)
    
    print("\nLoading Transformer results (non-occluded)...")
    try:
        transformer_nonocc = extract_category_metrics(TRANSFORMER_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading Transformer results: {e}")
        transformer_nonocc = {'precision': {}, 'recall': {}, 'f1': {}}
    
    print("Loading IV3-GRU results (non-occluded)...")
    try:
        iv3gru_nonocc = extract_category_metrics(IV3GRU_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading IV3-GRU results: {e}")
        iv3gru_nonocc = {'precision': {}, 'recall': {}, 'f1': {}}
    
    # Print non-occluded metrics
    print_metrics_table(transformer_nonocc['precision'], iv3gru_nonocc['precision'], 
                       'Precision', 'Non-Occluded')
    print_metrics_table(transformer_nonocc['recall'], iv3gru_nonocc['recall'], 
                       'Recall', 'Non-Occluded')
    print_metrics_table(transformer_nonocc['f1'], iv3gru_nonocc['f1'], 
                       'F1-Score', 'Non-Occluded')
    
    # Process occluded data
    print("\n\n" + "=" * 100)
    print("OCCLUDED DATA")
    print("=" * 100)
    
    print("\nLoading Transformer results (occluded)...")
    try:
        transformer_occ = extract_category_metrics(TRANSFORMER_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading Transformer results: {e}")
        transformer_occ = {'precision': {}, 'recall': {}, 'f1': {}}
    
    print("Loading IV3-GRU results (occluded)...")
    try:
        iv3gru_occ = extract_category_metrics(IV3GRU_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading IV3-GRU results: {e}")
        iv3gru_occ = {'precision': {}, 'recall': {}, 'f1': {}}
    
    # Print occluded metrics
    print_metrics_table(transformer_occ['precision'], iv3gru_occ['precision'], 
                       'Precision', 'Occluded')
    print_metrics_table(transformer_occ['recall'], iv3gru_occ['recall'], 
                       'Recall', 'Occluded')
    print_metrics_table(transformer_occ['f1'], iv3gru_occ['f1'], 
                       'F1-Score', 'Occluded')
    
    # Write CSV outputs
    print("\n\n" + "=" * 100)
    print("Writing CSV outputs...")
    print("=" * 100)
    
    # Write Transformer CSV
    transformer_csv_path = SCRIPT_DIR / "classification_metrics_transformer.csv"
    try:
        write_model_csv(transformer_nonocc, transformer_occ, 'Transformer', transformer_csv_path)
        print(f"\nTransformer CSV file written successfully: {transformer_csv_path}")
    except Exception as e:
        print(f"\nError writing Transformer CSV file: {e}")
    
    # Write IV3-GRU CSV
    iv3gru_csv_path = SCRIPT_DIR / "classification_metrics_iv3gru.csv"
    try:
        write_model_csv(iv3gru_nonocc, iv3gru_occ, 'IV3-GRU', iv3gru_csv_path)
        print(f"IV3-GRU CSV file written successfully: {iv3gru_csv_path}")
    except Exception as e:
        print(f"Error writing IV3-GRU CSV file: {e}")


if __name__ == "__main__":
    main()

