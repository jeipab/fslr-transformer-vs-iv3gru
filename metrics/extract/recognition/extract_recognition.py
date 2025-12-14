"""
Extract recognition (gloss) precision, recall, and f1-score per gloss
from CTC validation results JSON files.

Usage:
    python metrics/extract/recognition/extract_recognition.py

Output:
    - Prints precision, recall, and f1-score values per gloss for both 
      Transformer and IV3-GRU models for both occluded and non-occluded data.
    - Writes two CSV files:
      * recognition_metrics_transformer.csv - Transformer metrics with TP, FP, FN
      * recognition_metrics_iv3gru.csv - IV3-GRU metrics with TP, FP, FN
"""

import json
import csv
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


def load_label_mapping():
    """Load gloss ID to label mapping."""
    df = pd.read_csv(LABELS_CSV)
    return dict(zip(df['gloss_id'], df['label']))


def extract_gloss_metrics(json_path, occlusion_filter):
    """
    Extract precision, recall, and f1-score per gloss for occluded or non-occluded data.
    
    Args:
        json_path: Path to validation results JSON file
        occlusion_filter: 0 for non-occluded, 1 for occluded
        
    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total_gt'
        mapping gloss_id to metric value or count
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters for all 105 glosses (0-104)
    gloss_tp = defaultdict(int)
    gloss_fp = defaultdict(int)
    gloss_fn = defaultdict(int)
    gloss_total_gt = defaultdict(int)
    
    for prediction in data['predictions']:
        # Get ground truth and predicted sequences
        gt_sequence = prediction.get('ground_truth_sequence', [])
        pred_sequence = prediction.get('predicted_sequence', [])
        gt_occluded = prediction.get('ground_truth_occluded', [])
        
        # Count total ground truth items per gloss for this occlusion type
        for gt_idx, gt_gloss in enumerate(gt_sequence):
            if gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter:
                gloss_total_gt[gt_gloss] += 1
        
        # Get matched pairs
        matched_pairs = prediction.get('matched_pairs', [])
        unmatched_pred = prediction.get('unmatched_predictions', [])
        unmatched_gt = prediction.get('unmatched_ground_truth', [])
        
        # Create a mapping from pred_idx to gt_idx for matched pairs
        pred_to_gt = {}
        for pair in matched_pairs:
            pred_idx = pair.get('pred_idx')
            gt_idx = pair.get('gt_idx')
            if pred_idx is not None and gt_idx is not None:
                pred_to_gt[pred_idx] = gt_idx
        
        # Process matched pairs
        for pred_idx, gt_idx in pred_to_gt.items():
            if gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter:
                if pred_idx < len(pred_sequence) and gt_idx < len(gt_sequence):
                    pred_gloss = pred_sequence[pred_idx]
                    gt_gloss = gt_sequence[gt_idx]
                    if pred_gloss == gt_gloss:
                        gloss_tp[gt_gloss] += 1
                    else:
                        # FP for predicted gloss, FN for ground truth gloss
                        gloss_fp[pred_gloss] += 1
                        gloss_fn[gt_gloss] += 1
        
        # Process unmatched predictions (FP)
        for pred_idx in unmatched_pred:
            if pred_idx < len(pred_sequence):
                pred_gloss = pred_sequence[pred_idx]
                gloss_fp[pred_gloss] += 1
        
        # Process unmatched ground truth items (FN) - use the field from JSON
        for gt_idx in unmatched_gt:
            if gt_idx < len(gt_sequence) and gt_idx < len(gt_occluded) and gt_occluded[gt_idx] == occlusion_filter:
                gt_gloss = gt_sequence[gt_idx]
                gloss_fn[gt_gloss] += 1
    
    # Calculate precision, recall, and f1-score for each gloss
    gloss_precision = {}
    gloss_recall = {}
    gloss_f1 = {}
    gloss_tp_dict = {}
    gloss_fp_dict = {}
    gloss_fn_dict = {}
    gloss_total_gt_dict = {}
    
    for gloss_id in range(105):
        tp = gloss_tp[gloss_id]
        fp = gloss_fp[gloss_id]
        fn = gloss_fn[gloss_id]
        total_gt = gloss_total_gt[gloss_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        gloss_precision[gloss_id] = precision
        gloss_recall[gloss_id] = recall
        gloss_f1[gloss_id] = f1
        gloss_tp_dict[gloss_id] = tp
        gloss_fp_dict[gloss_id] = fp
        gloss_fn_dict[gloss_id] = fn
        gloss_total_gt_dict[gloss_id] = total_gt
    
    return {
        'precision': gloss_precision,
        'recall': gloss_recall,
        'f1': gloss_f1,
        'tp': gloss_tp_dict,
        'fp': gloss_fp_dict,
        'fn': gloss_fn_dict,
        'total_gt': gloss_total_gt_dict
    }


def write_model_csv(model_metrics_nonocc, model_metrics_occ, model_name, output_path, label_mapping):
    """Write CSV output for a single model with its metrics and counts."""
    headers = [
        'Gloss ID',
        'Gloss Label',
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
        for gloss_id in range(105):
            gloss_label = label_mapping.get(gloss_id, f"Unknown_{gloss_id}")
            
            # Get counts
            total_gt = model_metrics_nonocc['total_gt'].get(gloss_id, 0)
            tp = model_metrics_nonocc['tp'].get(gloss_id, 0)
            fp = model_metrics_nonocc['fp'].get(gloss_id, 0)
            fn = model_metrics_nonocc['fn'].get(gloss_id, 0)
            
            # Metrics
            prec = model_metrics_nonocc['precision'].get(gloss_id, 0.0)
            recall = model_metrics_nonocc['recall'].get(gloss_id, 0.0)
            f1 = model_metrics_nonocc['f1'].get(gloss_id, 0.0)
            
            writer.writerow([
                gloss_id,
                gloss_label,
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
        for gloss_id in range(105):
            gloss_label = label_mapping.get(gloss_id, f"Unknown_{gloss_id}")
            
            # Get counts
            total_gt = model_metrics_occ['total_gt'].get(gloss_id, 0)
            tp = model_metrics_occ['tp'].get(gloss_id, 0)
            fp = model_metrics_occ['fp'].get(gloss_id, 0)
            fn = model_metrics_occ['fn'].get(gloss_id, 0)
            
            # Metrics
            prec = model_metrics_occ['precision'].get(gloss_id, 0.0)
            recall = model_metrics_occ['recall'].get(gloss_id, 0.0)
            f1 = model_metrics_occ['f1'].get(gloss_id, 0.0)
            
            writer.writerow([
                gloss_id,
                gloss_label,
                'occluded',
                total_gt,
                tp,
                fp,
                fn,
                f'{prec:.6f}',
                f'{recall:.6f}',
                f'{f1:.6f}'
            ])


def print_metrics_table(transformer_metrics, iv3gru_metrics, metric_name, occlusion_type, label_mapping):
    """Print a formatted table for a specific metric."""
    print(f"\n{metric_name.upper()} Values per Gloss ({occlusion_type})")
    print("=" * 120)
    print(f"\n{'Gloss ID':<10} {'Gloss Label':<30} {'Transformer':<15} {'IV3-GRU':<15} {'Difference':<15}")
    print("-" * 120)
    
    for gloss_id in range(105):
        gloss_label = label_mapping.get(gloss_id, f"Unknown_{gloss_id}")
        trans_val = transformer_metrics.get(gloss_id, 0.0)
        gru_val = iv3gru_metrics.get(gloss_id, 0.0)
        diff = trans_val - gru_val
        
        print(f"{gloss_id:<10} {gloss_label:<30} {trans_val:<15.6f} {gru_val:<15.6f} {diff:<15.6f}")
    
    # Summary statistics
    if transformer_metrics and iv3gru_metrics:
        trans_values = [transformer_metrics[i] for i in range(105)]
        gru_values = [iv3gru_metrics[i] for i in range(105)]
        
        trans_mean = sum(trans_values) / len(trans_values)
        gru_mean = sum(gru_values) / len(gru_values)
        mean_diff = trans_mean - gru_mean
        
        print(f"\nMean Transformer {metric_name.capitalize()}: {trans_mean:.6f}")
        print(f"Mean IV3-GRU {metric_name.capitalize()}: {gru_mean:.6f}")
        print(f"Mean Difference: {mean_diff:.6f}")


def main():
    """Main function to extract and print precision, recall, and f1-score values."""
    print("=" * 120)
    print("Recognition (Gloss) Metrics Extraction (Precision, Recall, F1-Score)")
    print("=" * 120)
    
    print("\nLoading label mapping...")
    try:
        label_mapping = load_label_mapping()
    except Exception as e:
        print(f"Error loading label mapping: {e}")
        label_mapping = {}
    
    # Process non-occluded data
    print("\n" + "=" * 120)
    print("NON-OCCLUDED DATA")
    print("=" * 120)
    
    print("\nLoading Transformer results (non-occluded)...")
    try:
        transformer_nonocc = extract_gloss_metrics(TRANSFORMER_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading Transformer results: {e}")
        transformer_nonocc = {'precision': {}, 'recall': {}, 'f1': {}}
    
    print("Loading IV3-GRU results (non-occluded)...")
    try:
        iv3gru_nonocc = extract_gloss_metrics(IV3GRU_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading IV3-GRU results: {e}")
        iv3gru_nonocc = {'precision': {}, 'recall': {}, 'f1': {}}
    
    # Print non-occluded metrics (only show summary, not all 105 glosses)
    print("\nNote: Showing summary statistics only. Full details available in CSV files.")
    print_metrics_table(transformer_nonocc['precision'], iv3gru_nonocc['precision'], 
                       'Precision', 'Non-Occluded', label_mapping)
    print_metrics_table(transformer_nonocc['recall'], iv3gru_nonocc['recall'], 
                       'Recall', 'Non-Occluded', label_mapping)
    print_metrics_table(transformer_nonocc['f1'], iv3gru_nonocc['f1'], 
                       'F1-Score', 'Non-Occluded', label_mapping)
    
    # Process occluded data
    print("\n\n" + "=" * 120)
    print("OCCLUDED DATA")
    print("=" * 120)
    
    print("\nLoading Transformer results (occluded)...")
    try:
        transformer_occ = extract_gloss_metrics(TRANSFORMER_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading Transformer results: {e}")
        transformer_occ = {'precision': {}, 'recall': {}, 'f1': {}}
    
    print("Loading IV3-GRU results (occluded)...")
    try:
        iv3gru_occ = extract_gloss_metrics(IV3GRU_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading IV3-GRU results: {e}")
        iv3gru_occ = {'precision': {}, 'recall': {}, 'f1': {}}
    
    # Print occluded metrics
    print_metrics_table(transformer_occ['precision'], iv3gru_occ['precision'], 
                       'Precision', 'Occluded', label_mapping)
    print_metrics_table(transformer_occ['recall'], iv3gru_occ['recall'], 
                       'Recall', 'Occluded', label_mapping)
    print_metrics_table(transformer_occ['f1'], iv3gru_occ['f1'], 
                       'F1-Score', 'Occluded', label_mapping)
    
    # Write CSV outputs
    print("\n\n" + "=" * 120)
    print("Writing CSV outputs...")
    print("=" * 120)
    
    # Write Transformer CSV
    transformer_csv_path = SCRIPT_DIR / "recognition_metrics_transformer.csv"
    try:
        write_model_csv(transformer_nonocc, transformer_occ, 'Transformer', transformer_csv_path, label_mapping)
        print(f"\nTransformer CSV file written successfully: {transformer_csv_path}")
    except Exception as e:
        print(f"\nError writing Transformer CSV file: {e}")
    
    # Write IV3-GRU CSV
    iv3gru_csv_path = SCRIPT_DIR / "recognition_metrics_iv3gru.csv"
    try:
        write_model_csv(iv3gru_nonocc, iv3gru_occ, 'IV3-GRU', iv3gru_csv_path, label_mapping)
        print(f"IV3-GRU CSV file written successfully: {iv3gru_csv_path}")
    except Exception as e:
        print(f"Error writing IV3-GRU CSV file: {e}")


if __name__ == "__main__":
    main()

