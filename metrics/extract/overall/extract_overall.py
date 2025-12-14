"""
Extract overall precision, recall, and f1-score metrics (regardless of occlusion status)
from CTC validation results JSON files.

Usage:
    python metrics/extract/overall/extract_overall.py

Output:
    - overall_metrics.csv: CSV file with columns Metric, Transformer, IV3-GRU
      containing overall Precision, Recall, F1-Score, Total TP, Total FP, and Total FN for both models
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
EXTRACT_DIR = SCRIPT_DIR.parent
SHARED_INPUTS_DIR = EXTRACT_DIR / "shared_inputs"
IV3GRU_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_iv3gru.json"
TRANSFORMER_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_transformer.json"
OUTPUT_CSV = SCRIPT_DIR / "overall_metrics.csv"


def extract_overall_metrics(json_path):
    """
    Extract overall precision, recall, and f1-score metrics regardless of occlusion status.
    
    Args:
        json_path: Path to validation results JSON file
        
    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for prediction in data['predictions']:
        # Get ground truth and predicted sequences
        gt_sequence = prediction.get('ground_truth_sequence', [])
        pred_sequence = prediction.get('predicted_sequence', [])
        
        # Get matched pairs
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
            matched_gt_indices.add(gt_idx)
            if pred_idx < len(pred_sequence) and gt_idx < len(gt_sequence):
                pred_gloss = pred_sequence[pred_idx]
                gt_gloss = gt_sequence[gt_idx]
                if pred_gloss == gt_gloss:
                    total_tp += 1
                else:
                    # FP for predicted gloss, FN for ground truth gloss
                    total_fp += 1
                    total_fn += 1
        
        # Process unmatched predictions (FP)
        total_fp += len(unmatched_pred)
        
        # Process unmatched ground truth items (FN)
        for gt_idx, gt_gloss in enumerate(gt_sequence):
            if gt_idx not in matched_gt_indices:
                total_fn += 1
    
    # Calculate precision, recall, and f1-score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def write_overall_csv(transformer_metrics, iv3gru_metrics, output_path):
    """Write CSV output with overall metrics for both models."""
    headers = ['Metric', 'Transformer', 'IV3-GRU']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write Precision
        writer.writerow([
            'Precision',
            f'{transformer_metrics["precision"]:.6f}',
            f'{iv3gru_metrics["precision"]:.6f}'
        ])
        
        # Write Recall
        writer.writerow([
            'Recall',
            f'{transformer_metrics["recall"]:.6f}',
            f'{iv3gru_metrics["recall"]:.6f}'
        ])
        
        # Write F1-Score
        writer.writerow([
            'F1-Score',
            f'{transformer_metrics["f1"]:.6f}',
            f'{iv3gru_metrics["f1"]:.6f}'
        ])
        
        # Write Total TP
        writer.writerow([
            'Total TP',
            f'{transformer_metrics["tp"]}',
            f'{iv3gru_metrics["tp"]}'
        ])
        
        # Write Total FP
        writer.writerow([
            'Total FP',
            f'{transformer_metrics["fp"]}',
            f'{iv3gru_metrics["fp"]}'
        ])
        
        # Write Total FN
        writer.writerow([
            'Total FN',
            f'{transformer_metrics["fn"]}',
            f'{iv3gru_metrics["fn"]}'
        ])


def main():
    """Main function to extract and write overall metrics."""
    print("=" * 80)
    print("Overall Metrics Extraction (Precision, Recall, F1-Score)")
    print("=" * 80)
    
    # Load Transformer results
    print("\nLoading Transformer results...")
    try:
        transformer_metrics = extract_overall_metrics(TRANSFORMER_JSON)
        print(f"Transformer - TP: {transformer_metrics['tp']}, FP: {transformer_metrics['fp']}, FN: {transformer_metrics['fn']}")
        print(f"Transformer - Precision: {transformer_metrics['precision']:.6f}, Recall: {transformer_metrics['recall']:.6f}, F1: {transformer_metrics['f1']:.6f}")
    except Exception as e:
        print(f"Error loading Transformer results: {e}")
        return
    
    # Load IV3-GRU results
    print("\nLoading IV3-GRU results...")
    try:
        iv3gru_metrics = extract_overall_metrics(IV3GRU_JSON)
        print(f"IV3-GRU - TP: {iv3gru_metrics['tp']}, FP: {iv3gru_metrics['fp']}, FN: {iv3gru_metrics['fn']}")
        print(f"IV3-GRU - Precision: {iv3gru_metrics['precision']:.6f}, Recall: {iv3gru_metrics['recall']:.6f}, F1: {iv3gru_metrics['f1']:.6f}")
    except Exception as e:
        print(f"Error loading IV3-GRU results: {e}")
        return
    
    # Write CSV output
    print("\n" + "=" * 80)
    print("Writing CSV output...")
    print("=" * 80)
    try:
        write_overall_csv(transformer_metrics, iv3gru_metrics, OUTPUT_CSV)
        print(f"\nCSV file written successfully: {OUTPUT_CSV}")
    except Exception as e:
        print(f"\nError writing CSV file: {e}")


if __name__ == "__main__":
    main()
