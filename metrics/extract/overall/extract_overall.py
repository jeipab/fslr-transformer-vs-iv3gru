"""
Extract overall precision, recall, and f1-score metrics (regardless of occlusion status)
from CTC validation results JSON files.

Usage:
    python metrics/extract/overall/extract_overall.py

Output:
    - overall_recognition_metrics.csv: CSV file with columns Metric, Transformer, IV3-GRU
      containing overall Precision, Recall, F1-Score, Total TP, Total FP, and Total FN for recognition (gloss-based)
    - overall_classification_metrics.csv: CSV file with columns Metric, Transformer, IV3-GRU
      containing overall Precision, Recall, F1-Score, Total TP, Total FP, and Total FN for classification (category-based)
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
OUTPUT_RECOGNITION_CSV = SCRIPT_DIR / "overall_recognition_metrics.csv"
OUTPUT_CLASSIFICATION_CSV = SCRIPT_DIR / "overall_classification_metrics.csv"


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


def extract_overall_classification_metrics(json_path):
    """
    Extract overall classification precision, recall, and f1-score metrics regardless of occlusion status.
    Aggregates across all categories.
    
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
    
    def _max_iou_with_gt(pred_ts, gt_ts_list):
        """Find the ground truth item with maximum IoU overlap for a prediction timestamp."""
        best_iou = 0.0
        best_idx = -1
        
        if not pred_ts or not gt_ts_list:
            return best_iou, best_idx
        
        pred_start = pred_ts.get('start_ms', 0)
        pred_end = pred_ts.get('end_ms', 0)
        
        for j, gt_ts in enumerate(gt_ts_list):
            gt_start = gt_ts.get('start_ms', 0)
            gt_end = gt_ts.get('end_ms', 0)
            
            # Calculate overlap
            overlap_start = max(pred_start, gt_start)
            overlap_end = min(pred_end, gt_end)
            overlap = max(0.0, overlap_end - overlap_start)
            
            # Calculate union
            union_start = min(pred_start, gt_start)
            union_end = max(pred_end, gt_end)
            union = union_end - union_start
            
            # Calculate IoU
            iou = overlap / union if union > 0 else 0.0
            
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        
        return best_iou, best_idx
    
    for prediction in data['predictions']:
        # Get ground truth and predicted categories
        gt_categories = prediction.get('ground_truth_categories', [])
        pred_categories = prediction.get('predicted_categories', [])
        
        # Get timestamps for IoU calculation
        pred_timestamps = prediction.get('predicted_timestamps', [])
        gt_timestamps = prediction.get('ground_truth_timestamps', [])
        
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
            if pred_idx < len(pred_categories) and gt_idx < len(gt_categories):
                pred_cat = pred_categories[pred_idx]
                gt_cat = gt_categories[gt_idx]
                if pred_cat == gt_cat:
                    total_tp += 1
                else:
                    # FP for predicted category, FN for ground truth category
                    total_fp += 1
                    total_fn += 1
        
        # Process unmatched predictions (FP) - assign based on IoU with GT
        for pred_idx in unmatched_pred:
            if pred_idx < len(pred_categories):
                pred_cat = pred_categories[pred_idx]
                
                # Find the GT item with maximum IoU overlap
                pred_ts = None
                if pred_timestamps:
                    for ts in pred_timestamps:
                        if ts.get('index') == pred_idx:
                            pred_ts = ts
                            break
                
                if pred_ts and gt_timestamps:
                    max_iou, best_gt_idx = _max_iou_with_gt(pred_ts, gt_timestamps)
                    
                    # Only count FP if there's overlap
                    if max_iou > 0:
                        total_fp += 1
                    # If no overlap, don't count it (consistent with predict_ctc.py logic)
                else:
                    # Fallback: if timestamps are not available, don't count (to avoid double counting)
                    pass
        
        # Process unmatched ground truth items (FN)
        for gt_idx, gt_cat in enumerate(gt_categories):
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
    """Main function to extract and write overall metrics for recognition and classification."""
    print("=" * 80)
    print("Overall Metrics Extraction (Precision, Recall, F1-Score)")
    print("=" * 80)
    
    # ===== RECOGNITION METRICS =====
    print("\n" + "=" * 80)
    print("RECOGNITION METRICS (Gloss-based)")
    print("=" * 80)
    
    # Load Transformer recognition results
    print("\nLoading Transformer recognition results...")
    try:
        transformer_recognition = extract_overall_metrics(TRANSFORMER_JSON)
        print(f"Transformer - TP: {transformer_recognition['tp']}, FP: {transformer_recognition['fp']}, FN: {transformer_recognition['fn']}")
        print(f"Transformer - Precision: {transformer_recognition['precision']:.6f}, Recall: {transformer_recognition['recall']:.6f}, F1: {transformer_recognition['f1']:.6f}")
    except Exception as e:
        print(f"Error loading Transformer recognition results: {e}")
        return
    
    # Load IV3-GRU recognition results
    print("\nLoading IV3-GRU recognition results...")
    try:
        iv3gru_recognition = extract_overall_metrics(IV3GRU_JSON)
        print(f"IV3-GRU - TP: {iv3gru_recognition['tp']}, FP: {iv3gru_recognition['fp']}, FN: {iv3gru_recognition['fn']}")
        print(f"IV3-GRU - Precision: {iv3gru_recognition['precision']:.6f}, Recall: {iv3gru_recognition['recall']:.6f}, F1: {iv3gru_recognition['f1']:.6f}")
    except Exception as e:
        print(f"Error loading IV3-GRU recognition results: {e}")
        return
    
    # Write recognition CSV output
    print("\n" + "=" * 80)
    print("Writing recognition CSV output...")
    print("=" * 80)
    try:
        write_overall_csv(transformer_recognition, iv3gru_recognition, OUTPUT_RECOGNITION_CSV)
        print(f"\nRecognition CSV file written successfully: {OUTPUT_RECOGNITION_CSV}")
    except Exception as e:
        print(f"\nError writing recognition CSV file: {e}")
    
    # ===== CLASSIFICATION METRICS =====
    print("\n\n" + "=" * 80)
    print("CLASSIFICATION METRICS (Category-based)")
    print("=" * 80)
    
    # Load Transformer classification results
    print("\nLoading Transformer classification results...")
    try:
        transformer_classification = extract_overall_classification_metrics(TRANSFORMER_JSON)
        print(f"Transformer - TP: {transformer_classification['tp']}, FP: {transformer_classification['fp']}, FN: {transformer_classification['fn']}")
        print(f"Transformer - Precision: {transformer_classification['precision']:.6f}, Recall: {transformer_classification['recall']:.6f}, F1: {transformer_classification['f1']:.6f}")
    except Exception as e:
        print(f"Error loading Transformer classification results: {e}")
        return
    
    # Load IV3-GRU classification results
    print("\nLoading IV3-GRU classification results...")
    try:
        iv3gru_classification = extract_overall_classification_metrics(IV3GRU_JSON)
        print(f"IV3-GRU - TP: {iv3gru_classification['tp']}, FP: {iv3gru_classification['fp']}, FN: {iv3gru_classification['fn']}")
        print(f"IV3-GRU - Precision: {iv3gru_classification['precision']:.6f}, Recall: {iv3gru_classification['recall']:.6f}, F1: {iv3gru_classification['f1']:.6f}")
    except Exception as e:
        print(f"Error loading IV3-GRU classification results: {e}")
        return
    
    # Write classification CSV output
    print("\n" + "=" * 80)
    print("Writing classification CSV output...")
    print("=" * 80)
    try:
        write_overall_csv(transformer_classification, iv3gru_classification, OUTPUT_CLASSIFICATION_CSV)
        print(f"\nClassification CSV file written successfully: {OUTPUT_CLASSIFICATION_CSV}")
    except Exception as e:
        print(f"\nError writing classification CSV file: {e}")


if __name__ == "__main__":
    main()
