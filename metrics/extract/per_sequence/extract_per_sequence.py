"""
Extract per-sequence metrics (TP, FP, FN, Precision, Recall, F1-score) 
from CTC validation results JSON files.

Usage:
    python metrics/extract/per_sequence/extract_per_sequence.py

Input Files:
    - metrics/extract/shared_inputs/ctc_validation_results_transformer.json
    - metrics/extract/shared_inputs/ctc_validation_results_iv3gru.json

Output:
    Generates 4 CSV files in metrics/extract/per_sequence:
    - per_sequence_recognition_transformer.csv
    - per_sequence_classification_transformer.csv
    - per_sequence_recognition_iv3gru.csv
    - per_sequence_classification_iv3gru.csv

Columns:
    - File Name
    - Ground Truth Sequence (length of sequence)
    - Predicted Sequence (length of sequence)
    - TP
    - FP
    - FN
    - Precision
    - Recall
    - F1-score
"""

import json
import csv
import os
from pathlib import Path


def load_json_data(json_path):
    """Load JSON file with validation results."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('predictions', [])


def calculate_sequence_metrics(prediction, use_categories=False):
    """
    Calculate TP, FP, FN for a single sequence prediction.
    
    Args:
        prediction: Dictionary containing prediction data
        use_categories: If True, use categories; if False, use gloss sequences
        
    Returns:
        Dictionary with TP, FP, FN, Precision, Recall, F1-score
    """
    # Get sequences based on task type
    if use_categories:
        gt_sequence = prediction.get('ground_truth_categories', [])
        pred_sequence = prediction.get('predicted_categories', [])
    else:
        gt_sequence = prediction.get('ground_truth_sequence', [])
        pred_sequence = prediction.get('predicted_sequence', [])
    
    # Get matched pairs and unmatched items
    matched_pairs = prediction.get('matched_pairs', [])
    unmatched_pred = prediction.get('unmatched_predictions', [])
    unmatched_gt = prediction.get('unmatched_ground_truth', [])
    
    # Initialize counters
    tp = 0
    fp = 0
    fn = 0
    
    # Process matched pairs
    for pair in matched_pairs:
        pred_idx = pair.get('pred_idx')
        gt_idx = pair.get('gt_idx')
        
        if pred_idx is not None and gt_idx is not None:
            if pred_idx < len(pred_sequence) and gt_idx < len(gt_sequence):
                pred_item = pred_sequence[pred_idx]
                gt_item = gt_sequence[gt_idx]
                
                if pred_item == gt_item:
                    tp += 1
                else:
                    # Mismatch: FP for predicted, FN for ground truth
                    fp += 1
                    fn += 1
    
    # Process unmatched predictions (FP)
    fp += len(unmatched_pred)
    
    # Process unmatched ground truth (FN)
    fn += len(unmatched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gt_length': len(gt_sequence),
        'pred_length': len(pred_sequence)
    }


def extract_per_sequence_metrics(json_path, use_categories=False):
    """
    Extract per-sequence metrics from JSON file.
    
    Args:
        json_path: Path to JSON file
        use_categories: If True, extract classification metrics; if False, extract recognition metrics
        
    Returns:
        List of dictionaries, each containing metrics for one sequence
    """
    predictions = load_json_data(json_path)
    results = []
    
    for prediction in predictions:
        file_name = prediction.get('file_name', '')
        if not file_name:
            continue
        
        metrics = calculate_sequence_metrics(prediction, use_categories=use_categories)
        
        results.append({
            'file_name': file_name,
            'ground_truth_sequence': metrics['gt_length'],
            'predicted_sequence': metrics['pred_length'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    return results


def write_csv(results, output_path):
    """Write results to CSV file."""
    headers = [
        'File Name',
        'Ground Truth Sequence',
        'Predicted Sequence',
        'TP',
        'FP',
        'FN',
        'Precision',
        'Recall',
        'F1-score'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for result in results:
            writer.writerow([
                result['file_name'],
                result['ground_truth_sequence'],
                result['predicted_sequence'],
                result['tp'],
                result['fp'],
                result['fn'],
                f"{result['precision']:.6f}",
                f"{result['recall']:.6f}",
                f"{result['f1']:.6f}"
            ])


def main():
    """Main function to extract per-sequence metrics."""
    # Get script directory and paths
    script_dir = Path(__file__).parent
    extract_dir = script_dir.parent
    shared_inputs_dir = extract_dir / 'shared_inputs'
    output_dir = script_dir
    
    # Input JSON files
    transformer_json = shared_inputs_dir / 'ctc_validation_results_transformer.json'
    iv3gru_json = shared_inputs_dir / 'ctc_validation_results_iv3gru.json'
    
    # Output CSV files
    recognition_transformer_csv = output_dir / 'per_sequence_recognition_transformer.csv'
    classification_transformer_csv = output_dir / 'per_sequence_classification_transformer.csv'
    recognition_iv3gru_csv = output_dir / 'per_sequence_recognition_iv3gru.csv'
    classification_iv3gru_csv = output_dir / 'per_sequence_classification_iv3gru.csv'
    
    print("=" * 60)
    print("Extracting Per-Sequence Metrics")
    print("=" * 60)
    
    # Extract recognition metrics
    print("\nExtracting recognition metrics (Transformer)...")
    rec_transformer = extract_per_sequence_metrics(transformer_json, use_categories=False)
    write_csv(rec_transformer, recognition_transformer_csv)
    print(f"✓ Saved {len(rec_transformer)} sequences to {recognition_transformer_csv.name}")
    
    print("\nExtracting recognition metrics (IV3-GRU)...")
    rec_iv3gru = extract_per_sequence_metrics(iv3gru_json, use_categories=False)
    write_csv(rec_iv3gru, recognition_iv3gru_csv)
    print(f"✓ Saved {len(rec_iv3gru)} sequences to {recognition_iv3gru_csv.name}")
    
    # Extract classification metrics
    print("\nExtracting classification metrics (Transformer)...")
    cls_transformer = extract_per_sequence_metrics(transformer_json, use_categories=True)
    write_csv(cls_transformer, classification_transformer_csv)
    print(f"✓ Saved {len(cls_transformer)} sequences to {classification_transformer_csv.name}")
    
    print("\nExtracting classification metrics (IV3-GRU)...")
    cls_iv3gru = extract_per_sequence_metrics(iv3gru_json, use_categories=True)
    write_csv(cls_iv3gru, classification_iv3gru_csv)
    print(f"✓ Saved {len(cls_iv3gru)} sequences to {classification_iv3gru_csv.name}")
    
    print("\n" + "=" * 60)
    print("✓ All per-sequence metrics extracted successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

