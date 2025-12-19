"""
Extract per-frame recognition and classification results from CTC validation results.
Generates CSV files with frame ranges, ground truth, predictions, and match status.

Usage:
    python metrics/extract/per_frame/extract_per_frame.py

Input Files:
    - metrics/extract/shared_inputs/ctc_validation_results_transformer.json
    - metrics/extract/shared_inputs/ctc_validation_results_iv3gru.json

Output:
    Generates 4 CSV files in metrics/extract/per_frame:
    - per_frame_recognition_transformer.csv
    - per_frame_recognition_iv3gru.csv
    - per_frame_classification_transformer.csv
    - per_frame_classification_iv3gru.csv

Columns:
    - File Name
    - Frame Range
    - Ground Truth Gloss/Category
    - Predicted Gloss/Category
    - Match Status (TP/FP/FN)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent
EXTRACT_DIR = SCRIPT_DIR.parent
SHARED_INPUTS_DIR = EXTRACT_DIR / "shared_inputs"
IV3GRU_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_iv3gru.json"
TRANSFORMER_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_transformer.json"

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

# Frame rate (fps)
FPS = 30


def ms_to_frame(ms: int) -> int:
    """Convert milliseconds to frame number."""
    return int((ms / 1000.0) * FPS)


def format_frame_range(start_ms: int, end_ms: int) -> str:
    """Format frame range as 'start-end'."""
    start_frame = ms_to_frame(start_ms)
    end_frame = ms_to_frame(end_ms)
    return f"{start_frame}-{end_frame}"


def extract_per_frame_results(
    json_path: Path,
    use_categories: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract per-frame recognition or classification results.
    
    Args:
        json_path: Path to validation results JSON file
        use_categories: If True, extract classification; if False, extract recognition
        
    Returns:
        List of dictionaries with per-frame results
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for prediction in data.get('predictions', []):
        file_name = prediction.get('file_name', '')
        if not file_name:
            continue
        
        # Get sequences based on task type
        if use_categories:
            pred_labels = prediction.get('predicted_categories', [])
            gt_labels = prediction.get('ground_truth_categories', [])
            pred_label_names = [CATEGORY_NAMES.get(cat_id, str(cat_id)) for cat_id in pred_labels]
            gt_label_names = [CATEGORY_NAMES.get(cat_id, str(cat_id)) for cat_id in gt_labels]
        else:
            pred_label_names = prediction.get('predicted_labels', [])
            gt_label_names = prediction.get('ground_truth_labels', [])
        
        # Get timestamps
        pred_timestamps = prediction.get('predicted_timestamps', [])
        gt_timestamps = prediction.get('ground_truth_timestamps', [])
        
        # Get matching information
        matched_pairs = prediction.get('matched_pairs', [])
        unmatched_pred_indices = prediction.get('unmatched_predictions', [])
        unmatched_gt_indices = prediction.get('unmatched_ground_truth', [])
        
        # Build mapping from pred_idx to gt_idx for matched pairs
        pred_to_gt_map = {}
        gt_to_pred_map = {}  # Reverse mapping
        pred_to_match_status = {}  # Track if the match is TP (same label) or FP (different label)
        matched_gt_with_mismatch = set()  # GTs matched but labels differ (will be FN)
        
        for pair in matched_pairs:
            pred_idx = pair.get('pred_idx')
            gt_idx = pair.get('gt_idx')
            if pred_idx is not None and gt_idx is not None:
                pred_idx = int(pred_idx)
                gt_idx = int(gt_idx)
                pred_to_gt_map[pred_idx] = gt_idx
                gt_to_pred_map[gt_idx] = pred_idx
                
                # Check if labels match (TP) or differ (FP for pred, FN for GT)
                if (pred_idx < len(pred_label_names) and gt_idx < len(gt_label_names)):
                    if pred_label_names[pred_idx] == gt_label_names[gt_idx]:
                        pred_to_match_status[pred_idx] = 'TP'
                    else:
                        pred_to_match_status[pred_idx] = 'FP'
                        matched_gt_with_mismatch.add(gt_idx)
        
        # Process all predictions
        for pred_idx in range(len(pred_label_names)):
            if pred_idx < len(pred_timestamps):
                pred_ts = pred_timestamps[pred_idx]
                start_ms = pred_ts.get('start_ms', 0)
                end_ms = pred_ts.get('end_ms', 0)
                frame_range = format_frame_range(start_ms, end_ms)
                pred_label = pred_label_names[pred_idx] if pred_idx < len(pred_label_names) else '-'
                
                # Determine match status
                if pred_idx in unmatched_pred_indices:
                    # Unmatched prediction = FP (no GT)
                    match_status = 'FP'
                    gt_label = '-'
                elif pred_idx in pred_to_gt_map:
                    # Matched prediction
                    gt_idx = pred_to_gt_map[pred_idx]
                    match_status = pred_to_match_status.get(pred_idx, 'TP')
                    # For matched pairs, show the GT label (even if mismatch)
                    gt_label = gt_label_names[gt_idx] if gt_idx < len(gt_label_names) else '-'
                else:
                    # Should not happen, but handle gracefully
                    match_status = 'FP'
                    gt_label = '-'
                
                results.append({
                    'File Name': file_name,
                    'Frame Range': frame_range,
                    'Ground Truth Gloss/Category': gt_label,
                    'Predicted Gloss/Category': pred_label,
                    'Match Status': match_status
                })
        
        # Process unmatched ground truth (FN) and matched but mismatched GT (FN)
        all_matched_gt_indices = set(pred_to_gt_map.values())
        
        # Process unmatched_ground_truth (truly unmatched GTs)
        for gt_idx in unmatched_gt_indices:
            if gt_idx < len(gt_label_names) and gt_idx < len(gt_timestamps):
                gt_label = gt_label_names[gt_idx]
                gt_ts = gt_timestamps[gt_idx]
                start_ms = gt_ts.get('start_ms', 0)
                end_ms = gt_ts.get('end_ms', 0)
                frame_range = format_frame_range(start_ms, end_ms)
                
                results.append({
                    'File Name': file_name,
                    'Frame Range': frame_range,
                    'Ground Truth Gloss/Category': gt_label,
                    'Predicted Gloss/Category': '-',
                    'Match Status': 'FN'
                })
        
        # Process matched but mismatched GTs (they need their own row as FN)
        for gt_idx in matched_gt_with_mismatch:
            if gt_idx < len(gt_label_names) and gt_idx < len(gt_timestamps):
                gt_label = gt_label_names[gt_idx]
                pred_idx = gt_to_pred_map.get(gt_idx)
                pred_label = pred_label_names[pred_idx] if (pred_idx is not None and pred_idx < len(pred_label_names)) else '-'
                gt_ts = gt_timestamps[gt_idx]
                start_ms = gt_ts.get('start_ms', 0)
                end_ms = gt_ts.get('end_ms', 0)
                frame_range = format_frame_range(start_ms, end_ms)
                
                results.append({
                    'File Name': file_name,
                    'Frame Range': frame_range,
                    'Ground Truth Gloss/Category': gt_label,
                    'Predicted Gloss/Category': pred_label,
                    'Match Status': 'FN'
                })
    
    return results


def main():
    """Main function to extract per-frame metrics."""
    print("=" * 60)
    print("Extracting Per-Frame Recognition and Classification Results")
    print("=" * 60)
    
    # Extract recognition results
    print("\nExtracting recognition results (Transformer)...")
    rec_transformer = extract_per_frame_results(TRANSFORMER_JSON, use_categories=False)
    rec_transformer_df = pd.DataFrame(rec_transformer)
    rec_transformer_csv = SCRIPT_DIR / "per_frame_recognition_transformer.csv"
    rec_transformer_df.to_csv(rec_transformer_csv, index=False)
    print(f"Saved {len(rec_transformer)} entries to {rec_transformer_csv.name}")
    
    print("\nExtracting recognition results (IV3-GRU)...")
    rec_iv3gru = extract_per_frame_results(IV3GRU_JSON, use_categories=False)
    rec_iv3gru_df = pd.DataFrame(rec_iv3gru)
    rec_iv3gru_csv = SCRIPT_DIR / "per_frame_recognition_iv3gru.csv"
    rec_iv3gru_df.to_csv(rec_iv3gru_csv, index=False)
    print(f"Saved {len(rec_iv3gru)} entries to {rec_iv3gru_csv.name}")
    
    # Extract classification results
    print("\nExtracting classification results (Transformer)...")
    cls_transformer = extract_per_frame_results(TRANSFORMER_JSON, use_categories=True)
    cls_transformer_df = pd.DataFrame(cls_transformer)
    cls_transformer_csv = SCRIPT_DIR / "per_frame_classification_transformer.csv"
    cls_transformer_df.to_csv(cls_transformer_csv, index=False)
    print(f"Saved {len(cls_transformer)} entries to {cls_transformer_csv.name}")
    
    print("\nExtracting classification results (IV3-GRU)...")
    cls_iv3gru = extract_per_frame_results(IV3GRU_JSON, use_categories=True)
    cls_iv3gru_df = pd.DataFrame(cls_iv3gru)
    cls_iv3gru_csv = SCRIPT_DIR / "per_frame_classification_iv3gru.csv"
    cls_iv3gru_df.to_csv(cls_iv3gru_csv, index=False)
    print(f"Saved {len(cls_iv3gru)} entries to {cls_iv3gru_csv.name}")
    
    print("\n" + "=" * 60)
    print("All per-frame results extracted successfully!")
    print("=" * 60)
    
    # Print sample output
    print("\nSample Recognition Transformer output:")
    print(rec_transformer_df.head(10).to_string(index=False))
    print("\nSample Classification Transformer output:")
    print(cls_transformer_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()

