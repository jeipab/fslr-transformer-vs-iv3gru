"""
Extract TP, FP, FN counts per Ground Truth Gloss from CTC validation results.

Usage:
    python metrics/extract/extract_gloss_tp_fp_fn.py

Output:
    Creates a CSV file with TP, FP, FN counts for each of the 105 glosses
    for both iv3gru and transformer models.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
IV3GRU_JSON = SCRIPT_DIR / "ctc_validation_results_iv3gru.json"
TRANSFORMER_JSON = SCRIPT_DIR / "ctc_validation_results_transformer.json"
LABELS_CSV = Path(__file__).parent.parent.parent / "data" / "labels_reference.csv"
OUTPUT_CSV = SCRIPT_DIR / "gloss_tp_fp_fn_counts.csv"


def load_label_mapping():
    """Load gloss ID to label mapping."""
    df = pd.read_csv(LABELS_CSV)
    return dict(zip(df['gloss_id'], df['label']))


def extract_gloss_counts(json_path):
    """
    Extract TP, FP, FN counts per gloss from validation results.
    
    Args:
        json_path: Path to validation results JSON file
        
    Returns:
        Dictionary mapping gloss_id to {'tp': count, 'fp': count, 'fn': count}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters for all 105 glosses (0-104)
    gloss_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for prediction in data['predictions']:
        gt_sequence = prediction['ground_truth_sequence']
        pred_sequence = prediction['predicted_sequence']
        matched_pairs = prediction.get('matched_pairs', [])
        unmatched_pred = prediction.get('unmatched_predictions', [])
        unmatched_gt = prediction.get('unmatched_ground_truth', [])
        
        # Count TP: For each matched pair, count the ground truth gloss
        for pair in matched_pairs:
            gt_idx = pair['gt_idx']
            if 0 <= gt_idx < len(gt_sequence):
                gloss_id = gt_sequence[gt_idx]
                gloss_counts[gloss_id]['tp'] += 1
        
        # Count FP: For each unmatched prediction, count the predicted gloss
        for pred_idx in unmatched_pred:
            if 0 <= pred_idx < len(pred_sequence):
                gloss_id = pred_sequence[pred_idx]
                gloss_counts[gloss_id]['fp'] += 1
        
        # Count FN: For each unmatched ground truth, count the ground truth gloss
        for gt_idx in unmatched_gt:
            if 0 <= gt_idx < len(gt_sequence):
                gloss_id = gt_sequence[gt_idx]
                gloss_counts[gloss_id]['fn'] += 1
    
    return gloss_counts


def main():
    """Main function to extract and save gloss counts."""
    print("Loading label mapping...")
    label_mapping = load_label_mapping()
    
    print("Extracting counts from iv3gru results...")
    iv3gru_counts = extract_gloss_counts(IV3GRU_JSON)
    
    print("Extracting counts from transformer results...")
    transformer_counts = extract_gloss_counts(TRANSFORMER_JSON)
    
    # Create DataFrame with all 105 glosses
    results = []
    for gloss_id in range(105):
        gloss_label = label_mapping.get(gloss_id, f"Unknown_{gloss_id}")
        results.append({
            'gloss_id': gloss_id,
            'gloss_label': gloss_label,
            'iv3gru_tp': iv3gru_counts[gloss_id]['tp'],
            'iv3gru_fp': iv3gru_counts[gloss_id]['fp'],
            'iv3gru_fn': iv3gru_counts[gloss_id]['fn'],
            'transformer_tp': transformer_counts[gloss_id]['tp'],
            'transformer_fp': transformer_counts[gloss_id]['fp'],
            'transformer_fn': transformer_counts[gloss_id]['fn'],
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('gloss_id')
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Total glosses: {len(df)}")
    print(f"\nSample output:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

