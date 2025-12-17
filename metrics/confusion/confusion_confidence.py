"""
Visualize the distribution of TP, FP, FN in terms of their confidence scores.

Usage:
    python metrics/confusion/confusion_confidence.py

Output:
    Creates three PNG files:
    - tp_confidence_distribution.png - Distribution of confidence scores for True Positives
    - fp_confidence_distribution.png - Distribution of confidence scores for False Positives
    - fn_statistics.png - Statistics about False Negatives (no confidence scores available)
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
CONFUSION_DIR = SCRIPT_DIR
EXTRACT_DIR = CONFUSION_DIR.parent / "extract"
SHARED_INPUTS_DIR = EXTRACT_DIR / "shared_inputs"
TRANSFORMER_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_transformer.json"
IV3GRU_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_iv3gru.json"
OUTPUT_DIR = CONFUSION_DIR


def extract_confidence_scores(json_path, model_name):
    """
    Extract confidence scores for TP, FP, and count FN from validation results.
    
    Args:
        json_path: Path to validation results JSON file
        model_name: Name of the model (for labeling)
        
    Returns:
        Dictionary with 'tp_confidences', 'fp_confidences', 'fn_count'
    """
    print(f"  Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Processing {len(data['predictions'])} predictions...")
    tp_confidences = []
    fp_confidences = []
    fn_count = 0
    
    for i, prediction in enumerate(data['predictions']):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(data['predictions'])} predictions...")
        pred_sequence = prediction.get('predicted_sequence', [])
        gt_sequence = prediction.get('ground_truth_sequence', [])
        confidence_scores = prediction.get('confidence_scores', [])
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
        
        # Track which ground truth items have been matched
        matched_gt_indices = set()
        
        # Process matched pairs
        for pred_idx, gt_idx in pred_to_gt.items():
            matched_gt_indices.add(gt_idx)
            if pred_idx < len(pred_sequence) and pred_idx < len(confidence_scores):
                pred_gloss = pred_sequence[pred_idx]
                confidence = confidence_scores[pred_idx]
                
                if gt_idx < len(gt_sequence):
                    gt_gloss = gt_sequence[gt_idx]
                    if pred_gloss == gt_gloss:
                        # True Positive
                        tp_confidences.append(confidence)
                    else:
                        # False Positive (wrong prediction)
                        fp_confidences.append(confidence)
        
        # Process unmatched predictions (False Positives)
        for pred_idx in unmatched_pred:
            if pred_idx < len(pred_sequence) and pred_idx < len(confidence_scores):
                confidence = confidence_scores[pred_idx]
                fp_confidences.append(confidence)
        
        # Count unmatched ground truth items (False Negatives)
        # Note: FN don't have confidence scores since they weren't predicted
        fn_count += len(unmatched_gt)
    
    return {
        'tp_confidences': tp_confidences,
        'fp_confidences': fp_confidences,
        'fn_count': fn_count,
        'model_name': model_name
    }


def plot_tp_confidence_distribution(transformer_data, iv3gru_data):
    """Plot distribution of confidence scores for True Positives."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Transformer TP distribution
    ax1 = axes[0]
    tp_conf_transformer = transformer_data['tp_confidences']
    if tp_conf_transformer:
        ax1.hist(tp_conf_transformer, bins=50, alpha=0.7, color='#10b981', edgecolor='black')
        ax1.axvline(np.mean(tp_conf_transformer), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(tp_conf_transformer):.3f}')
        ax1.axvline(np.median(tp_conf_transformer), color='blue', linestyle='--', 
                   label=f'Median: {np.median(tp_conf_transformer):.3f}')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Transformer - TP Confidence Distribution\n(n={len(tp_conf_transformer)})', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # IV3GRU TP distribution
    ax2 = axes[1]
    tp_conf_iv3gru = iv3gru_data['tp_confidences']
    if tp_conf_iv3gru:
        ax2.hist(tp_conf_iv3gru, bins=50, alpha=0.7, color='#3b82f6', edgecolor='black')
        ax2.axvline(np.mean(tp_conf_iv3gru), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(tp_conf_iv3gru):.3f}')
        ax2.axvline(np.median(tp_conf_iv3gru), color='blue', linestyle='--', 
                   label=f'Median: {np.median(tp_conf_iv3gru):.3f}')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'IV3GRU - TP Confidence Distribution\n(n={len(tp_conf_iv3gru)})', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'tp_confidence_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_fp_confidence_distribution(transformer_data, iv3gru_data):
    """Plot distribution of confidence scores for False Positives."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Transformer FP distribution
    ax1 = axes[0]
    fp_conf_transformer = transformer_data['fp_confidences']
    if fp_conf_transformer:
        ax1.hist(fp_conf_transformer, bins=50, alpha=0.7, color='#ef4444', edgecolor='black')
        ax1.axvline(np.mean(fp_conf_transformer), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fp_conf_transformer):.3f}')
        ax1.axvline(np.median(fp_conf_transformer), color='blue', linestyle='--', 
                   label=f'Median: {np.median(fp_conf_transformer):.3f}')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Transformer - FP Confidence Distribution\n(n={len(fp_conf_transformer)})', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # IV3GRU FP distribution
    ax2 = axes[1]
    fp_conf_iv3gru = iv3gru_data['fp_confidences']
    if fp_conf_iv3gru:
        ax2.hist(fp_conf_iv3gru, bins=50, alpha=0.7, color='#f59e0b', edgecolor='black')
        ax2.axvline(np.mean(fp_conf_iv3gru), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fp_conf_iv3gru):.3f}')
        ax2.axvline(np.median(fp_conf_iv3gru), color='blue', linestyle='--', 
                   label=f'Median: {np.median(fp_conf_iv3gru):.3f}')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'IV3GRU - FP Confidence Distribution\n(n={len(fp_conf_iv3gru)})', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fp_confidence_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_fn_statistics(transformer_data, iv3gru_data):
    """Plot statistics about False Negatives (no confidence scores available)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = ['Transformer', 'IV3GRU']
    fn_counts = [transformer_data['fn_count'], iv3gru_data['fn_count']]
    colors = ['#8b5cf6', '#ec4899']
    
    bars = ax.bar(models, fn_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, fn_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('False Negative Count', fontsize=12)
    ax.set_title('False Negatives Count by Model\n(Note: FN items have no confidence scores as they were not predicted)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with explanation
    textstr = f'Transformer FN: {fn_counts[0]}\nIV3GRU FN: {fn_counts[1]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fn_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to extract and visualize confidence distributions."""
    try:
        print("Loading transformer validation results...")
        transformer_data = extract_confidence_scores(TRANSFORMER_JSON, 'transformer')
        
        print("\nLoading iv3gru validation results...")
        iv3gru_data = extract_confidence_scores(IV3GRU_JSON, 'iv3gru')
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nTransformer Statistics:")
    print(f"  TP count: {len(transformer_data['tp_confidences'])}")
    print(f"  FP count: {len(transformer_data['fp_confidences'])}")
    print(f"  FN count: {transformer_data['fn_count']}")
    if transformer_data['tp_confidences']:
        print(f"  TP mean confidence: {np.mean(transformer_data['tp_confidences']):.4f}")
    if transformer_data['fp_confidences']:
        print(f"  FP mean confidence: {np.mean(transformer_data['fp_confidences']):.4f}")
    
    print(f"\nIV3GRU Statistics:")
    print(f"  TP count: {len(iv3gru_data['tp_confidences'])}")
    print(f"  FP count: {len(iv3gru_data['fp_confidences'])}")
    print(f"  FN count: {iv3gru_data['fn_count']}")
    if iv3gru_data['tp_confidences']:
        print(f"  TP mean confidence: {np.mean(iv3gru_data['tp_confidences']):.4f}")
    if iv3gru_data['fp_confidences']:
        print(f"  FP mean confidence: {np.mean(iv3gru_data['fp_confidences']):.4f}")
    
    print("\nGenerating visualizations...")
    plot_tp_confidence_distribution(transformer_data, iv3gru_data)
    plot_fp_confidence_distribution(transformer_data, iv3gru_data)
    plot_fn_statistics(transformer_data, iv3gru_data)
    
    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    main()

