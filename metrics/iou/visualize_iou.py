"""
Visualize IoU (Intersection over Union) comparison between Transformer and IV3GRU models.

IoU measures temporal alignment accuracy - how well predicted timestamps overlap with 
ground truth timestamps. It only evaluates WHEN predictions occur, not WHAT is predicted.
Higher IoU means better temporal alignment, but doesn't necessarily mean better overall 
performance (which depends on recognition/classification accuracy).

Usage:
    python metrics/iou/visualize_iou.py

Input:
    metrics/extract/shared_inputs/ctc_validation_results_transformer.json
    metrics/extract/shared_inputs/ctc_validation_results_iv3gru.json

Output:
    Creates one PNG file:
    - iou_comparison.png - Bar chart comparing mean IoU between Transformer and IV3GRU
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
IOU_DIR = SCRIPT_DIR
EXTRACT_DIR = IOU_DIR.parent / "extract"
SHARED_INPUTS_DIR = EXTRACT_DIR / "shared_inputs"
TRANSFORMER_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_transformer.json"
IV3GRU_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_iv3gru.json"
OUTPUT_DIR = IOU_DIR


def extract_iou_values(json_path, model_name):
    """
    Extract IoU values from all matched pairs in validation results.
    
    Args:
        json_path: Path to validation results JSON file
        model_name: Name of the model (for labeling)
        
    Returns:
        Dictionary with 'iou_values', 'mean_iou', 'median_iou', 'std_iou', 'model_name'
    """
    print(f"  Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Processing {len(data['predictions'])} predictions...")
    iou_values = []
    
    for i, prediction in enumerate(data['predictions']):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(data['predictions'])} predictions...")
        
        matched_pairs = prediction.get('matched_pairs', [])
        
        # Extract IoU values from matched pairs
        for pair in matched_pairs:
            iou = pair.get('iou')
            if iou is not None:
                iou_values.append(float(iou))
    
    # Calculate statistics
    mean_iou = np.mean(iou_values) if iou_values else 0.0
    median_iou = np.median(iou_values) if iou_values else 0.0
    std_iou = np.std(iou_values) if iou_values else 0.0
    
    return {
        'iou_values': iou_values,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'std_iou': std_iou,
        'model_name': model_name,
        'count': len(iou_values)
    }


def plot_iou_comparison(transformer_data, iv3gru_data):
    """Plot bar chart comparing mean IoU between Transformer and IV3GRU."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    models = ['Transformer', 'IV3GRU']
    mean_ious = [transformer_data['mean_iou'], iv3gru_data['mean_iou']]
    colors = ['#8b5cf6', '#ec4899']
    
    # Create simple bar chart with only mean IoU
    bars = ax.bar(models, mean_ious, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, mean_ious):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{mean_val:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Mean IoU', fontsize=12)
    ax.set_title('Mean IoU Comparison: Transformer vs IV3GRU\n(Temporal Alignment Accuracy)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(mean_ious) * 1.15)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'iou_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to extract and visualize IoU comparison."""
    try:
        print("Loading transformer validation results...")
        transformer_data = extract_iou_values(TRANSFORMER_JSON, 'transformer')
        
        print("\nLoading iv3gru validation results...")
        iv3gru_data = extract_iou_values(IV3GRU_JSON, 'iv3gru')
        
        print(f"\nTransformer IoU Statistics:")
        print(f"  Mean IoU: {transformer_data['mean_iou']:.4f}")
        print(f"  Median IoU: {transformer_data['median_iou']:.4f}")
        print(f"  Std Dev: {transformer_data['std_iou']:.4f}")
        print(f"  Total matched pairs: {transformer_data['count']}")
        
        print(f"\nIV3GRU IoU Statistics:")
        print(f"  Mean IoU: {iv3gru_data['mean_iou']:.4f}")
        print(f"  Median IoU: {iv3gru_data['median_iou']:.4f}")
        print(f"  Std Dev: {iv3gru_data['std_iou']:.4f}")
        print(f"  Total matched pairs: {iv3gru_data['count']}")
        
        print("\nGenerating visualization...")
        plot_iou_comparison(transformer_data, iv3gru_data)
        
        print("\nVisualization saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

