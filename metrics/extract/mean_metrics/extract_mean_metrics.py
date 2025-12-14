"""
Extract mean precision, recall, and f1-score for recognition and classification
from CTC validation results JSON files.

Usage:
    python metrics/extract/mean_metrics/extract_mean_metrics.py

Output:
    - Writes two CSV files:
      * recognition_mean_metrics.csv - Mean metrics for gloss recognition
      * classification_mean_metrics.csv - Mean metrics for category classification
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

# Import extraction functions from existing scripts
import sys
EXTRACT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXTRACT_DIR / "recognition"))
sys.path.insert(0, str(EXTRACT_DIR / "classification"))

from extract_recognition import extract_gloss_metrics
from extract_classification import extract_category_metrics

# Paths
SCRIPT_DIR = Path(__file__).parent
SHARED_INPUTS_DIR = EXTRACT_DIR / "shared_inputs"
IV3GRU_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_iv3gru.json"
TRANSFORMER_JSON = SHARED_INPUTS_DIR / "ctc_validation_results_transformer.json"


def compute_mean_metrics(metrics_dict, num_classes):
    """
    Compute mean precision, recall, and f1-score from per-class metrics (macro-averaging).
    Only includes classes with actual data (TP + FP + FN > 0).
    
    Args:
        metrics_dict: Dictionary with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn' keys
        num_classes: Number of classes (105 for glosses, 10 for categories)
        
    Returns:
        Tuple of (mean_precision, mean_recall, mean_f1)
    """
    precision_values = []
    recall_values = []
    f1_values = []
    
    for i in range(num_classes):
        tp = metrics_dict.get('tp', {}).get(i, 0)
        fp = metrics_dict.get('fp', {}).get(i, 0)
        fn = metrics_dict.get('fn', {}).get(i, 0)
        
        # Only include classes with actual data
        if tp + fp + fn > 0:
            precision = metrics_dict['precision'].get(i, 0.0)
            recall = metrics_dict['recall'].get(i, 0.0)
            f1 = metrics_dict['f1'].get(i, 0.0)
            
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
    
    mean_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
    mean_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
    mean_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    
    return mean_precision, mean_recall, mean_f1


def compute_total_counts(metrics_dict, num_classes):
    """
    Compute total TP, FP, and FN counts across all classes.
    
    Args:
        metrics_dict: Dictionary with 'tp', 'fp', 'fn' keys mapping class_id to count
        num_classes: Number of classes (105 for glosses, 10 for categories)
        
    Returns:
        Tuple of (total_tp, total_fp, total_fn)
    """
    total_tp = sum(metrics_dict.get('tp', {}).get(i, 0) for i in range(num_classes))
    total_fp = sum(metrics_dict.get('fp', {}).get(i, 0) for i in range(num_classes))
    total_fn = sum(metrics_dict.get('fn', {}).get(i, 0) for i in range(num_classes))
    
    return total_tp, total_fp, total_fn


def write_mean_metrics_csv(recognition_data, classification_data, output_dir):
    """
    Write mean metrics to CSV files.
    
    Args:
        recognition_data: Dictionary with recognition metrics for both models and occlusion types
        classification_data: Dictionary with classification metrics for both models and occlusion types
        output_dir: Directory to write CSV files
    """
    # Write recognition metrics CSV
    recognition_path = output_dir / "recognition_mean_metrics.csv"
    with open(recognition_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Occlusion', 'Transformer', 'IV3-GRU'])
        
        # Precision
        writer.writerow([
            'Precision',
            'nonoccluded',
            f"{recognition_data['transformer_nonocc']['precision']:.6f}",
            f"{recognition_data['iv3gru_nonocc']['precision']:.6f}"
        ])
        writer.writerow([
            'Precision',
            'occluded',
            f"{recognition_data['transformer_occ']['precision']:.6f}",
            f"{recognition_data['iv3gru_occ']['precision']:.6f}"
        ])
        
        # Recall
        writer.writerow([
            'Recall',
            'nonoccluded',
            f"{recognition_data['transformer_nonocc']['recall']:.6f}",
            f"{recognition_data['iv3gru_nonocc']['recall']:.6f}"
        ])
        writer.writerow([
            'Recall',
            'occluded',
            f"{recognition_data['transformer_occ']['recall']:.6f}",
            f"{recognition_data['iv3gru_occ']['recall']:.6f}"
        ])
        
        # F1-score
        writer.writerow([
            'F1-score',
            'nonoccluded',
            f"{recognition_data['transformer_nonocc']['f1']:.6f}",
            f"{recognition_data['iv3gru_nonocc']['f1']:.6f}"
        ])
        writer.writerow([
            'F1-score',
            'occluded',
            f"{recognition_data['transformer_occ']['f1']:.6f}",
            f"{recognition_data['iv3gru_occ']['f1']:.6f}"
        ])
        
        # Total TP
        writer.writerow([
            'Total TP',
            'nonoccluded',
            f"{recognition_data['transformer_nonocc']['total_tp']}",
            f"{recognition_data['iv3gru_nonocc']['total_tp']}"
        ])
        writer.writerow([
            'Total TP',
            'occluded',
            f"{recognition_data['transformer_occ']['total_tp']}",
            f"{recognition_data['iv3gru_occ']['total_tp']}"
        ])
        
        # Total FP
        writer.writerow([
            'Total FP',
            'nonoccluded',
            f"{recognition_data['transformer_nonocc']['total_fp']}",
            f"{recognition_data['iv3gru_nonocc']['total_fp']}"
        ])
        writer.writerow([
            'Total FP',
            'occluded',
            f"{recognition_data['transformer_occ']['total_fp']}",
            f"{recognition_data['iv3gru_occ']['total_fp']}"
        ])
        
        # Total FN
        writer.writerow([
            'Total FN',
            'nonoccluded',
            f"{recognition_data['transformer_nonocc']['total_fn']}",
            f"{recognition_data['iv3gru_nonocc']['total_fn']}"
        ])
        writer.writerow([
            'Total FN',
            'occluded',
            f"{recognition_data['transformer_occ']['total_fn']}",
            f"{recognition_data['iv3gru_occ']['total_fn']}"
        ])
    
    # Write classification metrics CSV
    classification_path = output_dir / "classification_mean_metrics.csv"
    with open(classification_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Occlusion', 'Transformer', 'IV3-GRU'])
        
        # Precision
        writer.writerow([
            'Precision',
            'nonoccluded',
            f"{classification_data['transformer_nonocc']['precision']:.6f}",
            f"{classification_data['iv3gru_nonocc']['precision']:.6f}"
        ])
        writer.writerow([
            'Precision',
            'occluded',
            f"{classification_data['transformer_occ']['precision']:.6f}",
            f"{classification_data['iv3gru_occ']['precision']:.6f}"
        ])
        
        # Recall
        writer.writerow([
            'Recall',
            'nonoccluded',
            f"{classification_data['transformer_nonocc']['recall']:.6f}",
            f"{classification_data['iv3gru_nonocc']['recall']:.6f}"
        ])
        writer.writerow([
            'Recall',
            'occluded',
            f"{classification_data['transformer_occ']['recall']:.6f}",
            f"{classification_data['iv3gru_occ']['recall']:.6f}"
        ])
        
        # F1-score
        writer.writerow([
            'F1-score',
            'nonoccluded',
            f"{classification_data['transformer_nonocc']['f1']:.6f}",
            f"{classification_data['iv3gru_nonocc']['f1']:.6f}"
        ])
        writer.writerow([
            'F1-score',
            'occluded',
            f"{classification_data['transformer_occ']['f1']:.6f}",
            f"{classification_data['iv3gru_occ']['f1']:.6f}"
        ])
        
        # Total TP
        writer.writerow([
            'Total TP',
            'nonoccluded',
            f"{classification_data['transformer_nonocc']['total_tp']}",
            f"{classification_data['iv3gru_nonocc']['total_tp']}"
        ])
        writer.writerow([
            'Total TP',
            'occluded',
            f"{classification_data['transformer_occ']['total_tp']}",
            f"{classification_data['iv3gru_occ']['total_tp']}"
        ])
        
        # Total FP
        writer.writerow([
            'Total FP',
            'nonoccluded',
            f"{classification_data['transformer_nonocc']['total_fp']}",
            f"{classification_data['iv3gru_nonocc']['total_fp']}"
        ])
        writer.writerow([
            'Total FP',
            'occluded',
            f"{classification_data['transformer_occ']['total_fp']}",
            f"{classification_data['iv3gru_occ']['total_fp']}"
        ])
        
        # Total FN
        writer.writerow([
            'Total FN',
            'nonoccluded',
            f"{classification_data['transformer_nonocc']['total_fn']}",
            f"{classification_data['iv3gru_nonocc']['total_fn']}"
        ])
        writer.writerow([
            'Total FN',
            'occluded',
            f"{classification_data['transformer_occ']['total_fn']}",
            f"{classification_data['iv3gru_occ']['total_fn']}"
        ])
    
    return recognition_path, classification_path


def main():
    """Main function to extract and compute mean metrics."""
    print("=" * 100)
    print("Mean Metrics Extraction (Recognition & Classification)")
    print("=" * 100)
    
    # Extract recognition metrics
    print("\nExtracting recognition (gloss) metrics...")
    print("Loading Transformer results (non-occluded)...")
    try:
        transformer_recognition_nonocc = extract_gloss_metrics(TRANSFORMER_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading Transformer recognition results: {e}")
        transformer_recognition_nonocc = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    print("Loading IV3-GRU results (non-occluded)...")
    try:
        iv3gru_recognition_nonocc = extract_gloss_metrics(IV3GRU_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading IV3-GRU recognition results: {e}")
        iv3gru_recognition_nonocc = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    print("Loading Transformer results (occluded)...")
    try:
        transformer_recognition_occ = extract_gloss_metrics(TRANSFORMER_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading Transformer recognition results: {e}")
        transformer_recognition_occ = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    print("Loading IV3-GRU results (occluded)...")
    try:
        iv3gru_recognition_occ = extract_gloss_metrics(IV3GRU_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading IV3-GRU recognition results: {e}")
        iv3gru_recognition_occ = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    # Compute mean recognition metrics
    print("\nComputing mean recognition metrics...")
    recognition_data = {
        'transformer_nonocc': {
            'precision': compute_mean_metrics(transformer_recognition_nonocc, 105)[0],
            'recall': compute_mean_metrics(transformer_recognition_nonocc, 105)[1],
            'f1': compute_mean_metrics(transformer_recognition_nonocc, 105)[2],
            'total_tp': compute_total_counts(transformer_recognition_nonocc, 105)[0],
            'total_fp': compute_total_counts(transformer_recognition_nonocc, 105)[1],
            'total_fn': compute_total_counts(transformer_recognition_nonocc, 105)[2]
        },
        'iv3gru_nonocc': {
            'precision': compute_mean_metrics(iv3gru_recognition_nonocc, 105)[0],
            'recall': compute_mean_metrics(iv3gru_recognition_nonocc, 105)[1],
            'f1': compute_mean_metrics(iv3gru_recognition_nonocc, 105)[2],
            'total_tp': compute_total_counts(iv3gru_recognition_nonocc, 105)[0],
            'total_fp': compute_total_counts(iv3gru_recognition_nonocc, 105)[1],
            'total_fn': compute_total_counts(iv3gru_recognition_nonocc, 105)[2]
        },
        'transformer_occ': {
            'precision': compute_mean_metrics(transformer_recognition_occ, 105)[0],
            'recall': compute_mean_metrics(transformer_recognition_occ, 105)[1],
            'f1': compute_mean_metrics(transformer_recognition_occ, 105)[2],
            'total_tp': compute_total_counts(transformer_recognition_occ, 105)[0],
            'total_fp': compute_total_counts(transformer_recognition_occ, 105)[1],
            'total_fn': compute_total_counts(transformer_recognition_occ, 105)[2]
        },
        'iv3gru_occ': {
            'precision': compute_mean_metrics(iv3gru_recognition_occ, 105)[0],
            'recall': compute_mean_metrics(iv3gru_recognition_occ, 105)[1],
            'f1': compute_mean_metrics(iv3gru_recognition_occ, 105)[2],
            'total_tp': compute_total_counts(iv3gru_recognition_occ, 105)[0],
            'total_fp': compute_total_counts(iv3gru_recognition_occ, 105)[1],
            'total_fn': compute_total_counts(iv3gru_recognition_occ, 105)[2]
        }
    }
    
    # Extract classification metrics
    print("\nExtracting classification (category) metrics...")
    print("Loading Transformer results (non-occluded)...")
    try:
        transformer_classification_nonocc = extract_category_metrics(TRANSFORMER_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading Transformer classification results: {e}")
        transformer_classification_nonocc = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    print("Loading IV3-GRU results (non-occluded)...")
    try:
        iv3gru_classification_nonocc = extract_category_metrics(IV3GRU_JSON, occlusion_filter=0)
    except Exception as e:
        print(f"Error loading IV3-GRU classification results: {e}")
        iv3gru_classification_nonocc = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    print("Loading Transformer results (occluded)...")
    try:
        transformer_classification_occ = extract_category_metrics(TRANSFORMER_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading Transformer classification results: {e}")
        transformer_classification_occ = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    print("Loading IV3-GRU results (occluded)...")
    try:
        iv3gru_classification_occ = extract_category_metrics(IV3GRU_JSON, occlusion_filter=1)
    except Exception as e:
        print(f"Error loading IV3-GRU classification results: {e}")
        iv3gru_classification_occ = {'precision': {}, 'recall': {}, 'f1': {}, 'tp': {}, 'fp': {}, 'fn': {}}
    
    # Compute mean classification metrics
    print("\nComputing mean classification metrics...")
    classification_data = {
        'transformer_nonocc': {
            'precision': compute_mean_metrics(transformer_classification_nonocc, 10)[0],
            'recall': compute_mean_metrics(transformer_classification_nonocc, 10)[1],
            'f1': compute_mean_metrics(transformer_classification_nonocc, 10)[2],
            'total_tp': compute_total_counts(transformer_classification_nonocc, 10)[0],
            'total_fp': compute_total_counts(transformer_classification_nonocc, 10)[1],
            'total_fn': compute_total_counts(transformer_classification_nonocc, 10)[2]
        },
        'iv3gru_nonocc': {
            'precision': compute_mean_metrics(iv3gru_classification_nonocc, 10)[0],
            'recall': compute_mean_metrics(iv3gru_classification_nonocc, 10)[1],
            'f1': compute_mean_metrics(iv3gru_classification_nonocc, 10)[2],
            'total_tp': compute_total_counts(iv3gru_classification_nonocc, 10)[0],
            'total_fp': compute_total_counts(iv3gru_classification_nonocc, 10)[1],
            'total_fn': compute_total_counts(iv3gru_classification_nonocc, 10)[2]
        },
        'transformer_occ': {
            'precision': compute_mean_metrics(transformer_classification_occ, 10)[0],
            'recall': compute_mean_metrics(transformer_classification_occ, 10)[1],
            'f1': compute_mean_metrics(transformer_classification_occ, 10)[2],
            'total_tp': compute_total_counts(transformer_classification_occ, 10)[0],
            'total_fp': compute_total_counts(transformer_classification_occ, 10)[1],
            'total_fn': compute_total_counts(transformer_classification_occ, 10)[2]
        },
        'iv3gru_occ': {
            'precision': compute_mean_metrics(iv3gru_classification_occ, 10)[0],
            'recall': compute_mean_metrics(iv3gru_classification_occ, 10)[1],
            'f1': compute_mean_metrics(iv3gru_classification_occ, 10)[2],
            'total_tp': compute_total_counts(iv3gru_classification_occ, 10)[0],
            'total_fp': compute_total_counts(iv3gru_classification_occ, 10)[1],
            'total_fn': compute_total_counts(iv3gru_classification_occ, 10)[2]
        }
    }
    
    # Write CSV files
    print("\n" + "=" * 100)
    print("Writing CSV files...")
    print("=" * 100)
    try:
        rec_path, class_path = write_mean_metrics_csv(recognition_data, classification_data, SCRIPT_DIR)
        print(f"\nRecognition mean metrics CSV written: {rec_path}")
        print(f"Classification mean metrics CSV written: {class_path}")
        
        # Print summary
        print("\n" + "=" * 100)
        print("Summary")
        print("=" * 100)
        print("\nRecognition (Gloss) Mean Metrics:")
        print(f"  Non-occluded - Transformer: P={recognition_data['transformer_nonocc']['precision']:.6f}, "
              f"R={recognition_data['transformer_nonocc']['recall']:.6f}, "
              f"F1={recognition_data['transformer_nonocc']['f1']:.6f}")
        print(f"  Non-occluded - IV3-GRU: P={recognition_data['iv3gru_nonocc']['precision']:.6f}, "
              f"R={recognition_data['iv3gru_nonocc']['recall']:.6f}, "
              f"F1={recognition_data['iv3gru_nonocc']['f1']:.6f}")
        print(f"  Occluded - Transformer: P={recognition_data['transformer_occ']['precision']:.6f}, "
              f"R={recognition_data['transformer_occ']['recall']:.6f}, "
              f"F1={recognition_data['transformer_occ']['f1']:.6f}")
        print(f"  Occluded - IV3-GRU: P={recognition_data['iv3gru_occ']['precision']:.6f}, "
              f"R={recognition_data['iv3gru_occ']['recall']:.6f}, "
              f"F1={recognition_data['iv3gru_occ']['f1']:.6f}")
        
        print("\nClassification (Category) Mean Metrics:")
        print(f"  Non-occluded - Transformer: P={classification_data['transformer_nonocc']['precision']:.6f}, "
              f"R={classification_data['transformer_nonocc']['recall']:.6f}, "
              f"F1={classification_data['transformer_nonocc']['f1']:.6f}")
        print(f"  Non-occluded - IV3-GRU: P={classification_data['iv3gru_nonocc']['precision']:.6f}, "
              f"R={classification_data['iv3gru_nonocc']['recall']:.6f}, "
              f"F1={classification_data['iv3gru_nonocc']['f1']:.6f}")
        print(f"  Occluded - Transformer: P={classification_data['transformer_occ']['precision']:.6f}, "
              f"R={classification_data['transformer_occ']['recall']:.6f}, "
              f"F1={classification_data['transformer_occ']['f1']:.6f}")
        print(f"  Occluded - IV3-GRU: P={classification_data['iv3gru_occ']['precision']:.6f}, "
              f"R={classification_data['iv3gru_occ']['recall']:.6f}, "
              f"F1={classification_data['iv3gru_occ']['f1']:.6f}")
        
    except Exception as e:
        print(f"\nError writing CSV files: {e}")


if __name__ == "__main__":
    main()

