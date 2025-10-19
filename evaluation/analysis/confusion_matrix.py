"""
Confusion Matrix Analysis Script for Sign Language Recognition Models

This script generates a comprehensive analysis of model predictions, including
a confusion matrix, classification metrics (Precision, Recall, F1-Score),
and identification of the most confused class pairs.

It supports analysis for both isolated sign predictions and continuous (CTC)
sequence predictions.

Usage Examples:
    # Analyze gloss-level predictions for all signers
    python evaluation/analysis/confusion_matrix.py \\
        --input results/isolated/predictions.json \\
        --output-dir results/isolated/analysis \\
        --level gloss

    # Analyze category-level predictions for a specific signer (S2)
    python evaluation/analysis/confusion_matrix.py \\
        --input results/isolated/predictions.json \\
        --output-dir results/isolated/analysis_S2 \\
        --level category \\
        --signer S2

    # Analyze continuous signing results (WER evaluation)
    python evaluation/analysis/confusion_matrix.py \\
        --input results/continuous/detailed_results.csv \\
        --output-dir results/continuous/analysis \\
        --level gloss
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_metrics(cm):
    """Calculates TP, FP, TN, FN, Precision, Recall, and F1-score from a confusion matrix."""
    num_classes = cm.shape[0]
    metrics = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[i] = {
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'F1-Score': f1
        }
    return metrics

def get_most_confused_pairs(cm, labels, top_n=10):
    """Identifies the most confused pairs of classes from a confusion matrix."""
    np.fill_diagonal(cm, 0)
    flat_indices = np.argsort(cm.flatten())[::-1]
    
    confused_pairs = []
    for index in flat_indices[:top_n]:
        true_idx, pred_idx = np.unravel_index(index, cm.shape)
        count = cm[true_idx, pred_idx]
        if count == 0:
            break
        confused_pairs.append({
            'True': labels.get(true_idx, f"ID {true_idx}"),
            'Predicted': labels.get(pred_idx, f"ID {pred_idx}"),
            'Count': count
        })
    return confused_pairs

def plot_confusion_matrix(cm, labels, output_path, normalize=None):
    """Plots and saves the confusion matrix as a heatmap."""
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (by True Label)'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (by Predicted Label)'
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (by All Samples)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    df_cm = pd.DataFrame(cm, index=labels.values(), columns=labels.values())
    
    plt.figure(figsize=(max(10, len(labels) // 4), max(8, len(labels) // 5)))
    sns.heatmap(df_cm, annot=False, cmap='viridis', fmt=fmt)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Confusion matrix heatmap saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Confusion Matrix Analysis Script")
    parser.add_argument('--input', type=str, required=True, help='Path to prediction results JSON or CSV file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save analysis outputs')
    parser.add_argument('--level', choices=['gloss', 'category'], default='gloss', help='Analysis level')
    parser.add_argument('--signer', type=str, default='all', help='Filter by a specific signer ID or "all"')
    parser.add_argument('--normalize', choices=['true', 'pred', 'all'], default=None, help='Normalization method for the heatmap')
    parser.add_argument('--labels-ref', type=str, default='data/labels_reference.csv', help='Path to the labels reference CSV')
    
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    input_path = Path(args.input)
    if input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # For isolated predictions, true/pred are direct columns
        true_col, pred_col = f"{args.level}_true", f"{args.level}_pred"
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
        # For continuous, we need to flatten the sequences
        df = df.dropna(subset=['ground_truth', 'prediction'])
        y_true, y_pred = [], []
        for _, row in df.iterrows():
            gt = eval(row['ground_truth'])
            pred = eval(row['prediction'])
            # Pad the shorter sequence to align for CM
            max_len = max(len(gt), len(pred))
            gt.extend([-1] * (max_len - len(gt)))
            pred.extend([-1] * (max_len - len(pred)))
            y_true.extend(gt)
            y_pred.extend(pred)
        df = pd.DataFrame({'gloss_true': y_true, 'gloss_pred': y_pred})
        true_col, pred_col = "gloss_true", "gloss_pred" # Continuous is always gloss level
    else:
        raise ValueError("Input file must be a .json or .csv file")

    # Filter by signer if specified
    if args.signer != 'all' and 'signer' in df.columns:
        df = df[df['signer'] == args.signer]
        print(f"Filtered results for signer: {args.signer}")

    if df.empty:
        print("No data available after filtering. Exiting.")
        return

    # Load label mappings
    labels_df = pd.read_csv(args.labels_ref)
    if args.level == 'gloss':
        labels_map = dict(zip(labels_df['gloss_id'], labels_df['label']))
    else:
        labels_map = dict(zip(labels_df['cat_id'], labels_df['category']))
        labels_map = {k: v for k, v in sorted(labels_map.items())} # Ensure consistent order

    y_true = df[true_col].astype(int)
    y_pred = df[pred_col].astype(int)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(labels_map.keys()))
    cm_df = pd.DataFrame(cm, index=labels_map.values(), columns=labels_map.values())
    cm_csv_path = output_dir / f"{args.level}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"✅ Confusion matrix data saved to: {cm_csv_path}")

    # Plot and save heatmap
    heatmap_path = output_dir / f"{args.level}_confusion_matrix.png"
    plot_confusion_matrix(cm, labels_map, heatmap_path, args.normalize)

    # Calculate and save metrics
    metrics = calculate_metrics(cm)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df['label'] = metrics_df.index.map(labels_map)
    metrics_df = metrics_df[['label', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1-Score']]
    
    # Generate and save report
    report_path = output_dir / f"{args.level}_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"CLASSIFICATION REPORT (Level: {args.level.capitalize()}, Signer: {args.signer.capitalize()})\n")
        f.write("="*80 + "\n")
        f.write(metrics_df.to_string())
        f.write("\n\n" + "="*80 + "\n")
        
        # Most confused pairs
        confused_pairs = get_most_confused_pairs(cm.copy(), labels_map, top_n=15)
        f.write("\nMOST CONFUSED PAIRS:\n")
        f.write("--------------------\n")
        for pair in confused_pairs:
            f.write(f"True: {pair['True']:<20} | Predicted: {pair['Predicted']:<20} | Count: {pair['Count']}\n")

    print(f"✅ Classification report saved to: {report_path}")

if __name__ == "__main__":
    main()