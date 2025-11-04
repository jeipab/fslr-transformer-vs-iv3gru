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
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_indices = np.argsort(cm_copy.flatten())[::-1]
    
    confused_pairs = []
    for index in flat_indices[:top_n]:
        true_idx, pred_idx = np.unravel_index(index, cm_copy.shape)
        count = cm_copy[true_idx, pred_idx]
        if count == 0:
            break
        confused_pairs.append({
            'True': labels.get(true_idx, f"ID {true_idx}"),
            'Predicted': labels.get(pred_idx, f"ID {pred_idx}"),
            'Count': count
        })
    return confused_pairs


def analyze_within_category_confusion(cm, labels_df, level='gloss'):
    """Analyze confusion patterns within the same category."""
    if level == 'category':
        return {}  # No within-category analysis for category level
    
    # Group glosses by category
    category_groups = labels_df.groupby('cat_id')['gloss_id'].apply(list).to_dict()
    
    within_category_confusion = {}
    for cat_id, gloss_ids in category_groups.items():
        if len(gloss_ids) < 2:
            continue
            
        # Get confusion matrix subset for this category
        cat_indices = [i for i, gloss_id in enumerate(labels_df['gloss_id']) if gloss_id in gloss_ids]
        cat_cm = cm[np.ix_(cat_indices, cat_indices)]
        
        # Calculate within-category confusion rate
        total_predictions = cat_cm.sum()
        correct_predictions = np.diag(cat_cm).sum()
        within_category_errors = total_predictions - correct_predictions
        
        category_name = labels_df[labels_df['cat_id'] == cat_id]['category'].iloc[0]
        within_category_confusion[category_name] = {
            'total_predictions': int(total_predictions),
            'correct_predictions': int(correct_predictions),
            'within_category_errors': int(within_category_errors),
            'confusion_rate': within_category_errors / total_predictions if total_predictions > 0 else 0,
            'num_glosses': len(gloss_ids)
        }
    
    return within_category_confusion


def analyze_cross_category_confusion(cm, labels_df, level='gloss'):
    """Analyze confusion patterns across different categories."""
    if level == 'category':
        return {}  # No cross-category analysis for category level
    
    # Group glosses by category
    category_groups = labels_df.groupby('cat_id')['gloss_id'].apply(list).to_dict()
    category_names = dict(zip(labels_df['cat_id'], labels_df['category']))
    
    cross_category_confusion = {}
    categories = list(category_groups.keys())
    
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:  # Avoid duplicates and self-comparison
                continue
                
            cat1_indices = [idx for idx, gloss_id in enumerate(labels_df['gloss_id']) if gloss_id in category_groups[cat1]]
            cat2_indices = [idx for idx, gloss_id in enumerate(labels_df['gloss_id']) if gloss_id in category_groups[cat2]]
            
            # Cross-category confusion: cat1 predicted as cat2
            cat1_to_cat2 = cm[np.ix_(cat1_indices, cat2_indices)].sum()
            cat2_to_cat1 = cm[np.ix_(cat2_indices, cat1_indices)].sum()
            
            total_cross_confusion = cat1_to_cat2 + cat2_to_cat1
            total_predictions = cm[np.ix_(cat1_indices + cat2_indices, cat1_indices + cat2_indices)].sum()
            
            pair_name = f"{category_names[cat1]} ↔ {category_names[cat2]}"
            cross_category_confusion[pair_name] = {
                'cat1_to_cat2': int(cat1_to_cat2),
                'cat2_to_cat1': int(cat2_to_cat1),
                'total_cross_confusion': int(total_cross_confusion),
                'total_predictions': int(total_predictions),
                'cross_confusion_rate': total_cross_confusion / total_predictions if total_predictions > 0 else 0
            }
    
    return cross_category_confusion


def plot_per_category_breakdown(metrics_df, labels_df, output_dir, level='gloss'):
    """Generate bar charts showing confusion metrics by category."""
    if level == 'category':
        return  # No category breakdown for category level
    
    # Merge metrics with category information
    merged_df = metrics_df.merge(labels_df[['gloss_id', 'category']], left_index=True, right_on='gloss_id')
    
    # Calculate category-level metrics
    category_metrics = merged_df.groupby('category').agg({
        'Precision': 'mean',
        'Recall': 'mean', 
        'F1-Score': 'mean',
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum'
    }).round(3)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Per-Category Classification Metrics', fontsize=16)
    
    # Precision by category
    category_metrics['Precision'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Precision by Category')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Recall by category
    category_metrics['Recall'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Recall by Category')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # F1-Score by category
    category_metrics['F1-Score'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('F1-Score by Category')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Error distribution by category
    category_metrics['Error_Rate'] = (category_metrics['FP'] + category_metrics['FN']) / (category_metrics['TP'] + category_metrics['FP'] + category_metrics['FN'])
    category_metrics['Error_Rate'].plot(kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Error Rate by Category')
    axes[1,1].set_ylabel('Error Rate')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{level}_per_category_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save category metrics to CSV
    category_metrics.to_csv(output_dir / f"{level}_category_metrics.csv")
    print(f"✅ Per-category breakdown saved to: {output_dir / f'{level}_per_category_breakdown.png'}")


def plot_error_distribution(metrics_df, output_dir, level='gloss'):
    """Generate plots showing error distribution patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Error Distribution Analysis ({level.capitalize()} Level)', fontsize=16)
    
    # F1-Score distribution
    axes[0,0].hist(metrics_df['F1-Score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('F1-Score Distribution')
    axes[0,0].set_xlabel('F1-Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(metrics_df['F1-Score'].mean(), color='red', linestyle='--', label=f'Mean: {metrics_df["F1-Score"].mean():.3f}')
    axes[0,0].legend()
    
    # Precision vs Recall scatter
    axes[0,1].scatter(metrics_df['Recall'], metrics_df['Precision'], alpha=0.6, color='coral')
    axes[0,1].set_title('Precision vs Recall')
    axes[0,1].set_xlabel('Recall')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
    
    # Error rate distribution
    error_rate = (metrics_df['FP'] + metrics_df['FN']) / (metrics_df['TP'] + metrics_df['FP'] + metrics_df['FN'])
    axes[1,0].hist(error_rate, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Error Rate Distribution')
    axes[1,0].set_xlabel('Error Rate')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(error_rate.mean(), color='red', linestyle='--', label=f'Mean: {error_rate.mean():.3f}')
    axes[1,0].legend()
    
    # Support (total samples) vs F1-Score
    support = metrics_df['TP'] + metrics_df['FN']
    axes[1,1].scatter(support, metrics_df['F1-Score'], alpha=0.6, color='green')
    axes[1,1].set_title('F1-Score vs Support (Sample Count)')
    axes[1,1].set_xlabel('Support')
    axes[1,1].set_ylabel('F1-Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{level}_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Error distribution plots saved to: {output_dir / f'{level}_error_distribution.png'}")

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
        with open(input_path, 'r', encoding='utf-8') as f:
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
    elif args.signer != 'all' and 'signer' not in df.columns:
        print(f"Warning: 'signer' column not found in data, ignoring signer filter")

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
    
    # Generate advanced analysis (only for gloss level)
    if args.level == 'gloss':
        print("🔍 Generating advanced confusion analysis...")
        
        # Within-category confusion analysis
        within_category = analyze_within_category_confusion(cm, labels_df, args.level)
        if within_category:
            within_category_df = pd.DataFrame.from_dict(within_category, orient='index')
            within_category_df.to_csv(output_dir / f"{args.level}_within_category_confusion.csv")
            print(f"✅ Within-category confusion analysis saved")
        
        # Cross-category confusion analysis
        cross_category = analyze_cross_category_confusion(cm, labels_df, args.level)
        if cross_category:
            cross_category_df = pd.DataFrame.from_dict(cross_category, orient='index')
            cross_category_df.to_csv(output_dir / f"{args.level}_cross_category_confusion.csv")
            print(f"✅ Cross-category confusion analysis saved")
        
        # Generate advanced visualizations
        plot_per_category_breakdown(metrics_df, labels_df, output_dir, args.level)
        plot_error_distribution(metrics_df, output_dir, args.level)

    # Generate and save comprehensive report
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
        
        # Add advanced analysis to report (gloss level only)
        if args.level == 'gloss' and within_category:
            f.write("\n\n" + "="*80 + "\n")
            f.write("WITHIN-CATEGORY CONFUSION ANALYSIS:\n")
            f.write("="*80 + "\n")
            for category, stats in within_category.items():
                f.write(f"{category}:\n")
                f.write(f"  Total Predictions: {stats['total_predictions']}\n")
                f.write(f"  Correct Predictions: {stats['correct_predictions']}\n")
                f.write(f"  Within-Category Errors: {stats['within_category_errors']}\n")
                f.write(f"  Confusion Rate: {stats['confusion_rate']:.3f}\n")
                f.write(f"  Number of Glosses: {stats['num_glosses']}\n\n")
        
        if args.level == 'gloss' and cross_category:
            f.write("\n" + "="*80 + "\n")
            f.write("CROSS-CATEGORY CONFUSION ANALYSIS:\n")
            f.write("="*80 + "\n")
            # Sort by confusion rate
            sorted_cross = sorted(cross_category.items(), key=lambda x: x[1]['cross_confusion_rate'], reverse=True)
            for pair_name, stats in sorted_cross[:10]:  # Top 10 most confused pairs
                f.write(f"{pair_name}:\n")
                f.write(f"  Cross Confusion Rate: {stats['cross_confusion_rate']:.3f}\n")
                f.write(f"  Total Cross Confusion: {stats['total_cross_confusion']}\n")
                f.write(f"  Total Predictions: {stats['total_predictions']}\n\n")

    print(f"✅ Classification report saved to: {report_path}")

if __name__ == "__main__":
    main()