"""
Compare metrics between Transformer and IV3-GRU models.

This script generates 2 CSV files comparing metrics:
- gloss_comparison.csv: Comparison of per-gloss metrics
- category_comparison.csv: Comparison of per-category metrics

Usage:
    python metrics/comparison/metric_comparison.py

Input files (must exist):
    metrics/extract/overall/per_gloss_metrics.csv
    metrics/extract/overall/per_category_metrics.csv

Output files (saved to metrics/comparison):
    gloss_comparison.csv
    category_comparison.csv

Columns:
    - ID: Gloss ID or Category ID
    - Label: Gloss label or Category name
    - Transformer: Precision, Recall, F1-Score
    - IV3-GRU: Precision, Recall, F1-Score
    - Difference: Precision, Recall, F1-Score (Transformer - IV3-GRU)
    - Interpretation: Overall interpretation of the differences
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def load_label_mappings():
    """Load gloss and category mappings from labels reference CSV."""
    labels_ref_path = Path(__file__).parent.parent.parent / "data" / "labels_reference.csv"
    df = pd.read_csv(labels_ref_path)
    
    # Create gloss mapping: {gloss_id: gloss_label}
    gloss_mapping = dict(zip(df['gloss_id'], df['label']))
    
    # Create category mapping: {cat_id: category_name}
    category_mapping = dict(zip(df['cat_id'], df['category']))
    
    return gloss_mapping, category_mapping


def interpret_differences(precision_diff: float, recall_diff: float, f1_diff: float) -> str:
    """
    Interpret the differences in metrics with detailed information about which metrics favor which model.
    
    Args:
        precision_diff: Transformer Precision - IV3-GRU Precision
        recall_diff: Transformer Recall - IV3-GRU Recall
        f1_diff: Transformer F1-Score - IV3-GRU F1-Score
        
    Returns:
        Descriptive interpretation string
    """
    threshold = 0.05  # 5% threshold for significant difference
    
    # Track which metrics favor each model
    transformer_metrics = []
    iv3gru_metrics = []
    
    if precision_diff > threshold:
        transformer_metrics.append("Precision")
    elif precision_diff < -threshold:
        iv3gru_metrics.append("Precision")
    
    if recall_diff > threshold:
        transformer_metrics.append("Recall")
    elif recall_diff < -threshold:
        iv3gru_metrics.append("Recall")
    
    if f1_diff > threshold:
        transformer_metrics.append("F1-Score")
    elif f1_diff < -threshold:
        iv3gru_metrics.append("F1-Score")
    
    # Build descriptive interpretation
    parts = []
    
    if transformer_metrics:
        if len(transformer_metrics) == 3:
            parts.append("Transformer Better (Precision, Recall, F1-Score)")
        else:
            metrics_str = ", ".join(transformer_metrics)
            parts.append(f"Transformer Better ({metrics_str})")
    
    if iv3gru_metrics:
        if len(iv3gru_metrics) == 3:
            parts.append("IV3-GRU Better (Precision, Recall, F1-Score)")
        else:
            metrics_str = ", ".join(iv3gru_metrics)
            parts.append(f"IV3-GRU Better ({metrics_str})")
    
    if not parts:
        return "Similar (all metrics within 5%)"
    
    # If both models have advantages, it's mixed
    if transformer_metrics and iv3gru_metrics:
        return "Mixed: " + "; ".join(parts)
    else:
        return parts[0]


def create_comparison(input_csv: Path, label_mapping: Dict[int, str], id_column: str, label_type: str) -> pd.DataFrame:
    """
    Create comparison DataFrame from input metrics CSV.
    
    Args:
        input_csv: Path to input metrics CSV file
        label_mapping: Dictionary mapping IDs to label names
        id_column: Name of the ID column ('Gloss' or 'Category')
        label_type: Type of label ('Gloss' or 'Category')
        
    Returns:
        DataFrame with comparison metrics
    """
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    # Extract metrics
    transformer_precision = df['Transformer_Precision']
    transformer_recall = df['Transformer_Recall']
    transformer_f1 = df['Transformer_F1']
    
    iv3gru_precision = df['IV3-GRU_Precision']
    iv3gru_recall = df['IV3-GRU_Recall']
    iv3gru_f1 = df['IV3-GRU_F1']
    
    # Get IDs and labels
    ids = df[id_column].astype(int)
    labels = [label_mapping.get(id_val, f"Unknown_{id_val}") for id_val in ids]
    
    # Calculate differences (Transformer - IV3-GRU)
    precision_diff = transformer_precision - iv3gru_precision
    recall_diff = transformer_recall - iv3gru_recall
    f1_diff = transformer_f1 - iv3gru_f1
    
    # Create interpretations
    interpretations = [
        interpret_differences(p_diff, r_diff, f_diff)
        for p_diff, r_diff, f_diff in zip(precision_diff, recall_diff, f1_diff)
    ]
    
    # Build result DataFrame with multi-index columns
    columns = [
        ('ID', ''),
        ('Label', ''),
        ('Transformer', 'Precision'),
        ('Transformer', 'Recall'),
        ('Transformer', 'F1-Score'),
        ('IV3-GRU', 'Precision'),
        ('IV3-GRU', 'Recall'),
        ('IV3-GRU', 'F1-Score'),
        ('Difference', 'Precision'),
        ('Difference', 'Recall'),
        ('Difference', 'F1-Score'),
        ('Interpretation', '')
    ]
    
    data = {
        ('ID', ''): ids,
        ('Label', ''): labels,
        ('Transformer', 'Precision'): transformer_precision.round(6),
        ('Transformer', 'Recall'): transformer_recall.round(6),
        ('Transformer', 'F1-Score'): transformer_f1.round(6),
        ('IV3-GRU', 'Precision'): iv3gru_precision.round(6),
        ('IV3-GRU', 'Recall'): iv3gru_recall.round(6),
        ('IV3-GRU', 'F1-Score'): iv3gru_f1.round(6),
        ('Difference', 'Precision'): precision_diff.round(6),
        ('Difference', 'Recall'): recall_diff.round(6),
        ('Difference', 'F1-Score'): f1_diff.round(6),
        ('Interpretation', ''): interpretations
    }
    
    result_df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
    
    # Sort by ID
    result_df = result_df.sort_values(('ID', '')).reset_index(drop=True)
    
    return result_df


def main():
    """Main function to generate comparison CSV files."""
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.resolve()
    overall_dir = project_root / "metrics" / "extract" / "overall"
    output_dir = script_dir
    
    # Input files
    gloss_metrics_csv = overall_dir / "per_gloss_metrics.csv"
    category_metrics_csv = overall_dir / "per_category_metrics.csv"
    
    # Output files
    gloss_comparison_csv = output_dir / "gloss_comparison.csv"
    category_comparison_csv = output_dir / "category_comparison.csv"
    
    print("=" * 60)
    print("Metric Comparison: Transformer vs IV3-GRU")
    print("=" * 60)
    
    # Load label mappings
    try:
        gloss_mapping, category_mapping = load_label_mappings()
        print(f"\nLoaded {len(gloss_mapping)} gloss labels and {len(category_mapping)} categories")
    except Exception as e:
        print(f"Error loading label mappings: {e}")
        return
    
    # Process gloss comparison
    if not gloss_metrics_csv.exists():
        print(f"\nWarning: {gloss_metrics_csv} not found, skipping gloss comparison")
    else:
        print(f"\nProcessing gloss comparison...")
        try:
            gloss_df = create_comparison(gloss_metrics_csv, gloss_mapping, 'Gloss', 'Gloss')
            gloss_df.to_csv(gloss_comparison_csv, index=False)
            print(f"Saved {len(gloss_df)} gloss comparisons to {gloss_comparison_csv.name}")
            print(f"  Sample:")
            print(gloss_df.head(5).to_string(index=False))
        except Exception as e:
            print(f"Error processing gloss comparison: {e}")
            import traceback
            traceback.print_exc()
    
    # Process category comparison
    if not category_metrics_csv.exists():
        print(f"\nWarning: {category_metrics_csv} not found, skipping category comparison")
    else:
        print(f"\nProcessing category comparison...")
        try:
            category_df = create_comparison(category_metrics_csv, category_mapping, 'Category', 'Category')
            category_df.to_csv(category_comparison_csv, index=False)
            print(f"Saved {len(category_df)} category comparisons to {category_comparison_csv.name}")
            print(f"  Sample:")
            print(category_df.to_string(index=False))
        except Exception as e:
            print(f"Error processing category comparison: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All comparison files created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

