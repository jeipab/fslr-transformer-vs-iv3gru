#!/usr/bin/env python3
"""
Dataset Breakdown Script

Generates statistics breakdown for FSL-105, SMP-105, and CMB-105 datasets
showing total counts, occluded/not occluded counts by category and gloss.

Usage:
    python dataset_breakdown.py
    
    # Redirect output to a file:
    python dataset_breakdown.py > breakdown.txt
    
    # On Windows PowerShell:
    python dataset_breakdown.py | Out-File -FilePath output.txt -Encoding utf8

The script reads from data/processed/labels.csv by default and outputs statistics
for:
    - FSL-105: Signers S0-S3
    - SMP-105: Signers S4-S7
    - CMB-105: Signers S0-S7 (all signers)
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_dataset(df, dataset_name, signers):
    """
    Analyze a dataset subset filtered by signers.
    
    Args:
        df: Full DataFrame with all data
        dataset_name: Name of the dataset (e.g., "FSL-105")
        signers: List of signer IDs to include (e.g., ["S0", "S1", "S2", "S3"])
    
    Returns:
        Dictionary with analysis results
    """
    # Filter by signers
    filtered_df = df[df['signer'].isin(signers)].copy()
    
    # Overall statistics
    total = len(filtered_df)
    not_occluded = len(filtered_df[filtered_df['occluded'] == 0])
    occluded = len(filtered_df[filtered_df['occluded'] == 1])
    
    # By Category (0-9)
    category_stats = {}
    for cat in range(10):
        cat_df = filtered_df[filtered_df['cat'] == cat]
        cat_total = len(cat_df)
        cat_not_occluded = len(cat_df[cat_df['occluded'] == 0])
        cat_occluded = len(cat_df[cat_df['occluded'] == 1])
        category_stats[cat] = {
            'total': cat_total,
            'not_occluded': cat_not_occluded,
            'occluded': cat_occluded
        }
    
    # By Gloss (0-104)
    gloss_stats = {}
    for gloss in range(105):
        gloss_df = filtered_df[filtered_df['gloss'] == gloss]
        gloss_total = len(gloss_df)
        gloss_not_occluded = len(gloss_df[gloss_df['occluded'] == 0])
        gloss_occluded = len(gloss_df[gloss_df['occluded'] == 1])
        gloss_stats[gloss] = {
            'total': gloss_total,
            'not_occluded': gloss_not_occluded,
            'occluded': gloss_occluded
        }
    
    return {
        'name': dataset_name,
        'total': total,
        'not_occluded': not_occluded,
        'occluded': occluded,
        'category_stats': category_stats,
        'gloss_stats': gloss_stats
    }


def print_results(results):
    """
    Print results in the exact format matching the reference text file.
    
    Args:
        results: Dictionary returned from analyze_dataset()
    """
    print(f"{results['name']}: Total={results['total']}, Not Occluded={results['not_occluded']}, Occluded={results['occluded']}")
    print("Detailed Report:")
    print()
    print("By Category:")
    print()
    
    for cat in range(10):
        stats = results['category_stats'][cat]
        print(f"  Category {cat}: Total={stats['total']}, Not Occluded={stats['not_occluded']}, Occluded={stats['occluded']}")
    
    print()
    print("By Gloss:")
    print()
    
    for gloss in range(105):
        stats = results['gloss_stats'][gloss]
        print(f"  Gloss {gloss}: Total={stats['total']}, Not Occluded={stats['not_occluded']}, Occluded={stats['occluded']}")


def main():
    """Main function to run the dataset breakdown analysis."""
    # Default CSV path
    csv_path = Path("data/processed/labels.csv")
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return 1
    
    # Read CSV
    print(f"Loading data from {csv_path}...", file=sys.stderr)
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['file', 'gloss', 'cat', 'occluded', 'signer', 'duration']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}", file=sys.stderr)
        return 1
    
    # Analyze each dataset
    datasets = [
        ("FSL-105", ["S0", "S1", "S2", "S3"]),
        ("SMP-105", ["S4", "S5", "S6", "S7"]),
        ("CMB-105", ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]),
    ]
    
    results_list = []
    for dataset_name, signers in datasets:
        results = analyze_dataset(df, dataset_name, signers)
        results_list.append(results)
    
    # Print results
    for i, results in enumerate(results_list):
        print_results(results)
        if i < len(results_list) - 1:
            print()
            print("=" * 80)
            print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

