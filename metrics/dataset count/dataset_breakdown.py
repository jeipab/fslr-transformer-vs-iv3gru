#!/usr/bin/env python3
"""
Dataset Breakdown Script

Generates gloss and category count CSV files for FSL-105, SMP-105, and CMB-105 datasets
from train/test splits.

Usage:
    # From project root directory:
    python "metrics/dataset count/dataset_breakdown.py"
    
    # Or on Windows PowerShell:
    python "metrics\dataset count\dataset_breakdown.py"
    
Outputs 12 CSV files:
    - FSL-gloss_count.csv
    - SMP-gloss_count.csv
    - CMB-gloss_count.csv
    - FSL-category_count.csv
    - SMP-category_count.csv
    - CMB-category_count.csv
    - training-gloss_count.csv
    - training-category_count.csv
    - testing-gloss_count.csv
    - testing-category_count.csv
    - continuous-gloss_count.csv
    - continuous-category_count.csv

The script reads from:
    - data/processed/CMB105_test.csv
    - data/processed/CMB105_train.csv
    - data/processed/continuous_testing/*.json (for continuous sequences)

Dataset definitions:
    - FSL-105: Signers S0-S3
    - SMP-105: Signers S4-S7
    - CMB-105: Signers S0-S7 (all signers)
"""

import pandas as pd
import json
import sys
from pathlib import Path


def load_data():
    """Load test and train CSV files, returning both separately and combined."""
    test_path = Path("data/processed/CMB105_test.csv")
    train_path = Path("data/processed/CMB105_train.csv")
    
    if not test_path.exists():
        print(f"Error: CSV file not found at {test_path}", file=sys.stderr)
        return None, None, None
    
    if not train_path.exists():
        print(f"Error: CSV file not found at {train_path}", file=sys.stderr)
        return None, None, None
    
    print(f"Loading data from {test_path} and {train_path}...", file=sys.stderr)
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    
    # Validate required columns
    required_columns = ['file', 'gloss', 'cat', 'occluded', 'signer', 'duration']
    for df, name in [(df_test, "test"), (df_train, "train")]:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            print(f"Error: Missing required columns in {name} CSV: {missing_columns}", file=sys.stderr)
            return None, None, None
    
    df_combined = pd.concat([df_test, df_train], ignore_index=True)
    
    return df_test, df_train, df_combined


def load_label_mappings():
    """Load gloss and category label mappings from labels_reference.csv."""
    ref_path = Path("data/labels_reference.csv")
    
    if not ref_path.exists():
        print(f"Error: Labels reference file not found at {ref_path}", file=sys.stderr)
        return None, None
    
    print(f"Loading label mappings from {ref_path}...", file=sys.stderr)
    df_ref = pd.read_csv(ref_path)
    
    gloss_mapping = dict(zip(df_ref['gloss_id'], df_ref['label']))
    category_mapping = dict(zip(df_ref['cat_id'], df_ref['category']))
    
    return gloss_mapping, category_mapping


def count_glosses(df, signers):
    """Count occluded and non-occluded occurrences of each gloss for given signers."""
    filtered_df = df[df['signer'].isin(signers)]
    
    results = []
    # Include all gloss IDs from 0 to 104
    for gloss_id in range(105):
        gloss_df = filtered_df[filtered_df['gloss'] == gloss_id]
        occluded = len(gloss_df[gloss_df['occluded'] == 1])
        non_occluded = len(gloss_df[gloss_df['occluded'] == 0])
        total = len(gloss_df)
        
        results.append({
            'gloss_id': gloss_id,
            'occluded': occluded,
            'non_occluded': non_occluded,
            'total': total
        })
    
    return pd.DataFrame(results)


def count_categories(df, signers):
    """Count occluded and non-occluded occurrences of each category for given signers."""
    filtered_df = df[df['signer'].isin(signers)]
    
    results = []
    # Include all category IDs from 0 to 9
    for cat_id in range(10):
        cat_df = filtered_df[filtered_df['cat'] == cat_id]
        occluded = len(cat_df[cat_df['occluded'] == 1])
        non_occluded = len(cat_df[cat_df['occluded'] == 0])
        total = len(cat_df)
        
        results.append({
            'category_id': cat_id,
            'occluded': occluded,
            'non_occluded': non_occluded,
            'total': total
        })
    
    return pd.DataFrame(results)


def count_glosses_all(df):
    """Count occluded and non-occluded occurrences of each gloss for all signers."""
    results = []
    # Include all gloss IDs from 0 to 104
    for gloss_id in range(105):
        gloss_df = df[df['gloss'] == gloss_id]
        occluded = len(gloss_df[gloss_df['occluded'] == 1])
        non_occluded = len(gloss_df[gloss_df['occluded'] == 0])
        total = len(gloss_df)
        
        results.append({
            'gloss_id': gloss_id,
            'occluded': occluded,
            'non_occluded': non_occluded,
            'total': total
        })
    
    return pd.DataFrame(results)


def count_categories_all(df):
    """Count occluded and non-occluded occurrences of each category for all signers."""
    results = []
    # Include all category IDs from 0 to 9
    for cat_id in range(10):
        cat_df = df[df['cat'] == cat_id]
        occluded = len(cat_df[cat_df['occluded'] == 1])
        non_occluded = len(cat_df[cat_df['occluded'] == 0])
        total = len(cat_df)
        
        results.append({
            'category_id': cat_id,
            'occluded': occluded,
            'non_occluded': non_occluded,
            'total': total
        })
    
    return pd.DataFrame(results)


def load_continuous_json_data(json_dir: Path):
    """Load all JSON files from continuous_testing directory and extract segment data.
    
    Args:
        json_dir: Directory containing JSON files
        
    Returns:
        DataFrame with columns: gloss, cat, occluded
    """
    if not json_dir.exists():
        print(f"Error: JSON directory not found at {json_dir}", file=sys.stderr)
        return None
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}", file=sys.stderr)
        return None
    
    print(f"Loading {len(json_files)} JSON files from {json_dir}...", file=sys.stderr)
    
    segments = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract segments
            if 'segments' in data:
                for segment in data['segments']:
                    segments.append({
                        'gloss': int(segment['gloss']),
                        'cat': int(segment['category']),
                        'occluded': int(segment.get('occluded', 0))
                    })
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}", file=sys.stderr)
            continue
    
    if not segments:
        print(f"Error: No segments found in JSON files", file=sys.stderr)
        return None
    
    df = pd.DataFrame(segments)
    print(f"Loaded {len(df)} segments from {len(json_files)} JSON files", file=sys.stderr)
    
    return df


def count_glosses_from_json(df_json):
    """Count occluded and non-occluded occurrences of each gloss from JSON data."""
    results = []
    # Include all gloss IDs from 0 to 104
    for gloss_id in range(105):
        gloss_df = df_json[df_json['gloss'] == gloss_id]
        occluded = len(gloss_df[gloss_df['occluded'] == 1])
        non_occluded = len(gloss_df[gloss_df['occluded'] == 0])
        total = len(gloss_df)
        
        results.append({
            'gloss_id': gloss_id,
            'occluded': occluded,
            'non_occluded': non_occluded,
            'total': total
        })
    
    return pd.DataFrame(results)


def count_categories_from_json(df_json):
    """Count occluded and non-occluded occurrences of each category from JSON data."""
    results = []
    # Include all category IDs from 0 to 9
    for cat_id in range(10):
        cat_df = df_json[df_json['cat'] == cat_id]
        occluded = len(cat_df[cat_df['occluded'] == 1])
        non_occluded = len(cat_df[cat_df['occluded'] == 0])
        total = len(cat_df)
        
        results.append({
            'category_id': cat_id,
            'occluded': occluded,
            'non_occluded': non_occluded,
            'total': total
        })
    
    return pd.DataFrame(results)


def write_gloss_csv(df_counts, gloss_mapping, filename):
    """Write gloss count data to CSV file with labels."""
    output_dir = Path("metrics/dataset count")
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_out = df_counts.copy()
    df_out['Label'] = df_out['gloss_id'].map(gloss_mapping).fillna('')
    df_out = df_out[['gloss_id', 'Label', 'occluded', 'non_occluded', 'total']]
    df_out.columns = ['Gloss ID', 'Label', 'Occluded', 'NonOccluded', 'Total']
    df_out.to_csv(output_path, index=False)
    print(f"Written: {output_path}", file=sys.stderr)


def write_category_csv(df_counts, category_mapping, filename):
    """Write category count data to CSV file with labels."""
    output_dir = Path("metrics/dataset count")
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_out = df_counts.copy()
    df_out['Label'] = df_out['category_id'].map(category_mapping).fillna('')
    df_out = df_out[['category_id', 'Label', 'occluded', 'non_occluded', 'total']]
    df_out.columns = ['CategoryID', 'Label', 'Occluded', 'NonOccluded', 'Total']
    df_out.to_csv(output_path, index=False)
    print(f"Written: {output_path}", file=sys.stderr)


def main():
    """Main function to generate count CSV files."""
    df_test, df_train, df_combined = load_data()
    if df_combined is None:
        return 1
    
    gloss_mapping, category_mapping = load_label_mappings()
    if gloss_mapping is None or category_mapping is None:
        return 1
    
    # Dataset definitions
    datasets = [
        ("FSL", ["S0", "S1", "S2", "S3"]),
        ("SMP", ["S4", "S5", "S6", "S7"]),
        ("CMB", ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]),
    ]
    
    # Generate count files for each dataset (using combined data)
    for dataset_name, signers in datasets:
        # Gloss counts
        gloss_counts = count_glosses(df_combined, signers)
        write_gloss_csv(gloss_counts, gloss_mapping, f"{dataset_name}-gloss_count.csv")
        
        # Category counts
        cat_counts = count_categories(df_combined, signers)
        write_category_csv(cat_counts, category_mapping, f"{dataset_name}-category_count.csv")
    
    # Generate training and testing split count files
    training_gloss_counts = count_glosses_all(df_train)
    write_gloss_csv(training_gloss_counts, gloss_mapping, "training-gloss_count.csv")
    
    training_cat_counts = count_categories_all(df_train)
    write_category_csv(training_cat_counts, category_mapping, "training-category_count.csv")
    
    testing_gloss_counts = count_glosses_all(df_test)
    write_gloss_csv(testing_gloss_counts, gloss_mapping, "testing-gloss_count.csv")
    
    testing_cat_counts = count_categories_all(df_test)
    write_category_csv(testing_cat_counts, category_mapping, "testing-category_count.csv")
    
    # Generate continuous sequence count files from JSON
    json_dir = Path("data/processed/continuous_testing")
    df_continuous = load_continuous_json_data(json_dir)
    if df_continuous is not None:
        continuous_gloss_counts = count_glosses_from_json(df_continuous)
        write_gloss_csv(continuous_gloss_counts, gloss_mapping, "continuous-gloss_count.csv")
        
        continuous_cat_counts = count_categories_from_json(df_continuous)
        write_category_csv(continuous_cat_counts, category_mapping, "continuous-category_count.csv")
    
    print("All CSV files generated successfully.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
