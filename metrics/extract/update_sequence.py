"""
Script to update CSV files with actual sequence data from JSON validation results.

Usage:
    python metrics/extract/update_sequence.py

This script updates the following CSV files:
- raw data/Per Sequence/TABLE E2 — Per-Sequence Recognition Performance - I.csv
- raw data/Per Sequence/TABLE E2 — Per-Sequence Recognition Performance - T.csv
- raw data/Per Sequence/TABLE E4 — Per-Sequence Classification Performance - I.csv
- raw data/Per Sequence/TABLE E4 — Per-Sequence Classification Performance - T.csv

For Recognition (E2): Uses predicted_sequence/ground_truth_sequence (gloss IDs)
For Classification (E4): Uses predicted_categories/ground_truth_categories (category IDs)
"""

import json
import csv
import os
from pathlib import Path


def load_json_data(json_path):
    """Load JSON file and create mapping from file_name to sequences."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mapping = {}
    for prediction in data.get('predictions', []):
        file_name = prediction.get('file_name')
        if file_name:
            mapping[file_name] = {
                'ground_truth_sequence': prediction.get('ground_truth_sequence', []),
                'predicted_sequence': prediction.get('predicted_sequence', []),
                'ground_truth_categories': prediction.get('ground_truth_categories', []),
                'predicted_categories': prediction.get('predicted_categories', [])
            }
    return mapping


def format_sequence(sequence):
    """Format sequence list as 'ID1 -> ID2 -> ID3'."""
    if not sequence:
        return ''
    return ' -> '.join(str(item) for item in sequence)


def update_csv(csv_path, json_mapping, use_categories=False):
    """Update CSV file with sequence data from JSON mapping."""
    # Read CSV
    title_row = None
    header_row = None
    data_rows = []
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                # Title row (e.g., "IV3 - GRU" or "Transformer")
                title_row = row
            elif i == 1:
                # Header row
                header_row = row
            else:
                data_rows.append(row)
    
    # Find column indices
    try:
        file_name_idx = header_row.index('File Name')
        gt_seq_idx = header_row.index('Ground Truth Sequence')
        pred_seq_idx = header_row.index('Predicted Sequence')
    except ValueError as e:
        print(f"Error: Could not find required column in {csv_path}: {e}")
        return False
    
    # Update rows
    updated_count = 0
    missing_count = 0
    
    for row in data_rows:
        if len(row) <= file_name_idx:
            continue
        
        file_name = row[file_name_idx]
        
        if file_name in json_mapping:
            data = json_mapping[file_name]
            
            if use_categories:
                # Use category sequences for Classification
                gt_sequence = data.get('ground_truth_categories', [])
                pred_sequence = data.get('predicted_categories', [])
            else:
                # Use gloss sequences for Recognition
                gt_sequence = data.get('ground_truth_sequence', [])
                pred_sequence = data.get('predicted_sequence', [])
            
            # Ensure row has enough columns
            while len(row) <= max(gt_seq_idx, pred_seq_idx):
                row.append('')
            
            row[gt_seq_idx] = format_sequence(gt_sequence)
            row[pred_seq_idx] = format_sequence(pred_sequence)
            updated_count += 1
        else:
            missing_count += 1
            print(f"Warning: File name '{file_name}' not found in JSON data")
    
    # Write updated CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Write title row if it exists
        if title_row:
            writer.writerow(title_row)
        # Write header row
        writer.writerow(header_row)
        # Write data rows
        writer.writerows(data_rows)
    
    print(f"Updated {csv_path}: {updated_count} rows updated, {missing_count} rows not found")
    return True


def main():
    """Main function to update all CSV files."""
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Paths
    json_iv3gru_path = project_root / 'metrics' / 'extract' / 'shared_inputs' / 'ctc_validation_results_iv3gru.json'
    json_transformer_path = project_root / 'metrics' / 'extract' / 'shared_inputs' / 'ctc_validation_results_transformer.json'
    
    csv_e2_i_path = project_root / 'raw data' / 'Per Sequence' / 'TABLE E2 — Per-Sequence Recognition Performance - I.csv'
    csv_e2_t_path = project_root / 'raw data' / 'Per Sequence' / 'TABLE E2 — Per-Sequence Recognition Performance - T.csv'
    csv_e4_i_path = project_root / 'raw data' / 'Per Sequence' / 'TABLE E4 — Per-Sequence Classification Performance - I.csv'
    csv_e4_t_path = project_root / 'raw data' / 'Per Sequence' / 'TABLE E4 — Per-Sequence Classification Performance - T.csv'
    
    # Load JSON data
    print("Loading JSON data...")
    json_iv3gru = load_json_data(json_iv3gru_path)
    json_transformer = load_json_data(json_transformer_path)
    print(f"Loaded {len(json_iv3gru)} entries from IV3-GRU JSON")
    print(f"Loaded {len(json_transformer)} entries from Transformer JSON")
    
    # Update CSV files
    print("\nUpdating CSV files...")
    
    # Recognition files (use gloss sequences)
    update_csv(csv_e2_i_path, json_iv3gru, use_categories=False)
    update_csv(csv_e2_t_path, json_transformer, use_categories=False)
    
    # Classification files (use category sequences)
    update_csv(csv_e4_i_path, json_iv3gru, use_categories=True)
    update_csv(csv_e4_t_path, json_transformer, use_categories=True)
    
    print("\nAll CSV files updated successfully!")


if __name__ == '__main__':
    main()

