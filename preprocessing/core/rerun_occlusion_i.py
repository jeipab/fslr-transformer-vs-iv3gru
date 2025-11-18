"""
Rerun occlusion detection on individual/isolated sign sequences.

This script processes a folder containing NPZ files and a CSV file with labels,
reruns occlusion detection for each individual sign, and updates the 'occluded'
column in the CSV file.

Usage:
    python -m preprocessing.core.rerun_occlusion_i <folder_path> <csv_path>
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .occlusion_detection import compute_occlusion_detection_from_keypoints


def load_npz_file(npz_path: Path) -> Dict:
    """Load NPZ file and extract keypoint data.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dictionary containing X and mask
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        'X': data['X'],
        'mask': data['mask']
    }


def process_npz_file(npz_path: Path) -> int:
    """Run occlusion detection on a single NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Binary occlusion flag (0 or 1)
    """
    try:
        npz_data = load_npz_file(npz_path)
        X = npz_data['X']
        mask = npz_data['mask']
        
        # Check if arrays are empty
        if X.shape[0] == 0:
            return 0
        
        # Run occlusion detection with default parameters
        occluded = compute_occlusion_detection_from_keypoints(
            X=X,
            mask=mask,
            output_format='compatible'
        )
        return int(occluded)
    except Exception as e:
        print(f"  Warning: Failed to process {npz_path.name}: {e}")
        return 0


def process_csv_file(
    csv_path: Path,
    folder_path: Path,
    stats: Dict
) -> bool:
    """Process CSV file and update occlusion status.
    
    Args:
        csv_path: Path to CSV file
        folder_path: Folder containing NPZ files
        stats: Statistics dictionary to update
        
    Returns:
        True if processing was successful, False otherwise
    """
    # Load CSV with encoding fallbacks
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp1252')
    
    # Check required columns
    required_columns = ['file', 'occluded']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: CSV file missing required columns: {missing_columns}")
        return False
    
    # Clean file names (remove .npz extension if present, we'll add it back)
    df['file_clean'] = df['file'].str.replace('.npz', '', regex=False)
    
    # Process each row
    new_occluded_values = []
    processed = 0
    failed = 0
    
    for idx, row in df.iterrows():
        file_path_str = str(row['file_clean'])
        old_occluded = int(row['occluded']) if pd.notna(row['occluded']) else 0
        
        # Construct NPZ path (file might have subdirectory like "C0/clip_xxx.npz")
        npz_path = folder_path / f"{file_path_str}.npz"
        
        if not npz_path.exists():
            # Try without adding .npz (in case it's already in the path)
            npz_path = folder_path / file_path_str
            if not npz_path.exists():
                print(f"  Warning: NPZ file not found: {npz_path}")
                new_occluded_values.append(old_occluded)  # Keep old value
                # Still count in statistics (but no change)
                stats['old_occluded'] += old_occluded
                stats['old_not_occluded'] += (1 - old_occluded)
                stats['new_occluded'] += old_occluded  # No change, so new = old
                stats['new_not_occluded'] += (1 - old_occluded)
                failed += 1
                continue
        
        # Run occlusion detection
        if idx % 100 == 0:
            print(f"  Processing {idx + 1}/{len(df)}: {npz_path.name}")
        
        new_occluded = process_npz_file(npz_path)
        new_occluded_values.append(new_occluded)
        processed += 1
        
        # Update statistics
        stats['old_occluded'] += old_occluded
        stats['old_not_occluded'] += (1 - old_occluded)
        stats['new_occluded'] += new_occluded
        stats['new_not_occluded'] += (1 - new_occluded)
        
        # Track changes
        if old_occluded != new_occluded:
            stats['changed'] += 1
            stats['changes'].append({
                'row': idx + 2,  # +2 for 1-based and header row
                'file': row['file'],
                'old': old_occluded,
                'new': new_occluded
            })
    
    # Update DataFrame with new occlusion values
    df['occluded'] = new_occluded_values
    
    # Drop temporary column
    df = df.drop(columns=['file_clean'])
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    
    print(f"\nProcessed: {processed} file(s)")
    if failed > 0:
        print(f"Failed/Missing: {failed} file(s)")
    
    return True


def print_statistics(stats: Dict):
    """Print occlusion detection statistics.
    
    Args:
        stats: Statistics dictionary
    """
    print("\n" + "="*60)
    print("OCCLUSION DETECTION STATISTICS")
    print("="*60)
    
    print("\nOLD Counts:")
    print(f"  Occluded:     {stats['old_occluded']:5d}")
    print(f"  Not Occluded: {stats['old_not_occluded']:5d}")
    print(f"  Total:        {stats['old_occluded'] + stats['old_not_occluded']:5d}")
    
    print("\nNEW Counts:")
    print(f"  Occluded:     {stats['new_occluded']:5d}")
    print(f"  Not Occluded: {stats['new_not_occluded']:5d}")
    print(f"  Total:        {stats['new_occluded'] + stats['new_not_occluded']:5d}")
    
    print(f"\nChanges: {stats['changed']} files changed occlusion status")
    
    if stats['changed'] > 0 and stats['changes']:
        print("\nChanged Files (first 20):")
        for change in stats['changes'][:20]:
            print(f"  Row {change['row']}: {change['file']} - {change['old']} -> {change['new']}")
        if len(stats['changes']) > 20:
            print(f"  ... and {len(stats['changes']) - 20} more")
    
    print("="*60 + "\n")


def main():
    """Main function to rerun occlusion detection on individual signs."""
    parser = argparse.ArgumentParser(
        description="Rerun occlusion detection on individual/isolated sign sequences"
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing NPZ files'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to CSV file with labels'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    csv_path = Path(args.csv_path)
    
    # Validate inputs
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return
    
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        return
    
    print(f"Processing folder: {folder_path}")
    print(f"CSV file: {csv_path}\n")
    
    # Initialize statistics
    stats = {
        'old_occluded': 0,
        'old_not_occluded': 0,
        'new_occluded': 0,
        'new_not_occluded': 0,
        'changed': 0,
        'changes': []
    }
    
    # Process CSV file
    print("Processing CSV file...")
    if process_csv_file(csv_path, folder_path, stats):
        print(f"\n✓ CSV file updated: {csv_path}")
    else:
        print(f"\n✗ Failed to process CSV file")
        return
    
    # Print statistics
    print_statistics(stats)


if __name__ == '__main__':
    main()

