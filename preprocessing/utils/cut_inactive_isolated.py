"""
Script to remove inactive sections from isolated sign language NPZ files.

This script detects and removes periods where keypoints are inactive (e.g., both hands
not visible) from individual sign clips, then updates the duration in the corresponding CSV file.

Usage Examples:
    # Basic usage (auto-detects CSV file, overwrites input files)
    python preprocessing/utils/cut_inactive_isolated.py data/processed/FSL105_val/clip_00001_good_morning_S0.npz
    
    # Process entire folder (overwrites all NPZ files in folder and updates CSV)
    python preprocessing/utils/cut_inactive_isolated.py data/processed/FSL105_val/
    
    # Specify CSV file explicitly
    python preprocessing/utils/cut_inactive_isolated.py data/processed/FSL105_val/ --csv-file data/processed/FSL105_val.csv
    
    # Adjust visibility threshold (higher = more lenient, lower = more strict)
    python preprocessing/utils/cut_inactive_isolated.py input.npz --min-visible-ratio 0.2
    
    # Process folder with custom threshold
    python preprocessing/utils/cut_inactive_isolated.py data/processed/FSL105_val/ --min-visible-ratio 0.2
    
    # Use minimum keypoint count instead of ratio
    python preprocessing/utils/cut_inactive_isolated.py input.npz --min-visible-keypoints 10
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def detect_inactive_frames(
    mask: np.ndarray,
    timestamps_ms: np.ndarray,
    left_hand_start: int = 25,
    left_hand_end: int = 45,
    right_hand_start: int = 46,
    right_hand_end: int = 66,
    min_visible_keypoints: Optional[int] = None,
    min_visible_ratio: float = 0.1
) -> np.ndarray:
    """
    Detect frames with inactive keypoints.
    
    Args:
        mask: Keypoint visibility mask [T, 89] where True = visible
        timestamps_ms: Timestamp array [T] in milliseconds
        left_hand_start: Starting index for left hand keypoints (default 25)
        left_hand_end: Ending index for left hand keypoints (default 45)
        right_hand_start: Starting index for right hand keypoints (default 46)
        right_hand_end: Ending index for right hand keypoints (default 66)
        min_visible_keypoints: Minimum number of visible keypoints (overrides ratio if set)
        min_visible_ratio: Minimum ratio of visible keypoints (0.0-1.0)
        
    Returns:
        Boolean array [T] where True indicates inactive frames to remove
    """
    T = len(mask)
    inactive = np.zeros(T, dtype=bool)
    
    for t in range(T):
        # Check if both hands are inactive
        left_hand_mask = mask[t, left_hand_start:left_hand_end+1]
        right_hand_mask = mask[t, right_hand_start:right_hand_end+1]
        
        left_inactive = not np.any(left_hand_mask)
        right_inactive = not np.any(right_hand_mask)
        both_hands_inactive = left_inactive and right_inactive
        
        # Check overall keypoint visibility
        visible_count = np.sum(mask[t])
        total_keypoints = len(mask[t])
        visible_ratio = visible_count / total_keypoints
        
        if min_visible_keypoints is not None:
            is_inactive = both_hands_inactive or (visible_count < min_visible_keypoints)
        else:
            is_inactive = both_hands_inactive or (visible_ratio < min_visible_ratio)
        
        inactive[t] = is_inactive
    
    return inactive


def normalize_file_path(file_path: str) -> str:
    """Normalize file path for CSV matching.
    
    Removes subdirectory prefix (e.g., 'C0/') and .npz extension.
    
    Args:
        file_path: File path from CSV or filename (e.g., 'C0/clip_00001_good_morning_S0.npz')
        
    Returns:
        Normalized filename (e.g., 'clip_00001_good_morning_S0')
    """
    # Remove .npz extension if present
    if file_path.endswith('.npz'):
        file_path = file_path[:-4]
    
    # Remove subdirectory prefix if present (e.g., 'C0/', 'C1/')
    parts = file_path.split('/')
    if len(parts) > 1:
        return parts[-1]
    
    return file_path


def update_csv_duration(
    csv_path: Path,
    filename: str,
    new_duration_sec: float
) -> bool:
    """
    Update duration in CSV file for a specific NPZ file.
    
    Args:
        csv_path: Path to CSV file
        filename: NPZ filename (with or without extension, may have subdirectory prefix)
        new_duration_sec: New duration in seconds
        
    Returns:
        True if updated, False if not found
    """
    try:
        # Load CSV with encoding fallback
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='cp1252')
    except Exception as e:
        print(f"  Warning: Failed to load CSV: {e}")
        return False
    
    if 'file' not in df.columns:
        print(f"  Warning: CSV missing 'file' column")
        return False
    
    if 'duration' not in df.columns:
        print(f"  Warning: CSV missing 'duration' column, adding it")
        df['duration'] = 0.0
    
    # Normalize the input filename for matching
    normalized_input = normalize_file_path(filename)
    
    # Find matching row(s)
    matched = False
    for idx, row in df.iterrows():
        csv_file_path = str(row['file'])
        normalized_csv = normalize_file_path(csv_file_path)
        
        if normalized_csv == normalized_input:
            old_duration = row['duration']
            df.at[idx, 'duration'] = new_duration_sec
            matched = True
            print(f"  Updated CSV: {csv_file_path}: {old_duration:.3f}s -> {new_duration_sec:.3f}s")
    
    if not matched:
        return False
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    return True


def cut_inactive_sections(
    npz_path: Path,
    csv_path: Optional[Path] = None,
    output_npz_path: Optional[Path] = None,
    min_visible_keypoints: Optional[int] = None,
    min_visible_ratio: float = 0.1
) -> Dict:
    """
    Remove inactive sections from isolated NPZ file and update CSV duration.
    
    Args:
        npz_path: Path to input NPZ file
        csv_path: Optional path to CSV file (auto-detected if None)
        output_npz_path: Optional output NPZ path (overwrites input if None)
        min_visible_keypoints: Minimum visible keypoints per frame (overrides ratio if set)
        min_visible_ratio: Minimum ratio of visible keypoints (0.0-1.0)
        
    Returns:
        Dictionary with statistics about removed frames
    """
    # Load NPZ file
    print(f"Loading NPZ file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        X = np.array(data['X'])
        mask = np.array(data['mask'])
        timestamps_ms = np.array(data['timestamps_ms'])
        X2048 = data.get('X2048', None)
        if X2048 is not None:
            X2048 = np.array(X2048)
        meta = data.get('meta', None)
    
    T_original = len(X)
    print(f"Original frames: {T_original}")
    
    # Detect inactive frames
    inactive_mask = detect_inactive_frames(
        mask,
        timestamps_ms,
        min_visible_keypoints=min_visible_keypoints,
        min_visible_ratio=min_visible_ratio
    )
    
    # Get active frame indices
    active_mask = ~inactive_mask
    active_indices = np.where(active_mask)[0]
    n_removed = T_original - len(active_indices)
    
    print(f"Removed frames: {n_removed} ({n_removed/T_original*100:.1f}%)")
    print(f"Remaining frames: {len(active_indices)}")
    
    if len(active_indices) == 0:
        raise ValueError("All frames were marked as inactive! Check your detection parameters.")
    
    # Filter arrays
    X_filtered = X[active_indices].astype(np.float32)
    mask_filtered = mask[active_indices]
    timestamps_filtered = timestamps_ms[active_indices]
    
    # Recalculate timestamps to be continuous (starting from 0)
    first_timestamp = timestamps_filtered[0]
    timestamps_adjusted = timestamps_filtered - first_timestamp
    
    # Update timestamps to be evenly spaced based on original frame rate
    if len(timestamps_adjusted) > 1:
        # Estimate frame rate from original timestamps
        original_durations = np.diff(timestamps_ms)
        valid_durations = original_durations[original_durations > 0]
        if len(valid_durations) > 0:
            avg_frame_duration = np.median(valid_durations)
        else:
            avg_frame_duration = timestamps_adjusted[-1] / len(timestamps_adjusted)
        
        # Create evenly spaced timestamps
        timestamps_new = np.arange(len(timestamps_adjusted), dtype=np.int64) * int(avg_frame_duration)
    else:
        timestamps_new = np.array([0], dtype=np.int64)
    
    # Calculate new duration in seconds
    new_duration_sec = timestamps_new[-1] / 1000.0 if len(timestamps_new) > 0 else 0.0
    original_duration_sec = timestamps_ms[-1] / 1000.0 if len(timestamps_ms) > 0 else 0.0
    
    # Filter X2048 if present
    X2048_filtered = None
    if X2048 is not None:
        X2048_filtered = X2048[active_indices].astype(np.float32)
    
    # Update metadata duration if present
    if meta is not None:
        try:
            if isinstance(meta, (np.ndarray, np.generic)):
                meta_content = meta.item()
            else:
                meta_content = meta
            
            if isinstance(meta_content, str):
                meta_dict = json.loads(meta_content)
            elif isinstance(meta_content, dict):
                meta_dict = meta_content.copy()
            else:
                meta_dict = {}
            
            meta_dict['duration_sec'] = new_duration_sec
            meta = json.dumps(meta_dict)
        except Exception as e:
            print(f"  Warning: Failed to update metadata duration: {e}")
            # Keep original meta if update fails
            pass
    
    # Save updated NPZ file
    output_npz = output_npz_path or npz_path
    print(f"Saving updated NPZ file: {output_npz}")
    
    save_dict = {
        'X': X_filtered,
        'mask': mask_filtered,
        'timestamps_ms': timestamps_new,
    }
    if meta is not None:
        save_dict['meta'] = meta
    if X2048_filtered is not None:
        save_dict['X2048'] = X2048_filtered
    
    np.savez_compressed(output_npz, **save_dict)
    
    # Update CSV file if provided
    if csv_path and csv_path.exists():
        print(f"Updating CSV file: {csv_path}")
        update_csv_duration(csv_path, npz_path.name, new_duration_sec)
    
    # Return statistics
    return {
        'original_frames': T_original,
        'removed_frames': n_removed,
        'remaining_frames': len(active_indices),
        'removal_percentage': n_removed / T_original * 100,
        'original_duration_sec': original_duration_sec,
        'new_duration_sec': new_duration_sec
    }


def find_csv_file(npz_path: Path) -> Optional[Path]:
    """Try to find CSV file in parent directories.
    
    Looks for common CSV filenames in the same directory or parent directory.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Path to CSV file if found, None otherwise
    """
    # Common CSV filenames to check
    csv_names = ['FSL105_val.csv', 'FSL105_train.csv', 'labels.csv']
    
    # Check same directory
    for csv_name in csv_names:
        csv_path = npz_path.parent / csv_name
        if csv_path.exists():
            return csv_path
    
    # Check parent directory
    for csv_name in csv_names:
        csv_path = npz_path.parent.parent / csv_name
        if csv_path.exists():
            return csv_path
    
    return None


def process_folder(
    folder_path: Path,
    csv_path: Optional[Path] = None,
    min_visible_keypoints: Optional[int] = None,
    min_visible_ratio: float = 0.1
) -> Dict:
    """
    Process all NPZ files in a folder, overwriting originals and updating CSV.
    
    Args:
        folder_path: Path to folder containing NPZ files
        csv_path: Optional path to CSV file (auto-detected if None)
        min_visible_keypoints: Minimum visible keypoints per frame
        min_visible_ratio: Minimum ratio of visible keypoints
        
    Returns:
        Dictionary with processing statistics
    """
    npz_files = sorted(folder_path.glob('*.npz'))
    
    if not npz_files:
        print(f"No NPZ files found in {folder_path}")
        return {'processed': 0, 'failed': 0, 'total_removed_frames': 0}
    
    # Auto-detect CSV if not provided
    if csv_path is None:
        # Try to find CSV in folder or parent
        csv_path = find_csv_file(npz_files[0])
        if csv_path:
            print(f"Auto-detected CSV file: {csv_path}")
        else:
            print(f"Warning: No CSV file found, duration updates will be skipped")
    
    print(f"Found {len(npz_files)} NPZ file(s) in {folder_path}")
    if csv_path:
        print(f"CSV file: {csv_path}")
    print("="*70)
    
    stats = {
        'processed': 0,
        'failed': 0,
        'total_removed_frames': 0,
        'total_original_frames': 0,
        'total_remaining_frames': 0
    }
    
    for npz_file in npz_files:
        print(f"\nProcessing: {npz_file.name}")
        print("-"*70)
        
        try:
            file_stats = cut_inactive_sections(
                npz_path=npz_file,
                csv_path=csv_path,
                output_npz_path=npz_file,  # Overwrite original
                min_visible_keypoints=min_visible_keypoints,
                min_visible_ratio=min_visible_ratio
            )
            
            stats['processed'] += 1
            stats['total_removed_frames'] += file_stats['removed_frames']
            stats['total_original_frames'] += file_stats['original_frames']
            stats['total_remaining_frames'] += file_stats['remaining_frames']
            
        except Exception as e:
            print(f"  ERROR: Failed to process {npz_file.name}: {e}")
            stats['failed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Remove inactive sections from isolated sign language NPZ files'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to input NPZ file or folder containing NPZ files'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default=None,
        help='Path to CSV file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output-npz',
        type=str,
        default=None,
        help='Output NPZ path (overwrites input if not provided, single file only)'
    )
    parser.add_argument(
        '--min-visible-keypoints',
        type=int,
        default=None,
        help='Minimum number of visible keypoints per frame'
    )
    parser.add_argument(
        '--min-visible-ratio',
        type=float,
        default=0.1,
        help='Minimum ratio of visible keypoints (0.0-1.0, default: 0.1)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    csv_path = Path(args.csv_file) if args.csv_file else None
    
    # Check if path is a directory or file
    if path.is_dir():
        # Process folder
        print(f"Processing folder: {path}")
        stats = process_folder(
            folder_path=path,
            csv_path=csv_path,
            min_visible_keypoints=args.min_visible_keypoints,
            min_visible_ratio=args.min_visible_ratio
        )
        
        print("\n" + "="*70)
        print("Folder Processing Summary:")
        print("="*70)
        print(f"Files processed: {stats['processed']}")
        print(f"Files failed: {stats['failed']}")
        if stats['processed'] > 0:
            print(f"Total original frames: {stats['total_original_frames']}")
            print(f"Total removed frames: {stats['total_removed_frames']} "
                  f"({stats['total_removed_frames']/stats['total_original_frames']*100:.1f}%)")
            print(f"Total remaining frames: {stats['total_remaining_frames']}")
    else:
        # Process single file
        npz_path = path
        if not npz_path.suffix == '.npz':
            raise ValueError(f"Expected NPZ file, got: {npz_path}")
        
        # Auto-detect CSV file if not provided
        if csv_path is None:
            csv_path = find_csv_file(npz_path)
            if csv_path:
                print(f"Auto-detected CSV file: {csv_path}")
            else:
                print(f"Warning: No CSV file found, duration updates will be skipped")
        
        output_npz = Path(args.output_npz) if args.output_npz else None
        
        # Process file
        stats = cut_inactive_sections(
            npz_path=npz_path,
            csv_path=csv_path,
            output_npz_path=output_npz,
            min_visible_keypoints=args.min_visible_keypoints,
            min_visible_ratio=args.min_visible_ratio
        )
        
        print("\n" + "="*50)
        print("Processing Summary:")
        print("="*50)
        print(f"Original frames: {stats['original_frames']}")
        print(f"Removed frames: {stats['removed_frames']} ({stats['removal_percentage']:.1f}%)")
        print(f"Remaining frames: {stats['remaining_frames']}")
        print(f"Original duration: {stats['original_duration_sec']:.2f} sec")
        print(f"New duration: {stats['new_duration_sec']:.2f} sec")
        print(f"Duration reduction: {stats['original_duration_sec'] - stats['new_duration_sec']:.2f} sec")


if __name__ == '__main__':
    main()

