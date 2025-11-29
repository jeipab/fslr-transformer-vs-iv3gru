"""
Script to remove inactive sections from continuous sign language NPZ files.

This script detects and removes periods where keypoints are inactive (e.g., both hands
not visible), then updates timestamps and segment information in the corresponding JSON file.

Usage Examples:
    # Basic usage (auto-detects JSON file, overwrites input files)
    python preprocessing/utils/cut_inactive_continuous.py data/processed/diff_cat_npz-seq-400/continuous_0001_S4_strategy2.npz
    
    # Process entire folder (overwrites all NPZ files in folder)
    python preprocessing/utils/cut_inactive_continuous.py data/processed/diff_cat_npz-seq-400/
    
    # Save to new files instead of overwriting (single file only)
    python preprocessing/utils/cut_inactive_continuous.py data/processed/diff_cat_npz-seq-400/continuous_0001_S4_strategy2.npz \\
      --output-npz output.npz \\
      --output-json output.json
    
    # Adjust visibility threshold (higher = more lenient, lower = more strict)
    python preprocessing/utils/cut_inactive_continuous.py input.npz --min-visible-ratio 0.2
    
    # Process folder with custom threshold
    python preprocessing/utils/cut_inactive_continuous.py data/processed/diff_cat_npz-seq-400/ --min-visible-ratio 0.2
    
    # Use minimum keypoint count instead of ratio
    python preprocessing/utils/cut_inactive_continuous.py input.npz --min-visible-keypoints 10
    
    # Keep inactive frames between segments
    python preprocessing/utils/cut_inactive_continuous.py input.npz --keep-between-segments
    
    # Specify JSON file explicitly (single file only)
    python preprocessing/utils/cut_inactive_continuous.py input.npz --json-file metadata.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


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


def update_segment_timestamps(
    segments: List[Dict],
    original_timestamps_ms: np.ndarray,
    active_frame_indices: np.ndarray,
    new_timestamps_ms: np.ndarray
) -> List[Dict]:
    """
    Update segment timestamps after removing inactive frames.
    
    Args:
        segments: List of segment dictionaries with timestamp_start_ms and timestamp_end_ms
        original_timestamps_ms: Original timestamps [T] before filtering
        active_frame_indices: Indices of frames that were kept [N]
        new_timestamps_ms: New timestamps [N] after filtering
        
    Returns:
        Updated list of segments with adjusted timestamps
    """
    # Create mapping from original timestamp to new frame index
    timestamp_to_orig_idx = {ts: i for i, ts in enumerate(original_timestamps_ms)}
    
    updated_segments = []
    for seg in segments:
        start_ms = seg['timestamp_start_ms']
        end_ms = seg['timestamp_end_ms']
        
        # Find the closest active frames for segment boundaries
        start_idx = None
        end_idx = None
        
        # Find start: search for first active frame >= start_ms
        for i, orig_idx in enumerate(active_frame_indices):
            if original_timestamps_ms[orig_idx] >= start_ms:
                start_idx = i
                break
        
        # Find end: search for last active frame <= end_ms
        for i in range(len(active_frame_indices) - 1, -1, -1):
            orig_idx = active_frame_indices[i]
            if original_timestamps_ms[orig_idx] <= end_ms:
                end_idx = i
                break
        
        # If segment has valid active frames, update timestamps
        if start_idx is not None and end_idx is not None and start_idx <= end_idx:
            new_start_ms = int(new_timestamps_ms[start_idx])
            new_end_ms = int(new_timestamps_ms[end_idx])
            
            updated_seg = seg.copy()
            updated_seg['timestamp_start_ms'] = new_start_ms
            updated_seg['timestamp_end_ms'] = new_end_ms
            updated_segments.append(updated_seg)
        # If segment has no active frames, skip it (but warn)
        elif start_idx is None or end_idx is None:
            print(f"Warning: Segment {seg.get('gloss_label', seg.get('index', 'unknown'))} "
                  f"has no active frames, removing from output.")
    
    return updated_segments


def cut_inactive_sections(
    npz_path: Path,
    json_path: Optional[Path] = None,
    output_npz_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    min_visible_keypoints: Optional[int] = None,
    min_visible_ratio: float = 0.1,
    remove_between_segments: bool = True
) -> Dict:
    """
    Remove inactive sections from NPZ file and update JSON metadata.
    
    Args:
        npz_path: Path to input NPZ file
        json_path: Optional path to input JSON file (auto-detected if None)
        output_npz_path: Optional output NPZ path (overwrites input if None)
        output_json_path: Optional output JSON path (overwrites input if None)
        min_visible_keypoints: Minimum visible keypoints per frame (overrides ratio if set)
        min_visible_ratio: Minimum ratio of visible keypoints (0.0-1.0)
        remove_between_segments: If True, also remove inactive periods between segments
        
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
    
    # If JSON file exists, also consider gaps between segments
    segments_to_keep = None
    if json_path and json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        segments = json_data.get('segments', [])
        
        if remove_between_segments and segments:
            # Mark frames between segments as inactive
            segment_times = []
            for seg in segments:
                segment_times.append((seg['timestamp_start_ms'], seg['timestamp_end_ms']))
            
            # Sort segments by start time
            segment_times.sort(key=lambda x: x[0])
            
            # Mark frames outside all segments as inactive
            for t in range(T_original):
                ts = timestamps_ms[t]
                in_segment = any(start <= ts <= end for start, end in segment_times)
                if not in_segment:
                    inactive_mask[t] = True
    
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
    
    # Filter X2048 if present
    X2048_filtered = None
    if X2048 is not None:
        X2048_filtered = X2048[active_indices].astype(np.float32)
    
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
    
    # Update JSON file if provided
    if json_path and json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Update segment timestamps
        segments = json_data.get('segments', [])
        if segments:
            updated_segments = update_segment_timestamps(
                segments,
                timestamps_ms,
                active_indices,
                timestamps_new
            )
            
            json_data['segments'] = updated_segments
            json_data['num_segments'] = len(updated_segments)
            
            # Update total duration
            if timestamps_new.size > 0:
                json_data['total_duration_sec'] = timestamps_new[-1] / 1000.0
            else:
                json_data['total_duration_sec'] = 0.0
        
        # Save updated JSON
        output_json = output_json_path or json_path
        print(f"Saving updated JSON file: {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Return statistics
    return {
        'original_frames': T_original,
        'removed_frames': n_removed,
        'remaining_frames': len(active_indices),
        'removal_percentage': n_removed / T_original * 100,
        'original_duration_sec': timestamps_ms[-1] / 1000.0 if len(timestamps_ms) > 0 else 0.0,
        'new_duration_sec': timestamps_new[-1] / 1000.0 if len(timestamps_new) > 0 else 0.0
    }


def process_folder(
    folder_path: Path,
    min_visible_keypoints: Optional[int] = None,
    min_visible_ratio: float = 0.1,
    remove_between_segments: bool = True
) -> Dict:
    """
    Process all NPZ files in a folder, overwriting originals.
    
    Args:
        folder_path: Path to folder containing NPZ files
        min_visible_keypoints: Minimum visible keypoints per frame
        min_visible_ratio: Minimum ratio of visible keypoints
        remove_between_segments: Whether to remove frames between segments
        
    Returns:
        Dictionary with processing statistics
    """
    npz_files = sorted(folder_path.glob('*.npz'))
    
    if not npz_files:
        print(f"No NPZ files found in {folder_path}")
        return {'processed': 0, 'failed': 0, 'total_removed_frames': 0}
    
    print(f"Found {len(npz_files)} NPZ file(s) in {folder_path}")
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
        
        # Auto-detect JSON file
        json_file = npz_file.with_suffix('.json')
        if not json_file.exists():
            json_file = None
            print(f"  No JSON file found for {npz_file.name}, skipping JSON update")
        
        try:
            file_stats = cut_inactive_sections(
                npz_path=npz_file,
                json_path=json_file,
                output_npz_path=npz_file,  # Overwrite original
                output_json_path=json_file if json_file else None,  # Overwrite original
                min_visible_keypoints=min_visible_keypoints,
                min_visible_ratio=min_visible_ratio,
                remove_between_segments=remove_between_segments
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
        description='Remove inactive sections from continuous sign language NPZ files'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to input NPZ file or folder containing NPZ files'
    )
    parser.add_argument(
        '--json-file',
        type=str,
        default=None,
        help='Path to input JSON file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output-npz',
        type=str,
        default=None,
        help='Output NPZ path (overwrites input if not provided)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Output JSON path (overwrites input if not provided)'
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
    parser.add_argument(
        '--keep-between-segments',
        action='store_true',
        help='Keep inactive frames between segments (default: remove them)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Check if path is a directory or file
    if path.is_dir():
        # Process folder
        print(f"Processing folder: {path}")
        stats = process_folder(
            folder_path=path,
            min_visible_keypoints=args.min_visible_keypoints,
            min_visible_ratio=args.min_visible_ratio,
            remove_between_segments=not args.keep_between_segments
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
        
        # Auto-detect JSON file if not provided
        json_path = None
        if args.json_file:
            json_path = Path(args.json_file)
        else:
            # Try to find JSON file with same stem
            json_path = npz_path.with_suffix('.json')
            if not json_path.exists():
                print(f"Warning: JSON file not found at {json_path}, skipping JSON update")
                json_path = None
        
        output_npz = Path(args.output_npz) if args.output_npz else None
        output_json = Path(args.output_json) if args.output_json else None
        
        # Process file
        stats = cut_inactive_sections(
            npz_path=npz_path,
            json_path=json_path,
            output_npz_path=output_npz,
            output_json_path=output_json,
            min_visible_keypoints=args.min_visible_keypoints,
            min_visible_ratio=args.min_visible_ratio,
            remove_between_segments=not args.keep_between_segments
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

