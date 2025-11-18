"""
Rerun occlusion detection on continuous sign sequences.

This script processes a folder containing NPZ files and their corresponding JSON
metadata files, reruns occlusion detection for each segment, and updates the
'occluded' field in the JSON files.

Usage:
    python -m preprocessing.core.rerun_occlusion_c <folder_path>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from .occlusion_detection import compute_occlusion_detection_from_keypoints


def load_npz_file(npz_path: Path) -> Dict:
    """Load NPZ file and extract keypoint data.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dictionary containing X, mask, and timestamps_ms
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        'X': data['X'],
        'mask': data['mask'],
        'timestamps_ms': data['timestamps_ms']
    }


def extract_segment_frames(
    X: np.ndarray,
    mask: np.ndarray,
    timestamps_ms: np.ndarray,
    start_ms: int,
    end_ms: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frames for a specific time segment.
    
    Args:
        X: Keypoint coordinates [T, 178]
        mask: Visibility mask [T, 89]
        timestamps_ms: Timestamps in milliseconds [T]
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        
    Returns:
        Tuple of (X_segment, mask_segment) for the time range
    """
    # Find frames within the timestamp range
    frame_indices = np.where((timestamps_ms >= start_ms) & (timestamps_ms <= end_ms))[0]
    
    if len(frame_indices) == 0:
        # If no exact matches, use closest frames
        start_idx = np.searchsorted(timestamps_ms, start_ms)
        end_idx = np.searchsorted(timestamps_ms, end_ms, side='right')
        frame_indices = np.arange(start_idx, min(end_idx, len(timestamps_ms)))
    
    if len(frame_indices) == 0:
        # If still no frames, return empty arrays with correct shape
        return np.empty((0, 178), dtype=X.dtype), np.empty((0, 89), dtype=mask.dtype)
    
    X_segment = X[frame_indices]
    mask_segment = mask[frame_indices]
    
    return X_segment, mask_segment


def process_segment(
    X_segment: np.ndarray,
    mask_segment: np.ndarray
) -> int:
    """Run occlusion detection on a segment.
    
    Args:
        X_segment: Keypoint coordinates for segment [T, 178]
        mask_segment: Visibility mask for segment [T, 89]
        
    Returns:
        Binary occlusion flag (0 or 1)
    """
    if X_segment.shape[0] == 0:
        # No frames available, default to not occluded
        return 0
    
    try:
        # Run occlusion detection with default parameters
        occluded = compute_occlusion_detection_from_keypoints(
            X=X_segment,
            mask=mask_segment,
            output_format='compatible'
        )
        return int(occluded)
    except Exception as e:
        print(f"  Warning: Occlusion detection failed: {e}")
        return 0


def process_json_file(
    json_path: Path,
    folder_path: Path,
    stats: Dict
) -> bool:
    """Process a single JSON file and update occlusion status.
    
    Args:
        json_path: Path to JSON file
        folder_path: Folder containing NPZ files
        stats: Statistics dictionary to update
        
    Returns:
        True if processing was successful, False otherwise
    """
    # Load JSON metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Find corresponding NPZ file
    npz_filename = metadata.get('file_name', json_path.stem + '.npz')
    npz_path = folder_path / npz_filename
    
    if not npz_path.exists():
        print(f"  Error: NPZ file not found: {npz_path}")
        return False
    
    # Load NPZ file
    try:
        npz_data = load_npz_file(npz_path)
        X = npz_data['X']
        mask = npz_data['mask']
        timestamps_ms = npz_data['timestamps_ms']
    except Exception as e:
        print(f"  Error: Failed to load NPZ file: {e}")
        return False
    
    # Process each segment
    segments = metadata.get('segments', [])
    updated_segments = []
    
    for segment in segments:
        # Get old occlusion status
        old_occluded = int(segment.get('occluded', 0))
        
        # Extract frames for this segment
        start_ms = segment.get('timestamp_start_ms', 0)
        end_ms = segment.get('timestamp_end_ms', 0)
        
        X_segment, mask_segment = extract_segment_frames(
            X, mask, timestamps_ms, start_ms, end_ms
        )
        
        # Run occlusion detection
        new_occluded = process_segment(X_segment, mask_segment)
        
        # Update segment
        segment['occluded'] = new_occluded
        updated_segments.append(segment)
        
        # Update statistics
        stats['old_occluded'] += old_occluded
        stats['old_not_occluded'] += (1 - old_occluded)
        stats['new_occluded'] += new_occluded
        stats['new_not_occluded'] += (1 - new_occluded)
        
        # Track changes
        if old_occluded != new_occluded:
            stats['changed'] += 1
            stats['changes'].append({
                'file': json_path.name,
                'segment_index': segment.get('index', -1),
                'gloss': segment.get('gloss_label', 'unknown'),
                'old': old_occluded,
                'new': new_occluded
            })
    
    # Update metadata with new segments
    metadata['segments'] = updated_segments
    
    # Save updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
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
    
    print(f"\nChanges: {stats['changed']} segments changed occlusion status")
    
    if stats['changed'] > 0 and stats['changes']:
        print("\nChanged Segments (first 10):")
        for change in stats['changes'][:10]:
            print(f"  {change['file']} - Segment {change['segment_index']} ({change['gloss']}): "
                  f"{change['old']} -> {change['new']}")
        if len(stats['changes']) > 10:
            print(f"  ... and {len(stats['changes']) - 10} more")
    
    print("="*60 + "\n")


def main():
    """Main function to rerun occlusion detection on a folder of files."""
    parser = argparse.ArgumentParser(
        description="Rerun occlusion detection on continuous sign sequences"
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing NPZ and JSON files'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return
    
    # Find all JSON files
    json_files = sorted(folder_path.glob('*.json'))
    
    if len(json_files) == 0:
        print(f"Error: No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Processing folder: {folder_path}\n")
    
    # Initialize statistics
    stats = {
        'old_occluded': 0,
        'old_not_occluded': 0,
        'new_occluded': 0,
        'new_not_occluded': 0,
        'changed': 0,
        'changes': []
    }
    
    # Process each JSON file
    processed = 0
    failed = 0
    
    for json_path in json_files:
        print(f"Processing: {json_path.name}")
        if process_json_file(json_path, folder_path, stats):
            processed += 1
            print(f"  ✓ Updated")
        else:
            failed += 1
            print(f"  ✗ Failed")
    
    print(f"\nProcessed: {processed} file(s)")
    if failed > 0:
        print(f"Failed: {failed} file(s)")
    
    # Print statistics
    print_statistics(stats)


if __name__ == '__main__':
    main()

