#!/usr/bin/env python3
"""
Generate Ground Truth JSON Files for Continuous Sign Language Sequences

This script transforms metadata JSON files from continuous sequence generation
into ground truth JSON files formatted for CTC model validation.

Input: Metadata JSON files from preprocessing/continuous/create_continuous_signs.py
Output: Ground truth JSON files for CTC evaluation

Usage:
    python evaluation/validation/generate_ground_truth.py \
        --input-metadata data/processed/continuous_sequences \
        --output-dir data/processed/continuous_sequences

    python evaluation/validation/generate_ground_truth.py \
        --input-metadata data/processed/continuous_sequences \
        --output-dir data/processed/continuous_sequences/ground_truth
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def transform_metadata_to_ground_truth(metadata: Dict) -> Dict:
    """
    Transform continuous sequence metadata to ground truth format.
    
    Args:
        metadata: Metadata dictionary from create_continuous_signs.py
        
    Returns:
        Ground truth dictionary formatted for CTC validation
    """
    segments = metadata['segments']
    
    # Extract ground truth sequence and labels
    ground_truth_sequence = [seg['gloss'] for seg in segments]
    ground_truth_labels = [seg['gloss_label'] for seg in segments]

    # Extract per-segment categories (ids and labels)
    ground_truth_categories = [seg.get('category') for seg in segments]
    ground_truth_category_labels = [seg.get('category_label') for seg in segments]
    
    # Transform timestamps
    ground_truth_timestamps = []
    for seg in segments:
        ground_truth_timestamps.append({
            'index': seg['index'],
            'gloss': seg['gloss'],
            'gloss_label': seg['gloss_label'],
            'start_ms': seg['timestamp_start_ms'],
            'end_ms': seg['timestamp_end_ms'],
            'duration_ms': seg['timestamp_end_ms'] - seg['timestamp_start_ms']
        })
    
    # Sequence-level category (first segment). For strategy 2 categories can differ across segments.
    first_segment = segments[0]
    category = first_segment['category']
    category_label = first_segment['category_label']
    
    # Build ground truth dictionary
    ground_truth = {
        'file_name': metadata['file_name'],
        'ground_truth_sequence': ground_truth_sequence,
        'ground_truth_labels': ground_truth_labels,
        'ground_truth_categories': ground_truth_categories,
        'ground_truth_category_labels': ground_truth_category_labels,
        'ground_truth_timestamps': ground_truth_timestamps,
        'signer': metadata['signer'],
        'strategy': metadata['strategy_name'],
        'category': category,
        'category_label': category_label,
        'total_duration_sec': metadata['total_duration_sec'],
        'num_segments': metadata['num_segments']
    }
    
    return ground_truth


def validate_ground_truth(ground_truth: Dict, filename: str) -> None:
    """
    Validate ground truth consistency.
    
    Args:
        ground_truth: Ground truth dictionary
        filename: Source filename for error messages
        
    Raises:
        ValueError: If validation fails
    """
    # Check sequence length matches num_segments
    seq_len = len(ground_truth['ground_truth_sequence'])
    num_segments = ground_truth['num_segments']
    if seq_len != num_segments:
        raise ValueError(
            f"{filename}: Sequence length mismatch - "
            f"got {seq_len} glosses but num_segments={num_segments}"
        )
    
    # Check labels length matches sequence length
    labels_len = len(ground_truth['ground_truth_labels'])
    if labels_len != seq_len:
        raise ValueError(
            f"{filename}: Labels length mismatch - "
            f"got {labels_len} labels but {seq_len} glosses"
        )
    
    # Check timestamps length matches sequence length
    ts_len = len(ground_truth['ground_truth_timestamps'])
    if ts_len != seq_len:
        raise ValueError(
            f"{filename}: Timestamps length mismatch - "
            f"got {ts_len} timestamps but {seq_len} glosses"
        )
    
    # Check gloss IDs are valid (0-104)
    for idx, gloss_id in enumerate(ground_truth['ground_truth_sequence']):
        if not (0 <= gloss_id <= 104):
            raise ValueError(
                f"{filename}: Invalid gloss ID {gloss_id} at index {idx} "
                f"(must be 0-104)"
            )
    
    # Check timestamps are monotonically increasing and have no gaps
    timestamps = ground_truth['ground_truth_timestamps']
    for i in range(len(timestamps)):
        ts = timestamps[i]
        
        # Check duration is positive
        if ts['duration_ms'] <= 0:
            raise ValueError(
                f"{filename}: Invalid duration {ts['duration_ms']}ms "
                f"at index {i} (must be positive)"
            )
        
        # Check start < end
        if ts['start_ms'] >= ts['end_ms']:
            raise ValueError(
                f"{filename}: Invalid timestamp range at index {i} - "
                f"start_ms={ts['start_ms']} >= end_ms={ts['end_ms']}"
            )
        
        # Check no gaps between segments
        if i > 0:
            prev_end = timestamps[i-1]['end_ms']
            curr_start = ts['start_ms']
            if curr_start != prev_end:
                raise ValueError(
                    f"{filename}: Gap in timestamps between segments {i-1} and {i} - "
                    f"previous ends at {prev_end}ms but current starts at {curr_start}ms"
                )
    
    # Check first timestamp starts at 0
    if timestamps[0]['start_ms'] != 0:
        raise ValueError(
            f"{filename}: First timestamp must start at 0ms, "
            f"got {timestamps[0]['start_ms']}ms"
        )


def process_metadata_file(
    input_path: Path,
    output_dir: Path
) -> Tuple[Dict, str]:
    """
    Process a single metadata file.
    
    Args:
        input_path: Path to input metadata JSON file
        output_dir: Directory to save ground truth JSON
        
    Returns:
        Tuple of (ground_truth_dict, output_filename)
        
    Raises:
        ValueError: If validation fails
        json.JSONDecodeError: If input file is invalid JSON
    """
    # Load metadata
    with open(input_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Transform to ground truth format
    ground_truth = transform_metadata_to_ground_truth(metadata)
    
    # Validate consistency
    validate_ground_truth(ground_truth, input_path.name)
    
    # Generate output filename (add _gt suffix before .json)
    output_filename = input_path.stem + '_gt.json'
    output_path = output_dir / output_filename
    
    # Save ground truth JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    return ground_truth, output_filename


def generate_summary_statistics(ground_truths: List[Dict]) -> Dict:
    """
    Generate summary statistics from ground truth files.
    
    Args:
        ground_truths: List of ground truth dictionaries
        
    Returns:
        Summary statistics dictionary
    """
    if not ground_truths:
        return {
            'total_sequences': 0,
            'error': 'No ground truth files generated'
        }
    
    total_sequences = len(ground_truths)
    
    # Sequence lengths
    sequence_lengths = [gt['num_segments'] for gt in ground_truths]
    avg_length = sum(sequence_lengths) / total_sequences
    min_length = min(sequence_lengths)
    max_length = max(sequence_lengths)
    
    # Sequences per signer
    sequences_per_signer = defaultdict(int)
    for gt in ground_truths:
        sequences_per_signer[gt['signer']] += 1
    
    # Sequences per strategy
    sequences_per_strategy = defaultdict(int)
    for gt in ground_truths:
        sequences_per_strategy[gt['strategy']] += 1
    
    # Total duration
    total_duration_sec = sum(gt['total_duration_sec'] for gt in ground_truths)
    avg_duration_sec = total_duration_sec / total_sequences
    
    summary = {
        'total_sequences': total_sequences,
        'average_sequence_length': round(avg_length, 2),
        'min_sequence_length': min_length,
        'max_sequence_length': max_length,
        'sequences_per_signer': dict(sorted(sequences_per_signer.items())),
        'sequences_per_strategy': dict(sorted(sequences_per_strategy.items())),
        'total_duration_sec': round(total_duration_sec, 2),
        'total_duration_min': round(total_duration_sec / 60, 2),
        'average_duration_sec': round(avg_duration_sec, 2)
    }
    
    return summary


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Generate ground truth JSON files for continuous sign language sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ground truth in same directory as metadata
  python evaluation/validation/generate_ground_truth.py \\
      --input-metadata data/processed/continuous_sequences \\
      --output-dir data/processed/continuous_sequences

  # Generate ground truth in separate subdirectory
  python evaluation/validation/generate_ground_truth.py \\
      --input-metadata data/processed/continuous_sequences \\
      --output-dir data/processed/continuous_sequences/ground_truth
        """
    )
    
    parser.add_argument(
        '--input-metadata',
        type=Path,
        required=True,
        help='Directory containing metadata JSON files from create_continuous_signs.py'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for ground truth JSON files'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_metadata.exists():
        print(f"Error: Input directory not found: {args.input_metadata}")
        return 1
    
    if not args.input_metadata.is_dir():
        print(f"Error: Input path is not a directory: {args.input_metadata}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GROUND TRUTH GENERATION")
    print("=" * 80)
    print(f"Input directory:  {args.input_metadata.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print()
    
    # Find metadata JSON files (exclude _gt.json files)
    metadata_files = [
        f for f in args.input_metadata.glob('*.json')
        if not f.stem.endswith('_gt') and f.name != 'generation_summary.json'
    ]
    
    if not metadata_files:
        print(f"Error: No metadata JSON files found in {args.input_metadata}")
        print("Expected files like: continuous_0001_S0_strategy1.json")
        return 1
    
    print(f"Found {len(metadata_files)} metadata file(s)")
    print()
    
    # Process files
    ground_truths = []
    output_files = []
    
    print("Processing files...")
    for i, input_path in enumerate(sorted(metadata_files), 1):
        try:
            ground_truth, output_filename = process_metadata_file(
                input_path,
                args.output_dir
            )
            ground_truths.append(ground_truth)
            output_files.append(output_filename)
            print(f"  [{i}/{len(metadata_files)}] {input_path.name} → {output_filename}")
            
        except Exception as e:
            print(f"\nError processing {input_path.name}: {e}")
            return 1
    
    print()
    
    # Generate summary statistics
    summary = generate_summary_statistics(ground_truths)
    
    # Save summary
    summary_path = args.output_dir / 'ground_truth_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sequences:         {summary['total_sequences']}")
    print(f"Average sequence length: {summary['average_sequence_length']} glosses")
    print(f"Sequence length range:   {summary['min_sequence_length']}-{summary['max_sequence_length']} glosses")
    print(f"Total duration:          {summary['total_duration_sec']:.1f}s ({summary['total_duration_min']:.1f} min)")
    print(f"Average duration:        {summary['average_duration_sec']:.1f}s per sequence")
    print()
    print("Sequences per signer:")
    for signer, count in summary['sequences_per_signer'].items():
        print(f"  {signer}: {count}")
    print()
    print("Sequences per strategy:")
    for strategy, count in summary['sequences_per_strategy'].items():
        print(f"  {strategy}: {count}")
    print()
    print(f"Output saved to: {args.output_dir.resolve()}")
    print(f"  - {len(output_files)} ground truth JSON files")
    print(f"  - 1 summary file (ground_truth_summary.json)")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

