#!/usr/bin/env python3
"""
Generate continuous signing sequences from isolated sign videos.

This script concatenates isolated sign videos from the validation set to create
continuous signing sequences for model evaluation. It simulates real-world
continuous signing scenarios.

Features:
- Two concatenation strategies:
  * Strategy 1: Different glosses, same category
  * Strategy 2: Different glosses, different categories
- Signer-specific sequences (no mixing of signers)
- Configurable sequence length (min/max glosses)
- Proper timestamp offset handling
- JSON metadata generation for each sequence

Usage:
    # Strategy 1: Same category sequences
    python preprocessing/continuous/create_continuous_signs.py \
        --val-csv data/processed/fsl_val.csv \
        --val-dir data/processed/fsl_val \
        --output-dir data/processed/continuous_sequences \
        --strategy 1 \
        --sequences-per-signer 10 \
        --min-glosses 3 \
        --max-glosses 6

    # Strategy 2: Different category sequences
    python preprocessing/continuous/create_continuous_signs.py \
        --val-csv data/processed/fsl_val.csv \
        --val-dir data/processed/fsl_val \
        --output-dir data/processed/continuous_sequences \
        --strategy 2 \
        --sequences-per-signer 10 \
        --min-glosses 4 \
        --max-glosses 5

    # Dry run to preview
    python preprocessing/continuous/create_continuous_signs.py \
        --val-csv data/processed/fsl_val.csv \
        --val-dir data/processed/fsl_val \
        --strategy 1 \
        --dry-run
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import centralized label mapping utility
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.labels.label_mapping import load_label_mappings

# Global label mappings
GLOSS_LABELS = {}
CATEGORY_NAMES = {}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SequencePlan:
    """Plan for a continuous sequence."""
    sequence_id: int
    signer: str
    strategy: int
    samples: List[Dict]  # Each dict: {file, gloss, cat, occluded, signer, duration}
    total_duration: float
    num_glosses: int
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'sequence_id': self.sequence_id,
            'signer': self.signer,
            'strategy': self.strategy,
            'num_glosses': self.num_glosses,
            'total_duration': self.total_duration,
            'glosses': [s['gloss'] for s in self.samples],
            'categories': [s['cat'] for s in self.samples],
        }


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_validation_data(csv_path: Path, npz_dir: Path) -> pd.DataFrame:
    """Load validation CSV and validate NPZ files exist.
    
    Args:
        csv_path: Path to validation CSV file
        npz_dir: Directory containing NPZ files
        
    Returns:
        DataFrame with validation data
        
    Raises:
        ValueError: If CSV format is invalid
        FileNotFoundError: If CSV or NPZ directory not found
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")
    
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = {'file', 'gloss', 'cat'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")
    
    # Add optional columns with defaults if missing
    if 'occluded' not in df.columns:
        df['occluded'] = 0
        print("[INFO] Added default occluded=0 (column not found)")
    
    if 'signer' not in df.columns:
        df['signer'] = 'S0'
        print("[INFO] Added default signer=S0 (column not found)")
    
    if 'duration' not in df.columns:
        print("[INFO] Duration column not found, will calculate from NPZ files")
        df['duration'] = 0.0
    
    # Validate NPZ files exist
    missing_files = []
    for idx, row in df.iterrows():
        npz_path = npz_dir / row['file']
        if not npz_path.exists():
            missing_files.append(row['file'])
    
    if missing_files:
        print(f"[ERROR] {len(missing_files)} NPZ files not found:")
        for f in missing_files[:10]:
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        raise FileNotFoundError(f"{len(missing_files)} NPZ files missing")
    
    return df


def group_by_signer(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group validation data by signer.
    
    Args:
        df: DataFrame with validation data
        
    Returns:
        Dictionary mapping signer ID to DataFrame of samples
    """
    grouped = {}
    for signer in sorted(df['signer'].unique()):
        grouped[signer] = df[df['signer'] == signer].copy()
    
    return grouped


def validate_data_completeness(df: pd.DataFrame, strategy: int, min_glosses: int) -> bool:
    """Check if data is sufficient for sequence generation.
    
    Args:
        df: DataFrame with validation data
        strategy: Strategy number (2 or 3)
        min_glosses: Minimum glosses per sequence
        
    Returns:
        True if data is sufficient
    """
    grouped = group_by_signer(df)
    
    issues = []
    
    for signer, signer_df in grouped.items():
        if strategy == 1:
            # Check if signer has videos in at least one category with enough glosses
            category_counts = signer_df.groupby('cat').size()
            max_glosses_in_category = category_counts.max() if len(category_counts) > 0 else 0
            
            if max_glosses_in_category < min_glosses:
                issues.append(f"Signer {signer} has max {max_glosses_in_category} glosses in any category (need {min_glosses})")
        
        elif strategy == 2:
            # Check if signer has videos in multiple categories
            num_categories = signer_df['cat'].nunique()
            
            if num_categories < min_glosses:
                issues.append(f"Signer {signer} has videos in {num_categories} categories (need {min_glosses} for strategy 2)")
    
    if issues:
        print("[ERROR] Data completeness issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


# ============================================================================
# SEQUENCE PLANNING
# ============================================================================

def plan_sequence_strategy1(signer_df: pd.DataFrame, num_glosses: int, used_files: set) -> Optional[List[Dict]]:
    """Plan a sequence with same category (Strategy 1).
    
    Args:
        signer_df: DataFrame with signer's videos
        num_glosses: Number of glosses for this sequence
        used_files: Set of already used file names
        
    Returns:
        List of sample dictionaries, or None if not possible
    """
    # Get categories with enough available glosses
    available_df = signer_df[~signer_df['file'].isin(used_files)]
    
    category_counts = available_df.groupby('cat').size()
    valid_categories = category_counts[category_counts >= num_glosses].index.tolist()
    
    if not valid_categories:
        return None
    
    # Pick a random category
    category = random.choice(valid_categories)
    
    # Get available videos in this category
    category_df = available_df[available_df['cat'] == category]
    
    # Sample num_glosses unique videos (different glosses preferred, but allow same gloss if needed)
    if len(category_df) < num_glosses:
        return None
    
    # Sample randomly
    sampled = category_df.sample(n=num_glosses, replace=False)
    
    # Convert to list of dicts
    samples = []
    for _, row in sampled.iterrows():
        samples.append({
            'file': row['file'],
            'gloss': int(row['gloss']),
            'cat': int(row['cat']),
            'occluded': int(row['occluded']),
            'signer': row['signer'],
            'duration': float(row['duration']),
        })
    
    # Shuffle order
    random.shuffle(samples)
    
    return samples


def plan_sequence_strategy2(signer_df: pd.DataFrame, num_glosses: int, used_files: set) -> Optional[List[Dict]]:
    """Plan a sequence with different categories (Strategy 2).
    
    Args:
        signer_df: DataFrame with signer's videos
        num_glosses: Number of glosses for this sequence
        used_files: Set of already used file names
        
    Returns:
        List of sample dictionaries, or None if not possible
    """
    # Get available videos
    available_df = signer_df[~signer_df['file'].isin(used_files)]
    
    # Get categories with available videos
    available_categories = available_df['cat'].unique().tolist()
    
    if len(available_categories) < num_glosses:
        return None
    
    # Sample num_glosses different categories
    selected_categories = random.sample(available_categories, num_glosses)
    
    # Pick one random video from each category
    samples = []
    for cat in selected_categories:
        cat_df = available_df[available_df['cat'] == cat]
        
        if len(cat_df) == 0:
            return None
        
        # Pick one random video from this category
        row = cat_df.sample(n=1).iloc[0]
        samples.append({
            'file': row['file'],
            'gloss': int(row['gloss']),
            'cat': int(row['cat']),
            'occluded': int(row['occluded']),
            'signer': row['signer'],
            'duration': float(row['duration']),
        })
    
    # Shuffle order
    random.shuffle(samples)
    
    return samples


def generate_sequence_plans(
    grouped_data: Dict[str, pd.DataFrame],
    strategy: int,
    sequences_per_signer: int,
    min_glosses: int,
    max_glosses: int,
    seed: int = 42
) -> List[SequencePlan]:
    """Generate sequence plans for all signers.
    
    Args:
        grouped_data: Dictionary mapping signer to DataFrame
        strategy: Strategy number (2 or 3)
        sequences_per_signer: Number of sequences per signer
        min_glosses: Minimum glosses per sequence
        max_glosses: Maximum glosses per sequence
        seed: Random seed
        
    Returns:
        List of SequencePlan objects
    """
    random.seed(seed)
    
    plans = []
    sequence_id = 1
    
    # Strategy function selection
    if strategy == 1:
        plan_func = plan_sequence_strategy1
    elif strategy == 2:
        plan_func = plan_sequence_strategy2
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 1 or 2.")
    
    print(f"\n🎯 Generating sequence plans...")
    print(f"   Strategy: {strategy} ({'Same category' if strategy == 1 else 'Different categories'})")
    print(f"   Sequences per signer: {sequences_per_signer}")
    print(f"   Glosses per sequence: {min_glosses}-{max_glosses}\n")
    
    for signer in sorted(grouped_data.keys()):
        signer_df = grouped_data[signer]
        used_files = set()
        signer_plans = []
        
        print(f"   {signer}: ", end='', flush=True)
        
        for i in range(sequences_per_signer):
            # Random number of glosses
            num_glosses = random.randint(min_glosses, max_glosses)
            
            # Try to plan sequence
            max_attempts = 100
            samples = None
            
            for attempt in range(max_attempts):
                samples = plan_func(signer_df, num_glosses, used_files)
                if samples is not None:
                    break
            
            if samples is None:
                print(f"\n[WARN] Could not generate sequence {i+1} for {signer} (not enough unused videos)")
                continue
            
            # Mark files as used
            for sample in samples:
                used_files.add(sample['file'])
            
            # Calculate total duration
            total_duration = sum(s['duration'] for s in samples)
            
            # Create plan
            plan = SequencePlan(
                sequence_id=sequence_id,
                signer=signer,
                strategy=strategy,
                samples=samples,
                total_duration=total_duration,
                num_glosses=num_glosses
            )
            
            plans.append(plan)
            signer_plans.append(plan)
            sequence_id += 1
            
            print('█', end='', flush=True)
        
        print(f" {len(signer_plans)}/{sequences_per_signer}")
    
    print(f"\n✅ Generated {len(plans)} sequence plans\n")
    
    return plans


# ============================================================================
# NPZ CONCATENATION
# ============================================================================

def load_npz_data(npz_path: Path) -> Dict:
    """Load data from NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dictionary with arrays: X, X2048, mask, timestamps_ms, meta
    """
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'X': data['X'],
        'X2048': data.get('X2048', None),
        'mask': data['mask'],
        'timestamps_ms': data['timestamps_ms'],
        'meta': data.get('meta', None),
    }


def concatenate_npz_files(
    sample_list: List[Dict],
    npz_dir: Path
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, List[Dict], float]:
    """Concatenate multiple NPZ files into one continuous sequence.
    
    Args:
        sample_list: List of sample dictionaries
        npz_dir: Directory containing NPZ files
        
    Returns:
        Tuple of (X_concat, X2048_concat, mask_concat, timestamps_concat, segment_info, total_duration_ms)
    """
    X_list = []
    X2048_list = []
    mask_list = []
    timestamps_list = []
    segment_info = []
    
    cumulative_time_ms = 0
    has_x2048 = False
    
    for idx, sample in enumerate(sample_list):
        # Load NPZ
        npz_path = npz_dir / sample['file']
        data = load_npz_data(npz_path)
        
        # Extract arrays
        X = data['X']  # [T, 156]
        X2048 = data['X2048']  # [T, 2048] or None
        mask = data['mask']  # [T, 78]
        timestamps = data['timestamps_ms']  # [T]
        
        # Check if X2048 exists
        if X2048 is not None:
            has_x2048 = True
        
        # Calculate duration for this segment
        segment_duration_ms = int(timestamps[-1] - timestamps[0])
        
        # Apply offset to timestamps
        timestamps_offset = timestamps - timestamps[0] + cumulative_time_ms
        
        # Get labels
        gloss_label = GLOSS_LABELS.get(sample['gloss'], f"GLOSS_{sample['gloss']}")
        category_label = CATEGORY_NAMES.get(sample['cat'], f"CAT_{sample['cat']}")
        
        # Store segment info
        segment_info.append({
            'index': idx,
            'timestamp_start_ms': int(cumulative_time_ms),
            'timestamp_end_ms': int(cumulative_time_ms + segment_duration_ms),
            'gloss': int(sample['gloss']),
            'gloss_label': gloss_label,
            'category': int(sample['cat']),
            'category_label': category_label,
            'occluded': int(sample['occluded']),
            'signer': sample['signer'],
            'original_file': sample['file']
        })
        
        # Append to lists
        X_list.append(X)
        if X2048 is not None:
            X2048_list.append(X2048)
        mask_list.append(mask)
        timestamps_list.append(timestamps_offset)
        
        # Update cumulative time
        cumulative_time_ms += segment_duration_ms
    
    # Concatenate all arrays
    X_concat = np.concatenate(X_list, axis=0).astype(np.float32)
    mask_concat = np.concatenate(mask_list, axis=0)
    timestamps_concat = np.concatenate(timestamps_list, axis=0).astype(np.int64)
    
    X2048_concat = None
    if has_x2048 and X2048_list:
        X2048_concat = np.concatenate(X2048_list, axis=0).astype(np.float32)
    
    return X_concat, X2048_concat, mask_concat, timestamps_concat, segment_info, cumulative_time_ms


def save_continuous_npz(
    output_path: Path,
    X: np.ndarray,
    X2048: Optional[np.ndarray],
    mask: np.ndarray,
    timestamps: np.ndarray,
    meta: Dict
):
    """Save concatenated arrays to NPZ file.
    
    Args:
        output_path: Output file path
        X: Keypoint coordinates [T, 156]
        X2048: CNN features [T, 2048] or None
        mask: Visibility mask [T, 78]
        timestamps: Timestamps [T]
        meta: Metadata dictionary
    """
    save_dict = {
        'X': X,
        'mask': mask,
        'timestamps_ms': timestamps,
        'meta': json.dumps(meta)
    }
    
    if X2048 is not None:
        save_dict['X2048'] = X2048
    
    np.savez_compressed(output_path, **save_dict)


# ============================================================================
# JSON GENERATION
# ============================================================================

def create_sequence_metadata(plan: SequencePlan, segment_info: List[Dict], total_duration_ms: float) -> Dict:
    """Create JSON metadata for a continuous sequence.
    
    Args:
        plan: SequencePlan object
        segment_info: List of segment dictionaries
        total_duration_ms: Total duration in milliseconds
        
    Returns:
        Metadata dictionary
    """
    strategy_names = {
        1: "same_category",
        2: "different_categories"
    }
    
    metadata = {
        "file_name": f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.npz",
        "signer": plan.signer,
        "strategy": plan.strategy,
        "strategy_name": strategy_names.get(plan.strategy, f"strategy_{plan.strategy}"),
        "total_duration_sec": round(total_duration_ms / 1000.0, 2),
        "num_segments": plan.num_glosses,
        "segments": segment_info
    }
    
    return metadata


def save_sequence_json(output_path: Path, metadata: Dict):
    """Save sequence metadata to JSON file.
    
    Args:
        output_path: Output JSON file path
        metadata: Metadata dictionary
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_sequence_plan(
    plan: SequencePlan,
    npz_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Optional[Dict]:
    """Process a single sequence plan.
    
    Args:
        plan: SequencePlan object
        npz_dir: Directory with NPZ files
        output_dir: Output directory
        dry_run: If True, don't save files
        
    Returns:
        Metadata dictionary if successful, None otherwise
    """
    try:
        # Concatenate NPZ files
        X, X2048, mask, timestamps, segment_info, total_duration_ms = concatenate_npz_files(
            plan.samples, npz_dir
        )
        
        # Create metadata
        metadata = create_sequence_metadata(plan, segment_info, total_duration_ms)
        
        if not dry_run:
            # Save NPZ
            npz_filename = f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.npz"
            npz_path = output_dir / npz_filename
            
            meta_dict = {
                'strategy': plan.strategy,
                'signer': plan.signer,
                'num_segments': plan.num_glosses,
                'total_duration_sec': round(total_duration_ms / 1000.0, 2)
            }
            
            save_continuous_npz(npz_path, X, X2048, mask, timestamps, meta_dict)
            
            # Save JSON
            json_filename = f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.json"
            json_path = output_dir / json_filename
            save_sequence_json(json_path, metadata)
        
        return metadata
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process sequence {plan.sequence_id}: {e}")
        return None


def generate_summary_statistics(plans: List[SequencePlan], metadatas: List[Dict]) -> Dict:
    """Generate summary statistics for all sequences.
    
    Args:
        plans: List of SequencePlan objects
        metadatas: List of metadata dictionaries
        
    Returns:
        Summary dictionary
    """
    total_sequences = len(metadatas)
    total_duration = sum(m['total_duration_sec'] for m in metadatas)
    avg_duration = total_duration / total_sequences if total_sequences > 0 else 0
    
    glosses_per_sequence = [m['num_segments'] for m in metadatas]
    avg_glosses = sum(glosses_per_sequence) / len(glosses_per_sequence) if glosses_per_sequence else 0
    
    signers = defaultdict(int)
    for m in metadatas:
        signers[m['signer']] += 1
    
    summary = {
        'total_sequences': total_sequences,
        'total_duration_sec': round(total_duration, 2),
        'total_duration_min': round(total_duration / 60, 2),
        'avg_duration_sec': round(avg_duration, 2),
        'avg_glosses_per_sequence': round(avg_glosses, 2),
        'min_glosses': min(glosses_per_sequence) if glosses_per_sequence else 0,
        'max_glosses': max(glosses_per_sequence) if glosses_per_sequence else 0,
        'sequences_per_signer': dict(signers),
        'strategy': plans[0].strategy if plans else None,
    }
    
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate continuous signing sequences from isolated sign videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Strategy 1: Same category
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-csv data/processed/fsl_val.csv \\
      --val-dir data/processed/fsl_val \\
      --output-dir data/processed/continuous_sequences \\
      --strategy 1 --sequences-per-signer 10 --min-glosses 3 --max-glosses 6

  # Strategy 2: Different categories
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-csv data/processed/fsl_val.csv \\
      --val-dir data/processed/fsl_val \\
      --strategy 2 --sequences-per-signer 5 --min-glosses 4 --max-glosses 5

  # Dry run
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-csv data/processed/fsl_val.csv \\
      --val-dir data/processed/fsl_val \\
      --strategy 1 --dry-run
        """
    )
    
    parser.add_argument('--val-csv', type=Path, required=True,
                       help='Path to validation CSV file')
    parser.add_argument('--val-dir', type=Path, required=True,
                       help='Directory containing validation NPZ files')
    parser.add_argument('--output-dir', type=Path, default=Path('continuous_sequences'),
                       help='Output directory (default: continuous_sequences/)')
    parser.add_argument('--strategy', type=int, choices=[1, 2], default=1,
                       help='Concatenation strategy: 1=same category, 2=different categories (default: 1)')
    parser.add_argument('--sequences-per-signer', type=int, default=10,
                       help='Number of sequences per signer (default: 10)')
    parser.add_argument('--min-glosses', type=int, default=3,
                       help='Minimum glosses per sequence (default: 3)')
    parser.add_argument('--max-glosses', type=int, default=6,
                       help='Maximum glosses per sequence (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be generated without creating files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_glosses > args.max_glosses:
        print(f"[ERROR] min_glosses ({args.min_glosses}) cannot be greater than max_glosses ({args.max_glosses})")
        return 1
    
    if args.sequences_per_signer < 1:
        print(f"[ERROR] sequences_per_signer must be at least 1")
        return 1
    
    print("=" * 80)
    print("CONTINUOUS SEQUENCE GENERATION")
    print("=" * 80)
    
    if args.dry_run:
        print("\n🏃 DRY RUN MODE - No files will be created\n")
    
    # Load label mappings from centralized utility
    global GLOSS_LABELS, CATEGORY_NAMES
    try:
        GLOSS_LABELS, CATEGORY_NAMES = load_label_mappings()
        print(f"✓ Loaded {len(GLOSS_LABELS)} gloss labels and {len(CATEGORY_NAMES)} categories")
    except Exception as e:
        print(f"[WARN] Failed to load label mappings: {e}")
        print("[WARN] Labels will not be included in JSON output")
        GLOSS_LABELS = {}
        CATEGORY_NAMES = {}
    
    # Load validation data
    print(f"\n📊 Loading validation data...")
    print(f"   CSV: {args.val_csv}")
    print(f"   NPZ dir: {args.val_dir}")
    
    try:
        df = load_validation_data(args.val_csv, args.val_dir)
        print(f"   ✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"\n[ERROR] Failed to load validation data: {e}")
        return 1
    
    # Group by signer
    print(f"\n📋 Grouping by signer...")
    grouped = group_by_signer(df)
    
    for signer in sorted(grouped.keys()):
        signer_df = grouped[signer]
        print(f"   {signer}: {len(signer_df)} videos")
    
    # Validate data completeness
    print(f"\n🔍 Validating data completeness...")
    if not validate_data_completeness(df, args.strategy, args.min_glosses):
        return 1
    print("   ✓ Data is sufficient for sequence generation")
    
    # Generate sequence plans
    plans = generate_sequence_plans(
        grouped,
        args.strategy,
        args.sequences_per_signer,
        args.min_glosses,
        args.max_glosses,
        args.seed
    )
    
    if not plans:
        print("[ERROR] No sequence plans generated")
        return 1
    
    # Create output directory
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Output directory: {args.output_dir.resolve()}")
    
    # Process sequences
    print(f"\n🔄 Processing sequences...")
    
    metadatas = []
    
    for plan in tqdm(plans, desc="Generating sequences"):
        metadata = process_sequence_plan(plan, args.val_dir, args.output_dir, args.dry_run)
        if metadata:
            metadatas.append(metadata)
    
    # Generate summary
    summary = generate_summary_statistics(plans, metadatas)
    
    if not args.dry_run:
        summary_path = args.output_dir / "generation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"✅ Generated: {summary['total_sequences']} continuous sequences")
    print(f"📊 Total duration: {summary['total_duration_sec']:.1f}s ({summary['total_duration_min']:.1f} min)")
    print(f"⏱️  Average duration: {summary['avg_duration_sec']:.1f}s per sequence")
    print(f"📝 Average glosses: {summary['avg_glosses_per_sequence']:.1f} per sequence")
    print(f"📏 Gloss range: {summary['min_glosses']}-{summary['max_glosses']} per sequence")
    print(f"\n👥 Sequences per signer:")
    for signer in sorted(summary['sequences_per_signer'].keys()):
        count = summary['sequences_per_signer'][signer]
        print(f"   {signer}: {count}")
    
    if not args.dry_run:
        print(f"\n📁 Output saved to: {args.output_dir.resolve()}")
        print(f"   - {len(metadatas)} NPZ files")
        print(f"   - {len(metadatas)} JSON files")
        print(f"   - 1 summary JSON file")
    else:
        print(f"\n💡 Run without --dry-run to generate files")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

