#!/usr/bin/env python3
"""
Generate continuous signing sequences from isolated sign videos.

This script concatenates isolated sign videos from the validation set to create
continuous signing sequences for model evaluation. It simulates real-world
continuous signing scenarios.

Features:
- Two concatenation strategies (both support NPZ and video files):
  * Strategy 1: Same category (different glosses, same category)
  * Strategy 2: Different categories (different glosses, different categories)
- Signer-specific sequences (no mixing of signers)
- Configurable sequence length (min/max glosses)
- Proper timestamp offset handling
- JSON metadata generation for each sequence
- Support for both NPZ files and raw video files

Usage:
    # Strategy 1: Same category with NPZ files
    python preprocessing/continuous/create_continuous_signs.py \
        --val-csv data/processed/fsl_val.csv \
        --val-dir data/processed/fsl_val \
        --output-dir data/processed/continuous_sequences \
        --strategy 1 --sequences-per-signer 10 --min-glosses 3 --max-glosses 6

    # Strategy 1: Same category with video files
    python preprocessing/continuous/create_continuous_signs.py \
        --val-dir data/raw/continuous_sequences \
        --output-dir data/processed/continuous_sequences \
        --strategy 1 --sequences-per-signer 10 --min-glosses 3 --max-glosses 6

    # Strategy 2: Different categories with video files
    python preprocessing/continuous/create_continuous_signs.py \
        --val-dir data/raw/continuous_sequences \
        --output-dir data/processed/continuous_sequences \
        --strategy 2 --sequences-per-signer 10 --min-glosses 4 --max-glosses 5

    # Dry run to preview (any strategy)
    python preprocessing/continuous/create_continuous_signs.py \
        --val-dir data/raw/continuous_sequences \
        --strategy 1 --dry-run

    # Specify a single signer only (e.g., S1) with CSV/NPZ mode
    python preprocessing/continuous/create_continuous_signs.py \
        --val-csv data/processed/fsl_val.csv \
        --val-dir data/processed/fsl_val \
        --output-dir data/processed/continuous_sequences \
        --strategy 1 --signer S1 --sequences-per-signer 5 --min-glosses 3 --max-glosses 6

    # Use all files from CMB105_test (exhaustive mode, 6-7 signs per sequence, different categories)
    python preprocessing/continuous/create_continuous_signs.py \
        --val-csv data/processed/CMB105_test.csv \
        --val-dir data/processed/CMB105_test \
        --output-dir data/processed/continuous_testing \
        --strategy 2 --min-glosses 6 --max-glosses 7 --use-all-files
"""

import argparse
import json
import os
import random
import sys
import re
import tempfile
import cv2
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

# Video file extensions
VIDEO_EXTENSIONS = {'.mov', '.mp4', '.avi', '.mkv'}

# Temporary directory for preprocessing videos
TEMP_PREPROCESS_DIR = None


# DATA STRUCTURES

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


# VIDEO PROCESSING

def parse_video_filename(filename: str) -> Optional[Dict[str, any]]:
    """Parse video filename to extract gloss, signer, and other metadata.
    
    Expected format: clip_<number>_<gloss_name>_S<number>.<ext>
    Example: clip_0006_good_morning_S1.MOV
    
    Args:
        filename: Video filename
        
    Returns:
        Dictionary with gloss_name, signer, or None if parsing fails
    """
    basename = filename.rsplit('.', 1)[0]  # Remove extension
    
    # Pattern: clip_XXXX_<gloss>_S<N>
    # Example: clip_0006_good_morning_S1
    pattern = r'clip_\d+_(.+?)_S(\d+)$'
    match = re.match(pattern, basename, re.IGNORECASE)
    
    if not match:
        return None
    
    gloss_name = match.group(1).replace('_', ' ').upper().strip()
    signer = f"S{match.group(2)}"
    
    return {
        'gloss_name': gloss_name,
        'signer': signer
    }


def gloss_name_to_id(gloss_name: str) -> Optional[int]:
    """Convert gloss name to ID from labels_reference.
    
    Args:
        gloss_name: Gloss name (e.g., "GOOD MORNING")
        
    Returns:
        Gloss ID or None if not found
    """
    # Load mappings if not already loaded
    global GLOSS_LABELS, CATEGORY_NAMES
    if not GLOSS_LABELS:
        try:
            GLOSS_LABELS, CATEGORY_NAMES = load_label_mappings()
        except Exception as e:
            print(f"[WARN] Could not load label mappings: {e}")
            GLOSS_LABELS = {}
            CATEGORY_NAMES = {}
    
    # Search for matching gloss
    for gloss_id, label in GLOSS_LABELS.items():
        if gloss_name == label:
            return gloss_id
    
    return None


def load_video_dir(video_dir: Path) -> pd.DataFrame:
    """Load video files from a directory and create a DataFrame with metadata.
    
    This function scans a directory for video files, parses their filenames
    to extract gloss and signer information, and creates a DataFrame with
    the same structure as a CSV file.
    
    Args:
        video_dir: Directory containing video files
        
    Returns:
        DataFrame with columns: file, gloss, cat, occluded, signer, duration
    """
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(video_dir.glob(f'*{ext}')))

    # Remove duplicates (case-insensitive)
    video_files = list({f.resolve().as_posix().lower(): f for f in video_files}.values())
    
    if not video_files:
        raise ValueError(f"No video files found in {video_dir}")
    
    print(f"[INFO] Found {len(video_files)} video files")
    
    # Parse video files and create DataFrame
    records = []
    for video_file in video_files:
        filename = video_file.name
        parsed = parse_video_filename(filename)
        
        if parsed is None:
            print(f"[WARN] Could not parse filename: {filename}")
            continue
        
        # Get gloss ID from name
        gloss_id = gloss_name_to_id(parsed['gloss_name'])
        if gloss_id is None:
            print(f"[WARN] Gloss not found in label mapping: {parsed['gloss_name']}")
            continue
        
        # Get category from gloss ID (using label mapping)
        if GLOSS_LABELS and gloss_id in GLOSS_LABELS:
            # Find category from labels_reference.csv
            try:
                label_ref_path = Path(__file__).parent.parent.parent / "data" / "labels_reference.csv"
                if label_ref_path.exists():
                    df_ref = pd.read_csv(label_ref_path)
                    cat_id = int(df_ref[df_ref['gloss_id'] == gloss_id]['cat_id'].iloc[0])
                else:
                    cat_id = 0  # Default category
            except Exception:
                cat_id = 0  # Default category
        else:
            cat_id = 0
        
        records.append({
            'file': filename,
            'gloss': gloss_id,
            'cat': cat_id,
            'occluded': 0,  # Default: not occluded
            'signer': parsed['signer'],
            'duration': 0.0  # Will be calculated during processing
        })
    
    df = pd.DataFrame(records)
    print(f"[INFO] Created DataFrame with {len(df)} records")
    
    return df


def preprocess_video_to_npz(video_path: Path, temp_dir: Path) -> Path:
    """Preprocess a single video file to NPZ format.
    
    This is a simplified wrapper that calls the preprocessing module.
    It processes the video and returns the path to the generated NPZ file.
    
    Args:
        video_path: Path to video file
        temp_dir: Temporary directory for NPZ output
        
    Returns:
        Path to generated NPZ file
    """
    from preprocessing.core.preprocess import process_video
    
    # Ensure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video (this will extract keypoints and features)
    try:
        process_video(
            video_path=str(video_path),
            out_dir=str(temp_dir),
            target_fps=30,
            out_size=256,
            conf_thresh=0.35,
            max_gap=5,
            write_keypoints=True,
            write_iv3_features=True,
            compute_occlusion=False,  # Skip occlusion for speed
            labels_csv_path=None,
            signer_id=None,
            flip_horizontal=False
        )
        
        # Return path to NPZ file (same basename)
        npz_name = video_path.stem + '.npz'
        return temp_dir / npz_name
        
    except Exception as e:
        print(f"[ERROR] Failed to preprocess {video_path}: {e}")
        raise


# DATA LOADING & VALIDATION

def load_validation_data(csv_path: Optional[Path], npz_dir: Path) -> pd.DataFrame:
    """Load validation data from CSV or video directory.
    
    Args:
        csv_path: Path to validation CSV file (optional if using video mode)
        npz_dir: Directory containing NPZ files or video files
        
    Returns:
        DataFrame with validation data
        
    Raises:
        ValueError: If data format is invalid
        FileNotFoundError: If directory not found
    """
    # Check if we're in video mode (no CSV provided)
    video_mode = csv_path is None
    
    # Validate input
    if video_mode:
        # Check if directory contains video files
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(list(npz_dir.glob(f'*{ext}')))
            video_files.extend(list(npz_dir.glob(f'*{ext.upper()}')))
        
        if video_files:
            print("[INFO] Video mode detected: no CSV provided")
            df = load_video_dir(npz_dir)
        else:
            raise ValueError(f"Directory {npz_dir} contains no video files and no CSV was provided")
    else:
        # CSV mode
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
        print("[INFO] Duration column not found, will calculate during processing")
        df['duration'] = 0.0
    
    # In video mode, we'll process videos on-the-fly during sequence generation
    # In CSV mode, validate that NPZ files exist
    if not video_mode:
        # Validate NPZ files exist
        missing_files = []
        for idx, row in df.iterrows():
            # Add .npz extension if not present
            filename = row['file']
            if not filename.endswith('.npz'):
                filename = f"{filename}.npz"
            npz_path = npz_dir / filename
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


# SEQUENCE PLANNING

def plan_sequence_strategy1(signer_df: pd.DataFrame, num_glosses: int, used_files: set, allow_fewer: bool = False) -> Optional[List[Dict]]:
    """Plan a sequence with same category (Strategy 1).
    
    Args:
        signer_df: DataFrame with signer's videos
        num_glosses: Number of glosses for this sequence
        used_files: Set of already used file names
        allow_fewer: If True, allow sequences with fewer than num_glosses if needed
        
    Returns:
        List of sample dictionaries, or None if not possible
    """
    # Get categories with enough available glosses
    available_df = signer_df[~signer_df['file'].isin(used_files)]
    
    if len(available_df) == 0:
        return None
    
    if allow_fewer:
        # Use all remaining files if fewer than num_glosses
        actual_num_glosses = min(num_glosses, len(available_df))
        category_counts = available_df.groupby('cat').size()
        valid_categories = category_counts[category_counts >= actual_num_glosses].index.tolist()
        
        if not valid_categories:
            # If no category has enough, use the category with the most files
            if len(category_counts) > 0:
                category = category_counts.idxmax()
                category_df = available_df[available_df['cat'] == category]
                actual_num_glosses = min(num_glosses, len(category_df))
            else:
                return None
        else:
            # Pick a random category
            category = random.choice(valid_categories)
            category_df = available_df[available_df['cat'] == category]
            actual_num_glosses = min(num_glosses, len(category_df))
    else:
        category_counts = available_df.groupby('cat').size()
        valid_categories = category_counts[category_counts >= num_glosses].index.tolist()
        
        if not valid_categories:
            return None
        
        # Pick a random category
        category = random.choice(valid_categories)
        category_df = available_df[available_df['cat'] == category]
        
        if len(category_df) < num_glosses:
            return None
        
        actual_num_glosses = num_glosses
    
    # Sample randomly
    sampled = category_df.sample(n=actual_num_glosses, replace=False)
    
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


def plan_sequence_strategy2(signer_df: pd.DataFrame, num_glosses: int, used_files: set, allow_fewer: bool = False) -> Optional[List[Dict]]:
    """Plan a sequence with different categories (Strategy 2).
    
    Args:
        signer_df: DataFrame with signer's videos
        num_glosses: Number of glosses for this sequence
        used_files: Set of already used file names
        allow_fewer: If True, allow sequences with fewer than num_glosses if needed
        
    Returns:
        List of sample dictionaries, or None if not possible
    """
    # Get available videos
    available_df = signer_df[~signer_df['file'].isin(used_files)]
    
    if len(available_df) == 0:
        return None
    
    # Get categories with available videos
    available_categories = available_df['cat'].unique().tolist()
    
    if allow_fewer:
        actual_num_glosses = min(num_glosses, len(available_categories), len(available_df))
    else:
        if len(available_categories) < num_glosses:
            return None
        actual_num_glosses = num_glosses
    
    # Sample actual_num_glosses different categories
    selected_categories = random.sample(available_categories, actual_num_glosses)
    
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
    seed: int = 42,
    use_all_files: bool = False
) -> List[SequencePlan]:
    """Generate sequence plans for all signers.
    
    Args:
        grouped_data: Dictionary mapping signer to DataFrame
        strategy: Strategy number (1 or 2)
        sequences_per_signer: Number of sequences per signer (ignored if use_all_files=True)
        min_glosses: Minimum glosses per sequence
        max_glosses: Maximum glosses per sequence
        seed: Random seed
        use_all_files: If True, continue generating sequences until all files are used
        
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
    
    print(f"\nGenerating sequence plans...")
    print(f"   Strategy: {strategy} ({'Same category' if strategy == 1 else 'Different categories'})")
    if use_all_files:
        print(f"   Mode: Exhaustive (using all files)")
    else:
        print(f"   Sequences per signer: {sequences_per_signer}")
    print(f"   Glosses per sequence: {min_glosses}-{max_glosses}\n")
    
    for signer in sorted(grouped_data.keys()):
        signer_df = grouped_data[signer]
        used_files = set()
        signer_plans = []
        total_files = len(signer_df)
        
        print(f"   {signer}: ", end='', flush=True)
        
        if use_all_files:
            # Continue until all files are used
            max_iterations = total_files * 2  # Safety limit
            iteration = 0
            
            while len(used_files) < total_files and iteration < max_iterations:
                iteration += 1
                
                # Check remaining files
                remaining = total_files - len(used_files)
                if remaining == 0:
                    break
                
                # Determine number of glosses for this sequence
                if remaining < min_glosses:
                    # Use all remaining files
                    num_glosses = remaining
                    allow_fewer = True
                else:
                    # Random number of glosses within range
                    num_glosses = random.randint(min_glosses, min(max_glosses, remaining))
                    allow_fewer = False
                
                # Try to plan sequence
                max_attempts = 100
                samples = None
                
                for attempt in range(max_attempts):
                    samples = plan_func(signer_df, num_glosses, used_files, allow_fewer=allow_fewer)
                    if samples is not None:
                        break
                
                if samples is None:
                    # If we can't create a sequence, use remaining files but respect max_glosses
                    available_df = signer_df[~signer_df['file'].isin(used_files)]
                    if len(available_df) > 0:
                        # Limit to max_glosses to respect the constraint
                        # If remaining <= max_glosses, use all; otherwise, sample up to max_glosses
                        num_to_use = min(max_glosses, len(available_df))
                        sampled_df = available_df.sample(n=num_to_use, replace=False)
                        
                        samples = []
                        for _, row in sampled_df.iterrows():
                            samples.append({
                                'file': row['file'],
                                'gloss': int(row['gloss']),
                                'cat': int(row['cat']),
                                'occluded': int(row['occluded']),
                                'signer': row['signer'],
                                'duration': float(row['duration']),
                            })
                        random.shuffle(samples)
                    else:
                        break
                
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
                    num_glosses=len(samples)
                )
                
                plans.append(plan)
                signer_plans.append(plan)
                sequence_id += 1
                
                print('█', end='', flush=True)
            
            unused = total_files - len(used_files)
            if unused > 0:
                print(f"\n[WARN] {unused} files remain unused for {signer}")
        else:
            # Original behavior: fixed number of sequences
            for i in range(sequences_per_signer):
                # Random number of glosses
                num_glosses = random.randint(min_glosses, max_glosses)
                
                # Try to plan sequence
                max_attempts = 100
                samples = None
                
                for attempt in range(max_attempts):
                    samples = plan_func(signer_df, num_glosses, used_files, allow_fewer=False)
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
        
        if use_all_files:
            print(f" {len(signer_plans)} sequences, {len(used_files)}/{total_files} files used")
        else:
            print(f" {len(signer_plans)}/{sequences_per_signer}")
    
    print(f"\nGenerated {len(plans)} sequence plans\n")
    
    return plans


# CONCATENATION (Supports NPZ and Video)

def load_npz_data(npz_path: Path) -> Dict:
    """Load data from NPZ file."""
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
    """Concatenate multiple NPZ files into one continuous sequence."""
    X_list, X2048_list, mask_list, timestamps_list = [], [], [], []
    segment_info = []
    cumulative_time_ms = 0
    has_x2048 = False

    for idx, sample in enumerate(sample_list):
        filename = sample['file']
        if not filename.endswith('.npz'):
            filename += '.npz'
        npz_path = npz_dir / filename
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = load_npz_data(npz_path)
        X, X2048, mask, timestamps = data['X'], data['X2048'], data['mask'], data['timestamps_ms']

        if X2048 is not None:
            has_x2048 = True

        duration_ms = int(timestamps[-1] - timestamps[0])
        timestamps_offset = timestamps - timestamps[0] + cumulative_time_ms

        gloss_label = GLOSS_LABELS.get(sample['gloss'], f"GLOSS_{sample['gloss']}")
        cat_label = CATEGORY_NAMES.get(sample['cat'], f"CAT_{sample['cat']}")

        segment_info.append({
            'index': idx,
            'timestamp_start_ms': int(cumulative_time_ms),
            'timestamp_end_ms': int(cumulative_time_ms + duration_ms),
            'gloss': int(sample['gloss']),
            'gloss_label': gloss_label,
            'category': int(sample['cat']),
            'category_label': cat_label,
            'occluded': int(sample.get('occluded', 0)),
            'signer': sample['signer'],
            'original_file': sample['file']
        })

        X_list.append(X)
        mask_list.append(mask)
        timestamps_list.append(timestamps_offset)
        if X2048 is not None:
            X2048_list.append(X2048)

        cumulative_time_ms += duration_ms

    X_concat = np.concatenate(X_list, axis=0).astype(np.float32)
    mask_concat = np.concatenate(mask_list, axis=0)
    timestamps_concat = np.concatenate(timestamps_list, axis=0).astype(np.int64)
    X2048_concat = np.concatenate(X2048_list, axis=0).astype(np.float32) if has_x2048 else None

    return X_concat, X2048_concat, mask_concat, timestamps_concat, segment_info, cumulative_time_ms


def concatenate_video_files(
    sample_list: List[Dict],
    video_dir: Path,
    output_path: Path
) -> Tuple[List[Dict], float]:
    """Concatenate multiple video files into one continuous video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None
    segment_info = []
    cumulative_time_ms = 0.0

    for idx, sample in enumerate(sample_list):
        video_path = video_dir / sample['file']
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = (frame_count / fps) * 1000.0

        if out_writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_writer.write(frame)

        cap.release()

        gloss_label = GLOSS_LABELS.get(sample['gloss'], f"GLOSS_{sample['gloss']}")
        cat_label = CATEGORY_NAMES.get(sample['cat'], f"CAT_{sample['cat']}")

        segment_info.append({
            'index': idx,
            'timestamp_start_ms': int(cumulative_time_ms),
            'timestamp_end_ms': int(cumulative_time_ms + duration_ms),
            'gloss': int(sample['gloss']),
            'gloss_label': gloss_label,
            'category': int(sample['cat']),
            'category_label': cat_label,
            'occluded': int(sample.get('occluded', 0)),
            'signer': sample['signer'],
            'original_file': sample['file']
        })

        cumulative_time_ms += duration_ms

    if out_writer:
        out_writer.release()

    return segment_info, cumulative_time_ms


def save_continuous_npz(output_path: Path, X, X2048, mask, timestamps, meta):
    """Save concatenated arrays to NPZ."""
    save_dict = {'X': X, 'mask': mask, 'timestamps_ms': timestamps, 'meta': json.dumps(meta)}
    if X2048 is not None:
        save_dict['X2048'] = X2048
    np.savez_compressed(output_path, **save_dict)


def process_sequence_plan(
    plan: SequencePlan,
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Optional[Dict]:
    """Decide whether to concatenate NPZ or video files."""
    try:
        # Detect file type
        first_file = plan.samples[0]['file']
        ext = Path(first_file).suffix.lower()
        is_video = ext in VIDEO_EXTENSIONS

        if dry_run:
            print(f"[DRY RUN] Would process sequence {plan.sequence_id} ({'video' if is_video else 'npz'})")
            return None

        if is_video:
            # Output .mp4
            video_path = output_dir / f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.mp4"
            segment_info, total_duration_ms = concatenate_video_files(plan.samples, input_dir, video_path)
            metadata = create_sequence_metadata(plan, segment_info, total_duration_ms)
            metadata["file_name"] = video_path.name

        else:
            # Output .npz
            X, X2048, mask, timestamps, segment_info, total_duration_ms = concatenate_npz_files(plan.samples, input_dir)
            npz_path = output_dir / f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.npz"
            save_continuous_npz(npz_path, X, X2048, mask, timestamps, {
                'strategy': plan.strategy,
                'signer': plan.signer,
                'num_segments': plan.num_glosses,
                'total_duration_sec': round(total_duration_ms / 1000.0, 2)
            })
            metadata = create_sequence_metadata(plan, segment_info, total_duration_ms)
            metadata["file_name"] = npz_path.name

        # Save JSON metadata
        json_path = output_dir / f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.json"
        save_sequence_json(json_path, metadata)

        return metadata

    except Exception as e:
        print(f"\n[ERROR] Failed to process sequence {plan.sequence_id}: {e}")
        return None


# JSON GENERATION

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


# MAIN PROCESSING

def process_sequence_plan(
    plan: SequencePlan,
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    Process a single sequence plan, automatically handling either video or NPZ inputs.

    If the files in the sequence are videos (e.g., .mp4, .mov), it will concatenate
    them visually into a single continuous video (.mp4).

    If the files are NPZs, it will concatenate them numerically into a single .npz file.
    Both output types will include a corresponding JSON metadata file.

    Args:
        plan: SequencePlan object defining which files to concatenate.
        input_dir: Directory containing NPZ or video files.
        output_dir: Destination directory for output files.
        dry_run: If True, no files are written (for preview only).

    Returns:
        Metadata dictionary if successful, None otherwise.
    """
    try:
        # Detect file type based on the first file’s extension
        first_file = plan.samples[0]['file']
        ext = Path(first_file).suffix.lower()
        is_video = ext in VIDEO_EXTENSIONS

        # DRY-RUN MODE
        if dry_run:
            print(f"[DRY RUN] Would process sequence {plan.sequence_id} ({'video' if is_video else 'npz'})")
            return None

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_video:
            # 🎬 Concatenate video files
            video_path = output_dir / f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.mp4"
            segment_info, total_duration_ms = concatenate_video_files(plan.samples, input_dir, video_path)
            metadata = create_sequence_metadata(plan, segment_info, total_duration_ms)
            metadata["file_name"] = video_path.name

        else:
            # 📊 Concatenate NPZ feature files
            X, X2048, mask, timestamps, segment_info, total_duration_ms = concatenate_npz_files(plan.samples, input_dir)
            npz_path = output_dir / f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.npz"

            meta_dict = {
                'strategy': plan.strategy,
                'signer': plan.signer,
                'num_segments': plan.num_glosses,
                'total_duration_sec': round(total_duration_ms / 1000.0, 2)
            }

            save_continuous_npz(npz_path, X, X2048, mask, timestamps, meta_dict)
            metadata = create_sequence_metadata(plan, segment_info, total_duration_ms)
            metadata["file_name"] = npz_path.name

        # 🧾 Save JSON metadata for both cases
        json_path = output_dir / f"continuous_{plan.sequence_id:04d}_{plan.signer}_strategy{plan.strategy}.json"
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


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Generate continuous signing sequences from isolated sign videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Strategy 1: Same category (with CSV)
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-csv data/processed/fsl_val.csv \\
      --val-dir data/processed/fsl_val \\
      --output-dir data/processed/continuous_sequences \\
      --strategy 1 --sequences-per-signer 10 --min-glosses 3 --max-glosses 6

  # Strategy 2: With raw video files (no CSV)
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-dir data/raw/continuous_sequences \\
      --output-dir data/processed/continuous_sequences \\
      --strategy 1 --sequences-per-signer 10 --min-glosses 3 --max-glosses 6

  # Dry run
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-dir data/raw/continuous_sequences \\
      --strategy 1 --dry-run
  
  # Single signer only
  python preprocessing/continuous/create_continuous_signs.py \\
      --val-csv data/processed/fsl_val.csv \\
      --val-dir data/processed/fsl_val \\
      --output-dir data/processed/continuous_sequences \\
      --strategy 1 --signer S1
        """
    )
    
    parser.add_argument('--val-csv', type=Path, required=False,
                       help='Path to validation CSV file (optional if using video mode)')
    parser.add_argument('--val-dir', type=Path, required=True,
                       help='Directory containing validation NPZ files or video files')
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
    parser.add_argument('--signer', type=str, required=False,
                       help='Only process this signer (e.g., S1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be generated without creating files')
    parser.add_argument('--use-all-files', action='store_true',
                       help='Continue generating sequences until all files are used (exhaustive mode)')
    
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
    
    # Optional: filter to a specific signer
    if getattr(args, 'signer', None):
        df = df[df['signer'] == args.signer]
        if df.empty:
            print(f"[ERROR] No samples found for signer {args.signer}")
            return 1
        print(f"   ✓ Filtered to signer {args.signer}: {len(df)} samples")
    
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
        args.seed,
        use_all_files=getattr(args, 'use_all_files', False)
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
    print(f"Generated: {summary['total_sequences']} continuous sequences")
    print(f"📊 Total duration: {summary['total_duration_sec']:.1f}s ({summary['total_duration_min']:.1f} min)")
    print(f"Average duration: {summary['avg_duration_sec']:.1f}s per sequence")
    print(f"Average glosses: {summary['avg_glosses_per_sequence']:.1f} per sequence")
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
        print(f"\nRun without --dry-run to generate files")
    
    print("\n" + "=" * 80)
    
    # Clean up temp directory if video preprocessing was used
    global TEMP_PREPROCESS_DIR
    if TEMP_PREPROCESS_DIR is not None and TEMP_PREPROCESS_DIR.exists():
        import shutil
        print(f"\n🧹 Cleaning up temporary directory: {TEMP_PREPROCESS_DIR}")
        shutil.rmtree(TEMP_PREPROCESS_DIR)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

