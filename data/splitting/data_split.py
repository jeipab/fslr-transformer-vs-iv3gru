"""
Data splitting utility for Filipino sign language recognition datasets.

This module organizes preprocessed .npz files into train/validation splits
with proper stratified sampling and file management.

Features:
- Stratified splitting by gloss and category
- Signer-aware splitting (mixed or independent modes)
- Automatic category ID encoding
- File collision handling
- Support for filtering by specific categories/glosses
- Optional parquet file synchronization
- Strict validation of signer (S0-S7) and duration (> 0)

Usage:
- Default 80/20 split:
    python data/splitting/data_split.py \
        --processed-root data/processed \
        --labels data/processed/labels.csv \
        --out-root data/processed \
        --copy \
        --train-ratio 0.8

- Filtered split:
    python data/splitting/data_split.py \
        --processed-root data/processed \
        --labels data/processed/labels.csv \
        --out-root data/processed \
        --copy \
        --cats greeting survival number \
        --gloss yes no wrong \
        --label-ref data/splitting/labels_reference.csv
"""

import argparse
from pathlib import Path
import shutil
import sys
import csv
import random
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def _resolve_npz_path(processed_root: Path, file_entry: str) -> Path:
    """Resolve .npz file path from CSV file entry.
    
    Args:
        processed_root: Root directory containing .npz files
        file_entry: File entry from CSV (with or without .npz extension)
        
    Returns:
        Path to the actual .npz file
        
    Raises:
        FileNotFoundError: If .npz file cannot be found
    """
    fe = str(file_entry).strip()
    if fe.lower().endswith(".npz"):
        fe = fe[:-4]
    base = Path(fe).name
    candidates = [
        processed_root / f"{base}.npz",
        processed_root / "0" / f"{base}.npz",
    ]
    rel = Path(fe)
    if len(rel.parts) > 1:
        candidates.append(processed_root / f"{fe}.npz")
    for c in candidates:
        if c.exists():
            return c
    # Fallback: recursive search by basename
    for p in processed_root.rglob(f"{base}.npz"):
        return p
    raise FileNotFoundError(f"Could not resolve path for file entry '{file_entry}' under {processed_root}")

def _coerce_or_encode_cat(series: pd.Series, out_map_path: Path) -> pd.Series:
    """Convert category values to zero-based integer IDs.
    
    Args:
        series: Pandas series with category values
        out_map_path: Path to save category mapping CSV
        
    Returns:
        Series with integer category IDs starting from 0
    """
    # Try to treat as integer
    try:
        as_int = series.astype("Int64")
        if as_int.isna().any():
            raise ValueError("non-integer present")
        # Ensure starts at 0 but do not remap if user already chose ids
        return as_int.astype(int)
    except Exception:
        # Build mapping from sorted unique category names
        cats = sorted(str(x) for x in series.unique())
        mapping = {name: i for i, name in enumerate(cats)}
        # Write mapping for transparency
        out_map_path.parent.mkdir(parents=True, exist_ok=True)
        with out_map_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cat_name", "cat_id"])
            for name, i in mapping.items():
                w.writerow([name, i])
        return series.map(lambda x: mapping[str(x)])

def _stable_h8(path: Path) -> str:
    """Generate stable 8-character hash from file path."""
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]

def _move_or_copy_unique(src_npz: Path, dst_dir: Path, do_copy: bool) -> str:
    """Copy .npz file to destination directory with collision handling.
    
    Args:
        src_npz: Source .npz file path
        dst_dir: Destination directory
        do_copy: If True, copy files; if False, move files
        
    Returns:
        Final basename (without extension) for CSV entry
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src_npz.stem
    dst_npz = dst_dir / f"{stem}.npz"
    if dst_npz.exists():
        stem = f"{stem}-{_stable_h8(src_npz)}"
        dst_npz = dst_dir / f"{stem}.npz"
    
    # Always copy to preserve source files
    shutil.copy2(src_npz, dst_npz)

    # Handle optional parquet
    pq_src = src_npz.with_suffix(".parquet")
    if pq_src.exists():
        pq_dst = dst_dir / f"{stem}.parquet"
        shutil.copy2(pq_src, pq_dst)
    return stem

def _write_csv(path: Path, rows):
    """Write label rows to CSV file.
    
    Args:
        path: Output CSV file path
        rows: List of dictionaries with file, gloss, cat, occluded, signer, duration keys
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "gloss", "cat", "occluded", "signer", "duration"])
        for r in rows:
            writer.writerow([r["file"], r["gloss"], r["cat"], r["occluded"], r["signer"], r["duration"]])

def main():
    ap = argparse.ArgumentParser(description="Organize preprocessed dataset into train/val splits")
    ap.add_argument("--processed-root", required=True, type=Path, nargs="+", help="One or more directories containing NPZ files")
    ap.add_argument("--labels", required=True, type=Path, nargs="+", help="One or more labels CSV files (must match order of --processed-root)")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio if no split column is present")
    ap.add_argument("--cats", nargs="+",help="Restrict to specific categories (IDs or names). ",default=None)
    ap.add_argument("--gloss", nargs="+", help="Restrict to specific glosses (IDs or names). ",default=None)
    ap.add_argument("--label-ref", type=Path,help="Path to label_reference.csv (required if using names in --cats or --gloss).")
    ap.add_argument("--train-dir", type=str, default="keypoints_train", help="Name for train directory (default: keypoints_train)")
    ap.add_argument("--val-dir", type=str, default="keypoints_val", help="Name for val directory (default: keypoints_val)")
    ap.add_argument("--train-csv", type=str, default="train_labels.csv", help="Name for train CSV (default: train_labels.csv)")
    ap.add_argument("--val-csv", type=str, default="val_labels.csv", help="Name for val CSV (default: val_labels.csv)")
    ap.add_argument("--signer-split-mode", type=str, default="mixed", choices=["mixed", "independent"],
                   help="Signer splitting mode: 'mixed' (default, signers in both train/val) or "
                        "'independent' (each signer in train OR val only for generalization testing)")
    args = ap.parse_args()

    # Handle single or multiple input directories
    processed_roots = [Path(p).resolve() for p in (args.processed_root if isinstance(args.processed_root, list) else [args.processed_root])]
    labels_files = [Path(p).resolve() for p in (args.labels if isinstance(args.labels, list) else [args.labels])]
    
    if len(processed_roots) != len(labels_files):
        print(f"ERROR: Number of --processed-root ({len(processed_roots)}) must match --labels ({len(labels_files)})", file=sys.stderr)
        return 2
    
    out_root: Path = (args.out_root or processed_roots[0]).resolve()
    
    # Load and combine labels from all sources
    df_list = []
    for i, (proc_root, labels_file) in enumerate(zip(processed_roots, labels_files)):
        print(f"Loading dataset {i+1}/{len(processed_roots)}: {proc_root}")
        try:
            df_part = pd.read_csv(labels_file)
            df_part["__source_root"] = str(proc_root)  # Track source directory
            df_list.append(df_part)
        except Exception as e:
            print(f"ERROR: Could not read labels CSV {labels_file}: {e}", file=sys.stderr)
            return 2
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined {len(df)} samples from {len(processed_roots)} dataset(s)")

    # Validate required columns (strict - no defaults)
    required_cols = {"file", "gloss", "cat", "occluded", "signer", "duration"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: labels CSV missing required columns: {sorted(missing)}", file=sys.stderr)
        print(f"Required format: file, gloss, cat, occluded, signer, duration", file=sys.stderr)
        return 2
    
    # Validate signer format (S0-S7)
    invalid_signers = df[~df['signer'].astype(str).str.match(r'^S[0-7]$', na=False)]
    if len(invalid_signers) > 0:
        print(f"ERROR: {len(invalid_signers)} files have invalid signer IDs (must be S0-S7):", file=sys.stderr)
        for idx, row in invalid_signers.head(5).iterrows():
            print(f"  - {row['file']}: signer='{row['signer']}'", file=sys.stderr)
        if len(invalid_signers) > 5:
            print(f"  ... and {len(invalid_signers) - 5} more", file=sys.stderr)
        return 2
    
    # Validate duration is positive
    invalid_durations = df[(df['duration'] <= 0) | df['duration'].isna()]
    if len(invalid_durations) > 0:
        print(f"ERROR: {len(invalid_durations)} files have invalid duration (must be > 0):", file=sys.stderr)
        for idx, row in invalid_durations.head(5).iterrows():
            print(f"  - {row['file']}: duration={row['duration']}", file=sys.stderr)
        if len(invalid_durations) > 5:
            print(f"  ... and {len(invalid_durations) - 5} more", file=sys.stderr)
        return 2
    
    print(f"✓ Validation passed: {len(df)} samples with valid signer and duration")
    
    # Load label reference for mapping gloss/cat names to IDs (if provided)
    gloss_name_to_id, cat_name_to_id = {}, {}
    if args.label_ref is not None:
        try:
            ref_df = pd.read_csv(args.label_ref)
            gloss_name_to_id = {str(row["label"]).upper(): int(row["gloss_id"]) for _, row in ref_df.iterrows()}
            cat_name_to_id   = {str(row["category"]).upper(): int(row["cat_id"]) for _, row in ref_df.iterrows()}
        except Exception as e:
            print(f"ERROR: Could not read label reference CSV: {e}", file=sys.stderr)
            return 2


    # Resolve .npz paths (using source_root for each file)
    paths = []
    for idx, row in df.iterrows():
        file_entry = str(row["file"])
        source_root = Path(row["__source_root"])
        try:
            p = _resolve_npz_path(source_root, file_entry)
            paths.append(p)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    df = df.copy()
    df["__npz_path"] = paths

    # Ensure 'cat' is integer ids starting at 0 (write mapping for reference)
    cat_map_path = out_root / "cat_mapping.csv"
    df["cat"] = _coerce_or_encode_cat(df["cat"], cat_map_path)

    # Step 1: Map categories to all glosses
    cat_gloss_map = df.groupby("cat")["gloss"].unique().to_dict()

    # Step 2: Determine allowed categories
    allowed_cat_ids = set()
    if args.cats is not None:
        for c in args.cats:
            if str(c).isdigit():
                allowed_cat_ids.add(int(c))
            else:
                if not cat_name_to_id:
                    print(f"ERROR: Category name '{c}' requires --label-ref to resolve names to IDs", file=sys.stderr)
                    return 2
                key = str(c).strip().upper()
                if key in cat_name_to_id:
                    allowed_cat_ids.add(cat_name_to_id[key])
                else:
                    print(f"WARNING: Unknown category '{c}'", file=sys.stderr)

    # Step 3: Determine allowed glosses
    allowed_gloss_ids = set()
    if args.gloss is not None:
        for g in args.gloss:
            if str(g).isdigit():
                allowed_gloss_ids.add(int(g))
            else:
                key = str(g).strip().upper()
                if key in gloss_name_to_id:
                    allowed_gloss_ids.add(gloss_name_to_id[key])
                else:
                    # fallback: search in file column
                    matched = df.loc[df["file"].str.upper().str.contains(key), "gloss"].unique()
                    allowed_gloss_ids.update(matched.tolist())

    # Step 4: Combine categories and glosses
    combined_pairs = set()
    for cat_id in (allowed_cat_ids or cat_gloss_map.keys()):
        for gloss_id in cat_gloss_map.get(cat_id, []):
            if not allowed_gloss_ids or gloss_id in allowed_gloss_ids:
                combined_pairs.add((cat_id, gloss_id))

    # Step 5: Filter dataframe by these pairs
    df = df[df.apply(lambda row: (row["cat"], row["gloss"]) in combined_pairs, axis=1)].reset_index(drop=True)

    # Step 6: Print context-aware log messages
    if args.cats:
        print(f"Using subset of categories: {args.cats} (kept {len(df)} samples)")
    if args.gloss:
        if args.cats:
            print(f"Using subset of glosses (from subset of categories): {args.gloss} (kept {len(df)} samples)")
        else:
            print(f"Using subset of glosses: {args.gloss} (kept {len(df)} samples)")

    if df.empty:
        print(f"ERROR: No samples left after filtering by categories/glosses", file=sys.stderr)
        return 2

    # Handle split with signer-aware logic
    if "split" not in df.columns:
        if args.signer_split_mode == 'independent':
            # Signer-independent split: each signer goes to train OR val exclusively
            print(f"\nUsing signer-independent split (signer-exclusive for generalization testing)")
            
            unique_signers = sorted(df['signer'].unique())
            signer_splits = {}
            
            # Deterministically assign signers based on hash
            for signer in unique_signers:
                hash_val = int(hashlib.md5(signer.encode()).hexdigest(), 16)
                signer_splits[signer] = "train" if (hash_val % 100) < (args.train_ratio * 100) else "val"
            
            df["split"] = df["signer"].map(signer_splits)
            
            # Report signer assignment
            train_signers = sorted([s for s, split in signer_splits.items() if split == 'train'])
            val_signers = sorted([s for s, split in signer_splits.items() if split == 'val'])
            print(f"  Train signers: {', '.join(train_signers)} ({len(train_signers)} signers)")
            print(f"  Val signers: {', '.join(val_signers)} ({len(val_signers)} signers)")
            
        else:
            # Mixed mode: hash on filename (each signer appears in both splits)
            print(f"\nUsing mixed split (signers appear in both train/val)")
            
            def hash_to_split(filename, train_ratio):
                """Deterministically assign split based on filename hash"""
                hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
                return "train" if (hash_val % 100) < (train_ratio * 100) else "val"
            
            # Apply hash-based split
            splits = []
            for idx, row in df.iterrows():
                splits.append(hash_to_split(row['file'], args.train_ratio))
            
            df["split"] = splits
        
        # Shuffle the entire dataframe to randomize order for training
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)

    # Output directories (using custom names)
    d_train = out_root / args.train_dir
    d_val = out_root / args.val_dir

    # Clear existing content from output directories
    print(f"Clearing existing content from {d_train}...")
    if d_train.exists():
        shutil.rmtree(d_train)
    d_train.mkdir(parents=True, exist_ok=True)
    
    print(f"Clearing existing content from {d_val}...")
    if d_val.exists():
        shutil.rmtree(d_val)
    d_val.mkdir(parents=True, exist_ok=True)

    # Copy files and capture final basenames (after collision handling)
    basenames_train = []
    for p in df_train["__npz_path"]:
        stem = _move_or_copy_unique(p, d_train, args.copy)
        basenames_train.append(stem)
    basenames_val = []
    for p in df_val["__npz_path"]:
        stem = _move_or_copy_unique(p, d_val, args.copy)
        basenames_val.append(stem)

    # Write CSVs (file = final basenames) with 6 columns
    rows_train = [
        {
            "file": b, 
            "gloss": int(g), 
            "cat": int(c), 
            "occluded": int(o),
            "signer": str(s),
            "duration": float(d)
        }
        for b, g, c, o, s, d in zip(
            basenames_train, 
            df_train["gloss"], 
            df_train["cat"], 
            df_train["occluded"],
            df_train["signer"],
            df_train["duration"]
        )
    ]
    rows_val = [
        {
            "file": b, 
            "gloss": int(g), 
            "cat": int(c), 
            "occluded": int(o),
            "signer": str(s),
            "duration": float(d)
        }
        for b, g, c, o, s, d in zip(
            basenames_val, 
            df_val["gloss"], 
            df_val["cat"], 
            df_val["occluded"],
            df_val["signer"],
            df_val["duration"]
        )
    ]

    csv_train = out_root / args.train_csv
    csv_val = out_root / args.val_csv
    _write_csv(csv_train, rows_train)
    _write_csv(csv_val, rows_val)

    print("\n" + "="*80)
    print("SPLIT SUMMARY")
    print("="*80)
    print(f"Train samples:   {len(rows_train)}")
    print(f"Val samples:     {len(rows_val)}")
    print(f"Total samples:   {len(rows_train) + len(rows_val)}")
    print(f"Train ratio:     {len(rows_train)/(len(rows_train) + len(rows_val))*100:.1f}%")
    print(f"Val ratio:       {len(rows_val)/(len(rows_train) + len(rows_val))*100:.1f}%")
    
    # Signer distribution statistics
    print(f"\nSigner Distribution:")
    print(f"  Train:")
    for signer in sorted(df_train['signer'].unique()):
        count = (df_train['signer'] == signer).sum()
        pct = count / len(df_train) * 100 if len(df_train) > 0 else 0
        print(f"    {signer}: {count:4d} videos ({pct:.1f}%)")
    
    print(f"  Val:")
    for signer in sorted(df_val['signer'].unique()):
        count = (df_val['signer'] == signer).sum()
        pct = count / len(df_val) * 100 if len(df_val) > 0 else 0
        print(f"    {signer}: {count:4d} videos ({pct:.1f}%)")
    
    # Validate no leakage in independent mode
    if args.signer_split_mode == 'independent':
        train_signers = set(df_train['signer'].unique())
        val_signers = set(df_val['signer'].unique())
        overlap = train_signers & val_signers
        
        if overlap:
            print(f"\nERROR: Signer leakage detected in independent mode: {sorted(overlap)}", file=sys.stderr)
            print(f"In independent mode, each signer must be exclusively in train OR val", file=sys.stderr)
            return 2
        
        print(f"\n✓ Signer-independent validation passed:")
        print(f"  Train signers only: {sorted(train_signers)}")
        print(f"  Val signers only: {sorted(val_signers)}")
    
    print(f"\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"Train CSV:       {csv_train}")
    print(f"Val CSV:         {csv_val}")
    print(f"Train files dir: {d_train}")
    print(f"Val files dir:   {d_val}")
    print(f"Cat mapping:     {cat_map_path}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
