"""
Data splitting utility for Filipino sign language recognition datasets.

This module organizes preprocessed .npz files into train/validation splits
with proper stratified sampling and file management.

Features:
- Stratified splitting by gloss and category
- Automatic category ID encoding
- File collision handling
- Support for filtering by specific categories/glosses
- Optional parquet file synchronization

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
        rows: List of dictionaries with file, gloss, cat, occluded keys
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "gloss", "cat", "occluded"])
        for r in rows:
            writer.writerow([r["file"], r["gloss"], r["cat"], r["occluded"]])

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

    # Validate required columns  
    required_cols = {"file", "gloss", "cat", "occluded"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: labels CSV missing required columns: {sorted(missing)}", file=sys.stderr)
        return 2
    
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

    # Handle split
    if "split" not in df.columns:
        # Deterministic hash-based splitting per gloss (ensures consistency across runs)
        # Each file goes to same split based on its filename hash, regardless of dataset combination
        import hashlib
        
        def hash_to_split(filename, train_ratio):
            """Deterministically assign split based on filename hash"""
            hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
            return "train" if (hash_val % 100) < (train_ratio * 100) else "val"
        
        # Apply hash-based split per gloss to maintain stratification
        splits = []
        for idx, row in df.iterrows():
            splits.append(hash_to_split(row['file'], args.train_ratio))
        
        df["split"] = splits
        
        # Verify stratification is maintained
        for gloss_id in df['gloss'].unique():
            gloss_df = df[df['gloss'] == gloss_id]
            train_count = (gloss_df['split'] == 'train').sum()
            total_count = len(gloss_df)
            actual_ratio = train_count / total_count if total_count > 0 else 0
            # Should be close to args.train_ratio (small variations due to rounding)
        
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

    # Write CSVs (file = final basenames)
    rows_train = [
        {"file": b, "gloss": int(g), "cat": int(c), "occluded": int(o)}
        for b, g, c, o in zip(basenames_train, df_train["gloss"], df_train["cat"], df_train["occluded"])
    ]
    rows_val = [
        {"file": b, "gloss": int(g), "cat": int(c), "occluded": int(o)}
        for b, g, c, o in zip(basenames_val, df_val["gloss"], df_val["cat"], df_val["occluded"])
    ]

    csv_train = out_root / args.train_csv
    csv_val = out_root / args.val_csv
    _write_csv(csv_train, rows_train)
    _write_csv(csv_val, rows_val)

    print("Done!")
    print(f"- Train samples:   {len(rows_train)}")
    print(f"- Val samples:     {len(rows_val)}")
    print(f"- Total samples:   {len(rows_train) + len(rows_val)}")
    print(f"- Train ratio:     {len(rows_train)/(len(rows_train) + len(rows_val))*100:.1f}%")
    print(f"- Val ratio:       {len(rows_val)/(len(rows_train) + len(rows_val))*100:.1f}%")
    print(f"- Train CSV:       {csv_train}")
    print(f"- Val CSV:         {csv_val}")
    print(f"- Train files dir: {d_train}")
    print(f"- Val files dir:   {d_val}")
    print(f"- Cat mapping:     {cat_map_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
