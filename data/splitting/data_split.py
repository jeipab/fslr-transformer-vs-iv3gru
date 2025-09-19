"""
Splitter & organizer for preprocessed `.npz`/`.parquet` datasets.

Purpose:
- Take a `labels.csv` and corresponding preprocessed clip files.
- Verify required columns: `file`, `gloss`, `cat`, `occluded`.
- Encode `cat` values to zero-based integer IDs (writes `cat_mapping.csv`).
- Split into train/val sets (use provided `split` column if present, else shuffle and split by ratio).
- Move or copy `.npz` (and matching `.parquet`) files into `keypoints_train/` and `keypoints_val/`.
- Generate `train_labels.csv` and `val_labels.csv` containing only `file,gloss,cat,occluded`.

Usage:
- Split with default 80/20 ratio (copy files):
    python data/splitting/data_split.py \
        --processed-root data/processed \
        --labels data/processed/labels.csv \
        --out-root data/processed \
        --copy \
        --train-ratio 0.8

- Split with custom ratio (copy files):
    python data/splitting/data_split.py \
        --processed-root data/processed \
        --labels data/processed/labels.csv \
        --out-root data/processed \
        --copy \
        --train-ratio 0.8 \
        --cats greeting survival number \
        --gloss yes no wrong \
        --label-ref data/splitting/labels_reference.csv

Options:
- `--processed-root` : Path to directory of preprocessed `.npz`/`.parquet` files (required).
- `--labels`         : Path to `labels.csv` with columns `file,gloss,cat,occluded` (required).
- `--out-root`       : Destination root directory (default = `processed-root`).
- `--copy`           : Copy files instead of moving them.
- `--train-ratio`    : Train split ratio if no `split` column is present (default = 0.8).
- `--cats`           : Restrict to specific categories (by ID or name).
                      Examples:
                        --cats 0 1 2
                        --cats greeting survival number
                        --cats "daily routine" "sports"    # use quotes if the name has spaces or punctuation

- `--gloss`          : Restrict to specific glosses (by ID or name).
                      Examples:
                        --gloss 0 1 2
                        --gloss yes no wrong
                        --gloss "good morning" "thank you" "don't understand" # use quotes if the name has spaces or punctuation

- `--label-ref` data/splitting/labels_reference.csv      : Path to label_reference.csv (required if using names instead of numeric IDs for --cats or --gloss).

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
    """
    Resolve a .npz path for a given 'file' entry.
    Accepts values with or without '.npz', with or without subfolder prefixes like '0/'.
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
    """
    Ensure 'cat' is integer ids starting at 0.
    - If series is already integer-like, keep as-is (but reindex to int).
    - Else, encode unique sorted category strings to ids and write cat_mapping.csv.
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
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]

def _move_or_copy_unique(src_npz: Path, dst_dir: Path, do_copy: bool) -> str:
    """
    Move/copy src_npz to dst_dir, avoiding basename collisions.
    Returns the final basename (stem) without extension to be written in CSV.
    Keeps a sibling .parquet in sync if present.
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "gloss", "cat", "occluded"])
        for r in rows:
            writer.writerow([r["file"], r["gloss"], r["cat"], r["occluded"]])

def main():
    ap = argparse.ArgumentParser(description="Organize preprocessed dataset into train/val splits")
    ap.add_argument("--processed-root", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio if no split column is present")
    ap.add_argument("--cats", nargs="+",help="Restrict to specific categories (IDs or names). ",default=None)
    ap.add_argument("--gloss", nargs="+", help="Restrict to specific glosses (IDs or names). ",default=None)
    ap.add_argument("--label-ref", type=Path,help="Path to label_reference.csv (required if using names in --cats or --gloss).")
    args = ap.parse_args()

    processed_root: Path = args.processed_root.resolve()
    out_root: Path = (args.out_root or args.processed_root).resolve()
    # Load labels
    try:
        df = pd.read_csv(args.labels)
    except Exception as e:
        print(f"ERROR: Could not read labels CSV: {e}", file=sys.stderr)
        return 2

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


    # Resolve .npz paths
    paths = []
    for file_entry in df["file"].astype(str).tolist():
        try:
            p = _resolve_npz_path(processed_root, file_entry)
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
        # Stratify by gloss for proper 80/20 split per gloss
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=1-args.train_ratio, 
            random_state=42
        )
        train_indices, val_indices = next(splitter.split(df, df['gloss']))
        
        # Create split column
        df["split"] = "val"  # Initialize all as val
        df.loc[train_indices, "split"] = "train"
        
        # Shuffle the entire dataframe to randomize order for training
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)

    # Output directories
    d_train = out_root / "keypoints_train"
    d_val = out_root / "keypoints_val"

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

    csv_train = out_root / "train_labels.csv"
    csv_val = out_root / "val_labels.csv"
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
