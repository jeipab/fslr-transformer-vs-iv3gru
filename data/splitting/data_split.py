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
        --labels data/processed/labels_updated.csv \
        --out-root data/processed \
        --copy

Options:
- `--processed-root` : Path to directory of preprocessed `.npz`/`.parquet` files (required).
- `--labels`         : Path to `labels.csv` with columns `file,gloss,cat,occluded` (required).
- `--out-root`       : Destination root directory (default = `processed-root`).
- `--copy`           : Copy files instead of moving them.
- `--train-ratio`    : Train split ratio if no `split` column is present (default = 0.8).
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
    if do_copy:
        shutil.copy2(src_npz, dst_npz)
    else:
        shutil.move(src_npz, dst_npz)

    # Handle optional parquet
    pq_src = src_npz.with_suffix(".parquet")
    if pq_src.exists():
        pq_dst = dst_dir / f"{stem}.parquet"
        if do_copy:
            shutil.copy2(pq_src, pq_dst)
        else:
            shutil.move(pq_src, pq_dst)
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

    # Handle split
    if "split" not in df.columns:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        n_train = int(len(df) * args.train_ratio)
        df["split"] = ["train"] * n_train + ["val"] * (len(df) - n_train)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)

    # Output directories
    d_train = out_root / "keypoints_train"
    d_val = out_root / "keypoints_val"

    # Move/copy files and capture final basenames (after collision handling)
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
    print(f"- Train CSV:       {csv_train}")
    print(f"- Val CSV:         {csv_val}")
    print(f"- Train files dir: {d_train}")
    print(f"- Val files dir:   {d_val}")
    print(f"- Cat mapping:     {cat_map_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
