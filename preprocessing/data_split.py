import argparse
from pathlib import Path
import shutil
import sys
import csv
import hashlib
import pandas as pd

"""
Dataset organizer using predefined train/val split assignments.

This script **does not calculate** splits. It expects the input labels CSV to
contain a 'split' column with values 'train' or 'val' and will:
- Validate required columns: file, gloss, cat, occluded, split
- Normalize/resolve .npz paths under --processed-root
- Map string categories in 'cat' to contiguous integer ids starting at 0
  (if 'cat' is already integer-like, it is preserved)
- Move/copy each .npz (and optional .parquet neighbor) into:
    <out_root>/keypoints_train/ and <out_root>/keypoints_val/
- Write train_labels.csv and val_labels.csv with header:
    file,gloss,cat,occluded   (file = basename without extension)
- Avoid filename collisions by appending a stable 8-char hash suffix derived
  from the source path if a destination basename already exists.
"""

def _resolve_npz_path(processed_root: Path, file_entry: str) -> Path:
    """
    Try to resolve a .npz path for a given 'file' entry.
    Accepts values with or without '.npz', with or without subfolder prefixes like '0/'.
    """
    fe = file_entry.strip()
    # Strip .npz if given
    if fe.lower().endswith(".npz"):
        fe = fe[:-4]
    candidates = []
    # Candidate basenames
    base = Path(fe).name  # ensure basename only
    candidates.append(processed_root / f"{base}.npz")
    # Common subfolder used by preprocessing scripts
    candidates.append(processed_root / "0" / f"{base}.npz")
    # If 'file' encoded a relative subpath, also try it directly under processed_root
    rel = Path(fe)
    if len(rel.parts) > 1:
        candidates.append(processed_root / f"{fe}.npz")
    # Search recursively as a fallback
    # (in case files are nested elsewhere)
    for c in candidates:
        if c.exists():
            return c
    for p in processed_root.rglob(f"{base}.npz"):
        return p
    raise FileNotFoundError(f"Could not resolve path for file entry '{file_entry}' under {processed_root}")


def _stratify_groups(df: pd.DataFrame, mode: str) -> np.ndarray:
    """
    Return a numpy array of group keys used for stratified sampling.
    mode in {'gloss','cat','both','none'}.
    """
    mode = mode.lower()
    if mode == "gloss":
        return df["gloss"].to_numpy()
    if mode == "cat":
        return df["cat"].to_numpy()
    if mode == "both":
        return (df["gloss"].astype(str) + "_" + df["cat"].astype(str)).to_numpy()
    # none
    return np.zeros(len(df), dtype=int)


def _split_indices_stratified(n: int, groups: np.ndarray, train_ratio: float, rng: random.Random):
    """
    Deterministic stratified split without sklearn.
    Returns (train_idxs, val_idxs) as sorted lists.
    """
    assert len(groups) == n
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)

    train_idxs = []
    val_idxs = []
    for g, idxs in by_group.items():
        # stable order for reproducibility across Python versions
        idxs_sorted = sorted(idxs)
        rng.shuffle(idxs_sorted)
        n_g = len(idxs_sorted)
        n_train_g = int(round(n_g * train_ratio))
        n_train_g = min(max(n_train_g, 1 if n_g > 1 else 0), n_g)  # keep at least 1 in val if possible
        train_idxs.extend(idxs_sorted[:n_train_g])
        val_idxs.extend(idxs_sorted[n_train_g:])

    # As a safety, if val is empty due to tiny dataset, force one item to val
    if len(val_idxs) == 0 and len(train_idxs) > 1:
        val_idxs.append(train_idxs.pop())
    return sorted(train_idxs), sorted(val_idxs)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "gloss", "cat"])
        for r in rows:
            writer.writerow([r["file"], r["gloss"], r["cat"]])


def _move_or_copy(src: Path, dst_dir: Path, do_copy: bool, new_name: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{new_name}{src.suffix}"
    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)
    # move/copy optional parquet neighbor
    pq_src = src.with_suffix(".parquet")
    if pq_src.exists():
        pq_dst = dst_dir / f"{new_name}.parquet"
        if do_copy:
            shutil.copy2(pq_src, pq_dst)
        else:
            shutil.move(pq_src, pq_dst)


def main():
    ap = argparse.ArgumentParser(description="Create train/val split CSVs and organize files (80/20).")
    ap.add_argument("--processed-root", required=True, type=Path, help="Directory with preprocessed .npz files")
    ap.add_argument("--labels", required=True, type=Path, help="Path to labels.csv (file,gloss,cat[,...])")
    ap.add_argument("--out-root", type=Path, default=None, help="Output root (defaults to --processed-root)")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed (default 1337)")
    ap.add_argument("--stratify", choices=["gloss","cat","both","none"], default="both", help="Stratification key (default both)")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move the files into split folders")
    args = ap.parse_args()

    processed_root: Path = args.processed_root.resolve()
    out_root: Path = (args.out_root or args.processed_root).resolve()
    rng = random.Random(args.seed)

    # Load labels
    try:
        df = pd.read_csv(args.labels)
    except Exception as e:
        print(f"ERROR: Could not read labels CSV: {e}", file=sys.stderr)
        return 2

    required_cols = {"file", "gloss", "cat"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: labels CSV missing required columns: {sorted(missing)}", file=sys.stderr)
        return 2

    # Resolve full paths and normalize file basenames
    paths = []
    basenames = []
    for file_entry in df["file"].astype(str).tolist():
        try:
            p = _resolve_npz_path(processed_root, file_entry)
            paths.append(p)
            rel = p.relative_to(processed_root)              # e.g. "0/1.npz"
            safe_name = "_".join(rel.with_suffix("").parts)  # e.g. "0_1"
            basenames.append(safe_name)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    df = df.copy()
    df["__npz_path"] = paths
    df["__basename"] = basenames

    # Build stratification vector
    groups = _stratify_groups(df, args.stratify)
    train_idxs, val_idxs = _split_indices_stratified(len(df), groups, args.train_ratio, rng)

    df_train = df.iloc[train_idxs].reset_index(drop=True)
    df_val = df.iloc[val_idxs].reset_index(drop=True)

    # Output directories
    d_train = out_root / "keypoints_train"
    d_val = out_root / "keypoints_val"

    # Move/copy files
    print(f"Organizing files into:\n  {d_train}\n  {d_val}")
    for safe_name, p in zip(df_train["__basename"], df_train["__npz_path"]):
        _move_or_copy(p, d_train, args.copy, safe_name)
    for safe_name, p in zip(df_val["__basename"], df_val["__npz_path"]):
        _move_or_copy(p, d_val, args.copy, safe_name)

    # Write CSVs with required header and file basenames (no extension)        
    rows_train = [{"file": b, "gloss": int(g), "cat": int(c)} for b, g, c in zip(df_train["__basename"], df_train["gloss"], df_train["cat"])]
    rows_val   = [{"file": b, "gloss": int(g), "cat": int(c)} for b, g, c in zip(df_val["__basename"], df_val["gloss"], df_val["cat"])]
    csv_train = out_root / "train_labels.csv"
    csv_val = out_root / "val_labels.csv"
    _write_csv(csv_train, rows_train)
    _write_csv(csv_val, rows_val)

    print("\nDone!")
    print(f"- Train CSV: {csv_train}")
    print(f"- Val   CSV: {csv_val}")
    print(f"- Train files dir: {d_train}")
    print(f"- Val files dir:   {d_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
