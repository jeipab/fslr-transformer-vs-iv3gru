"""
Batch validator for preprocessed .npz / .parquet files.

How to run:
  - Validate a directory recursively (all *.npz):
      python -m preprocessing.validate_npz data/processed/npz_val

  - Validate and require X2048 to be present and shaped [T,2048]:
      python -m preprocessing.validate_npz data/processed/npz_val --require-x2048

  - Skip parquet checks (if pyarrow/fastparquet is not installed):
      python -m preprocessing.validate_npz data/processed/npz_val --skip-parquet

Exit code is non-zero if any file has issues.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # Parquet checks will be skipped if pandas is unavailable


def _load_meta(meta_any) -> dict:
    """Load JSON meta from npz object (may be bytes/str/object array)."""
    try:
        if hasattr(meta_any, "item"):
            meta_any = meta_any.item()
        if isinstance(meta_any, bytes):
            meta_any = meta_any.decode("utf-8")
        if isinstance(meta_any, str):
            return json.loads(meta_any)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"bad meta json: {exc}")
    raise ValueError("meta field is not a JSON-encoded string")


def validate_npz_file(npz_path: str, require_x2048: bool, check_parquet: bool) -> List[str]:
    """
    Validate one .npz (and sibling .parquet if present/required).

    Returns a list of error strings; empty list means OK.
    """
    errors: List[str] = []
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        return [f"npz load error: {exc}"]

    # Required keys present
    for key in ("X", "mask", "timestamps_ms", "meta"):
        if key not in data.files:
            errors.append(f"missing key: {key}")
            # If core key missing, further checks likely useless
            return errors

    X = data["X"]
    mask = data["mask"]
    timestamps_ms = data["timestamps_ms"]
    try:
        _ = _load_meta(data["meta"])  # parseability check
    except Exception as exc:
        errors.append(str(exc))

    # Shapes and dtypes
    if not (X.ndim == 2 and X.shape[1] == 156 and X.dtype == np.float32):
        errors.append(f"X bad shape/dtype: {X.shape} {X.dtype}")
    if not (mask.ndim == 2 and mask.shape[1] == 78 and mask.dtype == np.bool_):
        errors.append(f"mask bad shape/dtype: {mask.shape} {mask.dtype}")
    if not (timestamps_ms.ndim == 1 and timestamps_ms.dtype == np.int64):
        errors.append(
            f"timestamps_ms bad shape/dtype: {timestamps_ms.shape} {timestamps_ms.dtype}"
        )
    if X.shape[0] != mask.shape[0] or X.shape[0] != timestamps_ms.shape[0]:
        errors.append("T mismatch among X/mask/timestamps_ms")
    if X.size and (not np.isfinite(X).all()):
        errors.append("X has non-finite values")
    if X.size and ((X < 0).any() or (X > 1).any()):
        errors.append("X outside [0,1]")
    if timestamps_ms.size > 1 and not (timestamps_ms[1:] >= timestamps_ms[:-1]).all():
        errors.append("timestamps_ms not monotonic nondecreasing")

    # Optional X2048
    has_x2048 = "X2048" in data.files
    if require_x2048 and not has_x2048:
        errors.append("missing key: X2048 (required)")
    if has_x2048:
        X2048 = data["X2048"]
        if not (
            X2048.ndim == 2
            and X2048.shape[0] == X.shape[0]
            and X2048.shape[1] == 2048
            and X2048.dtype == np.float32
        ):
            errors.append(f"X2048 bad shape/dtype: {X2048.shape} {X2048.dtype}")

    # Parquet (optional)
    if check_parquet:
        pq_path = os.path.splitext(npz_path)[0] + ".parquet"
        if os.path.exists(pq_path):
            if pd is None:
                errors.append("parquet present but pandas not installed; use --skip-parquet or install pyarrow")
            else:
                try:
                    df = pd.read_parquet(pq_path)
                    if df.shape[0] != X.shape[0]:
                        errors.append(
                            f"parquet row count {df.shape[0]} != T {X.shape[0]}"
                        )
                    for col in ("t_ms", "mask_bits"):
                        if col not in df.columns:
                            errors.append(f"parquet missing column: {col}")
                    if "mask_bits" in df.columns:
                        ok_len = df["mask_bits"].map(lambda s: isinstance(s, str) and len(s) == 78)
                        if not bool(ok_len.all()):
                            errors.append("parquet mask_bits not length 78 for all rows")
                except Exception as exc:
                    errors.append(f"parquet read error: {exc}")

    return errors


def validate_directory(root_dir: str, require_x2048: bool, skip_parquet: bool) -> Tuple[int, list]:
    """
    Validate all .npz files under root_dir (recursive).

    Returns (num_files_checked, list_of_issues) where list_of_issues contains
    tuples of (file_path, [error_strings...]).
    """
    npz_files = glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True)
    issues: List[Tuple[str, List[str]]] = []
    for path in npz_files:
        errs = validate_npz_file(path, require_x2048=require_x2048, check_parquet=not skip_parquet)
        if errs:
            issues.append((path, errs))
    return len(npz_files), issues


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate preprocessed .npz/.parquet files")
    parser.add_argument("root", help="Root directory to scan recursively for .npz files")
    parser.add_argument(
        "--require-x2048",
        action="store_true",
        help="Require X2048 to be present with shape [T,2048]",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip .parquet validation even if files exist",
    )
    args = parser.parse_args(argv)

    num_checked, issues = validate_directory(
        args.root, require_x2048=args.require_x2048, skip_parquet=args.skip_parquet
    )

    print(f"Checked {num_checked} files; {len(issues)} with issues.")
    for file_path, errs in issues:
        print(f"- {file_path}")
        for e in errs:
            print(f"  * {e}")

    return 0 if len(issues) == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


