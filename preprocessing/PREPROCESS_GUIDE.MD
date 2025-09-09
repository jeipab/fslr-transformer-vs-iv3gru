## Preprocessing Guide

This guide explains how to turn raw videos into training-ready artifacts for the Transformer (keypoints) and IV3-GRU (CNN features). It covers required inputs, commands to run, expected outputs, and how to build the training CSVs.

---

### 1) Overview

- **Transformer model input**: `.npz` per clip, with key `X` containing keypoints shaped `[T,156]` (variable T). 156 = 78 keypoints × 2D coords.
- **IV3-GRU model input (optional)**: `.npz` per clip, with key `X2048` containing CNN features shaped `[T,2048]`.
- **Shared metadata**: `mask` (visibility per keypoint), `timestamps_ms`, `meta` JSON string, optional `.parquet` for inspection.
- **Labels CSVs**: `train_labels.csv` and `val_labels.csv` mapping each clip (by basename) to integer class ids: `gloss` and `cat` (0-based indices).

You can generate `.npz` files directly from videos using `preprocessing/preprocess.py` or `preprocessing/iv3_features.py`.

---

### 2) Prerequisites

- Python 3.9+
- Install repository requirements:

```bash
pip install -r requirments.txt
# Optional (recommended for inspecting outputs)
pip install pyarrow
```

- For video processing: FFmpeg and a working OpenCV build (installed via requirements).

---

### 3) Data layout

We recommend this structure:

```
data/
  raw/
    <your_raw_videos_or_archives>
  processed/
    keypoints_all/           # temporary dump of all processed .npz (optional)
    keypoints_train/         # final train split
    keypoints_val/           # final val split
    train_labels.csv         # file,gloss,cat (0-based)
    val_labels.csv           # file,gloss,cat (0-based)
```

Each `.npz` lives directly inside `keypoints_train/` or `keypoints_val/` (no nested class folders). The CSVs use the clip basename (with or without `.npz`) in the `file` column.

---

### 4) Running preprocessing from videos

There are two entry points that can create keypoints `X` and (optionally) IV3 features `X2048`:

- `preprocessing/preprocess.py` — supports both single video and directory modes.
- `preprocessing/iv3_features.py` — single-video processor (also supports labeling).

Pick one flow below.

#### Option A: Process ONE video (generic extractor)

```bash
# Single file (writes X and optionally X2048 into one .npz)
python preprocessing/preprocess.py \
  --write-keypoints --write-iv3-features \
  /path/to/video.mp4 /path/to/out_dir
```

#### Option A2: Process ONE video and write labels (curation helper)

```bash
# Also writes/updates <out_dir>/labels.csv with (file,gloss,cat) 0-based ints
python preprocessing/iv3_features.py \
  --write-keypoints --write-iv3-features \
  --gloss 12 --cat 3 \
  /path/to/video.mp4 /path/to/out_dir
```

Notes for A2:

- `--gloss` and `--cat` should be 0-based integers because training loaders parse them as ints.
- Optional: `--label-file` to override the default `<out_dir>/labels.csv`.

#### Option B: Batch process a WHOLE directory

```bash
# Directory mode: process all videos under the folder
python -m preprocessing.preprocess /path/to/videos /path/to/out_dir \
  --target-fps 30 --out-size 256 --conf-thresh 0.5 --max-gap 5 \
  --write-keypoints --write-iv3-features
```

Notes:

- Output files are saved as `/path/to/out_dir/0/<video_basename>.npz` (the script uses a subfolder `0` by default).
- Each `.npz` contains keys: `X` (and optionally `X2048`), `mask`, `timestamps_ms`, `meta`.
- A human-readable `.parquet` is also written for quick inspection (if `pyarrow` installed).

After batch extraction, move/split the `.npz` files into `keypoints_train/` and `keypoints_val/`.
When creating the split folders, place the `.npz` files directly inside `keypoints_train/` or `keypoints_val/` (no nested `0/` subfolder).

---

### 5) What’s inside each .npz

A typical file `<clip>.npz` contains:

- `X`: float32 array `[T,156]` — keypoints per frame (pose25, left_hand21, right_hand21, face11; normalized coords in [0,1]).
- `X2048` (optional): float32 array `[T,2048]` — per-frame InceptionV3 features.
- `mask`: bool array `[T,78]` — visibility of each keypoint; small gaps are interpolated.
- `timestamps_ms`: int64 array `[T]` — frame time in milliseconds.
- `meta`: JSON-encoded dict with fields such as `video`, `target_fps`, `dims_per_frame`, etc.

---

### 6) Creating train/val splits and CSVs

You need two CSVs:

- `train_labels.csv`
- `val_labels.csv`

Both must have header:

```
file,gloss,cat
```

Rows map clip basename → integer class ids:

```
clip_0001,12,3
clip_0002,12,3
clip_0101,45,7
```

Rules:

- `file`: the basename of the `.npz` file (with or without `.npz`). The training loader strips any extension.
- `gloss`, `cat`: 0-based integer IDs. Ensure ranges match `--num-gloss` and `--num-cat` during training.
- Each CSV should reference files present in its corresponding directory (`keypoints_train/` vs `keypoints_val/`).

How to build them:

- If you curated with `iv3_features.py` and used a `labels.csv`, you can split this file into two by sampling rows into train/val and moving the corresponding `.npz` files into the matching folders.
- If you started from directory names or a separate annotation source, write a small script to enumerate `.npz` basenames and join with your annotations to produce the CSVs.

Quick check:

- Verify every `file` exists in the directory: `keypoints_train/<file>.npz` or `keypoints_val/<file>.npz`.
- Confirm there are no out-of-range labels.

---

### 7) Training parameters that must match your data

When you run training for the Transformer, set:

- `--keypoints-train` → path to `keypoints_train/`
- `--keypoints-val` → path to `keypoints_val/`
- `--labels-train-csv` / `--labels-val-csv` → your CSVs
- `--num-gloss` and `--num-cat` → total counts in your dataset
- `--kp-key` → only if you stored keypoints under a different key than `X`

When you run training for IV3-GRU (precomputed features), set:

- `--features-train` / `--features-val` → your feature folders
- `--labels-train-csv` / `--labels-val-csv` → your CSVs
- `--feature-key` → only if you stored features under a different key than `X2048` (loader falls back to `X`)

Example command:

```bash
python -m training.train \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val   data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv   data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 \
  --output-dir data/processed --amp
```

---

### 8) Common pitfalls & validation

- Mismatched names: a `file` entry that doesn’t exist under the split directory.
- Wrong shape: ensure `X` is `[T,156]`. If you used a different key, pass `--kp-key`.
- Label ranges: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`.
- Empty sequences: preprocessing should yield at least 1 frame; otherwise the clip is skipped.

Sanity checks:

```bash
# Smoke test (no real data required)
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10

# Quick integrity check script idea (pseudo):
# - list .npz in train/val
# - for each row in CSV ensure matching .npz exists and loads X with shape [:,156]
```

Validation of preprocessed files:

```bash
# Validate all files under a directory for both models (both checks are enabled by default)
python -m preprocessing.validate_npz data/processed/keypoints_train

# Require X2048 to exist and be [T,2048]
python -m preprocessing.validate_npz data/processed/keypoints_train --require-x2048

# Skip parquet checks if you don't have pyarrow/fastparquet installed
python -m preprocessing.validate_npz data/processed/keypoints_train --skip-parquet
```

---

### 9) FAQ

- Do I need to create CSVs myself?  
  Yes. You provide `train_labels.csv` and `val_labels.csv`. Each maps `.npz` files to integer IDs for `gloss` and `cat`.

- Are the CSVs separate from the `.npz` files?  
  Yes. The training loader reads arrays from `.npz` and labels from CSVs. They are linked by the `file` basename.

- Can I store `X2048` and still train the Transformer?  
  Yes. The Transformer only uses `X`. Extra keys are ignored.

- How are keypoints ordered in `X`?  
  `pose25,left_hand21,right_hand21,face11`, normalized coordinates in [0,1]. See `preprocessing/keypoints_features.py`.

---

### 10) Where to look in code

- Dataset loaders: `training/train.py` (`FSLKeypointFileDataset`, `FSLFeatureFileDataset`).
- Transformer model: `models/transformer.py`.
- Preprocessing: `preprocessing/preprocess.py`, `preprocessing/iv3_features.py`, `preprocessing/keypoints_features.py`.
