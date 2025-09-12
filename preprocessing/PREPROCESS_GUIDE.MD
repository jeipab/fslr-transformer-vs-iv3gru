## Preprocessing Guide

This guide explains how to turn raw videos into training-ready artifacts for the Transformer (keypoints) and IV3-GRU (CNN features). It covers required inputs, commands to run, expected outputs, and how to build the training CSVs.

---

### 1) Overview

- **Transformer model input**: `.npz` per clip, with key `X` containing keypoints shaped `[T,156]` (variable T). 156 = 78 keypoints × 2D coords.
- **IV3-GRU model input (optional)**: `.npz` per clip, with key `X2048` containing CNN features shaped `[T,2048]`.
- **Shared metadata**: `mask` (visibility per keypoint), `timestamps_ms`, `meta` JSON string, optional `.parquet` for inspection.
- **Labels CSVs**: `train_labels.csv` and `val_labels.csv` mapping each clip (by basename) to integer class ids: `gloss` and `cat` (0-based indices). During preprocessing you can also auto-generate/update a labels CSV (with optional `occluded`) via `preprocessing/preprocess.py`.

You can generate `.npz` files directly from videos using `preprocessing/preprocess.py` or `preprocessing/iv3_features.py`.

---

### 2) Prerequisites

- Python 3.9+
- Install repository requirements:

```bash
pip install -r requirements.txt
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

### 4) Renaming Clips Before Preprocessing

Before preprocessing, rename the clips to ensure that file names are unique (e.g., `clip_0001_good_morning.MOV`, not `0.MOV`). This will prevent the system from assigning the same identifier to multiple clips.

To rename the clips:

1. Retrieve the master `labels.csv` from Tupal's drive (the one in the `fsl-105` drive).
2. Import/copy Tupal's `labels.csv` to `data\raw`.
3. Ensure the unflattened input clips are in `data\raw\` (i.e., `data\raw\clips\*\*.MOV`)
4. Execute `rename_clips.py`.

After renaming, a new `labels.csv` will be generated in `data\processed\keypoints_all` along with the renamed video clips (a total of 2,130 .mov files). This new CSV will be the master file for data splitting going forward.

Once the renaming and CSV generation are complete, the labels CSV from Tupal is no longer necessary.

---

### 5) Running preprocessing from videos

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

#### Option B: Batch process a WHOLE directory and store in `keypoints_all` (with optional CSV writing and occlusion)

```bash
# Directory mode: process all videos under the folder
python -m preprocessing.preprocess data\raw\videos data\processed\keypoints_all 
  --target-fps 30 --out-size 256 --conf-thresh 0.5 --max-gap 5
  --write-keypoints --write-iv3-features
  --id 0
# This writes data\processed\labels.csv by default. Use --append to accumulate.
```

Notes:

- Output files are saved as `/path/to/out_dir/<video_basename>.npz`.
- Each `.npz` contains keys: `X` (and optionally `X2048`), `mask`, `timestamps_ms`, `meta`.
- A human-readable `.parquet` is also written for quick inspection (if `pyarrow` installed).
- If `--id` (or `--gloss-id`/`--cat-id`) is provided, a row is appended per processed file to `<out_dir>/labels.csv` by default. Override with `--labels-csv <path>`.

After batch extraction, split the `.npz` files from `keypoints_all/` into `keypoints_train/` and `keypoints_val/`.
- Run `data_split.py`

When creating the split folders, place the `.npz` files directly inside `keypoints_train/` or `keypoints_val/` (no nested subfolder).

---

### 6) What’s inside each .npz

A typical file `<clip>.npz` contains:

- `X`: float32 array `[T,156]` — keypoints per frame (pose25, left_hand21, right_hand21, face11; normalized coords in [0,1]).
- `X2048` (optional): float32 array `[T,2048]` — per-frame InceptionV3 features.
- `mask`: bool array `[T,78]` — visibility of each keypoint; small gaps are interpolated.
- `timestamps_ms`: int64 array `[T]` — frame time in milliseconds.
- `meta`: JSON-encoded dict with fields such as `video`, `target_fps`, `dims_per_frame`, etc.

---

### 7) Creating train/val splits and CSVs

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

Optional column:

- `occluded`: 0/1 flag produced during preprocessing (currently ignored during training). If present, it will be ignored by loaders.

How to build them:

- If you curated with `iv3_features.py` and used a `labels.csv`, you can split this file into two by sampling rows into train/val and moving the corresponding `.npz` files into the matching folders.
- If you started from directory names or a separate annotation source, write a small script to enumerate `.npz` basenames and join with your annotations to produce the CSVs.
- Alternatively, auto-generate using `preprocessing/preprocess.py` with `--id` (see Option B above). By default it writes `<out_dir>/labels.csv`; use `--append` to accumulate multiple batches/classes or `--labels-csv` to customize the path.

Occlusion detection (auto):

- Enabled when you use `--write-keypoints` (or explicitly add `--occ-enable`).
- Heuristic: a frame is occluded if visible keypoints/78 < `--occ-vis-thresh` (default 0.6). A clip is marked occluded if either:
  - Proportion of occluded frames ≥ `--occ-frame-prop` (default 0.4), or
  - Longest consecutive occluded run ≥ `--occ-min-run` frames (default 15).
- Flags to tune:
  - `--occ-vis-thresh <float>`
  - `--occ-frame-prop <float>`
  - `--occ-min-run <int>`

CSV writing flags:

- `--id <int>`: use same id for `gloss` and `cat`.
- `--gloss-id <int>` and/or `--cat-id <int>`: override individually.
- `--labels-csv <path>`: file to create/append (default `<out_dir>/labels.csv`).
- `--append`: append rows; otherwise the header is rewritten before this run.

Quick check:

- Verify every `file` exists in the directory: `keypoints_train/<file>.npz` or `keypoints_val/<file>.npz`.
- Confirm there are no out-of-range labels.

---

### 8) Training parameters that must match your data

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

### 9) Common pitfalls & validation

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

### 10) FAQ

- Do I need to create CSVs myself?  
  Yes. You provide `train_labels.csv` and `val_labels.csv`. Each maps `.npz` files to integer IDs for `gloss` and `cat`.

- Are the CSVs separate from the `.npz` files?  
  Yes. The training loader reads arrays from `.npz` and labels from CSVs. They are linked by the `file` basename.

- Can I store `X2048` and still train the Transformer?  
  Yes. The Transformer only uses `X`. Extra keys are ignored.

- How are keypoints ordered in `X`?  
  `pose25,left_hand21,right_hand21,face11`, normalized coordinates in [0,1]. See `preprocessing/keypoints_features.py`.

---

### 11) Where to look in code

- Dataset loaders: `training/train.py` (`FSLKeypointFileDataset`, `FSLFeatureFileDataset`).
- Transformer model: `models/transformer.py`.
- Preprocessing: `preprocessing/preprocess.py`, `preprocessing/iv3_features.py`, `preprocessing/keypoints_features.py`.
