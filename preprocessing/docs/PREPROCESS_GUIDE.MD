# Preprocessing Guide

Convert raw sign language videos to training-ready data for both Transformer and IV3-GRU models.

> **For faster processing**: Use [Multi-Process Preprocessing Guide](MULTI_PREPROCESS_GUIDE.md) for 30-50x speedup.

## Overview

This preprocessing pipeline extracts two types of features from sign language videos:

- **Transformer input**: `.npz` files with MediaPipe keypoints `X` shaped `[T,156]` (78 keypoints × 2D coordinates)
- **IV3-GRU input**: `.npz` files with InceptionV3 CNN features `X2048` shaped `[T,2048]`
- **Metadata**: `mask` (keypoint visibility), `timestamps_ms`, `meta` JSON, optional `.parquet`
- **Labels**: `train_labels.csv` and `val_labels.csv` mapping clips to class IDs

## Prerequisites

```bash
pip install -r requirements.txt
pip install pyarrow  # optional, for parquet inspection
```

## Data Structure

```
data/
  raw/
    <videos>
  processed/
    keypoints_all/     # temporary dump
    keypoints_train/   # final train split
    keypoints_val/     # final val split
    train_labels.csv   # file,gloss,cat
    val_labels.csv     # file,gloss,cat
```

## Renaming Clips

Ensure unique filenames before preprocessing:

1. Copy master `labels.csv` to `data/raw/`
2. Place video clips in `data/raw/clips/*/*.MOV`
3. Run `rename_clips.py`

This generates a new `labels.csv` in `data/processed/keypoints_all/` with renamed video files.

## Processing Videos

Two processing options available:

- `preprocessing/preprocess.py` — single video or directory processing
- `preprocessing/iv3_features.py` — single video with labeling

### Single Video

```bash
# Extract both keypoints and CNN features
python preprocessing/preprocess.py \
  --write-keypoints --write-iv3-features \
  /path/to/video.mp4 /path/to/out_dir

# Extract with labels (0-based integers)
python preprocessing/iv3_features.py \
  --write-keypoints --write-iv3-features \
  --gloss 12 --cat 3 \
  /path/to/video.mp4 /path/to/out_dir
```

### Batch Processing

```bash
# Process entire directory of videos
python -m preprocessing.preprocess data/raw/videos data/processed/keypoints_all \
  --target-fps 30 --out-size 256 --conf-thresh 0.5 --max-gap 5 \
  --write-keypoints --write-iv3-features --id 0
```

After batch processing, split files using `data_split.py`.

## NPZ File Contents

Each processed `.npz` file contains:

- `X`: `[T,156]` MediaPipe keypoints (pose25, left_hand21, right_hand21, face11)
- `X2048`: `[T,2048]` InceptionV3 CNN features (optional)
- `mask`: `[T,78]` keypoint visibility mask
- `timestamps_ms`: `[T]` frame timestamps in milliseconds
- `meta`: JSON metadata with processing parameters

## Creating Labels CSVs

Required CSV format:

```csv
file,gloss,cat,occluded
clip_0001,12,3,0
clip_0002,12,3,1
```

Column requirements:

- `file`: basename of `.npz` file (without extension)
- `gloss`, `cat`: 0-based integer class IDs
- `occluded`: 0/1 flag (0=not occluded, 1=occluded)
- Files must exist in corresponding directories

### Auto-generation

Use `--id` flag during preprocessing to automatically generate labels:

```bash
python -m preprocessing.preprocess data/raw data/processed/keypoints_all \
  --write-keypoints --write-iv3-features --id 0
```

### Occlusion Detection

Automatically enabled when `--write-keypoints` is used:

- Frame marked as occluded if visible keypoints/78 < threshold (default 0.6)
- Clip marked as occluded if occluded frames ≥ 40% or consecutive run ≥ 15 frames

## Training Parameters

### Transformer Model

- `--keypoints-train` / `--keypoints-val`: paths to keypoint directories
- `--labels-train-csv` / `--labels-val-csv`: label CSV file paths
- `--num-gloss` / `--num-cat`: total number of classes

### IV3-GRU Model

- `--features-train` / `--features-val`: feature directories
- `--labels-train-csv` / `--labels-val-csv`: label CSV file paths

Training example:

```bash
python -m training.train \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --amp
```

## Validation

```bash
# Quick smoke test
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10

# Validate NPZ files
python -m preprocessing.validate_npz data/processed/keypoints_train
```

## Common Issues

- **Mismatched names**: CSV `file` entries must exist in directories
- **Wrong shape**: Ensure `X` is `[T,156]` for keypoints
- **Label ranges**: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`
- **Empty sequences**: Preprocessing skips clips with < 1 frame

## FAQ

- **CSVs separate from NPZ?** Yes, linked by `file` basename
- **X2048 with Transformer?** Yes, extra keys are ignored
- **Keypoint order?** `pose25,left_hand21,right_hand21,face11` (normalized [0,1])
