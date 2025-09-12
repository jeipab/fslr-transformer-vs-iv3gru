# Filipino Sign Language Recognition Tool

This repository contains the implementation of our thesis project:
**Multi-Head Attention Transformer for Filipino Sign Language**.

## Repository Structure

- `preprocessing/` → keypoint extraction and occlusion handling
- `models/` → model architectures (IV3-GRU, Transformer)
- `training/` → training scripts, utilities, evaluation
- `streamlit_app/` → Enhanced Streamlit demo application with modular architecture
- `notebooks/` → Jupyter notebooks for experiments

## Setup

We recommend using Python **3.9–3.11**, as these versions have the most stable support for PyTorch.

Clone the repository and install dependencies:

```bash
git clone https://github.com/jeipab/fslr-transformer-vs-iv3gru.git
cd fslr-transformer-vs-iv3gru
pip install -r requirments.txt
```

If you plan to inspect `.parquet` outputs from preprocessing:

```bash
pip install pyarrow
```

## Run the Streamlit demo (UI)

Use PowerShell from the repo root:

**1) (Optional) create & activate a virtual environment**

```bash
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

**2) Install dependencies**

```bash
pip install -r .\requirments.txt
```

**3) Run the app**

```bash
streamlit run run_app.py
```

Alternative (from streamlit_app directory):

```bash
cd streamlit_app
streamlit run main.py
```

The demo supports both preprocessed `.npz` files and video files. It includes animated keypoint visualization, feature analysis, and simulated predictions.

If the default port is busy:

```bash
streamlit run run_app.py --server.port 8502
```

## Quick start: Preprocessing

Generate training-ready `.npz` from videos. Use either the single-file or directory mode.

Single video (writes `X` and optionally `X2048` into one `.npz`):

```bash
python preprocessing/preprocess.py --write-keypoints --write-iv3-features \
  /path/to/video.mp4 /path/to/out_dir
```

Whole directory (recursively finds videos):

```bash
python -m preprocessing.preprocess /path/to/videos /path/to/out_dir \
  --target-fps 30 --out-size 256 --conf-thresh 0.5 --max-gap 5 \
  --write-keypoints --write-iv3-features
```

Notes:

- Files are written to `/path/to/out_dir/0/<basename>.npz` (script uses a `0/` subfolder by default).
- Each `.npz` contains: `X [T,156]`, optional `X2048 [T,2048]`, `mask [T,78]`, `timestamps_ms [T]`, `meta`.

After extraction, split into train/val; place `.npz` directly in these folders (no nested subfolders):

```
data/processed/
  keypoints_train/
  keypoints_val/
  train_labels.csv  # file,gloss,cat (0-based)
  val_labels.csv
```

## Quick start: Training

Transformer (uses keypoints `X [T,156]`):

```bash
python -m training.train \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val   data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv   data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data/processed
# If you used a different key for keypoints: add --kp-key MyKey
```

IV3-GRU (uses features `X2048 [T,2048]` or another key via `--feature-key`):

```bash
python -m training.train \
  --model iv3_gru \
  --features-train data/processed/keypoints_train \
  --features-val   data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv   data/processed/val_labels.csv \
  --feature-key X2048 \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data/processed
# If your features are under a different key: set --feature-key accordingly
```

Mixed precision and performance (optional): add `--amp`, `--num-workers N`, `--pin-memory`.

## Validate your data

Run shape/dtype/meta checks before training:

```bash
python -m preprocessing.validate_npz data/processed/keypoints_train
python -m preprocessing.validate_npz data/processed/keypoints_val --require-x2048
```

## Smoke tests (no real data required)

```bash
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10
python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10 --no-pretrained-backbone
```

## Guide

- Preprocessing guide: [preprocessing/PREPROCESS_GUIDE.md](preprocessing/PREPROCESS_GUIDE.md)
- Training guide: [training/TRAINING_GUIDE.md](training/TRAINING_GUIDE.md)
- Data guide: [data/DATA_GUIDE.md](data/DATA_GUIDE.md)
- Sharing guide: [shared/SHARING_GUIDE.md](shared/SHARING_GUIDE.md)

## Troubleshooting

- File not found: ensure each `file` in CSV matches a `.npz` basename in the corresponding split folder.
- Wrong shapes: Transformer needs `X [T,156]`; IV3-GRU needs `X2048 [T,2048]` (or pass `--feature-key`).
- Label ranges: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`.
- CPU vs GPU: the code auto-detects CUDA; you can still train on CPU for small tests.
