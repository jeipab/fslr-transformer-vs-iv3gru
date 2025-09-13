# Training Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Data Structure

```
data/processed/
  keypoints_train/        # or features_train/ for IV3-GRU
  keypoints_val/          # or features_val/ for IV3-GRU
  train_labels.csv        # file,gloss,cat,occluded
  val_labels.csv          # file,gloss,cat,occluded
```

**Requirements**:

- Label CSVs with columns: `file,gloss,cat,occluded` (0-based class IDs)
- `.npz` files directly in split folders (no nested subfolders)

## Quick Start

### Transformer (Keypoints)

```bash
python -m training.train \
  --model transformer \
  --keypoints-train path/to/keypoints_train \
  --keypoints-val path/to/keypoints_val \
  --labels-train-csv path/to/train_labels.csv \
  --labels-val-csv path/to/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data/processed
```

### IV3-GRU (Features)

```bash
python -m training.train \
  --model iv3_gru \
  --features-train path/to/features_train \
  --features-val path/to/features_val \
  --labels-train-csv path/to/train_labels.csv \
  --labels-val-csv path/to/val_labels.csv \
  --feature-key X2048 \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data/processed
```

## Data Requirements

- **Transformer**: `.npz` key `X` shaped `[T,156]` (variable lengths supported)
- **IV3-GRU**: `.npz` key `X2048` (or `X`) shaped `[T,2048]`
- **Labels CSV**: `file,gloss,cat,occluded` where `file` matches `.npz` basename

**Notes**:

- Variable-length sequences are padded automatically
- True lengths used for attention masks (Transformer) or packed sequences (IV3-GRU)
- Defaults: `--kp-key X`, `--feature-key X2048`

## Tips

- Use module mode: `python -m training.train`
- GPU auto-detected, CPU fallback available
- **AMP**: Mixed precision (CUDA only) - use `--amp` for faster GPU training
- **DataLoader**: `prefetch_factor` auto-set when `num_workers > 0`
- Ensure `--num-gloss/--num-cat` match your dataset
- CSV metrics flushed after each epoch

## Smoke Tests

Quick sanity checks without real data:

### Transformer

```bash
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10
```

### IV3-GRU

```bash
python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10 --no-pretrained-backbone
```

## Data Validation

```bash
# Validate NPZ files
python -m preprocessing.validate_npz path/to/keypoints_train

# Require X2048 for IV3-GRU
python -m preprocessing.validate_npz path/to/features_train --require-x2048
```

## Advanced Options

### Learning

- `--lr`, `--weight-decay`, `--grad-clip N`

### Scheduling

- `--scheduler [plateau|cosine]`, `--scheduler-patience K`
- `--early-stop K`

### Checkpoints

- `--resume path/to/{ModelName}_last.pt`
- Saved as `{ModelName}_best.pt` and `{ModelName}_last.pt`

### Logging

- `--log-csv logs/train.csv` (epoch, losses, accs, lr)

### DataLoader

- `--num-workers N`, `--pin-memory`, `--prefetch-factor K` (auto-handled)

### Reproducibility

- `--seed S`, `--deterministic`

## Common Issues

- **File not found**: CSV `file` values must match `.npz` basenames
- **Wrong shapes**: Transformer needs `[T,156]` in key `X`; IV3-GRU needs `[T,2048]` in key `X2048`
- **Label ranges**: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`
- **Empty data**: Check directories contain `.npz` files with matching CSV entries
- **AMP on CPU**: Mixed precision auto-disabled on CPU for safety

## Examples

### Transformer (Advanced)

```bash
python -m training.train --model transformer \
  --keypoints-train path/to/kp_train --keypoints-val path/to/kp_val \
  --labels-train-csv path/to/train.csv --labels-val-csv path/to/val.csv \
  --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 64 \
  --lr 3e-4 --weight-decay 1e-4 --amp --grad-clip 1.0 \
  --scheduler cosine --early-stop 10 --log-csv logs/transformer_train.csv \
  --num-workers 4 --pin-memory
```

### IV3-GRU (Advanced)

```bash
python -m training.train --model iv3_gru \
  --features-train path/to/feat_train --features-val path/to/feat_val \
  --labels-train-csv path/to/train.csv --labels-val-csv path/to/val.csv \
  --feature-key X2048 --num-gloss 105 --num-cat 10 --epochs 40 --batch-size 32 \
  --lr 1e-4 --scheduler plateau --scheduler-patience 3 --early-stop 8 \
  --log-csv logs/iv3_gru_train.csv --num-workers 4 --pin-memory
```

### CPU Training

```bash
python -m training.train --model transformer \
  --keypoints-train path/to/kp_train --keypoints-val path/to/kp_val \
  --labels-train-csv path/to/train.csv --labels-val-csv path/to/val.csv \
  --num-gloss 105 --num-cat 10 --epochs 20 --batch-size 16 \
  --lr 1e-4 --scheduler plateau --log-csv logs/cpu_train.csv
```
