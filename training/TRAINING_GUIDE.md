## Training Guide

### Prerequisites

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Prepare label CSVs with columns: `file,gloss,cat` (0-based class ids).

Data layout expected by the loaders (typical):

```
data/processed/
  keypoints_train/        # or features_train/ for IV3-GRU
  keypoints_val/          # or features_val/ for IV3-GRU
  train_labels.csv        # file,gloss,cat (0-based)
  val_labels.csv          # file,gloss,cat (0-based)
```

Each `.npz` file must live directly inside the split folder (no nested subfolders).

### Quick start

- Transformer (keypoints `[T,156]` in `.npz` key `X`):

```bash
python -m training.train \
  --model transformer \
  --keypoints-train path\to\keypoints_train \
  --keypoints-val   path\to\keypoints_val \
  --labels-train-csv path\to\train_labels.csv \
  --labels-val-csv   path\to\val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data\processed
# If your keypoints are stored under a different key, add: --kp-key MyKey
```

- IV3-GRU (features `[T,2048]` in `.npz` key `X2048` or `X`):

```bash
python -m training.train \
  --model iv3_gru \
  --features-train path\to\features_train \
  --features-val   path\to\features_val \
  --labels-train-csv path\to\train_labels.csv \
  --labels-val-csv   path\to\val_labels.csv \
  --feature-key X2048 \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data\processed
# If your features are stored under a different key, add: --feature-key MyKey
```

### Data requirements

- **Transformer**: `.npz` key `X` shaped `[T,156]` (use `--kp-key` to change); variable lengths supported.
- **IV3-GRU**: `.npz` key `X2048` (or `X`) shaped `[T,2048]`.
- **Labels CSV**: `file,gloss,cat` where `file` matches the `.npz` basename.

Notes:

- Variable-length sequences are padded in the DataLoader; true lengths are used to build attention masks (Transformer) or packed sequences (IV3-GRU).
- Defaults: `--kp-key X`, `--feature-key X2048`.

### Tips

- Use module mode: `python -m training.train`.
- GPU is auto-detected. CPU fallback works.
- **AMP (Mixed Precision)**: Only enabled on CUDA devices for safety. Use `--amp` for faster training on GPU.
- **DataLoader Safety**: `prefetch_factor` is automatically set only when `num_workers > 0`.
- For Transformer, an attention mask from lengths is applied automatically.
- Ensure `--num-gloss/--num-cat` match your dataset.
- Put `.npz` files directly under the train/val folders (no nested class directories).
- **CSV Logging**: Metrics are flushed after each epoch to prevent data loss.

### Smoke tests

Quick sanity checks without real data:

- Transformer:

```bash
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10
```

- IV3-GRU (no weights download):

```bash
python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10 --no-pretrained-backbone
```

Outputs saved to `data/processed` by default (override with `--output-dir`).

Integrity checks for your data (optional but recommended):

```bash
# Validate key shapes/dtypes and meta; both checks enabled by default
python -m preprocessing.validate_npz path\to\keypoints_train

# Require X2048 for IV3-GRU data
python -m preprocessing.validate_npz path\to\features_train --require-x2048
```

### Advanced training options

- **Learning**: `--lr`, `--weight-decay`
- **Precision**: `--amp` (mixed precision, CUDA only)
- **Stability**: `--grad-clip N`
- **Scheduling**: `--scheduler [plateau|cosine]`, `--scheduler-patience K`
- **Early stop**: `--early-stop K`
- **Checkpoints**: `--resume path\to\{ModelName}_last.pt`
- **Logging**: `--log-csv logs\train.csv` (epoch, losses, accs, lr)
- **DataLoader**: `--num-workers N`, `--pin-memory`, `--prefetch-factor K` (auto-handled)
- **Reproducibility**: `--seed S`, `--deterministic`

Notes:

- Best and last checkpoints are saved in `--output-dir` as `{ModelName}_best.pt` and `{ModelName}_last.pt`.
- Early stopping and scheduler use validation gloss accuracy.
- **Scheduler CLI**: Use `--scheduler plateau` or `--scheduler cosine` (no `None` option needed).
- **AMP Safety**: Mixed precision is automatically disabled on CPU to prevent errors.

Common errors and fixes:

- **File not found**: ensure `file` values in CSVs match `.npz` basenames and live in the right split folder.
- **Wrong shapes**: Transformer expects `[T,156]` under key `X`; IV3-GRU expects `[T,2048]` under key `X2048` (or pass `--feature-key`).
- **Label ranges**: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`.
- **Empty training data**: Check that your dataset directories contain `.npz` files and CSV has matching entries.
- **DataLoader errors**: `prefetch_factor` is now auto-handled; only set when `num_workers > 0`.
- **AMP on CPU**: Mixed precision is automatically disabled on CPU devices for safety.

### Examples

- **Transformer + keypoints** (with improved safety features):

```bash
python -m training.train --model transformer \
  --keypoints-train path\to\kp_train --keypoints-val path\to\kp_val \
  --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv \
  --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 64 \
  --lr 3e-4 --weight-decay 1e-4 --amp --grad-clip 1.0 \
  --scheduler cosine --early-stop 10 --log-csv logs\transformer_train.csv \
  --num-workers 4 --pin-memory --prefetch-factor 2
```

- **IV3-GRU + features** (with improved safety features):

```bash
python -m training.train --model iv3_gru \
  --features-train path\to\feat_train --features-val path\to\feat_val \
  --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv \
  --feature-key X2048 --num-gloss 105 --num-cat 10 --epochs 40 --batch-size 32 \
  --lr 1e-4 --scheduler plateau --scheduler-patience 3 --early-stop 8 \
  --log-csv logs\iv3_gru_train.csv --num-workers 4 --pin-memory --prefetch-factor 2
```

- **CPU-only training** (AMP automatically disabled):

```bash
python -m training.train --model transformer \
  --keypoints-train path\to\kp_train --keypoints-val path\to\kp_val \
  --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv \
  --num-gloss 105 --num-cat 10 --epochs 20 --batch-size 16 \
  --lr 1e-4 --scheduler plateau --log-csv logs\cpu_train.csv
```
