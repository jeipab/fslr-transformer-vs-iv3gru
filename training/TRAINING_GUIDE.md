## Training guide

### Prerequisites

- Install dependencies:

```bash
pip install -r requirments.txt
```

- Prepare label CSVs with columns: `file,gloss,cat` (0-based class ids).

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
```

### Data requirements

- **Transformer**: `.npz` key `X` shaped `[T,156]` (use `--kp-key` to change); variable lengths supported.
- **IV3-GRU**: `.npz` key `X2048` (or `X`) shaped `[T,2048]`.
- **Labels CSV**: `file,gloss,cat` where `file` matches the `.npz` basename.

### Tips

- Use module mode: `python -m training.train`.
- GPU is auto-detected. CPU fallback works.
- For Transformer, an attention mask from lengths is applied automatically.
- Ensure `--num-gloss/--num-cat` match your dataset.

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

### Advanced training options

- **Learning**: `--lr`, `--weight-decay`
- **Precision**: `--amp` (mixed precision)
- **Stability**: `--grad-clip N`
- **Scheduling**: `--scheduler [plateau|cosine]`, `--scheduler-patience K`
- **Early stop**: `--early-stop K`
- **Checkpoints**: `--resume path\to\{ModelName}_last.pt`
- **Logging**: `--log-csv logs\train.csv` (epoch, losses, accs, lr)
- **DataLoader**: `--num-workers N`, `--pin-memory`, `--prefetch-factor K`
- **Reproducibility**: `--seed S`, `--deterministic`

Notes:

- Best and last checkpoints are saved in `--output-dir` as `{ModelName}_best.pt` and `{ModelName}_last.pt`.
- Early stopping and scheduler use validation gloss accuracy.

Examples:

- Transformer + keypoints:

```bash
python -m training.train --model transformer \
  --keypoints-train path\to\kp_train --keypoints-val path\to\kp_val \
  --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv \
  --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 64 \
  --lr 3e-4 --weight-decay 1e-4 --amp --grad-clip 1.0 \
  --scheduler cosine --early-stop 10 --log-csv logs\transformer_train.csv \
  --num-workers 4 --pin-memory
```

- IV3-GRU + features:

```bash
python -m training.train --model iv3_gru \
  --features-train path\to\feat_train --features-val path\to\feat_val \
  --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv \
  --feature-key X2048 --num-gloss 105 --num-cat 10 --epochs 40 --batch-size 32 \
  --lr 1e-4 --scheduler plateau --scheduler-patience 3 --early-stop 8 \
  --log-csv logs\iv3_gru_train.csv --num-workers 4 --pin-memory
```
