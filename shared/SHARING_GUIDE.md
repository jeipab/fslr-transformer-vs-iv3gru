## Shared data dropzone

### Purpose

- A simple folder for teammates to drop preprocessed artifacts needed for training.

### Location

- `shared/`

### Expected contents

- Unified per-video NPZs (contain both modalities):
  - `shared/npz_train/*.npz` (each contains keys: `X: [T,156]`, `X2048: [T,2048]`, `mask: [T,78]`, `timestamps_ms`, `meta`)
  - `shared/npz_val/*.npz`
  - `shared/train_labels.csv` (columns: `file,gloss,cat`; 0-based ids)
  - `shared/val_labels.csv`

### How to use

1. Teammates export/copy files into `shared/` as above.
2. Copy into `data/processed/` before training, e.g. (Windows PowerShell):

```powershell
robocopy shared\npz_train data\processed\npz_train *.npz
robocopy shared\npz_val   data\processed\npz_val   *.npz
copy shared\train_labels.csv data\processed\
copy shared\val_labels.csv   data\processed\
```

### Notes

- Only `.npz` and CSVs are needed for training.
- `.parquet` files (if present) are for inspection and can be ignored.
- Keep class counts consistent with labels (`--num-gloss`, `--num-cat`).

### Using the same NPZs for both models

- Transformer training flags (point to the same unified folders):

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\npz_train ^
  --keypoints-val   data\processed\npz_val ^
  --labels-train-csv data\processed\train_labels.csv ^
  --labels-val-csv   data\processed\val_labels.csv
```

- IV3-GRU training flags (reuse the same unified folders):

```powershell
python -m training.train ^
  --model iv3_gru ^
  --features-train data\processed\npz_train ^
  --features-val   data\processed\npz_val ^
  --labels-train-csv data\processed\train_labels.csv ^
  --labels-val-csv   data\processed\val_labels.csv
```
