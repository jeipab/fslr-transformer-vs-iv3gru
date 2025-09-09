## Shared data dropzone

### Purpose

- A simple folder for teammates to drop preprocessed artifacts needed for training.

### Location

- `shared/`

### Expected contents

- Transformer (keypoints):
  - `shared/keypoints_train/*.npz` (each contains `X: [T,156]`)
  - `shared/keypoints_val/*.npz`
  - `shared/train_labels.csv` (columns: `file,gloss,cat`; 0-based ids)
  - `shared/val_labels.csv`
- IV3-GRU (features):
  - `shared/features_train/*.npz` (each contains `X2048: [T,2048]` or `X`)
  - `shared/features_val/*.npz`
  - `shared/train_labels.csv`
  - `shared/val_labels.csv`

### How to use

1. Teammates export/copy files into `shared/` as above.
2. Copy into `data/processed/` before training, e.g. (Windows PowerShell):

```powershell
robocopy shared\keypoints_train data\processed\keypoints_train *.npz
robocopy shared\keypoints_val   data\processed\keypoints_val   *.npz
copy shared\train_labels.csv data\processed\
copy shared\val_labels.csv   data\processed\
# If using IV3-GRU features
robocopy shared\features_train data\processed\features_train *.npz
robocopy shared\features_val   data\processed\features_val   *.npz
```

### Notes

- Only `.npz` and CSVs are needed for training.
- `.parquet` files (if present) are for inspection and can be ignored.
- Keep class counts consistent with labels (`--num-gloss`, `--num-cat`).
