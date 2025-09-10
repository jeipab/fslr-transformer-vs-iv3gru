## Data layout

### Subfolders

- `raw/`: original, unmodified data
- `processed/`: preprocessed/ready-to-use artifacts

### Common file types

- `raw/`: source files (e.g., .mp4, .jpg, .json, .zip)
- `processed/`: ready-to-train artifacts (e.g., `.npz`, `.npy`, `.pt`, cleaned `.csv`)

### Expected training artifacts

- Transformer keypoints: `.npz` with key `X` shaped `[T,156]`
- IV3-GRU features: `.npz` with key `X2048` (or `X`) shaped `[T,2048]`
- Labels CSV format for each split: `file,gloss,cat` (0-based)

### Examples

```
data/
  raw/
    ...
  processed/
    keypoints_train/
      sample_0001.npz  # contains X: [T,156]
    keypoints_val/
      sample_0101.npz
    features_train/
      clip_0001.npz    # contains X2048: [T,2048]
    features_val/
      clip_0101.npz
    train_labels.csv   # file,gloss,cat (file without extension ok)
    val_labels.csv
```
