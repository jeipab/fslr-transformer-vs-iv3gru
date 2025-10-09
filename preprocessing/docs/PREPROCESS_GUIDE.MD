# Preprocessing Guide

Convert raw sign language videos to training-ready data for both Transformer and IV3-GRU models.

> **Performance**: Use `--workers` parameter for 30-50x faster parallel processing.

## Overview

This preprocessing pipeline extracts two types of features from sign language videos:

- **Transformer input**: `.npz` files with MediaPipe keypoints `X` shaped `[T,156]` (78 keypoints × 2D coordinates)
- **IV3-GRU input**: `.npz` files with InceptionV3 CNN features `X2048` shaped `[T,2048]`
- **Metadata**: `mask` (keypoint visibility), `timestamps_ms`, `meta` JSON, optional `.parquet`
- **Occlusion detection**: Automatic frame and clip-level occlusion flagging

## Prerequisites

```powershell
pip install -r requirements.txt
pip install pyarrow  # optional, for parquet inspection
```

## Data Structure

```
data/
├── raw/
│   ├── fsl-105/              # Main dataset videos
│   └── sample-105/           # Supplementary dataset videos
├── processed/
│   ├── fsl-105_10-08/       # Processed fsl-105 with labels.csv
│   ├── smp-105_10-08/       # Processed sample-105 with labels.csv
│   ├── cmb_train/           # Combined training split (80%)
│   ├── cmb_val/             # Combined validation split (20%)
│   ├── cmb_train.csv        # Training labels
│   └── cmb_val.csv          # Validation labels
```

## Processing Videos

### Preprocessing Module

The `preprocessing.core.preprocess` module provides both sequential and parallel processing:

- **Sequential Processing**: `--workers 1` (default) - Process videos one by one
- **Parallel Processing**: `--workers N` - Process N videos simultaneously with GPU optimization
- **Batch GPU Processing**: Automatic batched InceptionV3 inference for efficiency
- **Occlusion Detection**: Automatic detection when `--write-keypoints` is used

### Single Video

```powershell
# Extract both keypoints and CNN features (sequential)
python -m preprocessing.core.preprocess ^
  video.mp4 ^
  output_dir ^
  --write-keypoints ^
  --write-iv3-features

# Extract with parallel processing (8 workers)
python -m preprocessing.core.preprocess ^
  video.mp4 ^
  output_dir ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 8 ^
  --batch-size 32
```

### Batch Processing (Directory)

```powershell
# Process entire directory of videos
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\fsl-105_10-08 ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 8 ^
  --batch-size 32 ^
  --target-fps 30 ^
  --out-size 256
```

After batch processing:

1. Create `labels.csv` in the output directory
2. Run `assign.py` to add gloss and category IDs
3. Use `data_split.py` to create train/val splits

## Performance Optimization

### Multi-Process Parallel Processing (30-50x Speedup)

For large datasets, use parallel processing with optimized parameters:

```powershell
# High-performance configuration (RTX 4090/5090, 24GB+ VRAM)
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\fsl-105_output ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 12 ^
  --batch-size 64 ^
  --target-fps 15 ^
  --disable-parquet

# Balanced configuration (RTX 3080/4080, 10-16GB VRAM)
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\fsl-105_output ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 8 ^
  --batch-size 32 ^
  --target-fps 20

# CPU-only processing
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\fsl-105_output ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 4 ^
  --batch-size 8 ^
  --target-fps 30
```

### Parameter Guidelines

- **`--workers`**: Number of parallel processes (recommended: CPU cores - 2)
- **`--batch-size`**: GPU batch size for InceptionV3 (32 for 16GB VRAM, 64 for 24GB+)
- **`--target-fps`**: Lower FPS = faster processing (15-30 recommended)
- **`--disable-parquet`**: Skip parquet files for maximum I/O speed
- **`--out-size`**: Image size for InceptionV3 (default: 256)

### Expected Performance

| Hardware               | Workers | Batch Size | Videos/Hour |
| ---------------------- | ------- | ---------- | ----------- |
| RTX 5090 + 12-core CPU | 12      | 64         | 800-1200    |
| RTX 4090 + 8-core CPU  | 8       | 48         | 600-900     |
| RTX 3080 + 6-core CPU  | 6       | 32         | 400-600     |

## NPZ File Contents

Each processed `.npz` file contains:

- `X`: `[T,156]` MediaPipe keypoints (pose25, left_hand21, right_hand21, face11)
- `X2048`: `[T,2048]` InceptionV3 CNN features
- `mask`: `[T,78]` keypoint visibility mask (boolean)
- `timestamps_ms`: `[T]` frame timestamps in milliseconds
- `meta`: JSON metadata with processing parameters and occlusion info

**Keypoint Structure (156 dimensions):**

- Pose: 25 landmarks × 2 coords = 50 dims
- Left hand: 21 landmarks × 2 coords = 42 dims
- Right hand: 21 landmarks × 2 coords = 42 dims
- Face: 11 landmarks × 2 coords = 22 dims

## Creating Labels CSV

After preprocessing, create a `labels.csv` file in the output directory:

```csv
file,occluded
clip_0001_hello,0
clip_0002_thank you,1
clip_0003_good morning,0
```

### Step 1: Generate Initial Labels

```python
import os
import numpy as np
import pandas as pd
import json

# Get all npz files in processed directory
processed_dir = 'data/processed/fsl-105_10-08'
files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]

# Extract occluded flag from each NPZ file's metadata
data = []
for f in files:
    npz_path = os.path.join(processed_dir, f)
    npz_data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(npz_data['meta']))
    occluded = meta.get('occluded_flag', 0)
    data.append({'file': f.replace('.npz', ''), 'occluded': occluded})

df = pd.DataFrame(data)
df.to_csv(f'{processed_dir}/labels.csv', index=False)
print(f'Created labels.csv with {len(files)} files')
```

### Step 2: Assign Gloss and Category IDs

```powershell
# Copy labels to main processed directory
copy data\processed\fsl-105_10-08\labels.csv data\processed\labels.csv

# Run assign script to add gloss and category IDs
python data\splitting\assign.py

# Copy updated labels back
copy data\processed\labels.csv data\processed\fsl-105_10-08\labels.csv
```

This updates the CSV to:

```csv
file,gloss,cat,occluded
clip_0001_hello,3,0,0
clip_0002_thank you,7,0,1
clip_0003_good morning,0,0,0
```

### Step 3: Split Data

```powershell
# Split into train/val sets
python data\splitting\data_split.py ^
  --processed-root data\processed\fsl-105_10-08 ^
  --labels data\processed\fsl-105_10-08\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir fsl_train ^
  --val-dir fsl_val ^
  --train-csv fsl_train.csv ^
  --val-csv fsl_val.csv
```

For combined datasets, see [Data Guide](../../data/DATA_GUIDE.md).

## Occlusion Detection

Automatically enabled when `--write-keypoints` is used:

**Frame-level occlusion:**

- Frame marked as occluded if `visible_keypoints / 78 < 0.6` (default threshold)

**Clip-level occlusion:**

- Clip marked as occluded if:
  - Occluded frames ≥ 40% of total frames, OR
  - Consecutive occluded run ≥ 15 frames

For detailed information, see [Occlusion Guide](OCCLUSION_GUIDE.md).

## Training Integration

### Transformer Model

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32
```

### IV3-GRU Model

```powershell
python -m training.train ^
  --model iv3_gru ^
  --features-train data\processed\cmb_train ^
  --features-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32
```

For detailed training instructions, see [Training Guide](../../training/TRAINING_GUIDE.md).

## Validation

### Validate NPZ Files

```powershell
# Check NPZ file structure and contents
python -m preprocessing.utils.validate_npz data\processed\cmb_train

# Check with InceptionV3 features requirement
python -m preprocessing.utils.validate_npz data\processed\cmb_val --require-x2048
```

### Quick Smoke Test

```powershell
# Test training pipeline
python -m training.train ^
  --model transformer ^
  --smoke-test ^
  --num-gloss 105 ^
  --num-cat 10
```

## Common Issues

**Mismatched names:**

- CSV `file` entries must match NPZ filenames (without `.npz` extension)

**Wrong shape:**

- Ensure `X` is `[T,156]` for keypoints
- Ensure `X2048` is `[T,2048]` for InceptionV3 features

**Label ranges:**

- `gloss` in `[0, 104]` (105 classes: 0-based)
- `cat` in `[0, 9]` (10 categories: 0-based)

**Empty sequences:**

- Preprocessing skips clips with < 1 valid frame
- Check video quality and MediaPipe detection

**Memory errors:**

- Reduce `--batch-size` for GPU processing
- Reduce `--workers` for CPU/RAM constraints

## FAQ

**Q: Are CSVs separate from NPZ files?**
A: Yes, they are linked by the `file` column (basename without extension).

**Q: Can I use X2048 with Transformer?**
A: Yes, extra keys in NPZ files are ignored by the training pipeline.

**Q: What is the keypoint order?**
A: `pose25, left_hand21, right_hand21, face11` - all normalized to `[0,1]`.

**Q: How do I process videos on Vast.ai?**
A: See [Vast.ai Preprocessing Guide](../../shared/for vast ai/PREPROCESS_VAST.md).

**Q: How long does preprocessing take?**
A: With 8 workers and batch size 32, expect 400-600 videos/hour on RTX 3080.

## Additional Resources

- [Data Guide](../../data/DATA_GUIDE.md): Data organization and splitting
- [Occlusion Guide](OCCLUSION_GUIDE.md): Occlusion detection details
- [Training Guide](../../training/TRAINING_GUIDE.md): Training instructions
- [Vast.ai Guide](../../shared/for vast ai/PREPROCESS_VAST.md): Remote preprocessing
