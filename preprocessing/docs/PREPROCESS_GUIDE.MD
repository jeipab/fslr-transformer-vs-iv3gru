# Preprocessing Guide

Convert raw videos to `.npz` files for model training and inference.

## Overview

Extracts features from sign language videos:

- **Keypoints** (`X`): MediaPipe keypoints `[T, 178]` - pose, hands, face
- **CNN Features** (`X2048`): InceptionV3 features `[T, 2048]`
- **Metadata**: Visibility masks, timestamps, occlusion flags

**Performance**: Parallel processing provides 30-50x speedup.

## Prerequisites

```powershell
pip install -r requirements.txt
```

## Data Structure

```
data/
├── raw/fsl-105/              # Input videos
├── processed/
│   ├── FSL105_train/        # Training split (80%)
│   ├── FSL105_val/          # Validation split (20%)
│   ├── FSL105_train.csv     # Training labels
│   └── FSL105_val.csv       # Validation labels
```

## Quick Start

### Single Video

```powershell
python -m preprocessing.core.preprocess ^
  video.mp4 ^
  output_dir ^
  --write-keypoints ^
  --write-iv3-features
```

### Batch Processing

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\FSL-105 ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 8 ^
  --batch-size 32
```

## Processing Pipeline

**Main Functions**:

- `process_video(video_path, out_dir, target_fps=30, out_size=256, ...)`
- `process_videos_multiprocess(video_files, out_dir, workers=None, batch_size=32, ...)`

### 1. Frame Extraction

- Sample frames at target FPS (default: 30)
- Resize to consistent dimensions (default: 256×256)

### 2. Background Removal

Function: `create_models(seg_model=1, detection_conf=0.5, tracking_conf=0.5)`

- MediaPipe Selfie Segmentation isolates signer from background
- Creates binary mask to focus on relevant features
- Improves generalization across environments

### 3. Feature Extraction

**MediaPipe Keypoints** (`X`):

Function: `extract_keypoints_from_frame(img_rgb, models, conf_thresh=0.5)`

- Pose: 25 upper body landmarks (50 dims)
- Hands: 21 landmarks per hand × 2 hands = 42 total (84 dims)
- Face: 22 key facial points (44 dims)
- Output: `[T, 178]` normalized coordinates `[0, 1]` (89 keypoints × 2)

**InceptionV3 Features** (`X2048`):

Function: `extract_iv3_features(frame_bgr, image_size=(299, 299), device=None)`  
Class: `BatchedInceptionV3Processor(device=None, batch_size=32)`

- Pretrained CNN (ImageNet, frozen)
- Global average pooling layer
- Output: `[T, 2048]` feature vectors

### 4. Post-Processing

**Gap Interpolation**:

Function: `interpolate_gaps(X, mask, max_gap=5)`

- Fill missing keypoints (max gap: 5 frames)
- Linear interpolation for temporal continuity

**Occlusion Detection**:

Function: `compute_occlusion_detection(X, mask, output_format='compatible', ...)`

- Flag hand-face interactions
- Multi-method detection with temporal filtering

**Metadata**: FPS, dimensions, processing parameters

## Performance Optimization

### Parallel Processing

**High-performance** (RTX 4090, 24GB VRAM):

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\FSL-105 ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 12 ^
  --batch-size 64
```

**Balanced** (RTX 3080, 16GB VRAM):

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\FSL-105 ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 8 ^
  --batch-size 32
```

**CPU-only**:

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\FSL-105 ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 4 ^
  --batch-size 8
```

### Parameters

- `--workers`: Parallel processes (recommended: CPU cores - 2)
- `--batch-size`: GPU batch size (32 for 16GB VRAM, 64 for 24GB+)
- `--target-fps`: Frame sampling rate (default: 30)
- `--out-size`: Frame dimensions (default: 256)
- `--disable-parquet`: Skip parquet export for faster I/O

### Benchmark

| Hardware | Workers | Batch Size | Videos/Hour |
| -------- | ------- | ---------- | ----------- |
| RTX 5090 | 12      | 64         | 800-1200    |
| RTX 4090 | 8       | 48         | 600-900     |
| RTX 3080 | 6       | 32         | 400-600     |

## Output Format

### NPZ File Structure

Each `.npz` file contains:

- `X`: `[T, 178]` - MediaPipe keypoints (normalized to [0, 1])
- `X2048`: `[T, 2048]` - InceptionV3 features
- `mask`: `[T, 89]` - Keypoint visibility (boolean)
- `timestamps_ms`: `[T]` - Frame timestamps (milliseconds)
- `meta`: JSON metadata (FPS, dimensions, occlusion flag)

**Keypoint Layout** (178 dimensions):

- Pose: 25 points × 2 coords = 50
- Left hand: 21 points × 2 coords = 42
- Right hand: 21 points × 2 coords = 42
- Face: 22 points × 2 coords = 44
- **Total**: 89 keypoints × 2 coordinates = 178 dimensions

## Labels CSV Workflow

### 1. Generate Initial Labels

```python
import os, json
import numpy as np
import pandas as pd

processed_dir = 'data/processed/FSL-105'
files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]

data = []
for f in files:
    npz_data = np.load(os.path.join(processed_dir, f), allow_pickle=True)
    meta = json.loads(str(npz_data['meta']))
    data.append({
        'file': f.replace('.npz', ''),
        'occluded': meta.get('occluded_flag', 0)
    })

pd.DataFrame(data).to_csv(f'{processed_dir}/labels.csv', index=False)
```

### 2. Add Gloss and Category IDs

```powershell
copy data\processed\FSL-105\labels.csv data\processed\labels.csv
python data\splitting\assign.py
copy data\processed\labels.csv data\processed\FSL-105\labels.csv
```

Result:

```csv
file,gloss,cat,occluded
clip_0001_hello,3,0,0
clip_0002_thank you,7,0,1
```

### 3. Split Train/Val

```powershell
python data\splitting\data_split.py ^
  --processed-root data\processed\FSL-105 ^
  --labels data\processed\FSL-105\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir FSL105_train ^
  --val-dir FSL105_val ^
  --train-csv FSL105_train.csv ^
  --val-csv FSL105_val.csv
```

## Occlusion Detection

Function: `compute_occlusion_detection(video_path=None, X=None, mask_bool_array=None, output_format='compatible', ...)`

Enabled automatically with `--write-keypoints`.

**Detection Methods**:

- Fingertip intersection with face regions
- Palm proximity analysis
- Hand trajectory tracking
- Multi-point hand orientation

**Temporal Filtering**:

- Requires 5 consecutive frames (allows 2 skips)
- Minimum confidence: 0.2 per frame
- Reduces false positives from tracking failures

**Output**: Binary flag in `meta['occluded_flag']` (0 = clean, 1 = occluded)

Details: [Occlusion Guide](OCCLUSION_GUIDE.md)

## Validation

Check NPZ files:

```powershell
python -m preprocessing.utils.validate_npz data\processed\FSL105_train
python -m preprocessing.utils.validate_npz data\processed\FSL105_val --require-x2048
```

Test training:

```powershell
python -m training.train --model transformer_isolated --smoke-test --num-gloss 105 --num-cat 10
```

## Common Issues

| Issue              | Solution                                                    |
| ------------------ | ----------------------------------------------------------- |
| Mismatched names   | CSV `file` column must match NPZ filenames (without `.npz`) |
| Wrong shape        | Check `X` is `[T, 178]` and `X2048` is `[T, 2048]`          |
| Label range errors | `gloss` in [0, 104], `cat` in [0, 9] (zero-indexed)         |
| Empty sequences    | Preprocessing skips videos with no valid frames             |
| Memory errors      | Reduce `--batch-size` and `--workers`                       |
| Slow processing    | Use `--workers 8` and GPU acceleration                      |

## FAQ

**CSV and NPZ relationship?**  
CSV links to NPZ via `file` column (basename without extension).

**Keypoint order?**  
`pose25, left_hand21, right_hand21, face22` - normalized to [0, 1].

**Use X2048 with Transformer?**  
Yes, Transformer supports keypoints (178-D), features (2048-D), or both (2226-D).

**Processing speed?**  
8 workers + batch size 32 on RTX 3080: 400-600 videos/hour.

## Related Guides

- [Data Guide](../../data/DATA_GUIDE.md) - Data organization and splitting
- [Occlusion Guide](OCCLUSION_GUIDE.md) - Occlusion detection details
- [Training Guide](../../training/TRAINING_GUIDE.md) - Model training
- [Vast.ai Guide](../../shared/for vast ai/PREPROCESS_VAST.md) - Remote processing
