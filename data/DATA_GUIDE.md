# Data Guide

File formats and structures for the Filipino Sign Language Recognition (FSLR) pipeline.

## Directory Structure

```
data/
├── raw/                    # Original videos (fsl-105, sample-105)
├── processed/              # Preprocessed NPZ files and splits
│   ├── FSL105_train/      # FSL-105 training split (80%)
│   ├── FSL105_val/        # FSL-105 validation split (20%)
│   ├── FSL105_train.csv   # Training labels
│   ├── FSL105_val.csv     # Validation labels
│   └── labels.csv         # Main labels file
├── demo/                   # Demo clips for testing
└── splitting/              # Data splitting utilities

trained_models/
└── transformer/        # Transformer model checkpoints
└── iv3_gru/           # InceptionV3-GRU model checkpoints
```

## File Types

- **Raw**: `.mp4` video files
- **Processed**: `.npz` (NumPy archives containing keypoints and features)
- **Labels**: `.csv` files with gloss/category mappings
- **Models**: `.pt` PyTorch checkpoint files
- **Logs**: `.log` and `.csv` training logs

---

## Raw Data

### Structure

```
data/raw/
├── fsl-105/               # Main FSL dataset (105 glosses)
│   └── *.mp4
└── sample-105/            # Sample/supplementary dataset
    └── *.mp4
```

### Requirements

- Video files: `.mp4` format (OpenCV-compatible)
- Naming convention: `clip_XXXX_<gloss_text>.mp4`
- Proper lighting and framing for keypoint detection

---

## Preprocessed Data (.npz)

Each `.npz` file contains both keypoint and feature data for both models.

### Format

- **File**: `.npz` (compressed NumPy archive)
- **Keys**:
  - `X`: `[T, 178]` - MediaPipe keypoints (for Transformer)
  - `X2048`: `[T, 2048]` - InceptionV3 features (for IV3-GRU)
  - `mask`: `[T, 89]` - Keypoint visibility mask
  - `timestamps_ms`: `[T]` - Frame timestamps in milliseconds
  - `meta`: JSON metadata (filename, source, occlusion info)

### Keypoint Structure (178 dimensions)

- **Pose landmarks** (25 points): 50 dims (x, y per point)
- **Left hand** (21 points): 42 dims (x, y per point)
- **Right hand** (21 points): 42 dims (x, y per point)
- **Face mesh** (22 points): 44 dims (x, y per point)
- **Total**: 89 keypoints × 2 coordinates = 178 dimensions

### Occlusion Detection

Automatic occlusion detection during preprocessing:

- **Frame occluded** if: `visible_keypoints / 89 < 0.6` (default threshold)
- **Clip marked occluded** if:
  - Occluded frames ≥ 40% of total frames, OR
  - Consecutive occluded frames ≥ 15

For details, see [Occlusion Guide](../preprocessing/docs/OCCLUSION_GUIDE.md)

---

## Data Splitting

### Overview

The project supports this dataset configurations:

1. **fsl-105**: Main FSL dataset only

### Splitting Strategy

- **Train**: 80% of data (default)
- **Validation**: 20% of data (default)
- **Stratified**: By gloss and category for balanced splits

### Label Assignment

Before splitting, assign numeric gloss IDs and category IDs:

```powershell
python data\splitting\assign.py
```

**Purpose**: Maps gloss text to numeric IDs using `labels_reference.csv`

**Input**: `data\processed\labels.csv` (with filename and text labels)

**Output**: Updated `labels.csv` with `gloss`, `cat`, and `occluded` columns

**Reference**: `data\splitting\labels_reference.csv` (105 glosses, 10 categories)

### Data Splitting Commands

**Split fsl-105 dataset:**

```powershell
python data\splitting\data_split.py ^
  --processed-root data\processed\FSL-105 ^
  --labels data\processed\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir FSL105_train ^
  --val-dir FSL105_val ^
  --train-csv FSL105_train.csv ^
  --val-csv FSL105_val.csv
```

### Label CSV Format

```csv
file,gloss,cat,occluded
clip_0315_yes,15,1,0
clip_1601_orange,79,7,0
clip_2062_no sugar,104,9,1
```

**Columns**:

- `file`: Filename without `.npz` extension
- `gloss`: 0-based gloss ID (0-104 for 105 glosses)
- `cat`: 0-based category ID (0-9 for 10 categories)
- `occluded`: Binary flag (0 = clean, 1 = occluded)

---

## Demo Data

### Location

```
data/demo/
├── clip_0138_nice to meet you.npz
├── clip_0585_nine.npz
├── clip_1146_grandfather.npz
├── clip_1493_green.npz
├── clip_1765_fish.npz
└── clip_1912_crab.npz
```

### Purpose

Sample NPZ files for testing and demonstration without loading full dataset.

---

## Label Reference

### File

`data/splitting/labels_reference.csv`

### Format

```csv
gloss_id,label,cat_id,category
0,GOOD MORNING,0,GREETING
1,GOOD AFTERNOON,0,GREETING
15,YES,1,SURVIVAL
79,ORANGE,7,COLOR
104,NO SUGAR,9,DRINK
```

### Categories (10 total)

0. GREETING
1. SURVIVAL
2. NUMBER
3. CALENDAR
4. DAYS
5. FAMILY
6. RELATIONSHIPS
7. COLOR
8. FOOD
9. DRINK

---

## Trained Models

### Directory Structure

```
trained_models/
├── transformer/
│   ├── FSL105_classification/
│   │   ├── SignTransformer_best.pt
│   │   ├── SignTransformer_last.pt
│   │   └── *.log
│   └── FSL105_ctc/
│       ├── SignTransformerCtc_best.pt
│       ├── SignTransformerCtc_last.pt
│       └── *.log
├── iv3_gru/
│   ├── FSL105_classification/
│   │   ├── InceptionV3GRU_best.pt
│   │   ├── InceptionV3GRU_last.pt
│   │   └── *.log
│   └── FSL105_ctc/
        ├── InceptionV3GRUCtc_best.pt
        ├── InceptionV3GRUCtc_last.pt
        └── *.log
```

### Checkpoint Format (.pt)

PyTorch checkpoint with keys:

- `model`: Model state_dict
- `epoch`: Training epoch number
- `best_metric`: Best validation metric
- `optimizer`: Optimizer state (optional)
- `scheduler`: Scheduler state (optional)

### Naming Convention

- `{ModelName}_best.pt`: Best validation performance
- `{ModelName}_last.pt`: Most recent epoch
- `{ModelName}_epoch_X.pt`: Specific epoch checkpoint

For details, see [Trained Model Guide](../trained_models/TRAINED_MODEL_GUIDE.md)

---

## Model Training

### Transformer Training (Classification)

```powershell
python -m training.train ^
  --model transformer_isolated ^
  --keypoints-train data\processed\FSL105_train ^
  --keypoints-val data\processed\FSL105_val ^
  --labels-train-csv data\processed\FSL105_train.csv ^
  --labels-val-csv data\processed\FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --output-dir trained_models\transformer\FSL105_classification
```

### IV3-GRU Training (Classification)

```powershell
python -m training.train ^
  --model iv3_gru_isolated ^
  --features-train data\processed\FSL105_train ^
  --features-val data\processed\FSL105_val ^
  --labels-train-csv data\processed\FSL105_train.csv ^
  --labels-val-csv data\processed\FSL105_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --output-dir trained_models\iv3_gru\FSL105_classification
```

For details, see [Training Guide](../training/TRAINING_GUIDE.md)

---

## Validation

### NPZ Validation

Validate preprocessed files before training:

```powershell
python -m preprocessing.utils.validate_npz data\processed\FSL105_train
python -m preprocessing.utils.validate_npz data\processed\FSL105_val --require-x2048
```

### Model Validation

Validate trained models:

```powershell
python -m evaluation.validation.validate ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv
```

For details, see [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md)

---

## File Sizes

- **NPZ files**: 50KB - 2MB per file (varies with sequence length)
- **Model checkpoints**:
  - Transformer: ~50-100 MB
  - IV3-GRU: ~100-200 MB
- **Label CSVs**: 1-10 KB per split
- **Full dataset**: ~5-10 GB total

---

## Validation Checklist

### Before Training

- [ ] All `.npz` files load without errors
- [ ] Label CSV has required columns: `file`, `gloss`, `cat`, `occluded`
- [ ] Data types correct: string, int, int, int
- [ ] Gloss IDs in range [0, 104]
- [ ] Category IDs in range [0, 9]
- [ ] Occlusion flags are 0 or 1
- [ ] No missing files referenced in label CSV

### Before Evaluation

- [ ] Model checkpoint loads successfully
- [ ] Test data matches training format
- [ ] Model architecture matches checkpoint
- [ ] Correct num_gloss (105) and num_cat (10)

---

## Complete Example

```
data/
├── raw/
│   ├── fsl-105/
│   │   ├── clip_0001_hello.mp4
│   │   └── clip_0002_thank you.mp4
│   └── sample-105/
│       └── clip_0003_good morning.mp4
├── processed/
│   ├── FSL105_train/               # Training split (80%)
│   │   ├── clip_0001_hello.npz
│   │   └── clip_0003_good morning.npz
│   ├── FSL105_val/                 # Validation split (20%)
│   │   └── clip_0002_thank you.npz
│   ├── FSL105_train.csv            # file,gloss,cat,occluded
│   ├── FSL105_val.csv
│   └── labels.csv
├── demo/
│   ├── clip_0138_nice to meet you.npz
│   └── clip_0585_nine.npz
└── splitting/
    ├── assign.py
    ├── data_split.py
    └── labels_reference.csv

trained_models/
├── transformer/
│   ├── FSL105_classification/
│   │   ├── SignTransformer_best.pt
│   │   ├── SignTransformer_last.pt
│   │   └── training_*.log
│   └── FSL105_ctc/
│       ├── SignTransformerCtc_best.pt
│       └── training_*.log
└── iv3_gru/
    ├── FSL105_classification/
    │   ├── InceptionV3GRU_best.pt
    │   ├── InceptionV3GRU_last.pt
    │   └── training_*.log
    └── FSL105_ctc/
        ├── InceptionV3GRUCtc_best.pt
        └── training_*.log
```
