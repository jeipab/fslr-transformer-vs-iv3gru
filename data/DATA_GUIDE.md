# Data Guide

File formats and structures for the Filipino Sign Language Recognition (FSLR) pipeline.

## Directory Structure

```
data/
├── raw/                    # Original videos (fsl-105, sample-105)
├── processed/              # Preprocessed NPZ files and splits
│   ├── fsl-105_10-08/     # Processed fsl-105 source
│   ├── smp-105_10-08/     # Processed sample-105 source
│   ├── fsl_train/         # FSL-105 training split
│   ├── fsl_val/           # FSL-105 validation split
│   ├── smp_train/         # Sample-105 training split
│   ├── smp_val/           # Sample-105 validation split
│   ├── cmb_train/         # Combined training split
│   ├── cmb_val/           # Combined validation split
│   └── *.csv              # Label files for each split
├── demo/                   # Demo clips for testing
└── splitting/              # Data splitting utilities

trained_models/
└── cmb/                    # Models trained on combined dataset
    ├── transformer/        # Transformer model checkpoints
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
  - `X`: `[T, 156]` - MediaPipe keypoints (for Transformer)
  - `X2048`: `[T, 2048]` - InceptionV3 features (for IV3-GRU)
  - `mask`: `[T, 78]` - Keypoint visibility mask
  - `timestamps_ms`: `[T]` - Frame timestamps in milliseconds
  - `meta`: JSON metadata (filename, source, occlusion info)

### Keypoint Structure (156 dimensions)

- **Pose landmarks** (25 points): 50 dims (x, y per point)
- **Left hand** (21 points): 42 dims (x, y per point)
- **Right hand** (21 points): 42 dims (x, y per point)
- **Face mesh** (11 points): 22 dims (x, y per point)

### Occlusion Detection

Automatic occlusion detection during preprocessing:

- **Frame occluded** if: `visible_keypoints / 78 < 0.6` (default threshold)
- **Clip marked occluded** if:
  - Occluded frames ≥ 40% of total frames, OR
  - Consecutive occluded frames ≥ 15

For details, see [Occlusion Guide](../preprocessing/docs/OCCLUSION_GUIDE.md)

---

## Data Splitting

### Overview

The project supports three dataset configurations:

1. **fsl-105**: Main FSL dataset only
2. **sample-105**: Supplementary dataset only
3. **cmb**: Combined dataset (fsl-105 + sample-105)

Current trained models use the combined dataset (`cmb`).

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

**Split sample-105 dataset:**

```powershell
python data\splitting\data_split.py ^
  --processed-root data\processed\smp-105_10-08 ^
  --labels data\processed\smp-105_10-08\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir smp_train ^
  --val-dir smp_val ^
  --train-csv smp_train.csv ^
  --val-csv smp_val.csv
```

**Split combined dataset:**

```powershell
python data\splitting\data_split.py ^
  --processed-root data\processed\fsl-105_10-08 data\processed\smp-105_10-08 ^
  --labels data\processed\fsl-105_10-08\labels.csv data\processed\smp-105_10-08\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir cmb_train ^
  --val-dir cmb_val ^
  --train-csv cmb_train.csv ^
  --val-csv cmb_val.csv
```

### Output Structure

```
data/processed/
├── cmb_train/             # Combined training set
│   └── *.npz
├── cmb_val/               # Combined validation set
│   └── *.npz
├── cmb_train.csv          # Training labels
└── cmb_val.csv            # Validation labels
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
15,YES,1,COMMON
79,ORANGE,7,FOOD
104,NO SUGAR,9,HEALTH
```

### Categories (10 total)

0. GREETING
1. COMMON
2. FAMILY
3. NATURE
4. COLORS
5. NUMBERS
6. PEOPLE
7. FOOD
8. TIME
9. HEALTH

---

## Trained Models

### Directory Structure

```
trained_models/cmb/
├── transformer/
│   ├── SignTransformer_best.pt
│   ├── SignTransformer_last.pt
│   └── *.log
└── iv3_gru/
    ├── InceptionV3GRU_best.pt
    ├── InceptionV3GRU_last.pt
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

### Transformer Training

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

### IV3-GRU Training

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

For details, see [Training Guide](../training/TRAINING_GUIDE.md)

---

## Validation

### NPZ Validation

Validate preprocessed files before training:

```powershell
python -m preprocessing.utils.validate_npz data\processed\cmb_train
python -m preprocessing.utils.validate_npz data\processed\cmb_val --require-x2048
```

### Model Validation

Validate trained models:

```powershell
python -m evaluation.validation.validate ^
  --model-type transformer ^
  --model-path trained_models\cmb\transformer\SignTransformer_best.pt ^
  --data-dir data\processed\cmb_val ^
  --labels-csv data\processed\cmb_val.csv
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
│   ├── fsl-105_10-08/              # Preprocessing output
│   │   ├── clip_0001_hello.npz
│   │   ├── clip_0002_thank you.npz
│   │   └── labels.csv
│   ├── smp-105_10-08/
│   │   ├── clip_0003_good morning.npz
│   │   └── labels.csv
│   ├── cmb_train/                  # Combined training split (80%)
│   │   ├── clip_0001_hello.npz
│   │   └── clip_0003_good morning.npz
│   ├── cmb_val/                    # Combined validation split (20%)
│   │   └── clip_0002_thank you.npz
│   ├── cmb_train.csv               # file,gloss,cat,occluded
│   └── cmb_val.csv
├── demo/
│   ├── clip_0138_nice to meet you.npz
│   └── clip_0585_nine.npz
└── splitting/
    ├── assign.py
    ├── data_split.py
    └── labels_reference.csv

trained_models/cmb/
├── transformer/
│   ├── SignTransformer_best.pt
│   ├── SignTransformer_last.pt
│   └── transformer_train.log
└── iv3_gru/
    ├── InceptionV3GRU_best.pt
    ├── InceptionV3GRU_last.pt
    └── iv3_train.log
```
