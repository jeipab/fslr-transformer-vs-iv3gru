# Data Guide

File formats and structures for the FSLR (Filipino Sign Language Recognition) pipeline.

## Directory Structure

```
data/
├── raw/           # Original videos and metadata
├── processed/     # Preprocessed artifacts ready for training
└── splitting/     # Data splitting utilities and configurations
```

## File Types

- **Raw**: `.mp4`, `.avi`, `.mov`, `.json`, `.csv`, `.zip`
- **Processed**: `.npz`, `.pt`, cleaned `.csv`

## Raw Data

### Structure

```
data/raw/
├── videos/
│   ├── sample_0001.mp4
│   └── sample_0002.mp4
├── labels.csv     # Main label file
└── metadata.json  # Optional: dataset information
```

### Requirements

- Video files: `.mp4`, `.avi`, `.mov` (OpenCV-supported)
- Labels: `.csv` with required columns
- Metadata: `.json` with dataset information (optional)

## Preprocessed Data (.npz)

Each `.npz` contains both keypoint and feature data for both models.

### Format

- **File**: `.npz` (compressed NumPy archive)
- **Keys**:
  - `X`: `[T, 156]` keypoints (Transformer model)
  - `X2048`: `[T, 2048]` InceptionV3 features (IV3-GRU model)
  - `mask`: `[T, 78]` keypoint visibility
  - `timestamps_ms`: `[T]` frame timestamps
  - `meta`: JSON metadata

### Keypoint Structure (156 dims)

- Pose landmarks (25 points): 50 dims
- Left hand (21 points): 42 dims
- Right hand (21 points): 42 dims
- Face mesh (11 points): 22 dims

### Occlusion Detection

- Frame occluded if visible keypoints/78 < 0.6 (default)
- Clip marked occluded if:
  - Occluded frames ≥ 40%, or
  - Consecutive occluded run ≥ 15 frames

## Data Splitting

### Overview

Data splitting is performed to create train/validation/test sets for model training and evaluation. The splitting process ensures proper distribution of classes and categories across splits.

### Splitting Strategy

- **Train**: 80% of data (default)
- **Validation**: 20% of data (default)
- **Test**: X% of data (default)

### Configuration

Splitting parameters are configured in the data splitting utilities located in `data/splitting/`:

- Stratified splitting by gloss and category
- Random seed for reproducibility
- Customizable split ratios
- Handling of occluded samples

### Output Structure

```
data/processed/
├── train/
│   ├── sample_0001.npz
│   └── sample_0002.npz
├── val/
│   └── sample_0101.npz
├── test/                    # Optional
│   └── sample_0201.npz
├── train_labels.csv
├── val_labels.csv
└── test_labels.csv
```

### Label CSV Format

```csv
file,gloss,cat,occluded
sample_0001,42,3,0
sample_0002,15,1,1
```

**Columns**:

- `file`: filename (with or without `.npz`)
- `gloss`: 0-based class ID (0 to num_gloss-1)
- `cat`: 0-based category ID (0 to num_cat-1)
- `occluded`: 0/1 flag (auto-generated during preprocessing)

## Model Training

### Checkpoints (.pt)

**Format**: PyTorch checkpoint with keys:

- `model`: model state_dict
- `epoch`: training epoch number
- `best_metric`: best validation metric
- `optimizer`: optimizer state (optional)
- `scheduler`: scheduler state (optional)

**Naming**:

- `SignTransformer_best.pt`
- `InceptionV3GRU_best.pt`
- `SignTransformer_epoch_X.pt`

### Training Logs (.csv)

```csv
epoch,train_loss,val_loss,gloss_acc,cat_acc,lr
1,2.456,2.123,0.234,0.567,0.001
2,2.134,1.987,0.289,0.612,0.001
```

### Configuration (.json)

```json
{
  "model_type": "transformer",
  "num_gloss": 105,
  "num_cat": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "num_epochs": 100
}
```

## Model Results

### Evaluation Files

- `summary_metrics_TIMESTAMP.csv`: Main performance metrics
- `gloss_per_class_TIMESTAMP.csv`: Per-class gloss performance
- `cat_per_class_TIMESTAMP.csv`: Per-class category performance
- `detailed_results_TIMESTAMP.json`: Complete results with confidence intervals
- `predictions_TIMESTAMP.csv`: All predictions with confidence scores

### Visualizations

- Confusion matrices (`.png`)
- Performance plots (HTML/images)
- Error analysis charts

## File Sizes

- **NPZ files**: 50KB-2MB per file (depends on sequence length)
- **Model checkpoints**: 10-200 MB
- **Label CSVs**: 1-10 KB per split

## Validation Checklist

### Before Training

- [ ] All `.npz` files load without errors
- [ ] Label CSV has required columns: `file,gloss,cat,occluded`
- [ ] Data types correct: string,int,int,int
- [ ] Class IDs within expected ranges (0-based)
- [ ] Occlusion flags are 0 or 1
- [ ] No missing files in label CSV

### Before Evaluation

- [ ] Model checkpoint has required keys
- [ ] Test data matches training format
- [ ] Model architecture matches checkpoint

## Complete Example

```
data/
├── raw/
│   ├── videos/
│   │   ├── gesture_001.mp4
│   │   └── gesture_002.mp4
│   └── labels.csv
├── processed/
│   ├── all/                        # Preprocessing output
│   │   ├── gesture_001.npz         # X: [45,156], X2048: [45,2048]
│   │   └── gesture_002.npz
│   ├── train/                      # After data splitting
│   │   ├── gesture_001.npz
│   │   └── gesture_002.npz
│   ├── val/
│   │   └── gesture_101.npz
│   ├── train_labels.csv            # gesture_001,12,2,0
│   ├── val_labels.csv              # gesture_101,5,1,1
│   ├── SignTransformer_best.pt     # Uses X key
│   ├── InceptionV3GRU_best.pt      # Uses X2048 key
│   ├── training_log.csv
│   └── evaluation_results_20240101_120000/
│       ├── summary_metrics.csv
│       └── predictions.csv
└── splitting/                      # Data splitting utilities
    └── split_config.json
```
