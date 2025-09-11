## Data layout

### Subfolders

- `raw/`: original, unmodified data
- `processed/`: preprocessed/ready-to-use artifacts

### Common file types

- `raw/`: source files (e.g., .mp4, .jpg, .json, .zip)
- `processed/`: ready-to-train artifacts (e.g., `.npz`, `.npy`, `.pt`, cleaned `.csv`)

## Pipeline File Specifications

This section details the expected file formats and structures for each stage of the FSLR (Few-Shot Learning Recognition) pipeline.

### 1. Raw Data Stage

**Input Requirements:**
- Video files: `.mp4`, `.avi`, `.mov` (any OpenCV-supported format)
- Metadata: `.json`, `.csv`, or annotation files with labels

**Expected Structure:**
```
data/raw/
├── videos/
│   ├── sample_0001.mp4
│   ├── sample_0002.mp4
│   └── ...
├── annotations.csv       # Optional: file,gloss,category mapping
└── metadata.json        # Optional: dataset information
```

### 2. Preprocessing Stage

**Outputs Generated:**

#### 2.1 Combined Data Files (.npz)
Each .npz file contains both keypoint and feature data for use by either model.

**Format Specification:**
- **File Extension**: `.npz` (compressed NumPy archive)
- **Required Keys:**
  - `X`: keypoint data, shape `[T, 156]`, dtype `float32` (for Transformer model)
  - `X2048`: InceptionV3 features, shape `[T, 2048]`, dtype `float32` (for IV3-GRU model)
  - `mask`: keypoint visibility mask, shape `[T, 78]`, dtype `bool`
  - `timestamps_ms`: frame timestamps, shape `[T]`, dtype `int64`
  - `meta`: JSON string with metadata (video info, preprocessing params)

**Keypoint Structure (156 dimensions):**
- Pose landmarks (25 points): x,y coordinates = 50 dims
- Left hand (21 points): x,y coordinates = 42 dims
- Right hand (21 points): x,y coordinates = 42 dims
- Face mesh (11 points): x,y coordinates = 22 dims
- Total: 50 + 42 + 42 + 22 = 156 dimensions

**Feature Structure (2048 dimensions):**
- InceptionV3 CNN features extracted from original video frames
- Shape: `[T, 2048]` where T matches the keypoint sequence length

#### 2.3 Debug Files (Optional)
- **Parquet files**: `.parquet` for quick inspection of keypoint data
- **Log files**: preprocessing statistics and error reports

### 3. Data Splitting Stage

**Inputs Required:**
- Preprocessed `.npz` files from preprocessing stage
- Master labels CSV with format: `file,gloss,cat`

**Outputs Generated:**

#### 3.1 Training Data
```
data/processed/
├── train/                     # Combined data for both models
│   ├── sample_0001.npz       # Contains both X and X2048 keys
│   ├── sample_0002.npz
│   └── ...
└── train_labels.csv
```

#### 3.2 Validation Data
```
data/processed/
├── val/                       # Combined data for both models
│   ├── sample_0101.npz       # Contains both X and X2048 keys
│   └── ...
└── val_labels.csv
```

#### 3.3 Test Data (Optional)
```
data/processed/
├── test/                      # Combined data for both models
│   ├── sample_0201.npz       # Contains both X and X2048 keys
│   └── ...
└── test_labels.csv
```

**Note**: The same .npz files are used by both Transformer (using `X` key) and IV3-GRU (using `X2048` key) models. Directory names may vary based on your specific setup (e.g., `keypoints_train/`, `data_train/`, etc.)

**Label CSV Format:**
- **Columns**: `file,gloss,cat,occluded`
- **Data Types**: `string,int,int,int`
- **Requirements**:
  - `file`: filename without extension or with `.npz`
  - `gloss`: 0-based integer class ID (range: 0 to num_gloss-1)
  - `cat`: 0-based integer category ID (range: 0 to num_cat-1)
  - `occluded`: 0/1 flag (0=not occluded, 1=occluded) - optional, auto-generated during preprocessing

**Example:**
```csv
file,gloss,cat,occluded
sample_0001,42,3,0
sample_0002,15,1,1
sample_0003,88,7,0
```

**Occlusion Detection:**
- Automatically computed during preprocessing based on keypoint visibility
- A frame is considered occluded if visible keypoints/78 < threshold (default: 0.6)
- A clip is marked as occluded (1) if either:
  - Proportion of occluded frames ≥ 0.4 (default), or
  - Longest consecutive occluded run ≥ 15 frames (default)
- Currently used for analysis/filtering but ignored during model training

### 4. Model Training Stage

**Inputs Required:**
- Training datasets from data splitting stage
- Configuration parameters (num_gloss, num_cat, etc.)

**Outputs Generated:**

#### 4.1 Model Checkpoints (.pt)
**Format Specification:**
- **File Extension**: `.pt` (PyTorch checkpoint)
- **Required Keys:**
  - `model`: OrderedDict containing model state_dict
  - `epoch`: integer, training epoch number
  - `best_metric`: float, best validation metric achieved
  - `optimizer`: optimizer state_dict (optional)
  - `scheduler`: scheduler state_dict (optional)

**Naming Convention:**
- `SignTransformer_best.pt`: best performing Transformer model
- `InceptionV3GRU_best.pt`: best performing IV3-GRU model
- `SignTransformer_epoch_X.pt`: checkpoint at specific epoch
- `SignTransformer_final.pt`: final epoch checkpoint

#### 4.2 Training Logs (.csv)
**Format**: CSV with training metrics per epoch
**Columns**: `epoch,train_loss,val_loss,gloss_acc,cat_acc,lr`

**Example:**
```csv
epoch,train_loss,val_loss,gloss_acc,cat_acc,lr
1,2.456,2.123,0.234,0.567,0.001
2,2.134,1.987,0.289,0.612,0.001
```

**Note**: Training logs are separate from label CSVs. The occlusion flag in label CSVs is currently not used during training but may be used for data analysis or filtering.

#### 4.3 Configuration Files (.json)
**Contents**: Complete training configuration for reproducibility
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

### 5. Model Results Stage

**Inputs Required:**
- Trained model checkpoints (.pt files)
- Test dataset from data splitting stage

**Outputs Generated:**

#### 5.1 Evaluation Results
**Summary Metrics (.csv):**
- `summary_metrics_TIMESTAMP.csv`: Main performance metrics
- `gloss_per_class_TIMESTAMP.csv`: Per-class gloss performance
- `cat_per_class_TIMESTAMP.csv`: Per-class category performance

**Detailed Results (.json):**
- `detailed_results_TIMESTAMP.json`: Complete evaluation results with confidence intervals

**Predictions (.csv):**
- `predictions_TIMESTAMP.csv`: All model predictions with confidence scores

#### 5.2 Visualization Outputs
- Confusion matrices (`.png` files)
- Performance plots (interactive HTML or static images)
- Error analysis charts

#### 5.3 Export Reports (.txt)
- `evaluation_report_TIMESTAMP.txt`: Human-readable summary report

### File Size Guidelines

**Expected File Sizes:**
- Combined `.npz` files: 50KB-2MB per file (depends on sequence length)
  - Contains both keypoint data (~10-500 KB) and CNN features (~40KB-1.5MB)
  - Longer sequences result in proportionally larger files
- Model checkpoints: 10-200 MB (depends on architecture)
- Label CSV files: 1-10 KB per split

### Validation Checklist

**Before Training:**
- [ ] All `.npz` files load without errors
- [ ] Label CSV contains required columns: `file,gloss,cat` (and optionally `occluded`)
- [ ] Data types are correct: string,int,int (and optionally int for occluded)
- [ ] Class IDs are within expected ranges (0-based)
- [ ] Occlusion flags are 0 or 1 (if present)
- [ ] No missing files referenced in label CSV
- [ ] Train/validation splits contain expected number of samples

**Before Evaluation:**
- [ ] Model checkpoint contains required keys
- [ ] Test data follows same format as training data
- [ ] Model architecture matches checkpoint parameters
- [ ] Device compatibility (CPU/GPU) verified

### Examples

```
data/
  raw/
    videos/
      gesture_001.mp4
      gesture_002.mp4
  processed/
    0/                          # Preprocessing output directory
      gesture_001.npz           # X: [45,156], X2048: [45,2048], mask: [45,78]
      gesture_002.npz           # Contains both keypoints and features
    train/                      # After data splitting
      gesture_001.npz           # Same file, moved/copied from 0/
      gesture_002.npz
    val/
      gesture_101.npz
    test/
      gesture_201.npz
    train_labels.csv            # gesture_001,12,2,0
    val_labels.csv              # gesture_101,5,1,1
    test_labels.csv             # gesture_201,8,3,0
    SignTransformer_best.pt     # Trained on X key from .npz files
    InceptionV3GRU_best.pt      # Trained on X2048 key from same .npz files
    training_log.csv
    evaluation_results_20240101_120000/
      summary_metrics.csv
      predictions.csv
      detailed_results.json
```
