# PANSINAYAN Training Pipeline Documentation

## Complete Machine Learning Pipeline from Data to Trained Models

**Document Purpose**: This document describes the complete end-to-end machine learning pipeline for PANSINAYAN, covering data preprocessing, splitting, training, and model evaluation. This is distinct from the Streamlit tool architecture and focuses on the research and development workflow.

---

## Table of Contents

1. [Data Collection & Preparation](#1-data-collection--preparation)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Data Splitting Strategy](#3-data-splitting-strategy)
4. [Training Workflow](#4-training-workflow)
5. [Model Evaluation](#5-model-evaluation)
6. [Complete Pipeline Flow](#6-complete-pipeline-flow)

---

## 1. Data Collection & Preparation

### 1.1 FSL-105 Dataset Overview

**Source**: Publicly available Filipino Sign Language dataset (Tupal, 2023)

**Key Statistics**:

- **Total videos**: 2,130 samples
- **Sign vocabulary**: 105 distinct FSL glosses
- **Semantic categories**: 10 thematic groups
- **Signers**: 4 native Deaf FSL users
- **Recording duration**: 4 seconds per video
- **Recording quality**: 1920×1080 pixels at 60 FPS

**Semantic Categories** (10 total):

1. GREETING (Hello, Good morning, etc.)
2. SURVIVAL (Yes, No, I understand, etc.)
3. NUMBER (One through Ten)
4. CALENDAR (Months)
5. DAYS (Days of the week, Today, Tomorrow, Yesterday)
6. FAMILY (Father, Mother, Grandfather, etc.)
7. RELATIONSHIPS (Boy, Girl, Deaf, Married, etc.)
8. COLOR (Red, Blue, Green, etc.)
9. FOOD (Rice, Bread, Fish, etc.)
10. DRINK (Coffee, Tea, Water, etc.)

### 1.2 Raw Data Structure

```
data/raw/fsl-105/
├── video_0001.mp4  (greeting_goodmorning_signer1_rep1)
├── video_0002.mp4  (greeting_goodmorning_signer1_rep2)
├── video_0003.mp4  (greeting_goodmorning_signer2_rep1)
└── ... (2,130 total videos)
```

Each video:

- Original format: .mp4 or .MOV
- Resolution: 1920×1080 pixels
- Frame rate: 60 FPS
- Duration: ~4 seconds
- Contains: Single signer performing one sign

---

## 2. Preprocessing Pipeline

### 2.1 Overview

The preprocessing pipeline transforms raw video files into structured feature representations suitable for deep learning models. This is a **one-time process** that prepares the dataset for training.

**Input**: Raw video files (.mp4, .mov)  
**Output**: Compressed .npz files containing keypoints, features, and metadata

### 2.2 Processing Steps

#### Step 1: Video Downsampling

**Purpose**: Reduce computational requirements while preserving sign language dynamics

**Process**:

```python
# Temporal downsampling
Original FPS: 60 → Target FPS: 30
Total frames: 240 → Sampled frames: ~120 (4 seconds × 30 FPS)

# Spatial resizing
Original: 1920×1080 → Resized: 256×256 (for keypoint extraction)
```

**Rationale**: 30 FPS captures sign language gestures adequately (most signs occur over 0.5-2 seconds), and 256×256 provides sufficient resolution for MediaPipe while reducing processing time by ~80%.

#### Step 2: Background Removal

**Tool**: MediaPipe Selfie Segmentation

**Process**:

```python
1. Run segmentation model on each frame
2. Generate binary mask (person=1, background=0)
3. Apply mask to isolate signer
4. Replace background with black (or transparent)
```

**Purpose**:

- Focus model attention on signer
- Remove environment-specific features that could cause overfitting
- Improve generalization across different recording locations
- Reduce visual complexity for downstream processing

**Technical Details**:

- Model: Lightweight semantic segmentation CNN
- Threshold: Pixels with probability > 0.5 classified as person
- Inference speed: ~10ms per frame on CPU

#### Step 3: Keypoint Extraction

**Tool**: MediaPipe Holistic

**Keypoint Structure** (78 total landmarks):

```
Pose (Upper Body):   25 landmarks
├─ Face region:      11 landmarks (nose, eyes, ears, mouth)
├─ Shoulders/Arms:   8 landmarks (shoulders, elbows, wrists)
├─ Hand bases:       4 landmarks (index/pinky base per hand)
└─ Torso:            2 landmarks (hips)

Left Hand:           21 landmarks
├─ Wrist:            1 landmark
├─ Thumb:            4 landmarks (joints)
├─ Index:            4 landmarks
├─ Middle:           4 landmarks
├─ Ring:             4 landmarks
└─ Pinky:            4 landmarks

Right Hand:          21 landmarks (same structure as left)

Face (Key Points):   11 landmarks
├─ Eyes:             4 landmarks (inner/outer left/right)
├─ Nose:             1 landmark (tip)
├─ Mouth:            2 landmarks (corners)
├─ Forehead:         1 landmark
├─ Chin:             1 landmark
└─ Cheeks:           2 landmarks

Total: 78 landmarks × 2 coordinates (x, y) = 156-dimensional vector
```

**Coordinate System**:

- MediaPipe outputs landmarks normalized to [0, 1] range
- (0, 0) = top-left corner of frame
- (1, 1) = bottom-right corner of frame
- **Translation invariant**: Signer position doesn't affect coordinates
- **Scale invariant**: Distance from camera doesn't affect values

**Visibility Tracking**:

- Each landmark has a visibility/confidence score
- Mask array [78] tracks which keypoints are reliably detected
- Missing/low-confidence keypoints marked as False in mask

#### Step 4: InceptionV3 Feature Extraction

**Tool**: Pretrained InceptionV3 (ImageNet weights)

**Process**:

```python
1. Resize frames to 299×299 (standard InceptionV3 input size)
2. Convert BGR → RGB
3. Normalize using ImageNet statistics:
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]
4. Forward pass through InceptionV3 backbone
5. Extract 2048-D feature vector from final layer
```

**Purpose**:

- Capture high-level visual patterns (shapes, edges, textures)
- Transfer learning from ImageNet provides robust representations
- Complements skeletal keypoints with holistic frame context

**GPU Acceleration**:

- Batched processing: 32 frames processed simultaneously
- 10-100× speedup on GPU vs CPU
- Automatic device detection and utilization

#### Step 5: Gap Interpolation

**Purpose**: Fill short gaps in keypoint sequences caused by momentary tracking failures

**Process**:

```python
For each keypoint across the sequence:
1. Identify valid (detected) frames
2. Find gaps between valid detections
3. If gap length ≤ 5 frames:
   - Linearly interpolate x, y coordinates
   - Update visibility mask to mark as filled
4. Skip gaps > 5 frames (too large to reliably interpolate)
```

**Rationale**: Short gaps (1-5 frames) are usually tracking failures, not actual missing keypoints. Linear interpolation maintains temporal continuity without introducing significant artifacts.

#### Step 6: Occlusion Detection

**Purpose**: Identify videos where hands obscure facial features (quality assessment)

**Method**: Multi-method detection with temporal consistency filtering

**Detection Algorithms**:

1. **Direct Intersection**: Check if fingertips enter face regions
2. **Proximity Analysis**: Measure palm-to-face distance
3. **Trajectory Tracking**: Analyze hand movement toward face
4. **Orientation Detection**: Check if hand is oriented toward face

**Temporal Filtering**:

- Require 5 consecutive frames with positive detection
- Allow up to 2 missed frames within window
- Minimum confidence: 0.2 per frame

**Binary Flag**:

- `0` = Clean video (no persistent occlusion)
- `1` = Occluded video (hands temporarily obscure face)

**Why This Matters**: Allows evaluation of model performance under clean vs occluded conditions, which is critical for assessing robustness in real-world sign language scenarios.

#### Step 7: NPZ File Generation

**Output Format**:

```python
# Compressed .npz file containing:
{
  'X': [T, 156] float32,           # Keypoint coordinates (normalized [0,1])
  'X2048': [T, 2048] float32,      # InceptionV3 features
  'mask': [T, 78] bool,             # Keypoint visibility flags
  'timestamps_ms': [T] int64,       # Frame timestamps in milliseconds
  'meta': JSON string               # Processing parameters and flags
}

# Metadata includes:
{
  'video': 'original_filename.mp4',
  'target_fps': 30,
  'out_size': 256,
  'dims_per_frame': 156,
  'keypoints_total': 78,
  'order': 'pose25,left_hand21,right_hand21,face11',
  'conf_thresh': 0.5,
  'interpolation_max_gap': 5,
  'occluded_flag': 0 or 1
}
```

**File Naming**: Same as original video filename (e.g., `video_0001.npz`)

**Compression**: Uses `np.savez_compressed()` for 3-5× file size reduction

### 2.3 Batch Processing Commands

**Single Video**:

```bash
python preprocessing/core/preprocess.py \
  video.mp4 output_dir \
  --write-keypoints \
  --write-iv3-features \
  --target-fps 30 \
  --out-size 256
```

**Directory of Videos (Parallel)**:

```bash
python preprocessing/core/preprocess.py \
  input_dir/ output_dir/ \
  --write-keypoints \
  --write-iv3-features \
  --workers 8 \
  --batch-size 32 \
  --target-fps 30
```

**Performance**:

- Sequential: ~30-60 seconds per video
- Parallel (8 workers, GPU): ~3-5 seconds per video
- Full FSL-105 dataset: ~2-3 hours on GPU workstation

---

## 3. Data Splitting Strategy

### 3.1 Split Methodology

**File**: `data/splitting/data_split.py`

**Approach**: Stratified 80/20 train-validation split with hash-based determinism

**Purpose**: Ensure both training and validation sets contain:

- All 105 glosses represented
- All 4 signers represented
- Balanced class distribution
- No temporal leakage (same video never in both sets)

### 3.2 Stratified Splitting

**Algorithm**:

```python
For each of the 105 glosses:
├─ Get all samples for this gloss (~20 samples)
├─ Hash-based assignment to train/val (deterministic)
├─ Target: 80% to train, 20% to val
└─ Result: ~16 train samples, ~4 val samples per gloss

Total:
├─ Training: ~1,704 samples (80%)
└─ Validation: ~426 samples (20%)
```

**Hash-Based Determinism**:

```python
def hash_to_split(filename, train_ratio=0.8):
    hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return "train" if (hash_val % 100) < (train_ratio * 100) else "val"
```

**Benefits**:

- Deterministic: Same file always goes to same split
- Consistent across different runs
- Enables dataset combination without conflicts
- Reproducible experiments

### 3.3 Output Structure

**Directory Organization**:

```
data/processed/
├── fsl_train/              # Training NPZ files
│   ├── video_0001.npz
│   ├── video_0003.npz
│   └── ... (~1,704 files)
├── fsl_train.csv           # Training labels
├── fsl_val/                # Validation NPZ files
│   ├── video_0002.npz
│   ├── video_0005.npz
│   └── ... (~426 files)
└── fsl_val.csv             # Validation labels
```

**Labels CSV Format**:

```csv
file,gloss,cat,occluded
video_0001,0,0,0
video_0003,0,0,1
video_0010,1,0,0
...
```

Where:

- `file`: NPZ filename (without extension)
- `gloss`: Gloss class ID (0-104)
- `cat`: Category class ID (0-9)
- `occluded`: Occlusion flag (0=clean, 1=occluded)

### 3.4 Running the Split

**Command**:

```bash
python data/splitting/data_split.py \
  --processed-root data/processed/all_preprocessed \
  --labels data/processed/all_labels.csv \
  --out-root data/processed \
  --copy \
  --train-ratio 0.8 \
  --train-dir fsl_train \
  --val-dir fsl_val \
  --train-csv fsl_train.csv \
  --val-csv fsl_val.csv
```

**Result**:

- Creates separate train/val directories
- Generates corresponding CSV files
- Maintains class balance across splits
- Ready for training

---

## 4. Training Workflow

### 4.1 Dataset Loading

**File**: `training/train.py`

**Dataset Classes**:

1. **`FSLKeypointFileDataset`**: Loads keypoint sequences [T, 156]
2. **`FSLFeatureFileDataset`**: Loads InceptionV3 features [T, 2048]

**DataLoader Configuration**:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=16,           # Training batch size
    shuffle=True,            # Randomize sample order
    num_workers=4,           # Parallel data loading
    pin_memory=True          # GPU optimization
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,           # Larger batches for validation
    shuffle=False,           # Keep consistent order
    num_workers=4,
    pin_memory=True
)
```

**Variable-Length Sequence Handling**:

```python
# Sequences have different lengths (60-150 frames typically)
# DataLoader pads to longest sequence in batch
# Attention masks track real vs padded positions
```

### 4.2 Model Initialization

#### Transformer Model

```python
model = SignTransformer(
    input_dim=156,           # Keypoint features
    emb_dim=256,            # Embedding dimension
    n_heads=8,              # Attention heads
    n_layers=4,             # Encoder layers
    num_gloss=105,          # Sign classes
    num_cat=10,             # Category classes
    dropout=0.1,            # Regularization
    max_len=300,            # Max sequence length
    pooling_method='mean'   # Sequence pooling strategy
)
```

**Note**: Configuration was selected after experimentation with larger models (768-dim, 6-layers) showed no significant improvement on the FSL-105 dataset size.

#### InceptionV3-GRU Model

```python
model = InceptionV3GRU(
    num_gloss=105,          # Sign classes
    num_cat=10,             # Category classes
    hidden1=16,             # First GRU hidden size
    hidden2=12,             # Second GRU hidden size
    dropout=0.3,            # Regularization
    pretrained_backbone=True,   # Use ImageNet weights
    freeze_backbone=True    # Freeze CNN (transfer learning)
)
```

### 4.3 Training Configuration

#### Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,                # Learning rate
    weight_decay=1e-5       # L2 regularization
)
```

**Why Adam?** Adaptive learning rates per parameter, works well with sparse gradients, and requires minimal hyperparameter tuning.

#### Loss Function

**Multi-Task Loss**:

```python
loss = loss_gloss + 0.5 * loss_category

where:
  loss_gloss = CrossEntropyLoss(pred_gloss, true_gloss)
  loss_category = CrossEntropyLoss(pred_category, true_category)
```

**Rationale**: Category task is easier (10 classes vs 105), so downweighting (0.5×) prevents it from dominating the training signal while still providing regularization benefits.

#### Learning Rate Scheduling

**Strategy**: ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',             # Monitor loss (minimize)
    factor=0.5,             # Reduce LR by 50%
    patience=5,             # Wait 5 epochs before reduction
    verbose=True            # Print LR changes
)
```

**Purpose**: Automatically reduce learning rate when validation loss plateaus, allowing the model to fine-tune in later epochs.

#### Early Stopping

**Configuration**:

```python
early_stopping_patience = 10  # Stop if no improvement for 10 epochs
best_val_loss = float('inf')
patience_counter = 0
```

**Purpose**: Prevent overfitting by stopping training when validation performance stops improving.

### 4.4 Training Loop

**Main Training Process**:

```python
max_epochs = 50

for epoch in range(max_epochs):
    # === TRAINING PHASE ===
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        videos, gloss_labels, category_labels = batch

        # Forward pass
        gloss_pred, category_pred = model(videos)

        # Compute combined loss
        loss_gloss = criterion_gloss(gloss_pred, gloss_labels)
        loss_category = criterion_category(category_pred, category_labels)
        loss = loss_gloss + 0.5 * loss_category

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        train_loss += loss.item()

    # === VALIDATION PHASE ===
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            videos, gloss_labels, category_labels = batch
            gloss_pred, category_pred = model(videos)

            loss = criterion_gloss(gloss_pred, gloss_labels) + \
                   0.5 * criterion_category(category_pred, category_labels)
            val_loss += loss.item()

    # Calculate averages
    train_loss_avg = train_loss / len(train_loader)
    val_loss_avg = val_loss / len(val_loader)

    # Logging
    print(f"Epoch {epoch+1}/{max_epochs}")
    print(f"  Train Loss: {train_loss_avg:.4f}")
    print(f"  Val Loss: {val_loss_avg:.4f}")

    # Learning rate scheduling
    scheduler.step(val_loss_avg)

    # Checkpointing: Save best model
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_avg,
            'val_loss': val_loss_avg
        }, 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping check
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("Training complete!")
```

### 4.5 Advanced Training Features

#### Automatic Mixed Precision (AMP)

**Purpose**: Speed up training and reduce memory usage using FP16 operations

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():  # Use FP16 for forward pass
        gloss_pred, category_pred = model(videos)
        loss = compute_loss(gloss_pred, category_pred, labels)

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefit**: 2-3× training speedup on modern GPUs with minimal accuracy impact

#### Gradient Clipping

**Purpose**: Prevent exploding gradients in deep networks

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Effect**: Scales down gradients if their norm exceeds 1.0, ensuring stable training

#### Checkpoint Resume

**Purpose**: Continue training from saved checkpoint

```python
# Save checkpoint with full state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss
}

# Resume from checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### 4.6 Training Output

**Saved Artifacts**:

```
trained_models/transformer/optimal/
├── SignTransformer_best.pt     # Best model checkpoint
└── training.log                # Complete training log
```

**Checkpoint Contents**:

- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state (for resuming)
- `epoch`: Training epoch number
- `train_loss`: Training loss value
- `val_loss`: Validation loss value

**Training Log Includes**:

- Per-epoch train/val loss
- Learning rate changes
- Early stopping notifications
- Best model save events
- System information (GPU, memory)

---

## 5. Model Evaluation

### 5.1 Validation Script

**File**: `evaluation/validation/validate.py`

**Purpose**: Comprehensive model evaluation on validation dataset

**Command**:

```bash
python evaluation/validation/validate.py \
  --model transformer \
  --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt \
  --data-dir data/processed/fsl_val \
  --labels-csv data/processed/fsl_val.csv \
  --batch-size 32 \
  --output-dir results-validate
```

### 5.2 Validation Process

**Steps**:

1. **Load Model**: Initialize architecture and load trained weights
2. **Load Dataset**: Read validation NPZ files and labels CSV
3. **Batch Inference**: Process samples in batches for efficiency
4. **Collect Predictions**: Store predicted classes and probabilities
5. **Compute Metrics**: Calculate comprehensive performance measures
6. **Generate Reports**: Create confusion matrices and per-class analysis

### 5.3 Metrics Computed

#### Overall Performance

**Metrics**:

- **Accuracy**: Proportion of correct predictions
- **Precision**: TP / (TP + FP) per class, macro-averaged
- **Recall**: TP / (TP + FN) per class, macro-averaged
- **F1-Score**: Harmonic mean of precision and recall

**Computed For**:

- Gloss classification (105 classes)
- Category classification (10 classes)

#### Occlusion-Based Analysis

**Automatic Subset Division**:

```python
# Validation set automatically partitioned by occlusion flag
clean_samples = [s for s in validation_data if s['occluded'] == 0]
occluded_samples = [s for s in validation_data if s['occluded'] == 1]

# Compute metrics for each subset
clean_metrics = compute_metrics(clean_samples)
occluded_metrics = compute_metrics(occluded_samples)

# Compare performance
accuracy_drop = clean_metrics['accuracy'] - occluded_metrics['accuracy']
```

**Purpose**: Assess model robustness to hand-face occlusions that occur naturally in sign language.

#### Per-Class Metrics

**Computation**:

```python
from sklearn.metrics import classification_report

report = classification_report(
    y_true=true_labels,
    y_pred=predicted_labels,
    output_dict=True,
    zero_division=0
)

# Extract per-class precision, recall, F1
for class_id in range(105):
    metrics[class_id] = {
        'precision': report[str(class_id)]['precision'],
        'recall': report[str(class_id)]['recall'],
        'f1-score': report[str(class_id)]['f1-score'],
        'support': report[str(class_id)]['support']
    }
```

**Purpose**: Identify which signs are easy/hard to recognize, revealing model strengths and weaknesses.

#### Confusion Matrices

**Generation**:

```python
from sklearn.metrics import confusion_matrix

# Gloss confusion matrix (105 × 105)
gloss_cm = confusion_matrix(y_true_gloss, y_pred_gloss)

# Category confusion matrix (10 × 10)
category_cm = confusion_matrix(y_true_cat, y_pred_cat)
```

**Interpretation**:

- Diagonal: Correct predictions
- Off-diagonal: Confusions between classes
- Reveals systematic misclassification patterns

### 5.4 Validation Output

**Results Structure**:

```python
{
  'model_info': {
    'model_type': 'transformer',
    'checkpoint_path': 'path/to/checkpoint.pt',
    'device': 'cuda:0',
    'timestamp': '2024-10-12 14:30:00'
  },
  'dataset_info': {
    'total_samples': 426,
    'occluded_samples': 127,
    'non_occluded_samples': 299
  },
  'overall_results': {
    'gloss_accuracy': 0.8732,
    'category_accuracy': 0.9201,
    'gloss_precision': 0.8645,
    'gloss_recall': 0.8598,
    'gloss_f1_score': 0.8621,
    'category_precision': 0.9156,
    'category_recall': 0.9089,
    'category_f1_score': 0.9122
  },
  'occluded_results': {...},        # Same metrics for occluded subset
  'non_occluded_results': {...},    # Same metrics for clean subset
  'per_class_results': {...},       # Per-class breakdown
  'confusion_matrices': {...},      # Confusion matrices
  'detailed_predictions': [...]     # Individual predictions
}
```

**Saved Files**:

```
results-validate/
├── complete_validation_results.json    # All results
├── overall_results.json                # Overall metrics
├── occluded_results.json              # Occluded subset metrics
├── non_occluded_results.json          # Clean subset metrics
├── per_class_results.json             # Per-class breakdown
└── confusion_matrices.json            # Confusion matrices
```

### 5.5 Model Comparison

**Process**:

1. Train both models (Transformer and InceptionV3-GRU) using same data
2. Validate both models using same validation set
3. Compare performance across all metrics
4. Analyze occlusion impact for each model

**Comparison Dimensions**:

- Overall accuracy (gloss and category)
- Performance on clean videos
- Performance on occluded videos
- Occlusion robustness (accuracy drop)
- Per-class strengths and weaknesses

**Note**: Statistical significance testing (paired t-tests, Wilcoxon tests, Holm-Bonferroni correction) described in thesis methodology represents the planned statistical analysis approach. Current implementation focuses on comprehensive metric computation and direct comparison.

---

## 6. Complete Pipeline Flow

### 6.1 End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA COLLECTION                          │
│  • FSL-105 dataset (2,130 videos)                              │
│  • 1920×1080 @ 60 FPS, 4 seconds each                          │
│  • 105 glosses across 10 categories                            │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
│  1. Video downsampling (60→30 FPS, resize to 256×256)          │
│  2. Background removal (MediaPipe Selfie Segmentation)          │
│  3. Keypoint extraction (MediaPipe Holistic → 156-D)           │
│  4. InceptionV3 feature extraction (→ 2048-D)                  │
│  5. Gap interpolation (max 5 frames)                           │
│  6. Occlusion detection (multi-method + temporal filtering)     │
│  7. NPZ file generation (compressed)                           │
│                                                                 │
│  Output: 2,130 .npz files with keypoints, features, metadata   │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING                               │
│  • Stratified 80/20 split (hash-based determinism)             │
│  • Train: ~1,704 samples                                       │
│  • Validation: ~426 samples                                    │
│  • All 105 glosses represented in both sets                    │
│  • Labels CSV generated for train and val                      │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                               │
│  Train Two Models:                                              │
│  ┌───────────────────┐        ┌──────────────────┐            │
│  │ SignTransformer   │        │ InceptionV3-GRU  │            │
│  │ (Attention-based) │        │ (CNN-RNN hybrid) │            │
│  └───────────────────┘        └──────────────────┘            │
│                                                                 │
│  Configuration:                                                 │
│  • Optimizer: Adam (lr=1e-4)                                   │
│  • Loss: Combined gloss + 0.5×category                         │
│  • Batch size: 16 (train), 32 (val)                           │
│  • Max epochs: 50 with early stopping (patience=10)            │
│  • LR scheduling: ReduceLROnPlateau                            │
│  • Gradient clipping: max_norm=1.0                             │
│  • Automatic Mixed Precision (AMP) for speed                   │
│                                                                 │
│  Output: Best model checkpoints saved                          │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                             │
│  • Load validation set (~426 samples)                          │
│  • Batch inference (batch_size=32)                             │
│  • Compute comprehensive metrics:                              │
│    - Overall accuracy, precision, recall, F1                   │
│    - Clean subset performance                                  │
│    - Occluded subset performance                               │
│    - Per-class metrics                                         │
│    - Confusion matrices                                        │
│  • Export results to JSON                                      │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RESULTS ANALYSIS                             │
│  • Compare Transformer vs InceptionV3-GRU                      │
│  • Analyze occlusion impact                                    │
│  • Identify challenging signs                                  │
│  • Generate performance reports                                │
│  • Visualize attention weights (Transformer only)              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Timeline Estimate

**For Complete FSL-105 Dataset**:

| Phase                  | Time (GPU Workstation) | Time (CPU Only)  |
| ---------------------- | ---------------------- | ---------------- |
| Preprocessing          | 2-3 hours              | 15-20 hours      |
| Data Splitting         | < 1 minute             | < 1 minute       |
| Training (Transformer) | 3-6 hours              | 20-30 hours      |
| Training (IV3-GRU)     | 2-4 hours              | 15-25 hours      |
| Validation             | 5-10 minutes           | 15-20 minutes    |
| **Total**              | **~8-14 hours**        | **~50-75 hours** |

**Hardware Recommendations**:

- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, NVIDIA GPU (8GB+ VRAM)
- **Optimal**: 64GB RAM, 16-core CPU, NVIDIA RTX 3090/4090 (24GB VRAM)

### 6.3 Reproducibility Checklist

**To Ensure Reproducible Results**:

- ✓ Set random seeds (Python, NumPy, PyTorch)
- ✓ Use deterministic algorithms (where possible)
- ✓ Hash-based data splitting (consistent across runs)
- ✓ Save complete training configuration in logs
- ✓ Version control training scripts
- ✓ Document hardware specifications
- ✓ Save preprocessing parameters in NPZ metadata
- ✓ Use fixed hyperparameters (avoid manual tuning during experiment)

**Random Seed Setup**:

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# For deterministic behavior (may reduce performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 7. Integration with Streamlit Tool

### 7.1 Trained Models → Streamlit Application

**Connection**:

```
Training Pipeline Output:
├── trained_models/transformer/optimal/SignTransformer_best.pt
└── trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt
         ↓
Streamlit Tool Configuration (streamlit_app/core/config.py):
MODEL_CONFIG = {
  'transformer': {
    'checkpoint_path': 'trained_models/transformer/optimal/SignTransformer_best.pt',
    ...
  },
  'iv3_gru': {
    'checkpoint_path': 'trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt',
    ...
  }
}
```

**Workflow**:

1. User uploads video to Streamlit tool
2. Tool preprocesses video using same pipeline (MediaPipe + InceptionV3)
3. Tool loads trained model checkpoint
4. Tool makes prediction using trained model
5. Tool displays results with visualization

### 7.2 Preprocessing Consistency

**Critical**: Streamlit tool uses **identical preprocessing** as training pipeline

```python
# Same parameters in both contexts:
target_fps = 30
out_size = 256
conf_thresh = 0.5
max_gap = 5
write_keypoints = True
write_iv3_features = True
```

**Ensures**: Model sees same data distribution at inference time as during training.

---

## 8. Key Files Reference

### 8.1 Preprocessing

| File                                             | Purpose                        |
| ------------------------------------------------ | ------------------------------ |
| `preprocessing/core/preprocess.py`               | Main preprocessing pipeline    |
| `preprocessing/extractors/keypoints_features.py` | MediaPipe keypoint extraction  |
| `preprocessing/extractors/iv3_features.py`       | InceptionV3 feature extraction |
| `preprocessing/core/occlusion_detection.py`      | Occlusion detection algorithms |

### 8.2 Data Management

| File                                  | Purpose                    |
| ------------------------------------- | -------------------------- |
| `data/splitting/data_split.py`        | Train/val splitting script |
| `data/labels/label_mapping.py`        | Label ID ↔ name mappings   |
| `data/splitting/labels_reference.csv` | Complete label reference   |

### 8.3 Training

| File                    | Purpose                        |
| ----------------------- | ------------------------------ |
| `training/train.py`     | Main training script           |
| `training/utils.py`     | Training utilities and helpers |
| `models/transformer.py` | SignTransformer architecture   |
| `models/iv3_gru.py`     | InceptionV3-GRU architecture   |

### 8.4 Evaluation

| File                                | Purpose             |
| ----------------------------------- | ------------------- |
| `evaluation/validation/validate.py` | Validation script   |
| `evaluation/prediction/predict.py`  | Inference utilities |

---

## 9. Common Issues & Solutions

### 9.1 Preprocessing Issues

**Issue**: GPU out of memory during InceptionV3 extraction

**Solution**:

```bash
# Reduce batch size
python preprocess.py ... --batch-size 8

# Or use CPU only
python preprocess.py ... --workers 1  # Sequential on CPU
```

**Issue**: MediaPipe fails to detect keypoints in some frames

**Solution**: Already handled by:

- Gap interpolation (fills gaps ≤ 5 frames)
- Visibility masking (tracks which keypoints are valid)
- Model handles variable-length sequences

### 9.2 Training Issues

**Issue**: Training loss not decreasing

**Solutions**:

- Check learning rate (try 1e-3 or 5e-5)
- Verify data loading (check labels match files)
- Reduce batch size if GPU memory is full
- Check for NaN values in features

**Issue**: Validation loss increases while training loss decreases (overfitting)

**Solutions**:

- Early stopping will catch this automatically
- Increase dropout rate (try 0.2 or 0.3)
- Reduce model capacity
- Add data augmentation

### 9.3 Validation Issues

**Issue**: Model checkpoint not loading

**Solutions**:

- Verify checkpoint path is correct
- Check that model architecture matches checkpoint
- Try different checkpoint format keys: `model_state_dict`, `state_dict`, `model`

**Issue**: Low accuracy on validation set

**Analysis**:

- Check per-class metrics to identify problematic signs
- Review confusion matrix for systematic errors
- Compare occluded vs non-occluded performance
- Verify preprocessing consistency with training

---

## 10. Research Workflow Summary

### For New Researchers/Developers:

**Step-by-Step Research Pipeline**:

1. **Obtain Dataset**: Download FSL-105 or collect your own sign language videos

2. **Preprocess All Videos**:

   ```bash
   python preprocessing/core/preprocess.py raw_videos/ processed/ \
     --write-keypoints --write-iv3-features --workers 8
   ```

3. **Create Train/Val Split**:

   ```bash
   python data/splitting/data_split.py \
     --processed-root processed/ \
     --labels processed/labels.csv \
     --out-root processed/ \
     --copy --train-ratio 0.8
   ```

4. **Train Models**:

   ```bash
   # Train Transformer
   python training/train.py --model transformer --epochs 50 \
     --output-dir trained_models/transformer/run1

   # Train InceptionV3-GRU
   python training/train.py --model iv3_gru --epochs 50 \
     --output-dir trained_models/iv3_gru/run1
   ```

5. **Validate Models**:

   ```bash
   # Validate Transformer
   python evaluation/validation/validate.py \
     --model transformer \
     --checkpoint trained_models/transformer/run1/best.pt \
     --output-dir results/transformer

   # Validate InceptionV3-GRU
   python evaluation/validation/validate.py \
     --model iv3_gru \
     --checkpoint trained_models/iv3_gru/run1/best.pt \
     --output-dir results/iv3_gru
   ```

6. **Compare Results**: Analyze validation outputs to compare model performance

7. **Deploy Best Model**: Copy best checkpoint to Streamlit tool's model directory

---

## 11. Connection to Thesis Methodology

This training pipeline implements the research methodology described in the thesis:

**From Thesis → Implementation Mapping**:

| Thesis Component                      | Implementation                              |
| ------------------------------------- | ------------------------------------------- |
| **Preprocessing** (video → keypoints) | `preprocessing/core/preprocess.py`          |
| **Transformer Architecture**          | `models/transformer.py`                     |
| **InceptionV3-GRU Baseline**          | `models/iv3_gru.py`                         |
| **80/20 Train-Test Split**            | `data/splitting/data_split.py`              |
| **Training Procedure**                | `training/train.py`                         |
| **Validation Metrics**                | `evaluation/validation/validate.py`         |
| **Occlusion Detection**               | `preprocessing/core/occlusion_detection.py` |

**Note**: The thesis describes the theoretical framework and experimental design. This pipeline document describes the actual implementation that realizes that framework.

---

**Document Version**: 1.0  
**Last Updated**: October 12, 2025  
**Status**: Complete Implementation Documentation  
**Related Documents**:

- `pansinayan_system_architecture.md`: Streamlit tool architecture
- `pansinayan_complete_pipeline.md`: User workflow guide
- `thesis_methodology.md`: Research framework and theoretical approach
