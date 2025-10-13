# Training Guide

This guide covers training sign language recognition models using Transformer or InceptionV3-GRU architectures.

## Prerequisites

```powershell
pip install -r requirements.txt
```

## Dataset Configuration

- **Glosses**: 105 sign words (IDs: 0-104)
- **Categories**: 10 semantic categories (IDs: 0-9)

Category mapping:

```
0: GREETING    5: FAMILY
1: SURVIVAL    6: RELATIONSHIPS
2: NUMBER      7: COLOR
3: CALENDAR    8: FOOD
4: DAYS        9: DRINK
```

For complete label mappings, see [Label Mapping Table](../data/labels/LABEL_MAPPING_TABLE.md).

---

## Quick Start

### Basic Training

**Transformer (Keypoints)**:

```powershell
python training/train.py ^
  --model transformer ^
  --keypoints-train data/processed/fsl_train ^
  --keypoints-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --output-dir trained_models/transformer/run1
```

**InceptionV3-GRU (Features)**:

```powershell
python training/train.py ^
  --model iv3_gru ^
  --features-train data/processed/fsl_train ^
  --features-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --output-dir trained_models/iv3_gru/run1
```

### Performance Options

**GPU Training**:

```powershell
python training/train.py ^
  --model transformer ^
  --keypoints-train data/processed/fsl_train ^
  --keypoints-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --amp ^
  --auto-workers ^
  --scheduler plateau ^
  --early-stop 10 ^
  --output-dir trained_models/transformer/run1
```

**Multi-GPU Training**:

```powershell
python training/train.py ^
  --model transformer ^
  --keypoints-train data/processed/fsl_train ^
  --keypoints-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 64 ^
  --amp ^
  --enable-parallel ^
  --auto-workers ^
  --output-dir trained_models/transformer/run1
```

**Smoke Test**:

```powershell
python training/train.py --smoke-test
```

---

## Data Structure

### Directory Layout

```
data/processed/
├── fsl_train/                    # Training data (80%)
│   ├── clip_0315_yes.npz
│   ├── clip_1601_orange.npz
│   └── ...
├── fsl_val/                      # Validation data (20%)
│   ├── clip_0138_nice to meet you.npz
│   ├── clip_1146_grandfather.npz
│   └── ...
├── fsl_train.csv                 # Training labels
└── fsl_val.csv                   # Validation labels
```

### NPZ File Format

**Transformer (Keypoints)**:

- Key: `X`
- Shape: `[T, 156]`
- Content: 78 keypoints × 2 coordinates (x, y)

**InceptionV3-GRU (Features)**:

- Key: `X2048`
- Shape: `[T, 2048]`
- Content: Precomputed InceptionV3 features

**Additional keys**:

- `mask`: `[T, 78]` - Keypoint visibility
- `timestamps_ms`: `[T]` - Frame timestamps
- `meta`: JSON metadata

### Labels CSV Format

Required columns: `file`, `gloss`, `cat`, `occluded`

Example:

```csv
file,gloss,cat,occluded
clip_0315_yes,15,1,0
clip_1601_orange,79,7,0
clip_2062_no sugar,104,9,1
```

- `file`: NPZ filename without extension
- `gloss`: Gloss ID (0-104)
- `cat`: Category ID (0-9)
- `occluded`: Binary flag (0=clean, 1=occluded)

---

## Training Parameters

### Essential Parameters

| Parameter      | Default       | Description                       |
| -------------- | ------------- | --------------------------------- |
| `--model`      | `transformer` | Model: `transformer` or `iv3_gru` |
| `--epochs`     | `20`          | Number of training epochs         |
| `--batch-size` | `32`          | Batch size                        |
| `--lr`         | `1e-4`        | Learning rate                     |
| `--num-gloss`  | `105`         | Number of gloss classes           |
| `--num-cat`    | `10`          | Number of category classes        |

### Data Parameters

| Parameter            | Required For | Description                        |
| -------------------- | ------------ | ---------------------------------- |
| `--keypoints-train`  | Transformer  | Training keypoints directory       |
| `--keypoints-val`    | Transformer  | Validation keypoints directory     |
| `--features-train`   | IV3-GRU      | Training features directory        |
| `--features-val`     | IV3-GRU      | Validation features directory      |
| `--labels-train-csv` | Both         | Training labels CSV                |
| `--labels-val-csv`   | Both         | Validation labels CSV              |
| `--kp-key`           | Transformer  | Keypoint NPZ key (default: `X`)    |
| `--feature-key`      | IV3-GRU      | Feature NPZ key (default: `X2048`) |

### Performance Parameters

| Parameter           | Default | Description                    |
| ------------------- | ------- | ------------------------------ |
| `--amp`             | `False` | Enable mixed precision         |
| `--compile-model`   | `False` | Compile model (PyTorch 2.0+)   |
| `--auto-workers`    | `False` | Auto-detect DataLoader workers |
| `--enable-parallel` | `False` | Enable multi-GPU DataParallel  |
| `--num-workers`     | `0`     | DataLoader workers (0=auto)    |
| `--pin-memory`      | `False` | Pin memory for GPU             |

### Training Control

| Parameter              | Default | Description                       |
| ---------------------- | ------- | --------------------------------- |
| `--weight-decay`       | `0.0`   | L2 regularization                 |
| `--grad-clip`          | `None`  | Gradient clipping max norm        |
| `--scheduler`          | `None`  | LR scheduler: `plateau`, `cosine` |
| `--scheduler-patience` | `5`     | Epochs before LR reduction        |
| `--early-stop`         | `None`  | Early stopping patience           |
| `--resume`             | `None`  | Resume from checkpoint path       |
| `--output-dir`         | `.`     | Output directory for models       |

### Loss Weighting

| Parameter | Default | Description          |
| --------- | ------- | -------------------- |
| `--alpha` | `0.5`   | Gloss loss weight    |
| `--beta`  | `0.5`   | Category loss weight |

### Curriculum Learning

| Parameter                 | Default  | Description                                          |
| ------------------------- | -------- | ---------------------------------------------------- |
| `--curriculum`            | `None`   | Strategy: `gloss-first`, `category-first`, `dynamic` |
| `--curriculum-epochs`     | `10`     | Epochs for curriculum phase                          |
| `--curriculum-min-weight` | `0.1`    | Minimum weight for secondary task                    |
| `--curriculum-schedule`   | `linear` | Schedule: `linear`, `cosine`, `exponential`          |

---

## Training Features

### Optimization

- **Automatic Mixed Precision (AMP)**: Faster training on CUDA devices
- **Model Compilation**: PyTorch 2.0+ optimization
- **Auto Workers**: Automatically detects optimal DataLoader workers
- **Multi-GPU**: Automatic DataParallel when multiple GPUs available

### Regularization

- **Dropout**: Applied in model layers
- **Weight Decay**: L2 regularization on model parameters
- **Gradient Clipping**: Prevents exploding gradients
- **Label Smoothing**: Optional for better generalization

### Scheduling

- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- **CosineAnnealingLR**: Smooth learning rate annealing
- **Early Stopping**: Stops training when validation performance stops improving

### Monitoring

- **CSV Logging**: Automatic metrics logging with timestamps
- **Console Output**: Real-time training progress
- **Checkpointing**: Best and last model saved automatically

---

## Curriculum Training

Curriculum training focuses on one task initially, then gradually introduces the other.

### Strategies

**Gloss-First**:

```powershell
python training/train.py ^
  --model transformer ^
  --keypoints-train data/processed/fsl_train ^
  --keypoints-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --curriculum gloss-first ^
  --curriculum-epochs 15 ^
  --output-dir trained_models/transformer/curriculum
```

**Category-First**:

```powershell
python training/train.py ^
  --model iv3_gru ^
  --features-train data/processed/fsl_train ^
  --features-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --curriculum category-first ^
  --curriculum-epochs 20 ^
  --output-dir trained_models/iv3_gru/curriculum
```

**Dynamic**:

```powershell
python training/train.py ^
  --model transformer ^
  --keypoints-train data/processed/fsl_train ^
  --keypoints-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 60 ^
  --curriculum dynamic ^
  --curriculum-epochs 20 ^
  --output-dir trained_models/transformer/dynamic
```

### Weight Schedules

- **Linear**: Gradual increase from min_weight to 0.5
- **Cosine**: Smooth transitions
- **Exponential**: Quick initial progress

---

## Checkpointing

### Automatic Saves

Models are saved to the output directory:

- `{ModelName}_best.pt`: Best validation performance
- `{ModelName}_last.pt`: Latest epoch

**Checkpoint contents**:

- Model state_dict
- Optimizer state
- Scheduler state (if used)
- Epoch number
- Best validation metrics

### Resume Training

```powershell
python training/train.py ^
  --resume trained_models/transformer/run1/SignTransformer_last.pt ^
  --keypoints-train data/processed/fsl_train ^
  --keypoints-val data/processed/fsl_val ^
  --labels-train-csv data/processed/fsl_train.csv ^
  --labels-val-csv data/processed/fsl_val.csv
```

### CSV Logging

Enable with `--log-csv`:

```powershell
python training/train.py --log-csv logs/training_metrics.csv
```

Logs include: epoch, train_loss, val_loss, val_gloss_acc, val_cat_acc, lr, epoch_time

---

## Troubleshooting

### Out of Memory

**Solutions**:

- Reduce `--batch-size`
- Enable `--amp`
- Increase `--gradient-accumulation-steps`
- Reduce sequence lengths in preprocessing

### Slow Training

**Solutions**:

- Enable `--amp` for faster computation
- Enable `--compile-model` for optimization
- Use `--auto-workers` for parallel data loading
- Enable `--enable-parallel` for multi-GPU

### Data Loading Issues

**Check**:

- File paths exist
- CSV has required columns: `file`, `gloss`, `cat`, `occluded`
- NPZ files contain correct keys (`X` or `X2048`)
- Gloss IDs in range [0, 104]
- Category IDs in range [0, 9]

### Convergence Problems

**Try**:

- Adjust `--lr` (try 1e-4, 5e-5, 3e-4)
- Use `--scheduler plateau` or `cosine`
- Adjust `--alpha` and `--beta` for loss balancing
- Enable `--grad-clip 1.0`
- Try curriculum learning

### Data Validation

```powershell
# Validate NPZ files
python preprocessing/utils/validate_npz.py data/processed/fsl_train

# Require X2048 features
python preprocessing/utils/validate_npz.py data/processed/fsl_val --require-x2048
```

---

## Best Practices

### Training Strategy

1. Validate data before training
2. Start with few epochs to verify setup
3. Monitor GPU memory during first epoch
4. Enable CSV logging for tracking
5. Use checkpoints for long runs
6. Enable early stopping to prevent overfitting

### Performance Tips

**GPU Training**:

- Enable `--amp` for mixed precision
- Enable `--compile-model` for PyTorch 2.0+ optimization
- Use `--auto-workers` for parallel data loading

**Multi-GPU**:

- Enable `--enable-parallel`
- Increase `--batch-size` to utilize GPUs
- Use gradient accumulation for larger effective batch sizes

**Memory Constraints**:

- Reduce `--batch-size` and increase `--gradient-accumulation-steps`
- Enable `--amp` to reduce memory footprint

### Hyperparameter Tuning

**Learning Rate**:

- Start with `1e-4`
- Try `5e-5` (multi-GPU) or `3e-4` (single GPU)
- Monitor loss curves

**Scheduler**:

- Use `plateau` for stable training
- Use `cosine` for faster convergence

**Loss Weights**:

- Start with defaults (0.5, 0.5)
- Adjust based on task difficulty
- Use curriculum learning if tasks differ significantly

---

## Model Notes

**Transformer**:

- Uses attention masks for variable-length sequences
- Benefits from larger batch sizes
- Lower memory footprint
- Provides attention weights for interpretability

**InceptionV3-GRU**:

- Uses pretrained ImageNet weights
- Processes precomputed features
- Requires `X2048` key in NPZ files
- Higher memory requirements

Both models support multi-task learning with configurable loss weights and curriculum strategies.

---

## Additional Resources

- [Model Guide](../models/MODEL_GUIDE.md): Model architectures and details
- [Data Guide](../data/DATA_GUIDE.md): Data preparation
- [Preprocessing Guide](../preprocessing/docs/PREPROCESS_GUIDE.MD): Video preprocessing
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md): Model evaluation
- [Trained Model Guide](../trained_models/TRAINED_MODEL_GUIDE.md): Model management
