# Training Guide

## Overview

This guide covers training sign language recognition models using either Transformer (keypoints) or InceptionV3+GRU (features) architectures. The training script has been **completely optimized** for real data training with performance optimizations for CUDA, memory management, data loading, and **automatic parallelization** for multi-GPU setups.

## Prerequisites

```powershell
pip install -r requirements.txt
```

## Dataset Configuration

- **Glosses**: 105 sign words (IDs: 0-104)
- **Categories**: 10 semantic categories (IDs: 0-9)
  - 0: GREETING
  - 1: SURVIVAL
  - 2: NUMBER
  - 3: CALENDAR
  - 4: DAYS
  - 5: FAMILY
  - 6: RELATIONSHIPS
  - 7: COLOR
  - 8: FOOD
  - 9: DRINK

For complete label mappings, see [Label Mapping Table](../data/labels/LABEL_MAPPING_TABLE.md).

## Quick Start

### Basic Training Commands

**Transformer (Keypoints)**:

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
  --batch-size 32 ^
  --auto-workers ^
  --auto-batch-size ^
  --amp ^
  --compile-model
```

**IV3-GRU (InceptionV3 Features)**:

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
  --batch-size 32 ^
  --auto-workers ^
  --auto-batch-size ^
  --amp ^
  --compile-model
```

### Multi-GPU Training

**For Vast AI or Multi-GPU Systems**:

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
  --batch-size 64 ^
  --auto-workers ^
  --auto-batch-size ^
  --enable-parallel ^
  --amp ^
  --compile-model ^
  --lr 5e-5 ^
  --weight-decay 1e-4 ^
  --scheduler plateau ^
  --grad-clip 1.0
```

**For Local Machine (CPU/Single GPU)**:

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --batch-size 16 ^
  --auto-workers ^
  --auto-batch-size ^
  --lr 1e-4 ^
  --weight-decay 1e-3 ^
  --scheduler plateau
```

## Data Structure Requirements

### Directory Structure

```
data/processed/
├── cmb_train/                    # Combined training data (80%)
│   ├── clip_0315_yes.npz
│   ├── clip_1601_orange.npz
│   └── ...
├── cmb_val/                      # Combined validation data (20%)
│   ├── clip_0138_nice to meet you.npz
│   ├── clip_1146_grandfather.npz
│   └── ...
├── cmb_train.csv                 # Training labels
└── cmb_val.csv                   # Validation labels
```

Alternative splits available:

- `fsl_train/`, `fsl_val/` - FSL-105 dataset only
- `smp_train/`, `smp_val/` - Sample-105 dataset only
- `cmb_train/`, `cmb_val/` - Combined dataset (recommended)

### Data Format Requirements

**NPZ Files**:

- **Transformer (Keypoints)**: Key `X` with shape `[T, 156]` (variable sequence lengths)
- **IV3-GRU (Features)**: Key `X2048` with shape `[T, 2048]` (variable sequence lengths)
- **Additional keys**: `mask` [T, 78], `timestamps_ms` [T], `meta` (JSON)

**Labels CSV**:

Required columns: `file`, `gloss`, `cat`, `occluded`

- `file`: NPZ filename without extension (e.g., `clip_0315_yes`)
- `gloss`: Gloss class ID (0-based, range: 0-104)
- `cat`: Category class ID (0-based, range: 0-9)
- `occluded`: Binary flag (0 = clean, 1 = occluded)

Example `cmb_train.csv`:

```csv
file,gloss,cat,occluded
clip_0315_yes,15,1,0
clip_1601_orange,79,7,0
clip_2062_no sugar,104,9,1
clip_1314_man,64,6,1
```

## Training Parameters

### Essential Parameters

| Parameter      | Description                | Default       | Notes                                    |
| -------------- | -------------------------- | ------------- | ---------------------------------------- |
| `--model`      | Model architecture         | `transformer` | `transformer` or `iv3_gru`               |
| `--epochs`     | Training epochs            | `20`          | Adjust based on convergence              |
| `--batch-size` | Batch size                 | `32`          | Reduce if OOM, increase if memory allows |
| `--lr`         | Learning rate              | `1e-4`        | Start conservative, adjust based on loss |
| `--num-gloss`  | Number of gloss classes    | `105`         | Must match dataset (105 for FSLR)        |
| `--num-cat`    | Number of category classes | `10`          | Must match dataset (10 for FSLR)         |

### Data Parameters

| Parameter            | Description                    | Default | Notes                        |
| -------------------- | ------------------------------ | ------- | ---------------------------- |
| `--keypoints-train`  | Training keypoints directory   | `None`  | Required for transformer     |
| `--keypoints-val`    | Validation keypoints directory | `None`  | Required for transformer     |
| `--features-train`   | Training features directory    | `None`  | Required for iv3_gru         |
| `--features-val`     | Validation features directory  | `None`  | Required for iv3_gru         |
| `--labels-train-csv` | Training labels CSV            | `None`  | Required                     |
| `--labels-val-csv`   | Validation labels CSV          | `None`  | Required                     |
| `--kp-key`           | Keypoint NPZ key               | `X`     | Key containing [T,156] data  |
| `--feature-key`      | Feature NPZ key                | `X2048` | Key containing [T,2048] data |

### Performance Parameters

| Parameter                       | Description                       | Default | Notes                       |
| ------------------------------- | --------------------------------- | ------- | --------------------------- |
| `--amp`                         | Enable mixed precision            | `False` | Faster training on CUDA     |
| `--compile-model`               | Compile model (PyTorch 2.0+)      | `False` | Better performance          |
| `--auto-workers`                | Auto-detect DataLoader workers    | `False` | Worker count (up to 8)      |
| `--auto-batch-size`             | Auto-calculate batch size         | `False` | Based on available memory   |
| `--enable-parallel`             | Enable DataParallel for multi-GPU | `False` | Automatic multi-GPU support |
| `--gradient-accumulation-steps` | Gradient accumulation             | `1`     | Effective larger batch size |
| `--num-workers`                 | DataLoader workers                | `0`     | 0 = auto-detect             |
| `--pin-memory`                  | Pin memory for GPU                | `False` | Faster GPU transfers        |

### Training Control Parameters

| Parameter              | Description             | Default | Notes                       |
| ---------------------- | ----------------------- | ------- | --------------------------- |
| `--weight-decay`       | Weight decay            | `0.0`   | L2 regularization           |
| `--grad-clip`          | Gradient clipping       | `None`  | Prevent exploding gradients |
| `--scheduler`          | Learning rate scheduler | `None`  | `plateau` or `cosine`       |
| `--scheduler-patience` | Scheduler patience      | `5`     | Epochs before LR reduction  |
| `--early-stop`         | Early stopping patience | `None`  | Stop if no improvement      |
| `--resume`             | Resume from checkpoint  | `None`  | Path to checkpoint file     |

### Loss Weighting Parameters

| Parameter | Description          | Default | Notes                               |
| --------- | -------------------- | ------- | ----------------------------------- |
| `--alpha` | Gloss loss weight    | `0.5`   | Higher = focus on gloss accuracy    |
| `--beta`  | Category loss weight | `0.5`   | Higher = focus on category accuracy |

### Curriculum Training Parameters

| Parameter                 | Description                               | Default  | Notes                                      |
| ------------------------- | ----------------------------------------- | -------- | ------------------------------------------ |
| `--curriculum`            | Curriculum strategy                       | `None`   | `gloss-first`, `category-first`, `dynamic` |
| `--curriculum-epochs`     | Number of epochs for curriculum phase     | `10`     | When to start balancing tasks              |
| `--curriculum-warmup`     | Warmup epochs before curriculum (dynamic) | `5`      | Only for dynamic strategy                  |
| `--curriculum-min-weight` | Minimum weight for secondary task         | `0.1`    | Range: 0.0-1.0                             |
| `--curriculum-schedule`   | Weight scheduling function                | `linear` | `linear`, `cosine`, `exponential`          |

## Performance Optimizations

The training script automatically optimizes for your hardware:

- **Device Detection**: CUDA, MPS (Apple Silicon), or CPU
- **Memory Management**: Optimized GPU memory allocation
- **DataLoader Optimization**: Auto-detects workers (up to 8) and prefetch settings
- **Mixed Precision**: AMP on CUDA devices
- **Dynamic Batch Sizing**: Calculates batch size based on available memory
- **Multi-GPU Support**: Automatic DataParallel when multiple GPUs detected

### Resource Adaptation

- **GPU Memory**: Batch size adjusts (8-64) based on available memory
- **CPU Cores**: Uses up to 8 DataLoader workers
- **Multi-GPU**: Distributes training across available GPUs

### Performance Examples

**Multi-GPU**:

```powershell
python -m training.train ^
  --model transformer ^
  --batch-size 64 ^
  --amp ^
  --compile-model ^
  --auto-workers ^
  --auto-batch-size ^
  --enable-parallel ^
  --gradient-accumulation-steps 2
```

**Limited Memory**:

```powershell
python -m training.train ^
  --model transformer ^
  --batch-size 16 ^
  --auto-batch-size ^
  --gradient-accumulation-steps 4 ^
  --amp ^
  --auto-workers
```

**CPU Training**:

```powershell
python -m training.train ^
  --model transformer ^
  --batch-size 8 ^
  --auto-workers ^
  --auto-batch-size
```

## Curriculum Training

Curriculum training focuses on one task initially, then gradually introduces the other task. Useful when tasks have different difficulty levels.

### Curriculum Examples

**Gloss-First**:

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --batch-size 32 ^
  --curriculum gloss-first ^
  --curriculum-epochs 15 ^
  --curriculum-min-weight 0.1 ^
  --curriculum-schedule linear ^
  --amp ^
  --compile-model ^
  --auto-workers ^
  --auto-batch-size
```

**Category-First**:

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
  --epochs 50 ^
  --batch-size 32 ^
  --curriculum category-first ^
  --curriculum-epochs 20 ^
  --curriculum-min-weight 0.05 ^
  --curriculum-schedule cosine ^
  --amp ^
  --compile-model ^
  --auto-workers ^
  --auto-batch-size
```

**Dynamic**:

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 60 ^
  --batch-size 32 ^
  --curriculum dynamic ^
  --curriculum-warmup 5 ^
  --curriculum-epochs 20 ^
  --curriculum-min-weight 0.1 ^
  --curriculum-schedule exponential ^
  --amp ^
  --compile-model ^
  --auto-workers ^
  --auto-batch-size
```

### Weight Scheduling

- **Linear**: Gradual increase from min_weight to 0.5
- **Cosine**: Smooth transitions with slower initial changes
- **Exponential**: Quick initial progress with rapid final balancing

### Benefits

- **Better Convergence**: More stable training with fewer oscillations
- **Higher Accuracy**: Often achieves better final performance
- **Faster Training**: May converge faster than balanced training
- **Use When**: Tasks have different difficulty levels or balanced training struggles

## Training Examples

**Multi-GPU (High Performance)**:

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
  --batch-size 64 ^
  --lr 5e-5 ^
  --amp ^
  --compile-model ^
  --auto-workers ^
  --auto-batch-size ^
  --enable-parallel ^
  --gradient-accumulation-steps 2 ^
  --grad-clip 1.0 ^
  --scheduler plateau ^
  --early-stop 15 ^
  --log-csv logs\transformer_multi_gpu.csv
```

**Single GPU (Balanced)**:

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --batch-size 32 ^
  --lr 3e-4 ^
  --amp ^
  --compile-model ^
  --auto-workers ^
  --auto-batch-size ^
  --gradient-accumulation-steps 2 ^
  --grad-clip 1.0 ^
  --scheduler cosine ^
  --early-stop 10 ^
  --log-csv logs\transformer_single_gpu.csv
```

**CPU Training**:

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 30 ^
  --batch-size 8 ^
  --auto-workers ^
  --auto-batch-size ^
  --scheduler plateau ^
  --log-csv logs\cpu_training.csv
```

## Monitoring & Checkpointing

### CSV Logging

Enable detailed logging:

```powershell
--log-csv logs\training_metrics.csv
```

Logs include: epoch, train_loss, val_loss, val_gloss_acc, val_cat_acc, lr, epoch_time, gpu_memory

### Automatic Checkpointing

Checkpoints are saved in the current directory:

- `{ModelName}_last.pt`: Latest checkpoint (saved every epoch)
- `{ModelName}_best.pt`: Best checkpoint (highest validation accuracy)

**Checkpoint contents:**

- Model state_dict
- Optimizer state
- Scheduler state (if used)
- Epoch number
- Best validation metrics

### Resume Training

```powershell
python -m training.train ^
  --resume SignTransformer_last.pt ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv
```

## Smoke Tests

Quick tests to verify setup without real data:

**Transformer**:

```powershell
python -m training.train ^
  --model transformer ^
  --smoke-test ^
  --num-gloss 105 ^
  --num-cat 10
```

**IV3-GRU**:

```powershell
python -m training.train ^
  --model iv3_gru ^
  --smoke-test ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --no-pretrained-backbone
```

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:

- Reduce `--batch-size`
- Increase `--gradient-accumulation-steps`
- Enable `--amp` for mixed precision
- Reduce sequence lengths in preprocessing

**Slow Training**:

- Enable `--amp` for faster computation
- Enable `--compile-model` for optimized execution
- Use `--auto-workers` for parallel data loading
- Use `--auto-batch-size` for optimal batch size
- Enable `--enable-parallel` for multi-GPU

**Data Loading Issues**:

- Verify file paths exist
- Check CSV format has required columns: `file`, `gloss`, `cat`, `occluded`
- Ensure NPZ files contain correct keys (`X` for keypoints, `X2048` for features)
- Verify gloss IDs in range [0, 104] and cat IDs in range [0, 9]

**Convergence Problems**:

- Adjust `--lr` (try 1e-4, 5e-5, 3e-4)
- Use `--scheduler` (plateau or cosine)
- Adjust `--alpha` and `--beta` for loss balancing
- Enable `--grad-clip` to prevent exploding gradients
- Try curriculum learning strategies

### Data Validation

```powershell
# Validate NPZ files
python -m preprocessing.utils.validate_npz data\processed\cmb_train

# Validate with InceptionV3 features requirement
python -m preprocessing.utils.validate_npz data\processed\cmb_val --require-x2048
```

## Best Practices

### Training Strategy

1. **Validate data** before training using `validate_npz`
2. **Start small** with few epochs to verify setup
3. **Monitor GPU memory** usage during first epoch
4. **Enable CSV logging** with `--log-csv` for tracking metrics
5. **Use checkpoints** with `--resume` for long training runs
6. **Early stopping** with `--early-stop` to prevent overfitting

### Performance Tips

**For GPU Training:**

- Enable `--amp` for mixed precision
- Enable `--compile-model` for PyTorch 2.0+ optimization
- Use `--auto-batch-size` for optimal memory utilization
- Use `--auto-workers` for parallel data loading

**For Multi-GPU:**

- Enable `--enable-parallel` for automatic DataParallel
- Increase `--batch-size` to utilize multiple GPUs
- Use gradient accumulation for even larger effective batch sizes

**For Memory Constraints:**

- Reduce `--batch-size` and increase `--gradient-accumulation-steps`
- Enable `--amp` to reduce memory footprint
- Reduce sequence lengths during preprocessing

### Hyperparameter Tuning

**Learning Rate:**

- Start with `1e-4` (conservative)
- Try `5e-5` (multi-GPU) or `3e-4` (single GPU)
- Monitor loss curves and adjust accordingly

**Scheduler:**

- Use `plateau` for stable training with automatic LR reduction
- Use `cosine` for faster convergence with smooth annealing

**Loss Weights:**

- Balance `--alpha` and `--beta` based on task priorities
- Start with default (0.5, 0.5) and adjust if one task is harder
- Use curriculum learning if tasks have very different difficulties

## Model Notes

**Transformer:**

- Uses attention masks for variable-length sequences
- Benefits from larger batch sizes
- Lighter memory footprint than IV3-GRU
- Provides attention weights for interpretability

**IV3-GRU:**

- Uses InceptionV3 backbone with pretrained ImageNet weights
- Processes precomputed visual features efficiently
- Requires `X2048` key in NPZ files
- Higher memory requirements than Transformer

Both models support multi-task learning with configurable loss weights and curriculum strategies.

## Output Location

Trained models are saved in the current working directory:

```
.
├── SignTransformer_best.pt
├── SignTransformer_last.pt
├── InceptionV3GRU_best.pt
├── InceptionV3GRU_last.pt
└── logs\
    └── training_metrics.csv
```

Move models to `trained_models/` directory for organization:

```powershell
# Organize trained models
New-Item -ItemType Directory -Force -Path trained_models\transformer\my_run
Move-Item SignTransformer_*.pt trained_models\transformer\my_run\

New-Item -ItemType Directory -Force -Path trained_models\iv3_gru\my_run
Move-Item InceptionV3GRU_*.pt trained_models\iv3_gru\my_run\
```

## Additional Resources

- [Model Guide](../models/MODEL_GUIDE.md): Model architectures and details
- [Data Guide](../data/DATA_GUIDE.md): Data preparation and organization
- [Preprocessing Guide](../preprocessing/docs/PREPROCESS_GUIDE.MD): Video preprocessing
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md): Model evaluation
