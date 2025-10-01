# Training Guide

## Overview

This guide covers training sign language recognition models using either Transformer (keypoints) or InceptionV3+GRU (features) architectures. The training script has been **completely optimized** for real data training with performance optimizations for CUDA, memory management, data loading, and **automatic parallelization** for multi-GPU setups.

## Prerequisites

```bash
pip install -r requirements.txt
```

## ⚠️ **Important: Real Data Required**

The training script now **requires real data files** and no longer supports synthetic data. You must provide:

- **For Transformer**: Keypoint data files [T, 156], feature data files [T, 2048], or combined [T, 2204]
- **For IV3-GRU**: Feature data files [T, 2048] with CSV labels

## Quick Start

### Basic Training Commands

**Transformer (Keypoints)**:

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 50 --batch-size 32 \
  --auto-workers --auto-batch-size --enable-parallel \
  --amp --compile-model
```

**Transformer (Combined: Keypoints + Features)**:

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --kp-key X \
  --feature-key X2048 \
  --combine-features \
  --num-gloss 105 --num-cat 10 \
  --epochs 50 --batch-size 32 \
  --auto-workers --auto-batch-size --enable-parallel \
  --amp --compile-model
```

**IV3-GRU (Features)**:

```bash
python training/train.py \
  --model iv3_gru \
  --features-train data/processed/prepro_09-18 \
  --features-val data/processed/prepro_09-18 \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 50 --batch-size 32 \
  --auto-workers --auto-batch-size --enable-parallel \
  --amp --compile-model
```

### Multi-GPU Training

**For Vast AI or Multi-GPU Systems**:

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 100 --batch-size 64 \
  --auto-workers --auto-batch-size --enable-parallel \
  --amp --compile-model \
  --lr 5e-5 --weight-decay 1e-4 \
  --scheduler plateau --grad-clip 1.0
```

**For Local Machine (CPU/Single GPU)**:

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 50 --batch-size 16 \
  --auto-workers --auto-batch-size \
  --lr 1e-4 --weight-decay 1e-3 \
  --scheduler plateau
```

## Data Structure Requirements

### Directory Structure

```
data/processed/
├── keypoints_train/          # Transformer training data
│   ├── clip_0089_how are you.npz
│   └── ...
├── keypoints_val/            # Transformer validation data
│   ├── clip_0161_thank you.npz
│   └── ...
├── prepro_09-18/            # IV3-GRU features (both train/val)
│   ├── clip_0032_good afternoon.npz
│   └── ...
├── train_labels.csv         # Training labels
└── val_labels.csv           # Validation labels
```

### Data Format Requirements

**NPZ Files**:

- **Transformer (Keypoints)**: Key `X` with shape `[T, 156]` (variable sequence lengths)
- **Transformer (Features)**: Key `X2048` with shape `[T, 2048]` (variable sequence lengths)
- **Transformer (Combined)**: Both keys `X` [T, 156] and `X2048` [T, 2048] in same file → concatenated to [T, 2204]
- **IV3-GRU**: Key `X2048` (or `X`) with shape `[T, 2048]` (variable sequence lengths)

**Labels CSV**:

- Required columns: `file`, `gloss`, `cat`
- `file`: NPZ filename without extension (e.g., `clip_0089_how are you`)
- `gloss`: Gloss class ID (0-based, range: 0 to num_gloss-1)
- `cat`: Category class ID (0-based, range: 0 to num_cat-1)

Example train_labels.csv:

```csv
file,gloss,cat
clip_0089_how are you,42,3
clip_0032_good afternoon,15,1
```

Example val_labels.csv:

```csv
file,gloss,cat
clip_0161_thank you,28,2
clip_0250_good morning,5,1
```

## Recent Improvements

### ✅ **Code Quality Enhancements**

- **Comprehensive Documentation**: Added detailed comments explaining every step of training
- **Clean Import Organization**: Consolidated imports for better readability
- **Enhanced Error Messages**: Clear descriptions for debugging issues
- **Improved Function Documentation**: Better docstrings for all functions and classes

### ✅ **Advanced Training Features**

- **Curriculum Learning**: Gloss-first, category-first, and dynamic strategies
- **Loss Weighting**: Static, grid-search, uncertainty, and gradnorm approaches
- **Performance Optimization**: Auto batch sizing, worker detection, and memory management

## Performance Optimizations

The training script automatically optimizes for your hardware:

- **Device Detection**: CUDA, MPS (Apple Silicon), or CPU
- **Memory Management**: Optimized GPU memory allocation
- **DataLoader Optimization**: Auto-detects workers (up to 8) and prefetch settings
- **Mixed Precision**: AMP on CUDA devices
- **Dynamic Batch Sizing**: Calculates batch size based on available memory
- **Multi-GPU Support**: Automatic DataParallel when multiple GPUs detected

### Multi-GPU Training

```bash
python training/train.py \
  --model transformer \
  --enable-parallel \
  --auto-batch-size \
  --auto-workers
```

**Resource Adaptation**:

- **GPU Memory**: Batch size adjusts (8-64) based on available memory
- **CPU Cores**: Uses up to 8 DataLoader workers
- **Multi-GPU**: Distributes training across available GPUs

### Performance Examples

**Multi-GPU**:

```bash
python training/train.py --model transformer --batch-size 64 --amp --compile-model --auto-workers --auto-batch-size --enable-parallel --gradient-accumulation-steps 2
```

**Limited Memory**:

```bash
python training/train.py --model transformer --batch-size 16 --auto-batch-size --gradient-accumulation-steps 4 --amp --auto-workers
```

**CPU Training**:

```bash
python training/train.py --model transformer --batch-size 8 --auto-workers --auto-batch-size
```

## Training Parameters

### Essential Parameters

| Parameter      | Description                | Default       | Notes                                    |
| -------------- | -------------------------- | ------------- | ---------------------------------------- |
| `--model`      | Model architecture         | `transformer` | `transformer` or `iv3_gru`               |
| `--epochs`     | Training epochs            | `20`          | Adjust based on convergence              |
| `--batch-size` | Batch size                 | `32`          | Reduce if OOM, increase if memory allows |
| `--lr`         | Learning rate              | `1e-4`        | Start conservative, adjust based on loss |
| `--num-gloss`  | Number of gloss classes    | `105`         | Must match your dataset                  |
| `--num-cat`    | Number of category classes | `10`          | Must match your dataset                  |

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

## Curriculum Training

Curriculum training focuses on one task initially, then gradually introduces the other task. Useful when tasks have different difficulty levels.

### Curriculum Examples

**Gloss-First**:

```bash
python training/train.py --model transformer --keypoints-train data/processed/keypoints_train --keypoints-val data/processed/keypoints_val --labels-train-csv data/processed/train_labels.csv --labels-val-csv data/processed/val_labels.csv --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 32 --curriculum gloss-first --curriculum-epochs 15 --curriculum-min-weight 0.1 --curriculum-schedule linear --amp --compile-model --auto-workers --auto-batch-size
```

**Category-First**:

```bash
python training/train.py --model iv3_gru --features-train data/processed/prepro_09-18 --features-val data/processed/prepro_09-18 --labels-train-csv data/processed/train_labels.csv --labels-val-csv data/processed/val_labels.csv --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 32 --curriculum category-first --curriculum-epochs 20 --curriculum-min-weight 0.05 --curriculum-schedule cosine --amp --compile-model --auto-workers --auto-batch-size
```

**Dynamic**:

```bash
python training/train.py --model transformer --keypoints-train data/processed/keypoints_train --keypoints-val data/processed/keypoints_val --labels-train-csv data/processed/train_labels.csv --labels-val-csv data/processed/val_labels.csv --num-gloss 105 --num-cat 10 --epochs 60 --batch-size 32 --curriculum dynamic --curriculum-warmup 5 --curriculum-epochs 20 --curriculum-min-weight 0.1 --curriculum-schedule exponential --amp --compile-model --auto-workers --auto-batch-size
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

**Multi-GPU**:

```bash
python training/train.py --model transformer --keypoints-train data/processed/keypoints_train --keypoints-val data/processed/keypoints_val --labels-train-csv data/processed/train_labels.csv --labels-val-csv data/processed/val_labels.csv --num-gloss 105 --num-cat 10 --epochs 100 --batch-size 64 --lr 5e-5 --amp --compile-model --auto-workers --auto-batch-size --enable-parallel --gradient-accumulation-steps 2 --grad-clip 1.0 --scheduler plateau --early-stop 15 --log-csv logs/transformer_multi_gpu.csv
```

**Single GPU**:

```bash
python training/train.py --model transformer --keypoints-train data/processed/keypoints_train --keypoints-val data/processed/keypoints_val --labels-train-csv data/processed/train_labels.csv --labels-val-csv data/processed/val_labels.csv --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 32 --lr 3e-4 --amp --compile-model --auto-workers --auto-batch-size --gradient-accumulation-steps 2 --grad-clip 1.0 --scheduler cosine --early-stop 10 --log-csv logs/transformer_single_gpu.csv
```

**CPU Training**:

```bash
python training/train.py --model transformer --keypoints-train data/processed/keypoints_train --keypoints-val data/processed/keypoints_val --labels-train-csv data/processed/train_labels.csv --labels-val-csv data/processed/val_labels.csv --num-gloss 105 --num-cat 10 --epochs 30 --batch-size 8 --auto-workers --auto-batch-size --scheduler plateau --log-csv logs/cpu_training.csv
```

## Monitoring & Checkpointing

### CSV Logging

Enable detailed logging: `--log-csv logs/training_metrics.csv`

Logs include: epoch, train_loss, val_loss, val_gloss_acc, val_cat_acc, lr, epoch_time, gpu_memory

### Automatic Checkpointing

- `{ModelName}_last.pt`: Latest checkpoint (every epoch)
- `{ModelName}_best.pt`: Best checkpoint (highest validation accuracy)

### Resume Training

```bash
python training/train.py --resume data/processed/SignTransformer_last.pt
```

## Smoke Tests

**Transformer**:

```bash
python training/train.py --model transformer --smoke-test --num-gloss 105 --num-cat 10
```

**IV3-GRU**:

```bash
python training/train.py --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10 --no-pretrained-backbone
```

## Troubleshooting

### Common Issues

**Out of Memory**: Reduce `--batch-size`, increase `--gradient-accumulation-steps`, use `--amp`

**Slow Training**: Enable `--amp`, `--compile-model`, `--auto-workers`, `--auto-batch-size`, `--enable-parallel`

**Data Issues**: Check file paths, CSV format (columns: file, gloss, cat), NPZ keys (X for keypoints, X2048 for features)

**Convergence**: Adjust `--lr`, try `--scheduler`, adjust `--alpha`/`--beta`, use `--grad-clip`

### Data Validation

```bash
python -m preprocessing.validate_npz data/processed/keypoints_train
python -m preprocessing.validate_npz data/processed/prepro_09-18 --require-x2048
```

## Best Practices

### Training Strategy

1. Validate data files and CSV labels before training
2. Start with few epochs to verify setup
3. Monitor GPU memory usage
4. Enable CSV logging with `--log-csv`
5. Use `--resume` for long training runs

### Performance Tips

- **GPU**: Use `--amp`, `--compile-model`, `--auto-batch-size`, `--auto-workers`
- **Multi-GPU**: Use `--enable-parallel` for automatic parallelization
- **Memory**: Use gradient accumulation for larger effective batch sizes

### Hyperparameter Tuning

- **Learning Rate**: Start with 1e-4, adjust based on loss curves
- **Scheduler**: Use `plateau` for stable training, `cosine` for faster convergence
- **Loss Weights**: Balance `--alpha` and `--beta` based on task priorities

## Model Notes

**Transformer**: Uses attention masks, benefits from larger batch sizes
**IV3-GRU**: Uses InceptionV3 backbone, processes visual features efficiently

Both support multi-task learning with configurable loss weights.
