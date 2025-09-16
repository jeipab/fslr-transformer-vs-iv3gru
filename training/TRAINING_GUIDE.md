# Training Guide

## Overview

This guide covers training sign language recognition models using either Transformer (keypoints) or InceptionV3+GRU (features) architectures. The training script includes comprehensive performance optimizations for CUDA, memory management, and data loading.

## Prerequisites

data\processed\seq prepro_30 fps_09-13\clip_0004_good morning.parquet```bash
pip install -r requirements.txt

````

## Quick Start

### Basic Training Commands

**Transformer (Keypoints)**:

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32
````

**IV3-GRU (Features)**:

```bash
python training/train.py \
  --model iv3_gru \
  --features-train data/processed/iv3_gru_only \
  --features-val data/processed/test_15fps \
  --labels-train-csv data/processed/iv3_gru_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32
```

## Data Structure Requirements

### Directory Structure

```
data/processed/
├── transformer_only/          # Transformer training data
│   ├── clip_0089_how are you.npz
│   └── labels.csv
├── iv3_gru_only/             # IV3-GRU training data
│   ├── clip_0032_good afternoon.npz
│   └── labels.csv
└── test_15fps/               # Validation data
    ├── clip_0161_thank you.npz
    └── labels.csv
```

### Data Format Requirements

**NPZ Files**:

- **Transformer**: Key `X` with shape `[T, 156]` (variable sequence lengths)
- **IV3-GRU**: Key `X2048` (or `X`) with shape `[T, 2048]` (variable sequence lengths)

**Labels CSV**:

- Required columns: `file`, `gloss`, `cat`
- `file`: NPZ filename without extension (e.g., `clip_0089_how are you`)
- `gloss`: Gloss class ID (0-based, range: 0 to num_gloss-1)
- `cat`: Category class ID (0-based, range: 0 to num_cat-1)

Example labels.csv:

```csv
file,gloss,cat
clip_0089_how are you,42,3
clip_0032_good afternoon,15,1
```

## Performance Optimizations

### Automatic Optimizations

The training script automatically optimizes for your hardware:

- **Device Detection**: Automatically uses CUDA, MPS (Apple Silicon), or CPU
- **Memory Management**: Optimized GPU memory allocation and cleanup
- **DataLoader Optimization**: Auto-detects optimal number of workers and prefetch settings
- **Mixed Precision**: Automatic mixed precision (AMP) on CUDA devices

### Manual Performance Tuning

**For High-End GPUs**:

```bash
python training/train.py \
  --model transformer \
  --batch-size 64 \
  --amp \
  --compile-model \
  --auto-workers \
  --gradient-accumulation-steps 2
```

**For Limited GPU Memory**:

```bash
python training/train.py \
  --model transformer \
  --batch-size 16 \
  --gradient-accumulation-steps 4 \
  --amp
```

**For CPU Training**:

```bash
python training/train.py \
  --model transformer \
  --batch-size 8 \
  --num-workers 4
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

| Parameter                       | Description                    | Default | Notes                       |
| ------------------------------- | ------------------------------ | ------- | --------------------------- |
| `--amp`                         | Enable mixed precision         | `False` | Faster training on CUDA     |
| `--compile-model`               | Compile model (PyTorch 2.0+)   | `False` | Better performance          |
| `--auto-workers`                | Auto-detect DataLoader workers | `False` | Optimal worker count        |
| `--gradient-accumulation-steps` | Gradient accumulation          | `1`     | Effective larger batch size |
| `--num-workers`                 | DataLoader workers             | `0`     | 0 = auto-detect             |
| `--pin-memory`                  | Pin memory for GPU             | `False` | Faster GPU transfers        |

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

## Advanced Training Examples

### High-Performance GPU Training

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 50 --batch-size 64 \
  --lr 3e-4 --weight-decay 1e-4 \
  --amp --compile-model --auto-workers \
  --gradient-accumulation-steps 2 \
  --grad-clip 1.0 \
  --scheduler cosine --early-stop 10 \
  --log-csv logs/transformer_performance.csv
```

### Memory-Efficient Training

```bash
python training/train.py \
  --model iv3_gru \
  --features-train data/processed/iv3_gru_only \
  --features-val data/processed/test_15fps \
  --labels-train-csv data/processed/iv3_gru_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 40 --batch-size 16 \
  --gradient-accumulation-steps 4 \
  --amp --auto-workers \
  --scheduler plateau --scheduler-patience 3 \
  --early-stop 8 \
  --log-csv logs/iv3_gru_efficient.csv
```

### CPU Training

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 20 --batch-size 8 \
  --num-workers 4 \
  --scheduler plateau \
  --log-csv logs/cpu_training.csv
```

## Monitoring Training

### Real-Time Monitoring

The training script provides comprehensive monitoring:

- **Device Information**: GPU specs, memory, compute capability
- **Performance Metrics**: Epoch time, validation time
- **Memory Usage**: GPU memory allocation and reservation
- **Training Progress**: Loss, accuracy, learning rate

### CSV Logging

Enable detailed logging with `--log-csv`:

```bash
--log-csv logs/training_metrics.csv
```

Logs include:

- `epoch`: Epoch number
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `val_gloss_acc`: Gloss accuracy
- `val_cat_acc`: Category accuracy
- `lr`: Current learning rate
- `epoch_time`: Time per epoch
- `gpu_memory_allocated`: GPU memory used
- `gpu_memory_reserved`: GPU memory reserved

## Checkpointing

### Automatic Checkpointing

The training script automatically saves:

- `{ModelName}_last.pt`: Latest checkpoint (every epoch)
- `{ModelName}_best.pt`: Best checkpoint (highest validation accuracy)

### Resuming Training

```bash
python training/train.py \
  --resume data/processed/SignTransformer_last.pt \
  # ... other parameters
```

## Smoke Tests

Quick sanity checks without real data:

### Transformer Smoke Test

```bash
python training/train.py \
  --model transformer \
  --smoke-test \
  --num-gloss 105 --num-cat 10
```

### IV3-GRU Smoke Test

```bash
python training/train.py \
  --model iv3_gru \
  --smoke-test \
  --num-gloss 105 --num-cat 10 \
  --no-pretrained-backbone
```

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:

- Reduce `--batch-size`
- Increase `--gradient-accumulation-steps`
- Use `--amp` for mixed precision

**Slow Training**:

- Enable `--amp` for GPU training
- Use `--compile-model` (PyTorch 2.0+)
- Enable `--auto-workers` for optimal DataLoader
- Increase `--batch-size` if memory allows

**Data Loading Issues**:

- Check NPZ file keys (`--kp-key`, `--feature-key`)
- Verify CSV column names (`file`, `gloss`, `cat`)
- Ensure file paths match between CSV and NPZ files

**Convergence Issues**:

- Adjust learning rate (`--lr`)
- Try different schedulers (`--scheduler`)
- Adjust loss weights (`--alpha`, `--beta`)
- Use gradient clipping (`--grad-clip`)

### Data Validation

Validate your data before training:

```bash
# Validate NPZ files
python -m preprocessing.validate_npz data/processed/transformer_only

# Require X2048 for IV3-GRU
python -m preprocessing.validate_npz data/processed/iv3_gru_only --require-x2048
```

## Best Practices

### Training Strategy

1. **Start Small**: Begin with smoke tests to verify setup
2. **Monitor Memory**: Watch GPU memory usage during training
3. **Use Validation**: Always use validation data for monitoring
4. **Save Logs**: Enable CSV logging for analysis
5. **Checkpoint**: Resume capability for long training runs

### Performance Tips

1. **GPU Training**: Always use `--amp` for CUDA devices
2. **Batch Size**: Start with 32, adjust based on memory
3. **Workers**: Use `--auto-workers` for optimal DataLoader performance
4. **Compilation**: Use `--compile-model` for PyTorch 2.0+ performance boost
5. **Memory**: Use gradient accumulation for larger effective batch sizes

### Hyperparameter Tuning

1. **Learning Rate**: Start with 1e-4, adjust based on loss curves
2. **Scheduler**: Use `plateau` for stable training, `cosine` for faster convergence
3. **Early Stopping**: Set patience based on validation curve stability
4. **Loss Weights**: Balance `--alpha` and `--beta` based on task priorities

## Model-Specific Notes

### Transformer Model

- Uses attention masks for variable-length sequences
- Benefits from larger batch sizes
- Good for sequence modeling tasks

### IV3-GRU Model

- Uses InceptionV3 backbone (can be pretrained or frozen)
- Processes visual features efficiently
- Good for feature-based recognition

Both models support multi-task learning with configurable loss weights for gloss and category classification.
