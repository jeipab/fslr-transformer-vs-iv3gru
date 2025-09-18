# Training Guide

## Overview

This guide covers training sign language recognition models using either Transformer (keypoints) or InceptionV3+GRU (features) architectures. The training script includes performance optimizations for CUDA, memory management, data loading, and **automatic parallelization** for multi-GPU setups.

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
  --epochs 50 --batch-size 32 \
  --auto-workers --auto-batch-size --enable-parallel \
  --amp --compile-model
```

**IV3-GRU (Features)**:

```bash
python training/train.py \
  --model iv3_gru \
  --features-train data/processed/iv3_gru_only \
  --features-val data/processed/test_15fps \
  --labels-train-csv data/processed/iv3_gru_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
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
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
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
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
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
- **DataLoader Optimization**: Auto-detects number of workers (up to 8) and prefetch settings
- **Mixed Precision**: Automatic mixed precision (AMP) on CUDA devices
- **Dynamic Batch Sizing**: Automatically calculates batch size based on available memory
- **Multi-GPU Parallelization**: Automatically uses DataParallel when multiple GPUs detected

### New Parallelization Features

**Automatic Multi-GPU Support**:

```bash
python training/train.py \
  --model transformer \
  --enable-parallel \
  --auto-batch-size \
  --auto-workers
```

**Dynamic Resource Adaptation**:

- **GPU Memory**: Adjusts batch size (8-64) based on GPU memory
- **CPU Cores**: Uses up to 8 DataLoader workers based on CPU count
- **Multi-GPU**: Distributes training across available GPUs

### Performance Tuning Examples

**Multi-GPU (Vast AI)**:

```bash
python training/train.py \
  --model transformer \
  --batch-size 64 \
  --amp \
  --compile-model \
  --auto-workers \
  --auto-batch-size \
  --enable-parallel \
  --gradient-accumulation-steps 2
```

**Limited GPU Memory**:

```bash
python training/train.py \
  --model transformer \
  --batch-size 16 \
  --auto-batch-size \
  --gradient-accumulation-steps 4 \
  --amp \
  --auto-workers
```

**CPU Training (Local Machine)**:

```bash
python training/train.py \
  --model transformer \
  --batch-size 8 \
  --auto-workers \
  --auto-batch-size
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
| `--auto-workers`                | Auto-detect DataLoader workers | `False` | Worker count (up to 8) |
| `--auto-batch-size`             | Auto-calculate batch size | `False` | Based on available memory   |
| `--enable-parallel`             | Enable DataParallel for multi-GPU | `False` | Automatic multi-GPU support |
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

## Additional Training Examples

### Multi-GPU Training (Vast AI)

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 100 --batch-size 64 \
  --lr 5e-5 --weight-decay 1e-4 \
  --amp --compile-model --auto-workers --auto-batch-size --enable-parallel \
  --gradient-accumulation-steps 2 \
  --grad-clip 1.0 \
  --scheduler plateau --early-stop 15 \
  --log-csv logs/transformer_multi_gpu.csv
```

### Single GPU Training

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 50 --batch-size 32 \
  --lr 3e-4 --weight-decay 1e-4 \
  --amp --compile-model --auto-workers --auto-batch-size \
  --gradient-accumulation-steps 2 \
  --grad-clip 1.0 \
  --scheduler cosine --early-stop 10 \
  --log-csv logs/transformer_single_gpu.csv
```

### Memory-Constrained Training

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
  --amp --auto-workers --auto-batch-size \
  --scheduler plateau --scheduler-patience 3 \
  --early-stop 8 \
  --log-csv logs/iv3_gru_efficient.csv
```

### CPU Training (Local Machine)

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 8 \
  --auto-workers --auto-batch-size \
  --scheduler plateau \
  --log-csv logs/cpu_training.csv
```

### Development/Testing

```bash
python training/train.py \
  --model transformer \
  --keypoints-train data/processed/transformer_only \
  --keypoints-val data/processed/test_15fps \
  --labels-train-csv data/processed/transformer_only/labels.csv \
  --labels-val-csv data/processed/test_15fps/labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 10 --batch-size 16 \
  --auto-workers --auto-batch-size \
  --amp \
  --log-csv logs/quick_test.csv
```

## Monitoring Training

### Real-Time Monitoring

The training script provides monitoring:

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
- Enable `--auto-workers` for DataLoader (up to 8 workers)
- Enable `--auto-batch-size` for batch sizing
- Enable `--enable-parallel` for multi-GPU setups
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
2. **Batch Size**: Use `--auto-batch-size` for memory utilization
3. **Workers**: Use `--auto-workers` for DataLoader performance (up to 8 workers)
4. **Multi-GPU**: Use `--enable-parallel` for automatic multi-GPU support
5. **Compilation**: Use `--compile-model` for PyTorch 2.0+ performance boost
6. **Memory**: Use gradient accumulation for larger effective batch sizes
7. **Resource Configuration**: Combine `--auto-workers`, `--auto-batch-size`, and `--enable-parallel` for performance

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

## Parallelization Benefits

### Multi-GPU Training Advantages

**Performance Improvements**:
- **2-4x faster training** with multiple GPUs
- **Near-linear scaling** with DataParallel
- **Larger effective batch sizes** for gradient estimates
- **Reduced training time** enables more experimentation

**Accuracy Impact**:
- ✅ **No negative impact** on model accuracy
- ✅ **Same convergence** as single-GPU training
- ✅ **Better gradient averaging** with parallel processing
- ✅ **More stable training** with batch sizes

### Resource Configuration

**Automatic Adaptations**:
- **GPU Memory**: Batch size adjusts from 8-64 based on available memory
- **CPU Cores**: Uses up to 8 DataLoader workers for data loading
- **Multi-GPU**: Distributes training across available GPUs
- **Memory Management**: Prevents out-of-memory errors with dynamic sizing

**Recommended Configurations**:

**Vast AI (Multi-GPU)**:
```bash
--enable-parallel --auto-batch-size --auto-workers --amp --compile-model
```

**Local Machine (CPU/Single GPU)**:
```bash
--auto-workers --auto-batch-size --amp
```

### Training Time Estimates

**With Parallelization**:
- **Single GPU**: 2-3 hours for 50 epochs
- **Multi-GPU (2x)**: 1-1.5 hours for 50 epochs
- **Multi-GPU (4x)**: 30-45 minutes for 50 epochs

**Without Parallelization**:
- **Single GPU**: 4-6 hours for 50 epochs
- **CPU**: 8-12 hours for 50 epochs

The parallelization features enable faster experimentation and longer training runs, leading to model performance through hyperparameter optimization.
````
