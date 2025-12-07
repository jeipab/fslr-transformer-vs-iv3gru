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
  --model transformer_isolated ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --output-dir trained_models/transformer/FSL105_classification
```

**InceptionV3-GRU (Features)**:

```powershell
python training/train.py ^
  --model iv3_gru_isolated ^
  --features-train data/processed/FSL105_train ^
  --features-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --output-dir trained_models/iv3_gru/FSL105_classification
```

### Performance Options

**GPU Training**:

```powershell
python training/train.py ^
  --model transformer_isolated ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --amp ^
  --auto-workers ^
  --scheduler plateau ^
  --early-stop 10 ^
  --output-dir trained_models/transformer/FSL105_classification
```

**Multi-GPU Training**:

```powershell
python training/train.py ^
  --model transformer_isolated ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 64 ^
  --amp ^
  --enable-parallel ^
  --auto-workers ^
  --output-dir trained_models/transformer/FSL105_classification
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
├── FSL105_train/                 # Training data (80%)
│   ├── clip_0315_yes.npz
│   ├── clip_1601_orange.npz
│   └── ...
├── FSL105_val/                   # Validation data (20%)
│   ├── clip_0138_nice to meet you.npz
│   ├── clip_1146_grandfather.npz
│   └── ...
├── FSL105_train.csv              # Training labels
└── FSL105_val.csv                # Validation labels
```

### NPZ File Format

**Transformer (Keypoints)**:

- Key: `X`
- Shape: `[T, 178]`
- Content: 89 keypoints × 2 coordinates (x, y)

**InceptionV3-GRU (Features)**:

- Key: `X2048`
- Shape: `[T, 2048]`
- Content: Precomputed InceptionV3 features

**Additional keys**:

- `mask`: `[T, 89]` - Keypoint visibility
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

| Parameter      | Default                | Description                                                                                       |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------- |
| `--model`      | `transformer_isolated` | Model: `transformer_isolated`, `iv3_gru_isolated`, `transformer_continuous`, `iv3_gru_continuous` |
| `--epochs`     | `20`                   | Number of training epochs                                                                         |
| `--batch-size` | `32`                   | Batch size                                                                                        |
| `--lr`         | `1e-4`                 | Learning rate                                                                                     |
| `--num-gloss`  | `105`                  | Number of gloss classes                                                                           |
| `--num-cat`    | `10`                   | Number of category classes                                                                        |

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
  --model transformer_isolated ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --curriculum gloss-first ^
  --curriculum-epochs 15 ^
  --output-dir trained_models/transformer/FSL105_classification
```

**Category-First**:

```powershell
python training/train.py ^
  --model iv3_gru_isolated ^
  --features-train data/processed/FSL105_train ^
  --features-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 50 ^
  --curriculum category-first ^
  --curriculum-epochs 20 ^
  --output-dir trained_models/iv3_gru/FSL105_classification
```

**Dynamic**:

```powershell
python training/train.py ^
  --model transformer_isolated ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 60 ^
  --curriculum dynamic ^
  --curriculum-epochs 20 ^
  --output-dir trained_models/transformer/FSL105_classification
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
  --resume trained_models/transformer/FSL105_classification/SignTransformer_last.pt ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv
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
python preprocessing/utils/validate_npz.py data/processed/FSL105_train

# Require X2048 features
python preprocessing/utils/validate_npz.py data/processed/FSL105_val --require-x2048
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

## CTC Training (Continuous Recognition)

Train sequence-to-sequence models for continuous sign language recognition using CTCLoss.

### Quick Start

**SignTransformerCtc** (Windows):

```powershell
python training/train.py ^
  --model transformer_continuous ^
  --keypoints-train data\processed\FSL105_train ^
  --keypoints-val data\processed\FSL105_val ^
  --labels-train-csv data\processed\FSL105_train.csv ^
  --labels-val-csv data\processed\FSL105_val.csv ^
  --epochs 100 ^
  --batch-size 32 ^
  --lr 0.0001 ^
  --grad-clip 1.0 ^
  --scheduler warmup_cosine ^
  --warmup-epochs 10 ^
  --amp ^
  --output-dir trained_models\transformer\FSL105_ctc
```

**InceptionV3GRUCtc** (Features):

```powershell
python training/train.py ^
  --model iv3_gru_continuous ^
  --training-mode ctc ^
  --features-train data\processed\FSL105_train ^
  --features-val data\processed\FSL105_val ^
  --labels-train-csv data\processed\FSL105_train.csv ^
  --labels-val-csv data\processed\FSL105_val.csv ^
  --feature-key X2048 ^
  --epochs 100 ^
  --batch-size 32 ^
  --lr 0.0001 ^
  --hidden1 16 ^
  --hidden2 12 ^
  --dropout 0.3 ^
  --grad-clip 1.0 ^
  --scheduler warmup_cosine ^
  --warmup-epochs 10 ^
  --amp ^
  --output-dir trained_models\iv3_gru\FSL105_ctc
```

### Production Training Examples

These examples show the actual training commands used to train the production models with optimized hyperparameters.

**SignTransformerCtc** (Production):

```powershell
python -m training.train ^
  --model transformer_continuous ^
  --training-mode ctc ^
  --epochs 30 ^
  --batch-size 128 ^
  --lr 0.0007 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --alpha 0.8 ^
  --beta 0.2 ^
  --num-ctc-classes 106 ^
  --ctc-blank-id 105 ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --kp-key X ^
  --amp ^
  --augment ^
  --augment-noise-std 0.0038 ^
  --augment-mask-prob 0.085 ^
  --augment-mask-ratio 0.085 ^
  --scheduler plateau ^
  --scheduler-patience 30 ^
  --early-stop 120 ^
  --weight-decay 0.00013 ^
  --label-smoothing 0.115 ^
  --grad-clip 1.0 ^
  --output-dir trained_models/transformer/FSL105_ctc
```

**InceptionV3GRUCtc** (Production):

```powershell
python -m training.train ^
  --model iv3_gru_continuous ^
  --training-mode ctc ^
  --epochs 30 ^
  --batch-size 128 ^
  --lr 0.0007 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --alpha 0.8 ^
  --beta 0.2 ^
  --num-ctc-classes 106 ^
  --ctc-blank-id 105 ^
  --features-train data/processed/FSL105_train ^
  --features-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --feature-key X2048 ^
  --hidden1 42 ^
  --hidden2 34 ^
  --dropout 0.34 ^
  --no-freeze-backbone ^
  --amp ^
  --augment ^
  --augment-noise-std 0.0038 ^
  --augment-mask-prob 0.085 ^
  --augment-mask-ratio 0.085 ^
  --scheduler plateau ^
  --scheduler-patience 30 ^
  --early-stop 120 ^
  --weight-decay 0.00013 ^
  --label-smoothing 0.115 ^
  --grad-clip 1.0 ^
  --output-dir trained_models/iv3_gru/FSL105_ctc
```

**Note**: Both models were trained with identical hyperparameters (where applicable) to ensure fair comparison. The only differences are:

- **Data input**: Keypoints for Transformer vs Features for InceptionV3GRU
- **Model-specific parameters**: Transformer uses default architecture parameters, while InceptionV3GRU uses `hidden1=42`, `hidden2=34`, `dropout=0.34`, and `--no-freeze-backbone`

### CTC-Specific Parameters

| Parameter           | Description       | Default        | Recommended                |
| ------------------- | ----------------- | -------------- | -------------------------- |
| `--training-mode`   | Training mode     | classification | Auto-detected from model   |
| `--ctc-blank-id`    | Blank token ID    | 105            | 105 (num_gloss)            |
| `--num-ctc-classes` | Total classes     | 106            | 106 (glosses + blank)      |
| `--grad-clip`       | Gradient clipping | None           | **1.0** (critical for CTC) |
| `--scheduler`       | LR scheduler      | None           | warmup_cosine              |

### Best Practices

1. **Always use gradient clipping**: `--grad-clip 1.0`
2. **Use LR warmup**: `--scheduler warmup_cosine --warmup-epochs 10`
3. **Enable AMP**: `--amp` (2x faster on GPU)
4. **Monitor for NaN/Inf**: Training loss should decrease smoothly
5. **Target loss**: < 5.0 for isolated signs

### Troubleshooting CTC Training

**CTCLoss is NaN**:

```bash
# Use gradient clipping (critical!)
--grad-clip 1.0

# Reduce learning rate
--lr 5e-5
```

**Model predicts only blanks**:

```bash
# Use learning rate warmup
--scheduler warmup_cosine --warmup-epochs 10
```

**Out of memory**:

```bash
# Reduce batch size
--batch-size 16

# Or use gradient accumulation
--batch-size 8 --gradient-accumulation-steps 4
```

---

## Mobile Export (Android)

Export trained CTC models to TorchScript format optimized for Android deployment.

### Overview

The mobile export feature converts trained CTC models into optimized `.ptl` files for Android applications. The export process:

- Converts models to TorchScript format
- Optimizes for mobile inference
- Generates metadata and label mappings
- Creates verification reports

**Supported Models**: CTC models (currently `transformer_continuous`)

### Export During Training

Automatically export after training completes:

```powershell
python -m training.train ^
  --model transformer_continuous ^
  --training-mode ctc ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --epochs 30 ^
  --batch-size 128 ^
  --output-dir trained_models/transformer/FSL105_ctc ^
  --export-mobile ^
  --export-output android_artifacts ^
  --export-example-T 120 ^
  --window-hint 120 ^
  --stride-hint 40
```

### Export Standalone (After Training)

Export an existing checkpoint without retraining:

**Option 1: Using training script with `--export-only`**:

```powershell
python -m training.train ^
  --model transformer_continuous ^
  --export-only ^
  --resume trained_models/transformer/FSL105_ctc/SignTransformerCtc_best.pt ^
  --export-output android_artifacts ^
  --export-example-T 120 ^
  --window-hint 120 ^
  --stride-hint 40 ^
  --num-cat 10
```

**Option 2: Using export script directly**:

```powershell
python training/export_mobile.py ^
  --model transformer_continuous ^
  --resume-path trained_models/transformer/FSL105_ctc/SignTransformerCtc_best.pt ^
  --output-dir android_artifacts ^
  --input-dim 178 ^
  --num-cat 10 ^
  --window-hint 120 ^
  --stride-hint 40 ^
  --example-T 120
```

### Export Parameters

| Parameter            | Default             | Description                                                     |
| -------------------- | ------------------- | --------------------------------------------------------------- |
| `--export-mobile`    | `False`             | Enable export after training                                    |
| `--export-only`      | `False`             | Skip training, only export (requires `--resume`)                |
| `--export-output`    | `android_artifacts` | Output directory for exported files                             |
| `--export-example-T` | `120`               | Representative sequence length for tracing (if scripting fails) |
| `--window-hint`      | `120`               | Window size hint for Android (frames)                           |
| `--stride-hint`      | `40`                | Stride hint for Android (frames)                                |
| `--input-dim`        | `178`               | Input dimension (178 for keypoints, 2048 for features)          |
| `--num-cat`          | `10`                | Number of category classes                                      |

### Export Output Structure

```
android_artifacts/
├── SignTransformerCtc_best.ptl          # Optimized mobile model
├── SignTransformerCtc_best.model.json     # Model metadata
├── label_mapping.json                    # Gloss and category mappings
└── pytorch_mobile_export_report.md       # Verification report
```

### Metadata File Format

The `.model.json` file contains:

```json
{
  "input_dim": 178,
  "num_gloss": 105,
  "blank_id": 105,
  "num_ctc": 106,
  "num_cat": 10,
  "window_size_hint": 120,
  "stride_hint": 40,
  "decode_default": "greedy",
  "model_type": "sign_transformer_ctc",
  "labels_file": "label_mapping.json",
  "version": "1.0.0"
}
```

### Label Mapping File

The `label_mapping.json` file contains:

```json
{
  "glosses": {
    "0": "HELLO",
    "1": "THANK YOU",
    ...
  },
  "categories": {
    "0": "GREETING",
    "1": "SURVIVAL",
    ...
  }
}
```

### Model Output Contract

Exported models return a tuple:

- **CTC Log Probabilities**: `[1, T, 106]` - Log probabilities for CTC decoding
- **Category Logits**: `[1, T, 10]` - Category predictions (if `num_cat > 0`)

### Android Integration

Load the exported model in Android:

```kotlin
import org.pytorch.LiteModuleLoader

// Load model
val module = LiteModuleLoader.load("SignTransformerCtc_best.ptl")

// Prepare input tensor [1, T, 178]
val inputTensor = Tensor.fromBlob(keypoints, longArrayOf(1, T, 178))

// Forward pass
val outputs = module.forward(IValue.from(inputTensor)).toTuple()
val ctcLogProbs = outputs[0].toTensor()  // [1, T, 106]
val categoryLogits = outputs[1].toTensor()  // [1, T, 10]
```

### Troubleshooting

**Scripting fails**:

- Provide `--export-example-T` for tracing fallback
- Ensure model architecture is compatible with TorchScript

**Checkpoint not found**:

- Verify checkpoint path exists
- Use `--resume-path` to specify exact checkpoint location

**Model type not supported**:

- Only `transformer_continuous` and `mediapipe_gru_continuous` are supported
- Classification models (`transformer_isolated`, `iv3_gru_isolated`) cannot be exported

**Export skipped during training**:

- Check that model is a CTC model
- Verify best checkpoint exists in output directory

### Best Practices

1. **Export after training**: Use `--export-mobile` flag during training for automatic export
2. **Verify export**: Check the generated `pytorch_mobile_export_report.md` for validation
3. **Test on device**: Always test exported models on target Android devices
4. **Window/Stride hints**: Set appropriate values based on your Android app's sliding window configuration
5. **Input dimension**: Ensure `--input-dim` matches your model's input (178 for keypoints, 2048 for features)

---

## Additional Resources

- [Model Guide](../models/MODEL_GUIDE.md): Model architectures including CTC models
- [Prediction Guide](../evaluation/prediction/PREDICTION_GUIDE.md): Making predictions with CTC models
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md): Evaluating CTC performance
- [Data Guide](../data/DATA_GUIDE.md): Data preparation
- [Preprocessing Guide](../preprocessing/docs/PREPROCESS_GUIDE.MD): Video preprocessing
- [Trained Model Guide](../trained_models/TRAINED_MODEL_GUIDE.md): Model management
