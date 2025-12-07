# Trained Model Guide

Management and usage of trained model artifacts for the FSLR pipeline.

## Overview

This directory contains trained models for Filipino Sign Language Recognition. Both SignTransformer and InceptionV3GRU models were trained under identical conditions using the same dataset, hyperparameters, and training configuration to ensure fair comparison.

## Directory Structure

```
trained_models/
├── transformer/
│   ├── FSL105_classification/
│   │   ├── SignTransformer_best.pt
│   │   ├── SignTransformer_last.pt
│   │   └── training_*.log
│   └── FSL105_ctc/
│       ├── SignTransformerCtc_best.pt
│       ├── SignTransformerCtc_last.pt
│       └── training_*.log
├── iv3_gru/
│   ├── FSL105_classification/
│   │   ├── InceptionV3GRU_best.pt
│   │   ├── InceptionV3GRU_last.pt
│   │   └── training_*.log
│   └── FSL105_ctc/
│       ├── InceptionV3GRUCtc_best.pt
│       ├── InceptionV3GRUCtc_last.pt
│       └── training_*.log
└── TRAINED_MODEL_GUIDE.md
```

---

## Model Organization

### Directory Structure

Models are organized by architecture and task type:

- **Classification models** (isolated sign recognition): `FSL105_classification/`
- **CTC models** (continuous sign recognition): `FSL105_ctc/`

### Training Configuration

Models in the `FSL105_classification/` directories were trained using:

**Dataset**:

- FSL-105 dataset
- 80/20 train/validation split
- 105 gloss classes, 10 category classes

**Training Parameters**:

- Epochs: 100
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: Adam
- Loss: Combined gloss + category (multi-task learning)
- Early stopping: 10 epochs patience

**Hardware**:

- GPU training with mixed precision (AMP)
- Gradient clipping: max_norm=1.0

### Model Performance

Both models were evaluated on the same validation set under identical conditions to enable direct comparison.

---

## Model Checkpoints

### File Format

PyTorch checkpoint files (`.pt`) containing:

- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch number
- `train_loss`: Training loss
- `val_loss`: Validation loss

### Loading Models

**SignTransformer**:

```python
import torch
from models.transformer import SignTransformer

# Load checkpoint
checkpoint = torch.load('trained_models/transformer/FSL105_classification/SignTransformer_best.pt')

# Initialize model with same architecture
model = SignTransformer(
    input_dim=178,
    emb_dim=256,
    num_heads=8,
    num_layers=4,
    num_gloss=105,
    num_cat=10,
    dropout=0.1,
    pooling_method='mean'
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**InceptionV3GRU**:

```python
import torch
from models.iv3_gru import InceptionV3GRU

# Load checkpoint
checkpoint = torch.load('trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt')

# Initialize model with same architecture
model = InceptionV3GRU(
    num_gloss=105,
    num_cat=10,
    hidden1=16,
    hidden2=12,
    dropout=0.3,
    pretrained_backbone=True,
    freeze_backbone=True
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Using Models for Inference

**Transformer (Keypoints)**:

```python
import numpy as np
import torch

# Load NPZ file with keypoints
data = np.load('path/to/clip.npz')
keypoints = data['X']  # Shape: [T, 178]

# Convert to tensor and add batch dimension
x = torch.from_numpy(keypoints).float().unsqueeze(0)  # [1, T, 178]

# Forward pass
with torch.no_grad():
    gloss_logits, cat_logits = model(x)

# Get predictions
gloss_pred = torch.argmax(gloss_logits, dim=-1).item()
cat_pred = torch.argmax(cat_logits, dim=-1).item()

print(f"Predicted gloss: {gloss_pred}")
print(f"Predicted category: {cat_pred}")
```

**InceptionV3GRU (Features)**:

```python
import numpy as np
import torch

# Load NPZ file with features
data = np.load('path/to/clip.npz')
features = data['X2048']  # Shape: [T, 2048]

# Convert to tensor and add batch dimension
x = torch.from_numpy(features).float().unsqueeze(0)  # [1, T, 2048]

# Forward pass with precomputed features
with torch.no_grad():
    gloss_logits, cat_logits = model(x, features_already=True)

# Get predictions
gloss_pred = torch.argmax(gloss_logits, dim=-1).item()
cat_pred = torch.argmax(cat_logits, dim=-1).item()

print(f"Predicted gloss: {gloss_pred}")
print(f"Predicted category: {cat_pred}")
```

---

## Training Logs

### Log Files

Training logs contain epoch-by-epoch metrics:

- `training_*.log`: Console output with progress
- `training_*_metrics.csv`: Structured metrics (if enabled)

### Metrics Tracked

- `epoch`: Epoch number
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `val_gloss_acc`: Gloss validation accuracy
- `val_cat_acc`: Category validation accuracy
- `lr`: Learning rate
- `epoch_time`: Time per epoch

---

## Integration with Pipeline

### Streamlit Application

The Streamlit app automatically loads models from these directories:

```python
# streamlit_app/core/config.py
MODEL_CONFIG = {
    'transformer': {
        'checkpoint_path': 'trained_models/transformer/FSL105_classification/SignTransformer_best.pt',
        ...
    },
    'iv3_gru': {
        'checkpoint_path': 'trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt',
        ...
    }
}
```

### Command-Line Inference

**Using trained Transformer**:

```powershell
python evaluation/prediction/predict.py ^
  --model transformer_isolated ^
  --checkpoint trained_models/transformer/FSL105_classification/SignTransformer_best.pt ^
  --input path/to/clip.npz
```

**Using trained InceptionV3GRU**:

```powershell
python evaluation/prediction/predict.py ^
  --model iv3_gru_isolated ^
  --checkpoint trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt ^
  --input path/to/clip.npz
```

### Model Validation

```powershell
python evaluation/validation/validate.py ^
  --model transformer_isolated ^
  --checkpoint trained_models/transformer/FSL105_classification/SignTransformer_best.pt ^
  --data-dir data/processed/FSL105_val ^
  --labels-csv data/processed/FSL105_val.csv
```

---

## Organizing New Models

### Training New Models

When training new models, specify output directory:

```powershell
python training/train.py ^
  --model transformer_isolated ^
  --output-dir trained_models/transformer/FSL105_classification ^
  --keypoints-train data/processed/FSL105_train ^
  --keypoints-val data/processed/FSL105_val ^
  --labels-train-csv data/processed/FSL105_train.csv ^
  --labels-val-csv data/processed/FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100
```

### Recommended Structure

```
trained_models/
├── transformer/
│   ├── FSL105_classification/    # Classification models
│   ├── FSL105_ctc/                # CTC models
│   └── experiment_name/           # Experimental runs
└── iv3_gru/
    ├── FSL105_classification/     # Classification models
    ├── FSL105_ctc/                # CTC models
    └── experiment_name/            # Experimental runs
```

### Naming Convention

Use descriptive folder names:

- `FSL105_classification/` - Classification models (isolated signs)
- `FSL105_ctc/` - CTC models (continuous recognition)
- `experiment_name/` - Descriptive experiment name

---

## Model Comparison

Models in the same task directory (e.g., `FSL105_classification/`) can be directly compared because they were:

1. **Trained on same data**: FSL-105 with identical train/val split
2. **Same hyperparameters**: Learning rate, batch size, epochs
3. **Same loss function**: Multi-task learning with same weights (classification) or CTCLoss (CTC)
4. **Same evaluation**: Identical validation set and metrics

This ensures any performance differences are due to model architecture, not training conditions.

---

## Best Practices

### File Management

- Use `FSL105_classification/` and `FSL105_ctc/` for production models
- Use descriptive folder names for experiments
- Document changes in separate README per experiment
- Keep training logs with checkpoints

### Model Updates

When updating production models:

1. Train new model in separate directory
2. Validate performance on test set
3. Compare with current production model
4. If better, replace production model
5. Archive old model with date suffix

### Checkpointing

- `*_best.pt`: Best validation performance (use for inference)
- `*_last.pt`: Latest epoch (use for resuming training)

---

## Additional Resources

- [Training Guide](../training/TRAINING_GUIDE.md): How to train models
- [Model Guide](../models/MODEL_GUIDE.md): Model architectures
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md): Model evaluation
- [Prediction Guide](../evaluation/prediction/PREDICTION_GUIDE.md): Inference usage
