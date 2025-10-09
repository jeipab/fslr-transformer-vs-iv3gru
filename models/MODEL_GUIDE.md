# Model Guide

Technical documentation for Filipino Sign Language Recognition (FSLR) model architectures.

## Overview

This project implements two architectures for Filipino Sign Language Recognition:

- **SignTransformer**: Transformer-based model for keypoint sequence processing
- **InceptionV3GRU**: CNN-RNN model for video frame processing

Both models predict sign words (gloss) and semantic categories.

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

## SignTransformer

### Architecture Overview

The SignTransformer processes sequences of body keypoints using a Transformer encoder:

```
Input: [B, T, 156] keypoint sequences
  ↓
Linear Embedding: [B, T, 156] → [B, T, E]
  ↓
Positional Encoding: Adds temporal order information
  ↓
Layer Normalization: Stabilizes training
  ↓
Transformer Encoder Stack (N layers):
  • Multi-Head Self-Attention + Residual Connection
  • Feed-Forward Network + Residual Connection
  ↓
Pooling Strategy (mean/max/cls): [B, T, E] → [B, E]
  ↓
Dual Output Heads:
  • Gloss Head: [B, E] → [B, num_gloss]
  • Category Head: [B, E] → [B, num_cat]
```

### Key Components

- **PositionalEncoding**: Sinusoidal temporal encoding
- **MultiHeadAttentionBlock**: Self-attention mechanism
- **FeedForwardBlock**: Position-wise feed-forward network
- **ResidualConnection**: Pre-layer normalization with residual connections
- **EncoderLayer**: Transformer layer with attention and feed-forward
- **Pooling Strategies**: Mean, max, or CLS token pooling
- **Classification Heads**: Separate heads for gloss and category prediction

### Usage

```python
from models.transformer import SignTransformer

# Initialize model with default parameters
model = SignTransformer(
    input_dim=156,        # 78 keypoints × 2 coordinates
    emb_dim=256,          # Embedding dimension
    num_heads=8,          # Number of attention heads
    num_layers=4,         # Number of encoder layers
    num_gloss=105,        # Number of gloss classes
    num_cat=10,           # Number of category classes
    dropout=0.1,          # Dropout rate
    pooling_method='mean' # Pooling strategy: 'mean', 'max', or 'cls'
)

# Forward pass with keypoint sequences
gloss_logits, cat_logits = model(x)  # x: [B, T, 156]

# Forward pass with attention mask for variable lengths
mask = torch.ones(B, T)  # 1 = valid frame, 0 = padding
gloss_logits, cat_logits = model(x, mask=mask)

# Get attention weights for visualization
attention_weights = model.get_attention_weights(x)
```

### Parameters

- `input_dim` (int): Input feature dimension per frame (default: 156)
- `emb_dim` (int): Embedding dimension (default: 256)
- `num_heads` (int): Number of attention heads (default: 8)
- `num_layers` (int): Number of encoder layers (default: 4)
- `num_gloss` (int): Number of gloss classes (default: 105)
- `num_cat` (int): Number of category classes (default: 10)
- `dropout` (float): Dropout rate (default: 0.1)
- `max_len` (int): Maximum sequence length (default: 300)
- `ff_dim` (int): Feed-forward hidden dimension (default: 4×emb_dim)
- `pooling_method` (str): Pooling strategy - 'mean', 'max', or 'cls' (default: 'mean')

## InceptionV3GRU

### Architecture Overview

The InceptionV3GRU combines visual feature extraction with temporal sequence modeling:

```
Input: [B, T, 2048] precomputed InceptionV3 features
  ↓
First GRU Layer: [B, T, 2048] → [B, T, hidden1] + Dropout
  ↓
Second GRU Layer: [B, T, hidden1] → [B, T, hidden2] + Dropout
  ↓
Final Hidden State: [B, hidden2]
  ↓
Dual Classification Heads:
  • Gloss Head: [B, hidden2] → [B, num_gloss]
  • Category Head: [B, hidden2] → [B, num_cat]
```

### Key Components

- **InceptionV3FeatureExtractor**: CNN backbone for visual feature extraction
- **GRU Layers**: Two-layer GRU network for temporal sequence modeling
- **Dropout Regularization**: Applied after each GRU layer
- **Weight Initialization**: Xavier/orthogonal initialization
- **Classification Heads**: Separate heads for gloss and category prediction
- **Packed Sequence Support**: Handling of variable-length sequences

### Usage

```python
from models.iv3_gru import InceptionV3GRU

# Initialize model
model = InceptionV3GRU(
    num_gloss=105,           # Number of gloss classes
    num_cat=10,              # Number of category classes
    hidden1=16,              # First GRU hidden dimension
    hidden2=12,              # Second GRU hidden dimension
    dropout=0.3,             # Dropout rate
    pretrained_backbone=True, # Load ImageNet weights
    freeze_backbone=True     # Freeze backbone for transfer learning
)

# Forward pass with precomputed features
gloss_logits, cat_logits = model(features, features_already=True)

# Forward pass with raw frames
gloss_logits, cat_logits = model(frames, features_already=False)

# Forward pass with variable-length sequences
lengths = torch.tensor([10, 15, 8])  # True sequence lengths
gloss_logits, cat_logits = model(frames, lengths=lengths, features_already=False)

# Get probabilities instead of logits
gloss_probs, cat_probs = model.predict_proba(frames, features_already=False)
```

### Parameters

- `num_gloss` (int): Number of gloss classes
- `num_cat` (int): Number of category classes
- `hidden1` (int): First GRU hidden dimension (default: 16)
- `hidden2` (int): Second GRU hidden dimension (default: 12)
- `dropout` (float): Dropout rate (default: 0.3)
- `pretrained_backbone` (bool): Load ImageNet weights (default: True)
- `freeze_backbone` (bool): Freeze InceptionV3 weights (default: True)

## Model Comparison

| Aspect               | SignTransformer              | InceptionV3GRU                          |
| :------------------- | :--------------------------- | :-------------------------------------- |
| **Input**            | Keypoints `[B, T, 156]`      | InceptionV3 features `[B, T, 2048]`     |
| **Architecture**     | Multi-head attention         | CNN + GRU                               |
| **Pretrained**       | No                           | InceptionV3 (ImageNet)                  |
| **Parameters**       | ~2M                          | ~25M (with frozen backbone)             |
| **Training**         | End-to-end                   | Can freeze backbone                     |
| **Memory**           | Lower                        | Higher                                  |
| **Interpretability** | Attention weights available  | Less interpretable                      |
| **Sequence Length**  | Fixed max length             | Variable length support                 |
| **Preprocessing**    | Requires keypoint extraction | Requires InceptionV3 feature extraction |

## Training Considerations

### SignTransformer

**Advantages:**

- Lower memory footprint and faster training
- Suitable for keypoint-based data
- Attention weights provide interpretability
- End-to-end training from scratch

**Considerations:**

- Requires keypoint extraction preprocessing
- Fixed maximum sequence length
- No pretrained weights available

**Training Tips:**

- Use attention masks for variable-length sequences
- Apply gradient clipping for training stability
- Monitor attention patterns for model interpretability
- Test different pooling strategies

### InceptionV3GRU

**Advantages:**

- Pretrained ImageNet features provide visual representations
- Precomputed features enable efficient training
- Variable-length sequence support with packed sequences
- Transfer learning capabilities

**Considerations:**

- Higher memory requirements
- Requires InceptionV3 feature extraction
- Less interpretable than attention-based models

**Training Tips:**

- Use precomputed features for faster training
- Start with frozen backbone for transfer learning
- Use mixed precision training (`--amp`) for efficiency
- Gradually unfreeze backbone after GRU head stabilizes

## Input Requirements

### SignTransformer

**Keypoint Sequences:**

- Shape: `[batch_size, sequence_length, 156]`
- 78 keypoints × 2 coordinates (x, y)
- Normalized coordinates in range `[0, 1]`
- Extracted using MediaPipe (pose, hands, face landmarks)
- Maximum sequence length: 300 frames (configurable)

**Keypoint Breakdown (78 points):**

- Pose: 25 landmarks
- Left hand: 21 landmarks
- Right hand: 21 landmarks
- Face: 11 landmarks

**Optional Attention Mask:**

- Shape: `[batch_size, sequence_length]`
- Values: 1 for valid frames, 0 for padding
- Used for variable-length sequence handling

### InceptionV3GRU

**Precomputed InceptionV3 Features:**

- Shape: `[batch_size, sequence_length, 2048]`
- Extracted using InceptionV3 backbone during preprocessing
- Stored in NPZ files with key `X2048`
- All training uses precomputed features for efficiency

**Variable-Length Sequences:**

- Supported through packed sequences
- Lengths tensor: `[batch_size]` with true sequence lengths

## Output Format

Both models return a tuple of two tensors:

- `gloss_logits`: `[batch_size, num_gloss]` - Raw classification scores for specific sign words
- `cat_logits`: `[batch_size, num_cat]` - Raw classification scores for semantic categories

**Converting to Predictions:**

```python
# Get probabilities
gloss_probs = torch.softmax(gloss_logits, dim=-1)
cat_probs = torch.softmax(cat_logits, dim=-1)

# Get predicted classes
gloss_pred = torch.argmax(gloss_logits, dim=-1)
cat_pred = torch.argmax(cat_logits, dim=-1)
```

## Performance Tips

### SignTransformer

**Training Optimization:**

- Use attention masks for variable-length sequences
- Apply gradient clipping (max_norm=1.0) for training stability
- Monitor attention patterns for model interpretability
- Experiment with different pooling strategies (mean, max, cls)

**Memory Optimization:**

- Use smaller batch sizes if memory is limited
- Consider reducing embedding dimension for faster training
- Use gradient accumulation for effective larger batch sizes

### InceptionV3GRU

**Training Optimization:**

- Use precomputed features stored in NPZ files
- Start with frozen backbone for transfer learning
- Use mixed precision training (`--amp`) for efficiency
- Gradually unfreeze backbone after GRU head stabilizes

**Memory Optimization:**

- Precomputed features reduce memory footprint significantly
- Use smaller hidden dimensions for GRU layers
- Implement gradient checkpointing for memory efficiency

## Training Integration

### SignTransformer Training

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
  --batch-size 32
```

### InceptionV3GRU Training

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
  --batch-size 32
```

For detailed training instructions, see [Training Guide](../training/TRAINING_GUIDE.md)

## Model Files

- `transformer.py`: SignTransformer implementation with documentation
- `iv3_gru.py`: InceptionV3GRU implementation with documentation
- `__init__.py`: Module initialization and imports
- `MODEL_GUIDE.md`: This guide

## Trained Models

Pre-trained models are available in `trained_models/`:

```
trained_models/
├── transformer/
│   └── cmb_optimal/
│       ├── SignTransformer_best.pt   # Best validation performance
│       └── SignTransformer_last.pt   # Most recent epoch
└── iv3_gru/
    └── cmb_optimal/
        ├── InceptionV3GRU_best.pt    # Best validation performance
        └── InceptionV3GRU_last.pt    # Most recent epoch
```

All models are trained on the combined dataset (fsl-105 + sample-105).

For model management details, see [Trained Model Guide](../trained_models/TRAINED_MODEL_GUIDE.md)

## Additional Resources

- [Training Guide](../training/TRAINING_GUIDE.md): Training instructions and hyperparameters
- [Data Guide](../data/DATA_GUIDE.md): Data preprocessing and preparation
- [Prediction Guide](../evaluation/prediction/PREDICTION_GUIDE.md): Model inference
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md): Model evaluation
- [Tool Guide](../streamlit_app/TOOL_GUIDE.md): Streamlit app usage

## Troubleshooting

### Common Issues

**SignTransformer:**

- `ValueError: Sequence length exceeds max_len`: Increase `max_len` parameter or reduce sequence length
- `ValueError: Expected 156 input features`: Ensure keypoint extraction produces correct dimensions
- Memory issues: Reduce batch size or embedding dimension

**InceptionV3GRU:**

- `ValueError: Expected features with 2048 dimensions`: Check NPZ files contain `X2048` key
- `KeyError: 'X2048'`: Use NPZ files with InceptionV3 features extracted during preprocessing
- CUDA out of memory: Use smaller batch sizes or mixed precision training

### Data Format Verification

**Check NPZ file contents:**

```python
import numpy as np

# Load NPZ file
data = np.load('data/processed/cmb_train/clip_0001.npz')

# Check available keys
print(data.files)  # Should include: ['X', 'X2048', 'mask', 'timestamps_ms', 'meta']

# Check shapes
print(f"X shape: {data['X'].shape}")        # [T, 156]
print(f"X2048 shape: {data['X2048'].shape}") # [T, 2048]
```

For NPZ validation, see [Preprocess Guide](../preprocessing/docs/PREPROCESS_GUIDE.MD)
