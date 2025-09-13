# Model Guide

Architecture details and usage for the FSLR models.

## Overview

Two model architectures for Filipino Sign Language Recognition:

- **SignTransformer**: Multi-head attention transformer for keypoint sequences
- **InceptionV3GRU**: CNN-GRU hybrid for video frame features

## SignTransformer

### Architecture

- **Input**: Keypoint sequences `[B, T, 156]` (pose, hands, face landmarks)
- **Encoder**: Multi-head attention with positional encoding
- **Output**: Gloss and category logits

### Key Components

- **PositionalEncoding**: Sinusoidal temporal encoding
- **TransformerEncoder**: Multi-head attention layers
- **Pooling**: Mean/max/CLS pooling options
- **Dual Heads**: Separate classification heads for gloss and category

### Usage

```python
from models.transformer import SignTransformer

model = SignTransformer(
    input_dim=156,
    emb_dim=256,
    num_heads=8,
    num_layers=6,
    num_gloss=105,
    num_cat=10,
    dropout=0.1
)

# Forward pass
gloss_logits, cat_logits = model(x)  # x: [B, T, 156]
```

### Parameters

- `input_dim`: Keypoint dimensions (156)
- `emb_dim`: Embedding dimension (256)
- `num_heads`: Attention heads (8)
- `num_layers`: Encoder layers (6)
- `num_gloss`: Number of gloss classes
- `num_cat`: Number of category classes
- `dropout`: Dropout rate (0.1)

## InceptionV3GRU

### Architecture

- **Input**: Video frames `[B, T, 3, 299, 299]` or features `[B, T, 2048]`
- **Backbone**: Pretrained InceptionV3 (frozen/fine-tunable)
- **Temporal**: Two-layer GRU with dropout
- **Output**: Gloss and category logits

### Key Components

- **InceptionV3FeatureExtractor**: 2048-D frame embeddings
- **GRU Layers**: Temporal sequence modeling
- **Dual Heads**: Classification outputs

### Usage

```python
from models.iv3_gru import InceptionV3GRU

model = InceptionV3GRU(
    num_gloss=105,
    num_cat=10,
    gru_hidden=16,
    dropout=0.3,
    freeze_backbone=True
)

# With precomputed features
gloss_logits, cat_logits = model(features, features_already=True)

# With raw frames
gloss_logits, cat_logits = model(frames, features_already=False)
```

### Parameters

- `num_gloss`: Number of gloss classes
- `num_cat`: Number of category classes
- `gru_hidden`: GRU hidden dimension (16)
- `dropout`: Dropout rate (0.3)
- `freeze_backbone`: Freeze InceptionV3 weights (True)

## Model Comparison

| Aspect           | SignTransformer         | InceptionV3GRU                                          |
| :--------------- | :---------------------- | :------------------------------------------------------ |
| **Input**        | Keypoints `[B, T, 156]` | Frames `[B, T, 3, 299, 299]` or features `[B, T, 2048]` |
| **Architecture** | Multi-head attention    | CNN + GRU                                               |
| **Pretrained**   | No                      | InceptionV3 (ImageNet)                                  |
| **Parameters**   | ~2M                     | ~25M (with frozen backbone)                             |
| **Training**     | End-to-end              | Can freeze backbone                                     |
| **Memory**       | Lower                   | Higher                                                  |

## Training Considerations

### SignTransformer

- **Advantages**: Lower memory, faster training, good for keypoint data
- **Data**: Requires keypoint extraction preprocessing
- **Training**: End-to-end from scratch

### InceptionV3GRU

- **Advantages**: Pretrained features, good for raw video
- **Data**: Can use raw frames or precomputed features
- **Training**: Start with frozen backbone, then fine-tune

## Input Requirements

### SignTransformer

- Keypoints extracted using MediaPipe
- Shape: `[batch_size, sequence_length, 156]`
- Normalized coordinates in `[0, 1]`

### InceptionV3GRU

- Raw frames: ImageNet normalized `[batch_size, sequence_length, 3, 299, 299]`
- Features: Precomputed `[batch_size, sequence_length, 2048]`
- Variable sequence lengths supported

## Output Format

Both models return:

- `gloss_logits`: `[batch_size, num_gloss]` - Gloss classification scores
- `cat_logits`: `[batch_size, num_cat]` - Category classification scores

Apply `torch.softmax()` for probabilities or `torch.argmax()` for predictions.

## Performance Tips

### SignTransformer

- Use attention masks for variable lengths
- Consider gradient clipping for stability
- Monitor attention patterns for interpretability

### InceptionV3GRU

- Start with frozen backbone
- Use mixed precision training (`--amp`)
- Consider data augmentation for raw frames

## Model Files

- `transformer.py`: SignTransformer implementation
- `iv3_gru.py`: InceptionV3GRU implementation
- `__init__.py`: Module initialization

## Integration

Both models integrate with the training pipeline:

```bash
# Train Transformer
python -m training.train --model transformer --keypoints-train ... --keypoints-val ...

# Train IV3-GRU
python -m training.train --model iv3_gru --features-train ... --features-val ...
```

See [Training Guide](../training/TRAINING_GUIDE.md) for detailed training instructions.
