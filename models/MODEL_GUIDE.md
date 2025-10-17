# Model Guide

Technical documentation for Filipino Sign Language Recognition model architectures.

## Overview

Two architectures for dual-task learning (gloss + category prediction):

- **SignTransformer**: Attention-based encoder for keypoint sequences
- **InceptionV3GRU**: CNN-RNN hybrid for visual features

Both models output `(gloss_logits, category_logits)` for 105 glosses and 10 categories.

---

## SignTransformer

Multi-head attention Transformer encoder that processes temporal sequences of body keypoints.

### Architecture Flow

```
Input: [B, T, 156]
  ↓
Linear Embedding → [B, T, E]
  ↓
+ Positional Encoding
  ↓
Layer Norm
  ↓
Transformer Encoder (×N layers)
  ├─ Multi-Head Self-Attention
  ├─ Residual + Layer Norm
  ├─ Feed-Forward Network
  └─ Residual + Layer Norm
  ↓
Pooling → [B, E]
  ↓
Dual Heads → [B, 105], [B, 10]
```

### Core Implementation

**Class Signature**:

```python
SignTransformer.__init__(
    input_dim=156,        # Keypoint features per frame
    emb_dim=256,          # Embedding dimension
    n_heads=8,            # Attention heads
    n_layers=4,           # Encoder layers
    num_gloss=105,        # Gloss classes
    num_cat=10,           # Category classes
    dropout=0.1,          # Dropout rate
    max_len=300,          # Max sequence length
    ff_dim=None,          # FFN hidden dim (default: 4×emb_dim)
    pooling_method='mean' # 'mean' | 'max' | 'cls'
)
```

**Forward Pass**:

```python
def forward(x, mask=None):
    """
    Args:
        x: [B, T, 156] keypoint sequences
        mask: [B, T] binary mask (1=valid, 0=padding)

    Returns:
        gloss_logits: [B, 105]
        cat_logits: [B, 10]
    """
```

**Attention Extraction**:

```python
def get_attention_weights(x, mask=None):
    """
    Returns: List of [B, H, T, T] attention matrices (one per layer)
    """
```

### Components

#### 1. Input Embedding

**Implementation**: `nn.Linear(input_dim, emb_dim)`

Projects 156-D keypoint vectors to embedding space. Trainable linear transformation applied per frame independently.

#### 2. Positional Encoding

**Class**: `PositionalEncoding(emb_dim, dropout=0.1, max_len=300)`

Adds temporal position information using sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where `pos` is frame position, `i` is dimension index. Fixed encodings computed at initialization, applied via addition in `forward()`. Enables model to distinguish frame order without recurrence.

#### 3. Multi-Head Self-Attention

**Class**: `MultiHeadAttentionBlock(emb_dim, num_heads, dropout=0.1)`

**Key Method**:

```python
@staticmethod
def SelfAttention(Q, K, V, mask=None, dropout=None):
    """
    Scaled dot-product attention:
    Attention(Q,K,V) = softmax(QK^T / √d_k) V

    Args:
        Q, K, V: [B, H, T, D] query/key/value matrices
        mask: [B, 1, 1, T] attention mask

    Returns:
        out: [B, H, T, D] attended values
        attn: [B, H, T, T] attention weights
    """
```

Splits embedding into `num_heads` parallel attention operations. Each head learns different attention patterns (e.g., hand-face proximity, bilateral coordination, temporal boundaries).

**Process**:

1. Linear projections: `Q = X·W_q`, `K = X·W_k`, `V = X·W_v`
2. Split into heads: `[B, T, E] → [B, H, T, D]` where `D = E/H`
3. Scaled attention: `scores = (Q·K^T) / √D`
4. Apply mask if provided
5. Softmax: `weights = softmax(scores)`
6. Weighted sum: `out = weights·V`
7. Concatenate heads: `[B, H, T, D] → [B, T, E]`
8. Final projection: `out·W_o`

#### 4. Feed-Forward Network

**Class**: `FeedForwardBlock(emb_dim, ff_dim=emb_dim*4, dropout=0.1)`

Position-wise two-layer MLP with expansion:

```python
FFN(x) = ReLU(x·W_1 + b_1)·W_2 + b_2
```

Where `W_1: [E, 4E]` expands, `W_2: [4E, E]` compresses. Applied identically to each frame. The expansion allows learning complex non-linear transformations.

#### 5. Residual Connection

**Class**: `ResidualConnection(emb_dim, dropout=0.1)`

Pre-layer normalization with residual:

```python
out = x + Dropout(Sublayer(LayerNorm(x)))
```

Enables gradient flow through deep networks (6+ layers). Layer norm before sublayer (pre-norm) provides more stable training than post-norm.

#### 6. Layer Normalization

**Class**: `LayerNormalization(features, eps=1e-6)`

Normalizes across feature dimension:

```
LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

Where `μ, σ²` computed per sample across features. Learnable `γ, β` parameters. More stable than batch norm for variable-length sequences.

#### 7. Encoder Layer

**Class**: `EncoderLayer(emb_dim, num_heads, ff_dim, dropout=0.1)`

Single Transformer block combining attention and feed-forward:

```python
def forward(x, mask=None, return_attn=False):
    # Attention sublayer
    x = x + dropout(attention(layer_norm(x)))
    # Feed-forward sublayer
    x = x + dropout(ffn(layer_norm(x)))
    return x
```

Stacked `n_layers` times (default: 4).

#### 8. Pooling Strategies

**Mean Pooling**:

```python
pooled = x.mean(dim=1)  # Average across time
```

**Max Pooling**:

```python
pooled = x.max(dim=1)[0]  # Max across time
```

**CLS Token**:

```python
cls = nn.Parameter(torch.randn(1, 1, emb_dim))
x = torch.cat([cls, x], dim=1)  # Prepend learnable token
# After encoding:
pooled = x[:, 0, :]  # Extract CLS representation
```

CLS token "attends to" all frames and learns to aggregate sequence-level information (inspired by BERT).

#### 9. Classification Heads

Two independent linear layers:

```python
gloss_head = nn.Sequential(
    nn.Linear(emb_dim, emb_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(emb_dim // 2, num_gloss)
)

category_head = nn.Sequential(
    nn.Linear(emb_dim, emb_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(emb_dim // 2, num_cat)
)
```

Both receive same pooled representation, enabling multi-task learning.

### Design Decisions

**Why Transformer?**

- Global attention allows any frame to attend to any other frame
- Parallel processing (faster than RNN sequential)
- Dynamic attention reweighting handles occlusions
- Attention weights provide interpretability

**Why these hyperparameters?**

- 256 embedding dimensions: Balance expressiveness and efficiency
- 8 attention heads: Captures multiple relationship types simultaneously
- 4 encoder layers: Sufficient depth for FSL-105 dataset size
- 4× FFN expansion: Standard Transformer ratio

**Why multi-task learning?**

- Category task provides regularization signal
- Shared representations learn semantic groupings
- Hierarchical evaluation (gloss + category accuracy)

---

## InceptionV3GRU

Hybrid CNN-RNN architecture combining pretrained visual features with temporal modeling.

### Architecture Flow

```
Input: [B, T, 2048]
  ↓
GRU Layer 1: 2048 → 16
  ↓
Dropout(0.3)
  ↓
GRU Layer 2: 16 → 12
  ↓
Dropout(0.3)
  ↓
Final Hidden State: [B, 12]
  ↓
Dual Heads → [B, 105], [B, 10]
```

### Core Implementation

**Class Signature**:

```python
InceptionV3GRU.__init__(
    num_gloss,              # Gloss classes
    num_cat,                # Category classes
    hidden1=16,             # First GRU hidden size
    hidden2=12,             # Second GRU hidden size
    dropout=0.3,            # Dropout rate
    pretrained_backbone=True,  # Load ImageNet weights
    freeze_backbone=True    # Freeze CNN weights
)
```

**Forward Pass**:

```python
def forward(
    frames_or_feats,        # Input data
    lengths=None,           # [B] true sequence lengths
    return_probs=False,     # Return probabilities vs logits
    features_already=False  # Input is precomputed features
):
    """
    Args:
        frames_or_feats: [B, T, 3, H, W] raw frames OR
                        [B, T, 2048] precomputed features
        lengths: [B] true lengths for packed sequences

    Returns:
        gloss_logits: [B, 105]
        cat_logits: [B, 10]
    """
```

**Feature Extraction**:

```python
def extract_features(frames):
    """
    Args:
        frames: [B, T, 3, H, W]

    Returns:
        features: [B, T, 2048]
    """
```

**Probability Output**:

```python
def predict_proba(frames_or_feats, lengths=None, features_already=False):
    """Returns softmax probabilities instead of logits"""
```

### Components

#### 1. InceptionV3 Feature Extractor

**Class**: `InceptionV3FeatureExtractor(pretrained=True, freeze=True)`

Pretrained CNN backbone (ImageNet weights):

- 48 convolutional layers
- ~23.8M parameters
- Input: 256×256 RGB
- Output: 2048-D feature vector per frame

**Freezing behavior**:

```python
if freeze:
    for param in self.backbone.parameters():
        param.requires_grad = False
    self.backbone.eval()  # Fix BatchNorm statistics
```

Frozen backbone enables transfer learning while reducing training time.

**Feature extraction**:

```python
def forward(x):
    # x: [N, 3, H, W]
    # Parallel convolutions at multiple scales (Inception modules)
    # Final layer removed, returns raw features
    return self.backbone(x)  # [N, 2048]
```

#### 2. Two-Layer GRU

**Configuration**:

```python
self.gru1 = nn.GRU(
    input_size=2048,
    hidden_size=16,
    num_layers=1,
    batch_first=True
)

self.gru2 = nn.GRU(
    input_size=16,
    hidden_size=12,
    num_layers=1,
    batch_first=True
)
```

**GRU Cell Operations**:

```
Update gate: z_t = σ(W_z·[h_{t-1}, x_t])
Reset gate:  r_t = σ(W_r·[h_{t-1}, x_t])
Candidate:   h̃_t = tanh(W_h·[r_t⊙h_{t-1}, x_t])
New state:   h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
```

Sequential processing maintains temporal dependencies.

**Weight Initialization**:

```python
for name, param in gru.named_parameters():
    if "weight_ih" in name:
        nn.init.xavier_uniform_(param)  # Input-hidden weights
    elif "weight_hh" in name:
        nn.init.orthogonal_(param)      # Hidden-hidden weights
    elif "bias" in name:
        nn.init.zeros_(param)           # Bias terms
```

Xavier/orthogonal initialization provides stability for small hidden sizes.

#### 3. Packed Sequences

**Helper Function**:

```python
def _dropout_packed(packed_seq, p, training):
    """Apply dropout to PackedSequence data tensor"""
    if p <= 0.0:
        return packed_seq
    data = F.dropout(packed_seq.data, p=p, training=training)
    return nn.utils.rnn.PackedSequence(
        data, packed_seq.batch_sizes,
        packed_seq.sorted_indices, packed_seq.unsorted_indices
    )
```

Handles variable-length sequences efficiently:

```python
# Pack sequences
packed = nn.utils.rnn.pack_padded_sequence(
    seq, lengths_cpu, batch_first=True, enforce_sorted=False
)

# Process through GRU
y1, h1 = self.gru1(packed)
y1 = _dropout_packed(y1, 0.3, training=True)

# Extract final hidden state
h_final = h2[-1]  # [B, 12]
```

Avoids computation on padding, more efficient than masking.

#### 4. Classification Heads

Simple linear projections from final GRU hidden state:

```python
self.gloss_head = nn.Linear(hidden2, num_gloss)      # 12 → 105
self.category_head = nn.Linear(hidden2, num_cat)     # 12 → 10
```

### Design Decisions

**Why InceptionV3-GRU?**

- Pretrained visual features provide strong baseline
- Transfer learning from ImageNet
- Efficient with precomputed features
- Standard baseline for comparison

**Why freeze backbone?**

- Reduces parameters from ~25M to ~50K trainable
- Faster training convergence
- Prevents overfitting on small dataset
- Transfer learning principle: keep pretrained features

**Why small GRU hidden sizes (16, 12)?**

- Precomputed features already contain rich information
- Small sizes prevent overfitting
- Faster training and inference
- Sufficient capacity for 105-class problem

**Why two GRU layers?**

- First layer: Temporal smoothing and local patterns
- Second layer: High-level sequence modeling
- Hierarchical temporal abstraction

---

## Model Comparison

| Aspect            | SignTransformer     | InceptionV3GRU           |
| ----------------- | ------------------- | ------------------------ |
| **Input**         | Keypoints [T, 156]  | Features [T, 2048]       |
| **Architecture**  | Attention encoder   | CNN + RNN                |
| **Context**       | Global (parallel)   | Local (sequential)       |
| **Pretrained**    | No                  | Yes (ImageNet)           |
| **Parameters**    | ~2M                 | ~25M (50K trainable)     |
| **Memory**        | Lower               | Higher                   |
| **Occlusion**     | Dynamic reweighting | Fixed feature extraction |
| **Interpretable** | Attention weights   | Hidden states (opaque)   |
| **Speed**         | GPU-parallelizable  | Sequential bottleneck    |

**When to use Transformer:**

- Need interpretability (attention visualization)
- Handle occlusions gracefully
- Lower memory constraints
- Prefer end-to-end learning

**When to use InceptionV3-GRU:**

- Have precomputed features
- Want transfer learning benefits
- Need strong baseline quickly
- Limited training data

---

## Input/Output Specifications

### SignTransformer

**Input**:

- Shape: `[batch, time, 156]`
- Type: `torch.FloatTensor`
- Range: Normalized [0, 1]
- Content: 78 keypoints × 2 coordinates

**Optional Mask**:

- Shape: `[batch, time]`
- Type: `torch.FloatTensor`
- Values: 1 (valid) or 0 (padding)

**Output**:

- `gloss_logits`: `[batch, 105]` raw scores
- `cat_logits`: `[batch, 10]` raw scores

### InceptionV3GRU

**Input (features)**:

- Shape: `[batch, time, 2048]`
- Type: `torch.FloatTensor`
- Content: Precomputed InceptionV3 features

**Input (raw frames)**:

- Shape: `[batch, time, 3, 256, 256]`
- Type: `torch.FloatTensor`
- Range: ImageNet normalized
- Content: RGB frames

**Optional Lengths**:

- Shape: `[batch]`
- Type: `torch.LongTensor`
- Content: True sequence lengths for packing

**Output**:

- `gloss_logits`: `[batch, 105]` raw scores
- `cat_logits`: `[batch, 10]` raw scores

### Converting to Predictions

```python
# Get class predictions
gloss_pred = torch.argmax(gloss_logits, dim=-1)
cat_pred = torch.argmax(cat_logits, dim=-1)

# Get probabilities
gloss_probs = torch.softmax(gloss_logits, dim=-1)
cat_probs = torch.softmax(cat_logits, dim=-1)

# Get confidence scores
gloss_confidence = gloss_probs.gather(1, gloss_pred.unsqueeze(1))
```

---

## CTC Models (Continuous Recognition)

Two CTC models for sequence-to-sequence sign language recognition without frame-level alignment.

### SignTransformerCtc

**Architecture**: Transformer encoder + CTC head (no pooling)

**Input/Output**:

- Input: `[B, T, 156]` keypoints
- Output: `[B, T, 106]` log probabilities (105 glosses + 1 blank)

**Usage**:

```python
from models import SignTransformerCtc

model = SignTransformerCtc(input_dim=156, num_ctc_classes=106)
log_probs = model(x)  # [B, T, 106]

# For CTCLoss
log_probs = log_probs.permute(1, 0, 2)  # [T, B, C]
```

**Key Differences from SignTransformer**:

- ❌ No pooling layer
- ❌ No category head
- ✅ Full temporal output
- ✅ Single CTC head

### MediaPipeGRUCtc

**Architecture**: Bidirectional 2-layer GRU + CTC head

**Input/Output**:

- Input: `[B, T, 156]` keypoints
- Output: `[B, T, 106]` log probabilities

**Usage**:

```python
from models import MediaPipeGRUCtc

model = MediaPipeGRUCtc(num_ctc_classes=106, hidden1=256, hidden2=128)
log_probs = model(x, lengths=lengths)  # [B, T, 106]
```

**Advantages**: Lightweight (~500KB), faster inference, mobile-friendly

### CTC vs Classification

| Feature      | Classification     | CTC                         |
| ------------ | ------------------ | --------------------------- |
| **Task**     | One sign per video | Multiple signs per sequence |
| **Output**   | `[B, num_classes]` | `[B, T, num_classes]`       |
| **Pooling**  | Required           | None                        |
| **Loss**     | CrossEntropy       | CTCLoss                     |
| **Use Case** | Isolated signs     | Continuous signs            |

### CTC Technical Details

**CTCLoss Requirements**:

```python
# Model output
log_probs = model(X)  # [B, T, C]
log_probs = log_probs.permute(1, 0, 2)  # [T, B, C] for CTCLoss

# Compute loss
criterion = nn.CTCLoss(blank=105, zero_infinity=True)
loss = criterion(log_probs, targets, input_lengths, target_lengths)
```

**Decoding Strategies**:

- **Greedy**: Fast (O(T)), deterministic, good for real-time
- **Beam Search**: Accurate, explores multiple paths, slower

```python
from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder

# Greedy
decoded = greedy_ctc_decoder(log_probs, blank_id=105)

# Beam search
decoded, score = beam_search_ctc_decoder(log_probs, blank_id=105, beam_width=10)
```

---

## Implementation Files

- `models/transformer.py`: SignTransformer + SignTransformerCtc (~1020 lines)
- `models/mediapipe_gru.py`: MediaPipeGRU + MediaPipeGRUCtc (~679 lines)
- `models/iv3_gru.py`: InceptionV3GRU implementation (~430 lines)
- `models/__init__.py`: Module exports
- `evaluation/ctc_utils.py`: CTC decoders and utilities

---

## Additional Resources

- [Training Guide](../training/TRAINING_GUIDE.md): How to train both classification and CTC models
- [Prediction Guide](../evaluation/prediction/PREDICTION_GUIDE.md): Making predictions with all models
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md): Evaluating model performance
- [Trained Model Guide](../trained_models/TRAINED_MODEL_GUIDE.md): Loading and using trained models
- [Data Guide](../data/DATA_GUIDE.md): Data preparation and format
- [Preprocessing Guide](../preprocessing/docs/PREPROCESS_GUIDE.MD): Video to NPZ conversion
