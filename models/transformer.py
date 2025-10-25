"""
Transformer model for sign language recognition from 178-d keypoint sequences.

This module implements a complete Transformer architecture specifically designed for
sign language recognition using body keypoint sequences. The model processes temporal
sequences of 178-dimensional keypoint features (89 keypoints × 2 coordinates) and
outputs predictions for both gloss classification and semantic category classification.

Architecture Overview:
Input: [B, T, 178] keypoint sequences
  ↓
Linear Embedding: [B, T, 178] → [B, T, E]
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

Key Components:
- Sinusoidal positional encoding for temporal sequence understanding
- Custom layer normalization with learnable scale/shift parameters
- Multi-head self-attention mechanism for capturing temporal dependencies
- Residual connections with pre-layer normalization for stable training
- Configurable pooling strategies (mean, max, or CLS token)
- Dual classification heads for hierarchical prediction

Usage:
    from models import SignTransformer

    # Initialize model with default parameters
    model = SignTransformer()
    
    # Forward pass
    gloss_logits, cat_logits = model(x)  # x: [B, T, 178]
    
    # Get attention weights for visualization
    attention_maps = model.get_attention_weights(x)
"""

# Standard library imports
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    This module adds temporal order information to input embeddings by encoding
    each position in the sequence with a unique sinusoidal pattern. This allows
    the model to understand the relative and absolute positions of frames in
    the sign language sequence.
    
    The encoding uses sine and cosine functions with different frequencies:
    - Even dimensions (0, 2, 4, ...): sin(pos / 10000^(2i/d_model))
    - Odd dimensions (1, 3, 5, ...): cos(pos / 10000^(2i/d_model))
    
    Where pos is the position and i is the dimension index.
    """
    
    def __init__(self, emb_dim, dropout=0.1, max_len=300):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            emb_dim (int): Embedding dimension (E). Must match the model's embedding size.
            dropout (float): Dropout rate applied after adding positional encoding.
            max_len (int): Maximum sequence length supported. Sequences longer than
                          this will raise a ValueError.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # Step 1: Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # Step 2: Create frequency terms for sinusoidal functions
        # Each dimension gets a different frequency: 1/10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )  # [emb_dim//2]
        
        # Step 3: Initialize encoding matrix
        pe = torch.zeros(max_len, emb_dim)  # [max_len, emb_dim]
        
        # Step 4: Apply sinusoidal functions
        # Even dimensions: sin(position * frequency)
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, emb_dim//2]
        # Odd dimensions: cos(position * frequency)  
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, emb_dim//2]
        
        # Step 5: Reshape for broadcasting across batch dimension
        pe = pe.unsqueeze(0)  # [1, max_len, emb_dim]
        
        # Step 6: Register as buffer (non-trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor): Input embeddings of shape [B, T, E] where:
                       B = batch size, T = sequence length, E = embedding dimension.
        Returns:
            Tensor: Positionally encoded embeddings of shape [B, T, E].
        """
        # Step 1: Validate sequence length
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_len}. "
                f"Consider increasing max_len parameter or reducing sequence length."
            )

        # Step 2: Add positional encoding to input embeddings
        # Broadcasting: [B, T, E] + [1, T, E] = [B, T, E]
        x = x + self.pe[:, :seq_len, :]
        
        # Step 3: Apply dropout for regularization
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Custom Layer Normalization implementation.
    
    This module normalizes inputs across the last dimension (features) and applies
    learnable scale (gamma) and shift (beta) parameters. Layer normalization helps
    stabilize training by reducing internal covariate shift.
    
    Formula: LN(x) = γ * (x - μ) / σ + β
    Where μ and σ are the mean and standard deviation across the last dimension.
    """
    
    def __init__(self, features, eps=1e-6):
        """
        Initialize layer normalization.
        
        Args:
            features (int): Number of features to normalize (embedding dimension E).
            eps (float): Small constant added to variance to prevent division by zero.
        """
        super(LayerNormalization, self).__init__()
        
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(features))   # Scale parameter (γ)
        self.beta = nn.Parameter(torch.zeros(features))   # Shift parameter (β)
        self.eps = eps

    def forward(self, x):
        """
        Apply layer normalization to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, E] or [B, E].
                        Normalization is applied across the last dimension (E).
        Returns:
            Tensor: Normalized tensor with the same shape as input.
        """
        # Step 1: Compute mean across the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1] or [B, 1]
        
        # Step 2: Compute variance across the last dimension
        # Use unbiased=False to avoid NaN for single-element tensors
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, T, 1] or [B, 1]
        
        # Step 3: Apply normalization formula: γ * (x - μ) / σ + β
        normalized = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        
        return normalized

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) used in Transformer layers.
    
    This module implements the position-wise feed-forward network that is applied
    to each position separately and identically. It consists of two linear transformations
    with a ReLU activation in between, followed by dropout for regularization.
    
    The FFN expands the input dimension (emb_dim) to a higher dimension (ff_dim),
    applies non-linearity, then projects back to the original embedding dimension.
    This allows the model to learn complex non-linear transformations.
    
    Architecture: Linear(emb_dim → ff_dim) → ReLU → Dropout → Linear(ff_dim → emb_dim)
    """
    
    def __init__(self, emb_dim, ff_dim=512, dropout=0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            emb_dim (int): Embedding dimension (input/output size).
            ff_dim (int): Hidden dimension of the feed-forward layer.
                          Typically 2-4 times the embedding dimension.
            dropout (float): Dropout rate applied after the ReLU activation.
        """
        super(FeedForwardBlock, self).__init__()
        
        # First linear transformation: expand dimension
        self.linear1 = nn.Linear(emb_dim, ff_dim)
        
        # Non-linear activation function
        self.activation = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Second linear transformation: project back to original dimension
        self.linear2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        """
        Apply feed-forward transformation to input.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, E].
        Returns:
            Tensor: Output tensor of shape [B, T, E].
        """
        # Step 1: Expand dimension [B, T, E] → [B, T, ff_dim]
        x = self.linear1(x)
        
        # Step 2: Apply non-linear activation
        x = self.activation(x)
        
        # Step 3: Apply dropout for regularization
        x = self.dropout(x)
        
        # Step 4: Project back to original dimension [B, T, ff_dim] → [B, T, E]
        x = self.linear2(x)
        
        return x    

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    This module implements the core attention mechanism of the Transformer.
    It splits the input embedding into multiple "heads" and computes scaled
    dot-product attention for each head independently, then concatenates the
    results. This allows the model to attend to different types of relationships
    simultaneously.
    
    Process:
    1. Linear projections to create Q, K, V matrices
    2. Split into multiple heads
    3. Compute scaled dot-product attention for each head
    4. Concatenate all heads
    5. Final linear projection
    
    Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention block.
        
        Args:
            emb_dim (int): Embedding dimension (E). Must be divisible by num_heads.
            num_heads (int): Number of attention heads (H).
            dropout (float): Dropout rate applied to attention weights.
        
        Raises:
            ValueError: If emb_dim is not divisible by num_heads.
        """
        super(MultiHeadAttentionBlock, self).__init__()
        
        # Validate that embedding dimension is divisible by number of heads
        if emb_dim % num_heads != 0:
            raise ValueError(f"Embedding dim {emb_dim} must be divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head (D = E / H)

        # Linear projections for queries, keys, and values
        # Each projection maps [B, T, E] → [B, T, E]
        self.W_q = nn.Linear(emb_dim, emb_dim)  # Query projection
        self.W_k = nn.Linear(emb_dim, emb_dim)  # Key projection
        self.W_v = nn.Linear(emb_dim, emb_dim)  # Value projection
        
        # Output projection after concatenating all heads
        self.W_o = nn.Linear(emb_dim, emb_dim)  # Output projection

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def SelfAttention(Q, K, V, mask=None, dropout=None):
        """
        Compute scaled dot-product attention.
        
        This is the core attention operation that computes attention weights
        and applies them to the values. The attention mechanism allows the
        model to focus on different parts of the input sequence.

        Args:
            Q (Tensor): Query matrix of shape [B, H, T, D].
            K (Tensor): Key matrix of shape [B, H, T, D].
            V (Tensor): Value matrix of shape [B, H, T, D].
            mask (Tensor or None): Optional mask broadcastable to [B, 1, 1, T].
                                  1 = keep, 0 = mask out.
            dropout (nn.Dropout or None): Optional dropout layer for attention weights.

        Returns:
            out (Tensor): Attention output of shape [B, H, T, D].
            attn (Tensor): Attention weights of shape [B, H, T, T].
        
        Raises:
            ValueError: If mask dimensions don't match attention scores.
        """
        # Get the dimension of keys (for scaling)
        d_k = Q.size(-1)

        # Step 1: Compute attention scores
        # QK^T gives similarity between queries and keys
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, T, T]
        
        # Step 2: Scale by square root of key dimension
        # This prevents the dot products from becoming too large
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask if provided
        if mask is not None:
            if mask.shape[-1] != scores.shape[-1]:
                raise ValueError(
                    f"Mask last dimension {mask.shape[-1]} doesn't match "
                    f"scores {scores.shape[-1]}"
                )
            # Set masked positions to -inf so softmax → 0
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Apply softmax to get attention probabilities
        attn = torch.softmax(scores, dim=-1)  # [B, H, T, T]

        # Step 5: Apply dropout to attention weights (regularization)
        if dropout is not None:
            attn = dropout(attn)

        # Step 6: Apply attention weights to values
        out = torch.matmul(attn, V)  # [B, H, T, D]
        
        return out, attn

    def forward(self, x, mask=None):
        """
        Apply multi-head self-attention to input embeddings.
        
        Args:
            x (Tensor): Input embeddings of shape [B, T, E].
            mask (Tensor or None): Optional mask broadcastable to [B, 1, 1, T].

        Returns:
            out (Tensor): Output embeddings of shape [B, T, E].
            attn (Tensor): Attention weights of shape [B, H, T, T].
        """
        B, T, E = x.size()

        # Step 1: Linear projections to create Q, K, V matrices
        # Each projection: [B, T, E] → [B, T, E]
        Q = self.W_q(x)  # Query matrix
        K = self.W_k(x)  # Key matrix
        V = self.W_v(x)  # Value matrix

        # Step 2: Reshape and split into multiple heads
        # [B, T, E] → [B, T, H, D] → [B, H, T, D]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention
        out, attn = MultiHeadAttentionBlock.SelfAttention(Q, K, V, mask, self.dropout)

        # Step 4: Concatenate heads back together
        # [B, H, T, D] → [B, T, H, D] → [B, T, E]
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # Step 5: Final linear projection
        out = self.W_o(out)

        return out, attn

class ResidualConnection(nn.Module):
    """
    Residual connection with pre-layer normalization.
    
    This module implements the residual connection pattern used in Transformer
    architectures. It wraps a sublayer (attention or feed-forward) with:
    1. Layer normalization applied to the input
    2. The sublayer operation
    3. Dropout for regularization
    4. Residual connection (addition with original input)
    
    Formula: output = x + Dropout(Sublayer(LayerNorm(x)))
    
    This pattern helps with gradient flow and training stability by allowing
    the model to learn identity mappings when needed.
    """
    
    def __init__(self, emb_dim, dropout=0.1):
        """
        Initialize residual connection.
        
        Args:
            emb_dim (int): Embedding dimension (E).
            dropout (float): Dropout rate applied after the sublayer.
        """
        super(ResidualConnection, self).__init__()
        
        # Layer normalization applied before the sublayer
        self.norm = LayerNormalization(emb_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, E].
            sublayer (callable): Function or layer applied to normalized x.

        Returns:
            Tensor: Output of shape [B, T, E] after residual connection.
        """
        # Step 1: Apply layer normalization to input
        normalized_x = self.norm(x)
        
        # Step 2: Apply the sublayer (attention or feed-forward)
        sublayer_output = sublayer(normalized_x)
        
        # Step 3: Apply dropout for regularization
        dropped_output = self.dropout(sublayer_output)
        
        # Step 4: Add residual connection
        return x + dropped_output

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    This module implements one complete Transformer encoder layer, which consists of:
    1. Multi-head self-attention mechanism with residual connection
    2. Position-wise feed-forward network with residual connection
    
    Each sublayer is wrapped with pre-layer normalization and residual connections
    to ensure stable training and good gradient flow.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Encoder Layer                             │
    ├─────────────────────────────────────────────────────────────┤
    │ Input: [B, T, E]                                            │
    │         ↓                                                  │
    │ Multi-Head Self-Attention + Residual                        │
    │         ↓                                                  │
    │ Feed-Forward Network + Residual                             │
    │         ↓                                                  │
    │ Output: [B, T, E]                                           │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, emb_dim, num_heads, ff_dim=512, dropout=0.1):
        """
        Initialize Transformer encoder layer.
        
        Args:
            emb_dim (int): Embedding dimension (E).
            num_heads (int): Number of attention heads (H).
            ff_dim (int): Hidden dimension in feed-forward network.
            dropout (float): Dropout rate applied to attention & FFN outputs.
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention mechanism
        self.attention = MultiHeadAttentionBlock(emb_dim, num_heads, dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = FeedForwardBlock(emb_dim, ff_dim, dropout)

        # Residual connections with pre-layer normalization
        self.residual1 = ResidualConnection(emb_dim, dropout)  # For attention
        self.residual2 = ResidualConnection(emb_dim, dropout)  # For feed-forward

    def forward(self, x, mask=None, return_attn=False):
        """
        Forward pass through the encoder layer.
        
        Args:
            x (Tensor): Input embeddings of shape [B, T, E].
            mask (Tensor or None): Attention mask of shape [B, 1, 1, T].
                                   1 = keep, 0 = mask out.
            return_attn (bool): If True, also return attention weights.

        Returns:
            Tensor: Encoded output of shape [B, T, E].
            (Optional) Tensor: Attention weights of shape [B, H, T, T].
        """
        # Step 1: Multi-head self-attention with residual connection
        # Apply layer normalization, then attention, then residual connection
        normed_x = self.residual1.norm(x)
        attn_out, attn = self.attention(normed_x, mask)
        x = x + self.residual1.dropout(attn_out)

        # Step 2: Feed-forward network with residual connection
        # Apply layer normalization, then feed-forward, then residual connection
        normed_x2 = self.residual2.norm(x)
        ff_out = self.feed_forward(normed_x2)
        x = x + self.residual2.dropout(ff_out)

        # Return attention weights if requested
        if return_attn:
            return x, attn
        return x

class SignTransformer(nn.Module):
    """
    Transformer-based model for Sign Language Recognition.
    
    This is the main model class that implements a complete Transformer architecture
    for sign language recognition. It processes sequences of body keypoints and
    outputs predictions for both gloss classification (specific sign words) and
    category classification (semantic groups).
    
    The model follows the standard Transformer encoder architecture with:
    - Input embedding and positional encoding
    - Stack of Transformer encoder layers
    - Configurable pooling strategy
    - Dual output heads for hierarchical classification
    
    Key Features:
    - Handles variable-length sequences with optional masking
    - Supports three pooling strategies: mean, max, or CLS token
    - Dual classification heads for gloss and category prediction
    - Attention weight extraction for visualization
    
    Args:
        input_dim (int): Input feature dimension per frame (default: 178).
                        Should match the number of keypoint features.
        emb_dim (int): Embedding dimension E (default: 256).
        n_heads (int): Number of attention heads H (default: 8).
        n_layers (int): Number of encoder layers (default: 4).
        num_gloss (int): Number of gloss classes (default: 105).
        num_cat (int): Number of category classes (default: 10).
        dropout (float): Dropout rate used throughout the model (default: 0.1).
        max_len (int): Maximum supported sequence length (default: 300).
        ff_dim (int): Hidden dimension of FFN (default: 4× emb_dim).
        pooling_method (str): Pooling strategy - 'mean', 'max', or 'cls' (default: 'mean').
        
    Note:
        Input sequences are expected to have shape [B, T, 178] where 178 represents
        the number of keypoint features (89 keypoints × 2 coordinates: x, y).
    """
    
    def __init__(self,
                    input_dim=178,     # 89 keypoints × 2 coordinates
                    emb_dim=256,       # embedding dimension
                    n_heads=8,         # number of attention heads
                    n_layers=4,        # number of encoder layers
                    num_gloss=105,     # number of gloss classes
                    num_cat=10,        # number of category classes
                    dropout=0.1,       # dropout rate
                    max_len=300,       # maximum sequence length
                    ff_dim=None,       # feed-forward hidden size (defaults to 4*emb_dim)
                    pooling_method='mean'  # 'mean' | 'max' | 'cls'
                ):
        super(SignTransformer, self).__init__()

        # ===== INPUT PROCESSING =====
        # Linear projection from raw keypoints to model embedding space
        self.embedding = nn.Linear(input_dim, emb_dim)

        # Positional encoding for temporal sequence understanding
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)

        # Input normalization for training stability
        self.input_norm = LayerNormalization(emb_dim)

        # ===== TRANSFORMER ENCODER =====
        # Set feed-forward dimension (typically 4× embedding dimension)
        if ff_dim is None:
            ff_dim = emb_dim * 4
            
        # Stack of Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(emb_dim, n_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ===== POOLING STRATEGY =====
        # Validate pooling method
        if pooling_method not in ('mean', 'max', 'cls'):
            raise ValueError(
                f"Invalid pooling_method: {pooling_method}. "
                f"Choose from 'mean', 'max', 'cls'"
            )
        self.pooling_method = pooling_method
        
        # CLS token for classification-based pooling
        # Always created but only used when pooling_method == 'cls'
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # ===== OUTPUT HEADS =====
        # Final dropout before classification
        self.dropout_final = nn.Dropout(dropout)
        
        # Gloss classification head (specific sign words)
        self.gloss_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),  # Reduce dimension
            nn.ReLU(),                          # Non-linearity
            nn.Dropout(dropout),                # Regularization
            nn.Linear(emb_dim // 2, num_gloss)  # Final classification
        )
        
        # Category classification head (semantic groups)
        self.category_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),  # Reduce dimension
            nn.ReLU(),                          # Non-linearity
            nn.Dropout(dropout),                # Regularization
            nn.Linear(emb_dim // 2, num_cat)   # Final classification
        )

    def forward(self, x, mask=None):
        """
        Forward pass through the complete Transformer model.

        Args:
            x (Tensor): Input keypoint sequence of shape [B, T, 178].
            mask (Tensor or None): Binary mask of shape [B, T] or [B, T+1] if using CLS token.
                                  1 = valid frame, 0 = padding.
                                  Internally broadcast to [B, 1, 1, T(+1)] for attention.

        Returns:
            gloss_out (Tensor): Gloss prediction logits of shape [B, num_gloss].
            cat_out (Tensor): Category prediction logits of shape [B, num_cat].
        
        Raises:
            ValueError: If input dimensions are invalid or mask dimensions don't match.
        """
        # ===== INPUT VALIDATION =====
        # Check input tensor dimensions
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.embedding.in_features:
            raise ValueError(
                f"Expected {self.embedding.in_features} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()

        # ===== EMBEDDING LAYER =====
        # Project raw keypoints to embedding space
        # [B, T, 178] → [B, T, E]
        x = self.embedding(x)

        # ===== CLS TOKEN HANDLING =====
        # If using CLS token pooling, prepend CLS token to sequence
        if self.pooling_method == 'cls':
            # Expand CLS token to batch size
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
            # Concatenate CLS token with input sequence
            x = torch.cat([cls_tokens, x], dim=1)          # [B, T+1, E]
            
            # Update mask to include CLS token (always valid)
            if mask is not None:
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)  # [B, T+1]

        # ===== POSITIONAL ENCODING =====
        # Add temporal order information to embeddings
        x = self.pos_encoder(x)

        # ===== INPUT NORMALIZATION =====
        # Apply layer normalization for training stability
        x = self.input_norm(x)

        # ===== ATTENTION MASK PREPARATION =====
        # Prepare mask for attention mechanism
        if mask is not None:
            # Validate mask dimensions
            expected_len = T + (1 if self.pooling_method == 'cls' else 0)
            if mask.shape[0] != B:
                raise ValueError(
                    f"Mask batch size {mask.shape[0]} doesn't match input batch size {B}"
                )
            if mask.shape[1] != expected_len:
                raise ValueError(
                    f"Mask sequence length {mask.shape[1]} doesn't match expected length {expected_len}"
                )
            # Broadcast mask for attention: [B, T] → [B, 1, 1, T]
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None

        # ===== TRANSFORMER ENCODER =====
        # Pass through stack of encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)

        # ===== SEQUENCE POOLING =====
        # Collapse sequence dimension to single vector for classification
        if self.pooling_method == 'cls':
            # Use CLS token as sequence representation
            pooled = x[:, 0, :]  # [B, E]
            
        elif self.pooling_method == 'mean':
            # Average pooling across time dimension
            if mask is not None:
                # Masked average pooling (ignore padded positions)
                mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, T, E]
                masked_x = x * mask_expanded                      # Zero out padded positions
                mask_sum = mask.sum(dim=1, keepdim=True)          # [B, 1]
                
                # Handle empty sequences (all masked)
                valid_lengths = mask_sum.clamp(min=1)            # Avoid division by zero
                pooled = masked_x.sum(dim=1) / valid_lengths     # [B, E]
                
                # Zero out results for completely masked sequences
                completely_masked = (mask_sum == 0).expand_as(pooled)
                pooled = pooled.masked_fill(completely_masked, 0.0)
            else:
                # Simple average pooling
                pooled = x.mean(dim=1)  # [B, E]
                
        elif self.pooling_method == 'max':
            # Max pooling across time dimension
            if mask is not None:
                # Masked max pooling (ignore padded positions)
                mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, T, E]
                # Set padded positions to -inf so they don't affect max
                masked_x = x.masked_fill(~mask_expanded.bool(), float('-inf'))
                pooled = masked_x.max(dim=1)[0]  # [B, E]
                
                # Handle completely masked sequences (replace -inf with zeros)
                has_valid_tokens = (mask.sum(dim=1) > 0).unsqueeze(-1)
                pooled = torch.where(has_valid_tokens, pooled, torch.zeros_like(pooled))
            else:
                # Simple max pooling
                pooled = x.max(dim=1)[0]  # [B, E]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # ===== FINAL DROPOUT =====
        # Apply dropout before classification
        pooled = self.dropout_final(pooled)

        # ===== CLASSIFICATION HEADS =====
        # Generate predictions for both gloss and category
        gloss_out = self.gloss_head(pooled)     # [B, num_gloss]
        cat_out = self.category_head(pooled)     # [B, num_cat]

        return gloss_out, cat_out

    def get_attention_weights(self, x, mask=None):
        """
        Extract attention weights from all encoder layers for visualization.
        
        This utility method performs a forward pass through the model while
        collecting attention weights from each encoder layer. The weights can
        be used to visualize what the model is attending to at each layer.

        Args:
            x (Tensor): Input keypoint sequence of shape [B, T, 178].
            mask (Tensor or None): Binary mask of shape [B, T] or [B, T+1] if using CLS token.
                                  1 = valid frame, 0 = padding.

        Returns:
            List[Tensor]: List of attention weight tensors, one per encoder layer.
                         Each tensor has shape [B, H, T, T] (or [B, H, T+1, T+1] if using 'cls').
                         Attention weights are detached and moved to CPU for visualization.
        """
        B, T, _ = x.size()
        attention_weights = []

        # ===== INPUT PROCESSING =====
        # Apply embedding transformation
        x = self.embedding(x)
        
        # Handle CLS token if needed
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Apply positional encoding and normalization
        x = self.pos_encoder(x)
        x = self.input_norm(x)

        # Prepare attention mask
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None

        # ===== COLLECT ATTENTION WEIGHTS =====
        # Pass through each encoder layer and collect attention weights
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, attention_mask, return_attn=True)
            # Detach and move to CPU for visualization
            attention_weights.append(attn_weights.detach().cpu())

        return attention_weights


class SignTransformerCtc(nn.Module):
    """
    Transformer-based model for Continuous Sign Language Recognition using CTC.
    
    This model is designed for sequence-to-sequence learning with CTC loss,
    enabling continuous sign language recognition without frame-level alignment.
    Unlike the classification-based SignTransformer, this model:
    - Does NOT pool the sequence (preserves temporal dimension)
    - Outputs per-frame predictions for CTC decoding
    - Uses a single CTC head instead of dual classification heads
    - Supports variable-length output sequences
    
    Architecture:
    Input: [B, T, 178] keypoint sequences
      ↓
    Linear Embedding: [B, T, 178] → [B, T, E]
      ↓
    Positional Encoding: Adds temporal order information
      ↓
    Layer Normalization: Stabilizes training
      ↓
    Transformer Encoder Stack (N layers)
      ↓
    CTC Head: [B, T, E] → [B, T, num_ctc_classes]
      ↓
    LogSoftmax: [B, T, num_ctc_classes] → log probabilities
    
    Output: [B, T, num_ctc_classes] log probabilities for CTC loss
    
    Args:
        input_dim (int): Input feature dimension per frame (default: 178).
        emb_dim (int): Embedding dimension E (default: 256).
        n_heads (int): Number of attention heads H (default: 8).
        n_layers (int): Number of encoder layers (default: 4).
        num_ctc_classes (int): Number of CTC classes including blank (default: 106).
        dropout (float): Dropout rate (default: 0.1).
        max_len (int): Maximum sequence length (default: 300).
        ff_dim (int): Feed-forward hidden dimension (default: 4× emb_dim).
    
    Usage:
        model = SignTransformerCtc(input_dim=178, num_ctc_classes=106)
        log_probs = model(x)  # x: [B, T, 178] → log_probs: [B, T, 106]
        
        # For CTC loss, permute to [T, B, C]
        log_probs = log_probs.permute(1, 0, 2)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    """
    
    def __init__(self,
                 input_dim=178,
                 emb_dim=512,  # Increased from 256
                 n_heads=8,
                 n_layers=6,   # Increased from 4
                 num_ctc_classes=106,
                 num_cat=None,
                 dropout=0.05, # Reduced from 0.1
                 max_len=300,
                 ff_dim=None):
        super(SignTransformerCtc, self).__init__()
        
        # ===== INPUT PROCESSING =====
        # Linear projection from raw keypoints to model embedding space
        self.embedding = nn.Linear(input_dim, emb_dim)
        
        # Positional encoding for temporal sequence understanding
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)
        
        # Input normalization for training stability
        self.input_norm = LayerNormalization(emb_dim)
        
        # ===== TRANSFORMER ENCODER =====
        # Set feed-forward dimension (typically 4× embedding dimension)
        if ff_dim is None:
            ff_dim = emb_dim * 4
        
        # Stack of Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(emb_dim, n_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization after encoder stack for better training stability
        self.output_norm = LayerNormalization(emb_dim)
        
        # ===== DUAL OUTPUT HEADS =====
        # CTC head for gloss sequence prediction (per-frame)
        self.ctc_head = nn.Linear(emb_dim, num_ctc_classes)
        
        # Optional category head for per-frame category classification
        self.num_cat = num_cat
        if num_cat is not None:
            self.category_head = nn.Linear(emb_dim, num_cat)
        else:
            self.category_head = None
        
        # Store configuration
        self.num_ctc_classes = num_ctc_classes
        self.emb_dim = emb_dim
        self.max_len = max_len
    
    def forward(self, x, mask=None):
        """
        Forward pass through the CTC Transformer model with optional category prediction.
        
        Args:
            x (Tensor): Input keypoint sequence of shape [B, T, 178].
            mask (Tensor or None): Binary mask of shape [B, T].
                                  1 = valid frame, 0 = padding.
                                  Internally broadcast to [B, 1, 1, T] for attention.
        
        Returns:
            If num_cat is None (CTC-only mode):
                Tensor: CTC log probabilities of shape [B, T, num_ctc_classes].
            
            If num_cat is provided (dual-task mode):
                Tuple[Tensor, Tensor]: (ctc_log_probs, cat_logits)
                    - ctc_log_probs: [B, T, num_ctc_classes] for CTC loss
                    - cat_logits: [B, T, num_cat] for per-frame category classification
        
        Raises:
            ValueError: If input dimensions are invalid or mask dimensions don't match.
        """
        # ===== INPUT VALIDATION =====
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.embedding.in_features:
            raise ValueError(
                f"Expected {self.embedding.in_features} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()
        
        # ===== EMBEDDING LAYER =====
        # Project raw keypoints to embedding space
        # [B, T, 178] → [B, T, E]
        x = self.embedding(x)
        
        # ===== POSITIONAL ENCODING =====
        # Add temporal order information to embeddings
        x = self.pos_encoder(x)
        
        # ===== INPUT NORMALIZATION =====
        # Apply layer normalization for training stability
        x = self.input_norm(x)
        
        # ===== ATTENTION MASK PREPARATION =====
        # Prepare mask for attention mechanism
        if mask is not None:
            # Validate mask dimensions
            if mask.shape[0] != B:
                raise ValueError(
                    f"Mask batch size {mask.shape[0]} doesn't match input batch size {B}"
                )
            if mask.shape[1] != T:
                raise ValueError(
                    f"Mask sequence length {mask.shape[1]} doesn't match input length {T}"
                )
            # Broadcast mask for attention: [B, T] → [B, 1, 1, T]
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None
        
        # ===== TRANSFORMER ENCODER =====
        # Pass through stack of encoder layers
        # Output shape: [B, T, E]
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        # Apply layer normalization after encoder stack
        x = self.output_norm(x)
        
        # ===== CTC HEAD (PER-FRAME PREDICTION) =====
        # Project to CTC vocabulary size
        # [B, T, E] → [B, T, num_ctc_classes]
        ctc_logits = self.ctc_head(x)
        
        # Apply log softmax for CTC loss
        # CTCLoss expects log probabilities, not raw logits
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2)
        
        # ===== CATEGORY HEAD (PER-FRAME PREDICTION) =====
        if self.category_head is not None:
            # Category prediction per frame: [B, T, E] → [B, T, num_cat]
            cat_logits = self.category_head(x)
            
            return ctc_log_probs, cat_logits
        else:
            # CTC-only mode (backward compatibility)
            return ctc_log_probs
    
    def get_attention_weights(self, x, mask=None):
        """
        Extract attention weights from all encoder layers for visualization.
        
        Args:
            x (Tensor): Input keypoint sequence of shape [B, T, 178].
            mask (Tensor or None): Binary mask of shape [B, T].
        
        Returns:
            List[Tensor]: List of attention weight tensors, one per encoder layer.
                         Each tensor has shape [B, H, T, T].
        """
        B, T, _ = x.size()
        attention_weights = []
        
        # ===== INPUT PROCESSING =====
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        
        # Prepare attention mask
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None
        
        # ===== COLLECT ATTENTION WEIGHTS =====
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, attention_mask, return_attn=True)
            attention_weights.append(attn_weights.detach().cpu())
        
        return attention_weights