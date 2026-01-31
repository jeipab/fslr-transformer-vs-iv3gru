"""
Transformer model for sign language recognition from keypoint sequences.

Implements Transformer architecture for processing 178-dimensional keypoint sequences
and outputting predictions for gloss and category classification.
"""

# Standard library imports
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence understanding."""
    
    def __init__(self, emb_dim, dropout=0.1, max_len=300):
        """Initialize sinusoidal positional encoding."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )
        
        pe = torch.zeros(max_len, emb_dim)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_len}. "
                f"Consider increasing max_len parameter or reducing sequence length."
            )

        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """Custom Layer Normalization implementation."""
    
    def __init__(self, features, eps=1e-6):
        """Initialize layer normalization."""
        super(LayerNormalization, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Apply layer normalization to input tensor."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return normalized

class FeedForwardBlock(nn.Module):
    """Position-wise Feed-Forward Network used in Transformer layers."""
    
    def __init__(self, emb_dim, ff_dim=512, dropout=0.1):
        """Initialize the feed-forward network."""
        super(FeedForwardBlock, self).__init__()
        
        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        """Apply feed-forward transformation to input."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x    

class MultiHeadAttentionBlock(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
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
        """Compute scaled dot-product attention."""
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            if mask.shape[-1] != scores.shape[-1]:
                raise ValueError(
                    f"Mask last dimension {mask.shape[-1]} doesn't match "
                    f"scores {scores.shape[-1]}"
                )
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        if dropout is not None:
            attn = dropout(attn)

        out = torch.matmul(attn, V)
        
        return out, attn

    def forward(self, x, mask=None):
        """Apply multi-head self-attention to input embeddings."""
        B, T, E = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out, attn = MultiHeadAttentionBlock.SelfAttention(Q, K, V, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.W_o(out)

        return out, attn

class ResidualConnection(nn.Module):
    """Residual connection with pre-layer normalization."""
    
    def __init__(self, emb_dim, dropout=0.1):
        """Initialize residual connection."""
        super(ResidualConnection, self).__init__()
        
        self.norm = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection with pre-layer normalization."""
        normalized_x = self.norm(x)
        sublayer_output = sublayer(normalized_x)
        dropped_output = self.dropout(sublayer_output)
        return x + dropped_output

class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, emb_dim, num_heads, ff_dim=512, dropout=0.1):
        """Initialize Transformer encoder layer."""
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttentionBlock(emb_dim, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_dim, ff_dim, dropout)
        self.residual1 = ResidualConnection(emb_dim, dropout)
        self.residual2 = ResidualConnection(emb_dim, dropout)

    def forward(self, x, mask=None, return_attn=False):
        """Forward pass through the encoder layer."""
        normed_x = self.residual1.norm(x)
        attn_out, attn = self.attention(normed_x, mask)
        x = x + self.residual1.dropout(attn_out)

        normed_x2 = self.residual2.norm(x)
        ff_out = self.feed_forward(normed_x2)
        x = x + self.residual2.dropout(ff_out)

        if return_attn:
            return x, attn
        return x

class SignTransformer(nn.Module):
    """Transformer-based model for Sign Language Recognition."""
    
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

        self.embedding = nn.Linear(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)
        self.input_norm = LayerNormalization(emb_dim)

        if ff_dim is None:
            ff_dim = emb_dim * 4
            
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(emb_dim, n_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        if pooling_method not in ('mean', 'max', 'cls'):
            raise ValueError(
                f"Invalid pooling_method: {pooling_method}. "
                f"Choose from 'mean', 'max', 'cls'"
            )
        self.pooling_method = pooling_method
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.dropout_final = nn.Dropout(dropout)
        
        self.gloss_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_gloss)
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_cat)
        )

    def forward(self, x, mask=None):
        """Forward pass through the complete Transformer model."""
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.embedding.in_features:
            raise ValueError(
                f"Expected {self.embedding.in_features} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()

        # Project raw keypoints to embedding space
        # [B, T, 178] → [B, T, E]
        x = self.embedding(x)

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

        # Add temporal order information to embeddings
        x = self.pos_encoder(x)

        # Apply layer normalization for training stability
        x = self.input_norm(x)

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

        # Pass through stack of encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)

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

        # Apply dropout before classification
        pooled = self.dropout_final(pooled)

        # Generate predictions for both gloss and category
        gloss_out = self.gloss_head(pooled)     # [B, num_gloss]
        cat_out = self.category_head(pooled)     # [B, num_cat]

        return gloss_out, cat_out

    def get_attention_weights(self, x, mask=None):
        """Extract attention weights from all encoder layers for visualization."""
        B, T, _ = x.size()
        attention_weights = []

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

        # Pass through each encoder layer and collect attention weights
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, attention_mask, return_attn=True)
            # Detach and move to CPU for visualization
            attention_weights.append(attn_weights.detach().cpu())

        return attention_weights


class SignTransformerCtc(nn.Module):
    """Transformer-based model for Continuous Sign Language Recognition using CTC."""
    
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
        
        # Linear projection from raw keypoints to model embedding space
        self.embedding = nn.Linear(input_dim, emb_dim)
        
        # Positional encoding for temporal sequence understanding
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)
        
        # Input normalization for training stability
        self.input_norm = LayerNormalization(emb_dim)
        
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
        """Forward pass through the CTC Transformer model."""
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.embedding.in_features:
            raise ValueError(
                f"Expected {self.embedding.in_features} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()
        
        # Project raw keypoints to embedding space
        # [B, T, 178] → [B, T, E]
        x = self.embedding(x)
        
        # Add temporal order information to embeddings
        x = self.pos_encoder(x)
        
        # Apply layer normalization for training stability
        x = self.input_norm(x)
        
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
        
        # Pass through stack of encoder layers
        # Output shape: [B, T, E]
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        # Apply layer normalization after encoder stack
        x = self.output_norm(x)
        
        # Project to CTC vocabulary size
        # [B, T, E] → [B, T, num_ctc_classes]
        ctc_logits = self.ctc_head(x)
        
        # Apply log softmax for CTC loss
        # CTCLoss expects log probabilities, not raw logits
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2)
        
        if self.category_head is not None:
            # Category prediction per frame: [B, T, E] → [B, T, num_cat]
            cat_logits = self.category_head(x)
            
            return ctc_log_probs, cat_logits
        else:
            # CTC-only mode (backward compatibility)
            return ctc_log_probs
    
    def get_attention_weights(self, x, mask=None):
        """Extract attention weights from all encoder layers for visualization."""
        B, T, _ = x.size()
        attention_weights = []
        
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        
        # Prepare attention mask
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None
        
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, attention_mask, return_attn=True)
            attention_weights.append(attn_weights.detach().cpu())
        
        return attention_weights