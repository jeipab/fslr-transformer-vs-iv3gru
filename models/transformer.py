# transformer.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in
    "Attention Is All You Need".
    
    Adds information about the order of sequence elements
    to the input embeddings.
    """
    
    def __init__(self, emb_dim, dropout=0.1, max_len=300):
        """
        Args:
            emb_dim (int): embedding dimension (E).
            dropout (float): dropout rate applied after adding positional encoding.
            max_len (int): maximum sequence length supported.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute sinusoidal positional encodings [max_len, emb_dim]
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        
        pe = pe.unsqueeze(0)  # [1, max_len, emb_dim] for broadcasting
        self.register_buffer('pe', pe) # non-trainable buffer

    def forward(self, x):
        """
        Args:
            x (Tensor): input embeddings of shape [B, T, E]
                        where B = batch size, T = sequence length, E = embedding dim.
        Returns:
            Tensor: positionally encoded embeddings of shape [B, T, E].
        """
        # Add positional encoding (up to sequence length T)
        x = x + self.pe[:, :x.size(1), :]
        
        # Apply dropout
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Custom implementation of Layer Normalization.
    
    Normalizes inputs across the last dimension (features) and
    applies learnable scale (gamma) and shift (beta) parameters.
    """
    
    def __init__(self, features, eps=1e-6):
        """
        Args:
            features (int): number of features (E) to normalize.
            eps (float): small constant to prevent division by zero.
        """
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # learnable scale
        self.beta = nn.Parameter(torch.zeros(features))  # learnable shift
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor of shape [B, T, E] or [B, E].
                        Normalization is applied across the last dimension (E).
        Returns:
            Tensor: normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)     # per-sample mean
        std = x.std(dim=-1, keepdim=True)       # per-sample std
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward network (FFN) used inside Transformer layers.

    Applies two linear transformations with a ReLU activation in between,
    and dropout for regularization.
    """
    
    def __init__(self, emb_dim, ff_dim=512, dropout=0.1):
        """
        Args:
            emb_dim (int): embedding dimension (input/output size).
            ff_dim (int): hidden dimension of the feed-forward layer (usually 2–4× emb_dim).
            dropout (float): dropout rate applied after the activation.
        """
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(emb_dim, ff_dim)   # expand to higher dimension
        self.activation = nn.ReLU()                 # non-linearity
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, emb_dim)   # project back to embedding dimension

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor of shape [B, T, E].
        Returns:
            Tensor: output tensor of shape [B, T, E].
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x    

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Splits the embedding into multiple heads, computes scaled dot-product
    attention for each head, and concatenates the results.
    """
    
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        """
        Args:
            emb_dim (int): embedding dimension (E).
            num_heads (int): number of attention heads (H).
                            Must divide evenly into emb_dim.
            dropout (float): dropout rate applied to attention weights.
        """
        super(MultiHeadAttentionBlock, self).__init__()
        assert emb_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads       # dimension per head (D = E / H)

        # Learnable linear projections for queries, keys, and values
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)

        # Final linear projection after concatenating all heads
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def SelfAttention(Q, K, V, mask=None, dropout=None):
        """
        Scaled dot-product attention (core operation of attention).

        Args:
            Q (Tensor): queries of shape [B, H, T, D].
            K (Tensor): keys of shape [B, H, T, D].
            V (Tensor): values of shape [B, H, T, D].
            mask (Tensor or None): optional attention mask of shape [B, 1, 1, T].
                                    1 = valid, 0 = masked.
            dropout (nn.Dropout or None): optional dropout layer for attention weights.

        Returns:
            out (Tensor): attention output of shape [B, H, T, D].
            attn (Tensor): attention weights of shape [B, H, T, T].
        """
        d_k = Q.size(-1)

        # Compute raw attention scores: [B, H, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # Apply mask (set masked positions to -inf so softmax → 0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Normalize scores into probabilities
        attn = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights (regularization)
        if dropout is not None:
            attn = dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)
        return out, attn

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): input embeddings of shape [B, T, E].
            mask (Tensor or None): optional attention mask of shape [B, T].

        Returns:
            out (Tensor): output embeddings of shape [B, T, E].
            attn (Tensor): attention weights of shape [B, H, T, T].
        """
        B, T, E = x.size()

        # 1. Linear projections: [B, T, E] → [B, T, E]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into multiple heads: [B, T, E] → [B, H, T, D]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply scaled dot-product attention
        out, attn = MultiHeadAttentionBlock.SelfAttention(Q, K, V, mask, self.dropout)

        # 4. Concatenate heads: [B, H, T, D] → [B, T, E]
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # 5. Final linear projection
        out = self.W_o(out)

        return out, attn

class ResidualConnection(nn.Module):
    """
    Implements a residual connection with pre-layer normalization.
    
    Each sublayer (e.g., attention or feed-forward) is wrapped with:
        x + Dropout(Sublayer(LayerNorm(x)))
    """
    
    def __init__(self, emb_dim, dropout=0.1):
        """
        Args:
            emb_dim (int): embedding dimension (E).
            dropout (float): dropout rate applied after the sublayer.
        """
        super(ResidualConnection, self).__init__()
        
        # Custom layer normalization
        self.norm = LayerNormalization(emb_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass for residual connection.

        Args:
            x (Tensor): input tensor of shape [B, T, E].
            sublayer (callable): function or layer applied to normalized x.

        Returns:
            Tensor of shape [B, T, E] after residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer.

    Consists of:
        1. Multi-head self-attention with residual connection.
        2. Position-wise feed-forward network with residual connection.
    """
    
    def __init__(self, emb_dim, num_heads, ff_dim=512, dropout=0.1):
        
        """
        Args:
            emb_dim (int): embedding dimension (E).
            num_heads (int): number of attention heads (H).
            ff_dim (int): hidden dimension in feed-forward network.
            dropout (float): dropout rate applied to attention & FFN outputs.
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head attention block
        self.attention = MultiHeadAttentionBlock(emb_dim, num_heads, dropout)
        
        # Position-wise feed-forward block
        self.feed_forward = FeedForwardBlock(emb_dim, ff_dim, dropout)

        # Residual connections (one after attention, one after feed-forward)
        self.residual1 = ResidualConnection(emb_dim, dropout)
        self.residual2 = ResidualConnection(emb_dim, dropout)

    def forward(self, x, mask=None, return_attn=False):
        """
        Forward pass of encoder layer.

        Args:
            x (Tensor): input of shape [B, T, E].
            mask (Tensor or None): attention mask of shape [B, 1, 1, T].
                                    1 = keep, 0 = mask out.
            return_attn (bool): if True, also return attention weights.

        Returns:
            Tensor: encoded output of shape [B, T, E].
            (Optional) Attention weights of shape [B, H, T, T].
        """
        # Multi-head attention with residual connection
        normed = self.residual1.norm(x)                 # Pre-norm
        attn_out, attn = self.attention(normed, mask)   # Self-attention
        x = x + self.residual1.dropout(attn_out)        # Residual

        # Feed-forward with residual connection
        normed2 = self.residual2.norm(x)                # Pre-norm
        ff_out = self.feed_forward(normed2)             # FFN
        x = x + self.residual2.dropout(ff_out)          # Residual

        if return_attn:
            return x, attn
        return x

class SignTransformer(nn.Module):
    def __init__(self,
                    input_dim=156,     # 78 keypoints × 2 coords
                    emb_dim=256,       # embedding dimension
                    n_heads=8,         # number of attention heads
                    n_layers=4,        # number of encoder layers
                    num_gloss=105,     # number of gloss classes
                    num_cat=10,        # number of category classes
                    dropout=0.1,       # dropout rate
                    max_len=300):      # maximum sequence length
        super(SignTransformer, self).__init__()

        # Input projection (Linear layer 156 → emb_dim)
        self.embedding = nn.Linear(input_dim, emb_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)

        # Input normalization layer
        self.input_norm = LayerNormalization(emb_dim)

        # Transformer Encoder (stack of encoder layers)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(emb_dim, n_heads, ff_dim=emb_dim*4, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Global pooling method (you can change this to 'cls' if you prefer CLS token)
        self.pooling_method = 'mean'  # options: 'mean', 'max', 'cls'
        
        # If using CLS token, add it
        if self.pooling_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Output heads (two classifiers: gloss + category)
        self.dropout_final = nn.Dropout(dropout)
        
        # Gloss classifier
        self.gloss_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_gloss)
        )
        
        # Category classifier
        self.category_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, num_cat)
        )

    def forward(self, x, mask=None):
        """
        x shape: [B, T, 156]
        mask shape: [B, T] (optional) - 1 for valid positions, 0 for padding
        B = batch size, T = sequence length, 156 = keypoint features
        """
        B, T, _ = x.size()

        # Project input into embedding space
        x = self.embedding(x)  # [B, T, emb_dim]

        # Add CLS token if using CLS pooling
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, emb_dim]
            if mask is not None:
                # Add mask for CLS token (always valid)
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)  # [B, T+1]

        # Add positional encoding to embeddings
        x = self.pos_encoder(x)  # [B, T(+1), emb_dim]

        # Input normalization
        x = self.input_norm(x)

        # Prepare attention mask if provided
        if mask is not None:
            # Convert mask to attention mask format [B, 1, 1, T(+1)] for broadcasting
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T(+1)]
        else:
            attention_mask = None

        # Pass through Transformer Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)

        # Pool sequence into a single vector
        if self.pooling_method == 'cls':
            # Use CLS token (first token)
            pooled = x[:, 0, :]  # [B, emb_dim]
        elif self.pooling_method == 'mean':
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, T, emb_dim]
                masked_x = x * mask_expanded
                pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                # Simple mean pooling
                pooled = x.mean(dim=1)  # [B, emb_dim]
        elif self.pooling_method == 'max':
            if mask is not None:
                # Masked max pooling with safe fallback if no valid positions exist
                mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, T, E]
                masked_x = x.masked_fill(~mask_expanded.bool(), float('-inf'))
                pooled = masked_x.max(dim=1)[0]  # [B, E]
                # If a sample had zero valid frames, replace -inf row with zeros
                valid = (mask.sum(dim=1) > 0).unsqueeze(-1)   # [B, 1]
                pooled = torch.where(valid, pooled, torch.zeros_like(pooled))
            else:
                pooled = x.max(dim=1)[0]  # [B, E]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Apply final dropout
        pooled = self.dropout_final(pooled)

        # Compute outputs for gloss and category
        gloss_out = self.gloss_head(pooled)      # [B, num_gloss]
        cat_out = self.category_head(pooled)     # [B, num_cat]

        return gloss_out, cat_out

    def get_attention_weights(self, x, mask=None):
        """
        Utility method to extract attention weights for visualization
        """
        B, T, _ = x.size()
        attention_weights = []

        # Forward pass through embedding and positional encoding
        x = self.embedding(x)
        
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)

        x = self.pos_encoder(x)
        x = self.input_norm(x)

        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None

        # Collect attention weights from each layer (in the same order as forward)
        for encoder_layer in self.encoder_layers:
            # call encoder_layer such that it performs its normal normalization/residuals
            x, attn_weights = encoder_layer(x, attention_mask, return_attn=True)
            # attn_weights shape: [B, H, T, T]
            attention_weights.append(attn_weights.detach().cpu())

        return attention_weights