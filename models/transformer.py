# transformer.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=300):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, emb_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, E]
        """
        x = x + (self.pe[:, :x.size(1), :]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # scale
        self.beta = nn.Parameter(torch.zeros(features))  # shift
        self.eps = eps

    def forward(self, x):
        """
        x: [B, T, E] or [B, E]
        Normalizes across the last dimension (E).
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_dim, ff_dim=512, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        """
        x: [B, T, E]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        assert emb_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Linear projections
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)

        # Dropout (can still be skipped in SelfAttention)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def SelfAttention(Q, K, V, mask=None, dropout=None):
        d_k = Q.size(-1)

        # 1. Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # 2. Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 3. Softmax to get attention weights
        attn = torch.softmax(scores, dim=-1)

        # 4. Apply dropout if provided
        if dropout is not None:
            attn = dropout(attn)

        # 5. Weighted sum of values
        out = torch.matmul(attn, V)
        return out, attn

    def forward(self, x, mask=None):
        B, T, E = x.size()

        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Call static SelfAttention
        out, attn = MultiHeadAttentionBlock.SelfAttention(Q, K, V, mask, self.dropout)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # Final projection
        out = self.W_o(out)

        return out, attn

class ResidualConnection(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(emb_dim)      # built-in LayerNorm is more efficient
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: [B, T, E]
        sublayer: a function/layer applied to normalized x
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim=512, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttentionBlock(emb_dim, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_dim, ff_dim, dropout)

        # Two residual connections: one for attention, one for feed-forward
        self.residual1 = ResidualConnection(emb_dim, dropout)
        self.residual2 = ResidualConnection(emb_dim, dropout)

    def forward(self, x, mask=None):
        # 1. Attention block with residual
        x = self.residual1(x, lambda x: self.attention(x, mask)[0])  # only take output, ignore attn weights

        # 2. Feed-forward block with residual
        x = self.residual2(x, self.feed_forward)

        return x

# Complete SignTransformer class with all missing parts

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
        self.input_norm = nn.LayerNorm(emb_dim)

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
                # Masked max pooling
                mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, T, emb_dim]
                masked_x = x.masked_fill(~mask_expanded.bool(), float('-inf'))
                pooled = masked_x.max(dim=1)[0]  # [B, emb_dim]
            else:
                # Simple max pooling
                pooled = x.max(dim=1)[0]  # [B, emb_dim]
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

        # Collect attention weights from each layer
        for encoder_layer in self.encoder_layers:
            # Get attention weights from the attention block
            attn_output, attn_weights = encoder_layer.attention(
                encoder_layer.residual1.norm(x), attention_mask
            )
            attention_weights.append(attn_weights.detach())
            
            # Continue forward pass
            x = encoder_layer.residual1(x, lambda x: attn_output)
            x = encoder_layer.residual2(x, encoder_layer.feed_forward)

        return attention_weights