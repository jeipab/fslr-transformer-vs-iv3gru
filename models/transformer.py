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

class SignTransformer(nn.Module):
    def __init__(self,
                    input_dim=156,     # 78 keypoints × 2 coords
                    emb_dim=256,       # embedding dimension
                    n_heads=8,         # number of attention heads
                    n_layers=4,        # number of encoder layers
                    num_gloss=105,     # number of gloss classes
                    num_cat=10):       # number of category classes
        super(SignTransformer, self).__init__()

        # TODO: Define input projection (Linear layer 156 → emb_dim)
        self.embedding = nn.Linear(input_dim, emb_dim)

        # TODO: Define positional encoding (sinusoidal or learnable)
        self.pos_encoder = PositionalEncoding(emb_dim)

        # TODO: Define Transformer Encoder (stack of encoder layers)

        # TODO: Define output heads (two classifiers: gloss + category)

    def forward(self, x):
        """
        x shape: [B, T, 156]
        B = batch size, T = sequence length, 156 = keypoint features
        """

        # TODO: Project input into embedding space
        x = self.embedding(x)  # [B, T, E]

        # TODO: Add positional encoding to embeddings
        x = self.pos_encoder(x)

        # TODO: Rearrange dimensions for Transformer (expected [T, B, E])
        x = self.input_norm(x)

        # TODO: Pass through Transformer Encoder

        # TODO: Pool sequence into a single vector (mean or [CLS] token)

        # TODO: Compute outputs for gloss and category

        # TODO: Return gloss_out, cat_out
