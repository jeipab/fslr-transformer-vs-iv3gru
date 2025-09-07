# transformer.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, emb_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

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

        # TODO: Add positional encoding to embeddings

        # TODO: Rearrange dimensions for Transformer (expected [T, B, E])

        # TODO: Pass through Transformer Encoder

        # TODO: Pool sequence into a single vector (mean or [CLS] token)

        # TODO: Compute outputs for gloss and category

        # TODO: Return gloss_out, cat_out
