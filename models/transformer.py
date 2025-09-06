# transformer.py
import torch
import torch.nn as nn

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

        # TODO: Define positional encoding (sinusoidal or learnable)

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
