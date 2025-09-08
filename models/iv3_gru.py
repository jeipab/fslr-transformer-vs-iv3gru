"""
inceptionv3_gru.py

This module implements an InceptionV3–GRU model for recognizing Filipino Sign Language
from short video clips. A pretrained InceptionV3 extracts 2048-D spatial features per frame,
which are modeled temporally by a lightweight two-layer GRU to predict gloss classes.

Key Components:
- InceptionV3FeatureExtractor: Pretrained CNN backbone (2048-D per frame; frozen by default)
- GRUStack: Two GRU layers with 16 and 12 hidden units to capture temporal dynamics
- Dropout: Regularization (p = 0.3) after GRU layers to reduce overfitting
- ClassifierHead: Linear → Softmax over gloss classes
- Variable-Length Support: Optional pack_padded_sequence handling for ragged clips
- Feature Flexibility: Accepts raw frames or precomputed 2048-D features (features_already=True)

Architecture:
Input (raw frames): [B, T=30, 3, H, W] (expected 299×299, ImageNet-normalized)
or Input (features): [B, T=30, 2048]
→ InceptionV3 (pretrained) → [B, T, 2048]
→ GRU (hidden=16) → Dropout(0.3)
→ GRU (hidden=12) → Dropout(0.3)
→ Linear → Softmax (gloss: N classes)

Usage:
    from models.inceptionv3_gru import InceptionV3GRU
    import torch

    # Initialize model (set num_classes to your gloss vocabulary size)
    model = InceptionV3GRU(
        num_classes=105,
        hidden1=16,
        hidden2=12,
        dropout=0.3,
        pretrained_backbone=True,
        freeze_backbone=True,  # unfreeze later for fine-tuning if desired
    )

    # Forward with raw frames (B, T=30, 3, 299, 299), normalized to ImageNet stats
    logits = model(frames, features_already=False)        # -> [B, 105]

    # Or forward with precomputed features (B, T=30, 2048)
    logits = model(feats, features_already=True)

    # Optional variable-length handling (when sequences are padded)
    logits = model(feats, lengths=seq_lengths, features_already=True)

    # Get probabilities instead of logits
    probs = model.predict_proba(feats, features_already=True)

Training Notes:
- Use CrossEntropyLoss with logits; apply standard augmentation at the frame level if using raw frames.
- Start with the backbone frozen; optionally unfreeze later for modest gains once the GRU head stabilizes.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3FeatureExtractor(nn.Module):
    """
    Pretrained InceptionV3 feature extractor producing 2048-D embeddings per frame.
    Replaces the final FC with Identity so forward() returns (N, 2048).
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = inception_v3(weights=weights, aux_logits=False, transform_input=False)
        self.backbone.fc = nn.Identity()  # output: (N, 2048)
        self.out_dim = 2048

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # keep BatchNorm stats stable when frozen

    @torch.no_grad()
    def _forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W), ideally H=W=299 and normalized to ImageNet stats.
        return self.backbone(x)  # (N, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not any(p.requires_grad for p in self.backbone.parameters()):
            return self._forward_frozen(x)
        return self.backbone(x)


def _dropout_packed(packed_seq, p: float, training: bool):
    if p <= 0.0:
        return packed_seq
    data = F.dropout(packed_seq.data, p=p, training=training)
    return nn.utils.rnn.PackedSequence(
        data, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices
    )


class InceptionV3GRU(nn.Module):
    """
    InceptionV3-GRU for sign gloss recognition.

    Args:
        num_classes: number of gloss classes.
        hidden1: hidden units for first GRU layer (default 16).
        hidden2: hidden units for second GRU layer (default 12).
        dropout: dropout rate applied after each GRU layer (default 0.3).
        pretrained_backbone: load ImageNet weights for InceptionV3.
        freeze_backbone: freeze CNN weights (recommended to start).
    Forward inputs:
        frames_or_feats:
            - If features_already=False: Tensor of shape (B, T, 3, H, W)
            - If features_already=True: Tensor of shape (B, T, 2048)
        lengths (optional): 1D Tensor of sequence lengths before padding (B,)
        return_probs: if True returns probabilities (softmax); else logits.
        features_already: set True if passing precomputed 2048-D features.
    Returns:
        logits or probabilities of shape (B, num_classes)
    """
    def __init__(
        self,
        num_gloss: int,
        num_cat: int,
        hidden1: int = 16,
        hidden2: int = 12,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.feat_extractor = InceptionV3FeatureExtractor(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )
        self.input_dim = self.feat_extractor.out_dim  # 2048
        # Two GRU layers with different hidden sizes (stacked modules)
        self.gru1 = nn.GRU(input_size=self.input_dim, hidden_size=hidden1, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden1, hidden_size=hidden2, num_layers=1, batch_first=True)
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.gloss_head = nn.Linear(hidden2, num_gloss)
        self.category_head = nn.Linear(hidden2, num_cat)


        # Kaiming/orthogonal init for GRUs helps stability on small hidden sizes
        for gru in (self.gru1, self.gru2):
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, H, W) → features: (B, T, 2048)
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        feats = self.feat_extractor(x)                     # (B*T, 2048)
        feats = feats.reshape(B, T, -1)                    # (B, T, 2048)
        return feats

    def forward(
        self,
        frames_or_feats: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        features_already: bool = False,
    ) -> torch.Tensor:
        # Build (B, T, 2048) sequence
        if features_already:
            seq = frames_or_feats  # (B, T, 2048)
        else:
            seq = self.extract_features(frames_or_feats)  # (B, T, 2048)

        # Packed or plain sequence through GRUs
        if lengths is not None:
            # ensure lengths on CPU for pack_padded_sequence
            lengths_cpu = lengths.to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(seq, lengths_cpu, batch_first=True, enforce_sorted=False)
            y1, h1 = self.gru1(packed)                     # y1: PackedSequence, h1: (1, B, hidden1)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)     # dropout on packed.data
            y2, h2 = self.gru2(y1)                         # h2: (1, B, hidden2)
            h = h2[-1]                                     # (B, hidden2)
        else:
            y1, h1 = self.gru1(seq)                        # y1: (B, T, hidden1)
            y1 = self.do1(y1)
            y2, h2 = self.gru2(y1)                         # h2: (1, B, hidden2)
            h = h2[-1]                                     # (B, hidden2)

        h = self.do2(h)                                    # final dropout on summary
        gloss_logits = self.gloss_head(h)                  # (B, num_gloss)
        cat_logits = self.category_head(h)                 # (B, num_cat)
        if return_probs:
            gloss_probs = F.softmax(gloss_logits, dim=-1)
            cat_probs = F.softmax(cat_logits, dim=-1)
            return gloss_probs, cat_probs
        return gloss_logits, cat_logits

    def predict_proba(
        self,
        frames_or_feats: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        features_already: bool = False,
    ) -> torch.Tensor:
        return self.forward(frames_or_feats, lengths=lengths, return_probs=True, features_already=features_already)