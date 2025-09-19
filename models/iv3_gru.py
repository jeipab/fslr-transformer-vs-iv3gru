"""
InceptionV3 + GRU model for sign language recognition from short video clips.

This module provides:
- InceptionV3 backbone to extract 2048-D frame embeddings (optionally frozen)
- Two-layer GRU head for temporal modeling, with dropout regularization
- Dual heads for gloss and category classification
- Support for raw frames or precomputed 2048-D features and variable-length clips

Architecture (typical):
Input (raw): [B, T, 3, 299, 299] (ImageNet-normalized)
or Input (features): [B, T, 2048]
→ InceptionV3 (pretrained) → [B, T, 2048]
→ GRU (hidden=16) → Dropout(0.3)
→ GRU (hidden=12) → Dropout(0.3)
→ Linear heads → logits (gloss, category)

Usage:
    from models import InceptionV3GRU
    
    model = InceptionV3GRU(num_gloss=105, num_cat=10)
    gloss_logits, cat_logits = model(feats, features_already=True)

Training notes:
- Use CrossEntropyLoss on logits. If using raw frames, apply standard ImageNet normalization
  and consider light augmentations per-frame.
- Start with the backbone frozen; unfreeze for fine-tuning after the GRU head stabilizes.
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
        self.backbone = inception_v3(weights=weights)
        self.backbone.aux_logits = False
        self.backbone.fc = nn.Identity()  # output: (N, 2048)
        self.out_dim = 2048

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # keep BatchNorm stats stable when frozen

    @torch.no_grad()
    def _forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass when backbone is frozen (more efficient)."""
        # x: (N, 3, H, W), ideally H=W=299 and normalized to ImageNet stats.
        return self.backbone(x)  # (N, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not any(p.requires_grad for p in self.backbone.parameters()):
            return self._forward_frozen(x)
        return self.backbone(x)


def _dropout_packed(packed_seq, p: float, training: bool):
    """
    Apply dropout to a PackedSequence by dropping on its `.data` field.

    Args:
        packed_seq: torch.nn.utils.rnn.PackedSequence to be dropped out.
        p: Dropout probability.
        training: Whether in training mode.

    Returns:
        PackedSequence with dropout applied to underlying data.
    """
    if p <= 0.0:
        return packed_seq
    data = F.dropout(packed_seq.data, p=p, training=training)
    return nn.utils.rnn.PackedSequence(
        data, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices
    )


class InceptionV3GRU(nn.Module):
    """
    Hierarchical InceptionV3-GRU for gloss and category classification from video sequences.

    Uses a hierarchical architecture where:
        1. Category classification (semantic class/group) - first level
        2. Gloss classification (word/sign ID) - second level, conditioned on category

    The model uses category-specific gloss heads, where each category has its own
    dedicated head for gloss prediction, reflecting the natural hierarchy where
    glosses belong to categories.

    Args:
        num_gloss: Number of gloss classes.
        num_cat: Number of category classes.
        hidden1: Hidden units for first GRU layer (default 16).
        hidden2: Hidden units for second GRU layer (default 12).
        dropout: Dropout rate applied after GRU layers (default 0.3).
        pretrained_backbone: Load ImageNet weights for InceptionV3.
        freeze_backbone: Freeze CNN weights (recommended at start).

    Forward inputs:
        frames_or_feats:
            - If features_already=False: Tensor (B, T, 3, H, W)
            - If features_already=True: Tensor (B, T, 2048)
        lengths: Optional 1D Tensor (B,) with true sequence lengths.
        return_probs: If True, return probabilities; otherwise logits.
        features_already: Set True when passing precomputed 2048-D features.

    Returns:
        Tuple[Tensor, Tensor]: (gloss, category) tensors of shape (B, num_classes).
        
    Note:
        The hierarchical architecture predicts category first, then uses the
        predicted category to select the appropriate gloss head for prediction.
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
        
        # Hierarchical architecture: Category first, then category-conditioned gloss
        self.category_head = nn.Linear(hidden2, num_cat)
        
        # Category-conditioned gloss heads - each category has its own head
        self.gloss_heads = nn.ModuleList([
            nn.Linear(hidden2 + num_cat, num_gloss) for _ in range(num_cat)
        ])
        
        # Category embedding layer for conditioning gloss prediction
        self.category_embedding = nn.Embedding(num_cat, num_cat)


        # Xavier/orthogonal init for GRUs helps stability on small hidden sizes
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
        Extract per-frame 2048-D features using InceptionV3.

        Args:
            frames: Tensor of raw frames (B, T, 3, H, W), ImageNet-normalized.

        Returns:
            Tensor: features of shape (B, T, 2048).
        """
        if len(frames.shape) != 5:
            raise ValueError(f"Expected frames with 5 dimensions [B, T, C, H, W], got shape {frames.shape}")
        
        B, T, C, H, W = frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 color channels, got {C}")
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical forward pass with either raw frames or precomputed features.

        Args:
            frames_or_feats: (B, T, 3, H, W) if features_already=False, else (B, T, 2048).
            lengths: Optional true lengths (B,) for packed-sequence processing.
            return_probs: If True, return softmax probabilities instead of logits.
            features_already: Whether `frames_or_feats` are 2048-D features.

        Returns:
            Tuple[Tensor, Tensor]: (gloss, category) logits or probabilities of
            shapes (B, num_gloss) and (B, num_cat).
            
        Note:
            The hierarchical architecture predicts category first, then uses the
            predicted category to select the appropriate gloss head for prediction.
        """
        # Build (B, T, 2048) sequence
        if features_already:
            seq = frames_or_feats  # (B, T, 2048)
            if seq.shape[-1] != 2048:
                raise ValueError(f"Expected features with 2048 dimensions, got {seq.shape[-1]}")
        else:
            if len(frames_or_feats.shape) != 5:
                raise ValueError(f"Expected raw frames with shape [B, T, 3, H, W], got {frames_or_feats.shape}")
            seq = self.extract_features(frames_or_feats)  # (B, T, 2048)
        
        B = seq.shape[0]  # Get batch size

        # Packed or plain sequence through GRUs
        if lengths is not None:
            # Validate lengths tensor
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > seq.shape[1]:
                raise ValueError(f"Maximum length {lengths.max()} exceeds sequence length {seq.shape[1]}")
            
            # ensure lengths on CPU for pack_padded_sequence (avoid unnecessary device transfer)
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
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
        
        # Hierarchical predictions: Category first, then category-conditioned gloss
        cat_logits = self.category_head(h)                 # (B, num_cat)
        
        # Get predicted category for each sample in batch
        cat_pred = torch.argmax(cat_logits, dim=-1)        # (B,)
        
        # Create category embeddings for conditioning
        cat_emb = self.category_embedding(cat_pred)        # (B, num_cat)
        
        # Concatenate hidden features with category embeddings
        conditioned_features = torch.cat([h, cat_emb], dim=-1)  # (B, hidden2 + num_cat)
        
        # Predict gloss using category-specific head
        # For each sample, use the head corresponding to its predicted category
        gloss_logits = torch.zeros(B, self.gloss_heads[0].out_features, device=h.device, dtype=h.dtype)
        for i in range(B):
            category_idx = cat_pred[i].item()
            gloss_logits[i] = self.gloss_heads[category_idx](conditioned_features[i])
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper to return probability outputs from `forward`.

        Args:
            frames_or_feats: See `forward`.
            lengths: See `forward`.
            features_already: See `forward`.

        Returns:
            Tuple[Tensor, Tensor]: (gloss_probs, cat_probs), both softmaxed along class dim.
        """
        return self.forward(
            frames_or_feats,
            lengths=lengths,
            return_probs=True,
            features_already=features_already,
        )