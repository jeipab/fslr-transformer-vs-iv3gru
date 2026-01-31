"""
InceptionV3 + GRU model for sign language recognition.

Hybrid CNN-RNN architecture combining InceptionV3 visual features with GRU temporal modeling.
Supports both raw frames and precomputed features.
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for frame-level visual features."""
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """Initialize InceptionV3 feature extractor."""
        super().__init__()
        
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = inception_v3(weights=weights)
        self.backbone.aux_logits = False
        self.backbone.fc = nn.Identity()
        self.out_dim = 2048

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    @torch.no_grad()
    def _forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass when backbone is frozen."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor."""
        if not any(param.requires_grad for param in self.backbone.parameters()):
            return self._forward_frozen(x)
        return self.backbone(x)


def _dropout_packed(packed_seq, p: float, training: bool):
    """Apply dropout to a PackedSequence."""
    if p <= 0.0:
        return packed_seq
        
    data = F.dropout(packed_seq.data, p=p, training=training)
    return nn.utils.rnn.PackedSequence(
        data, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices
    )


class InceptionV3GRU(nn.Module):
    """
    InceptionV3-GRU hybrid model for sign language recognition.
    
    Combines InceptionV3 visual features with GRU temporal modeling.
    Supports both raw frames and precomputed features.
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
        """Initialize the InceptionV3-GRU model."""
        super().__init__()
        
        self.feat_extractor = InceptionV3FeatureExtractor(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )
        self.input_dim = self.feat_extractor.out_dim
        
        self.gru1 = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=hidden1, 
            num_layers=1, 
            batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=hidden1, 
            hidden_size=hidden2, 
            num_layers=1, 
            batch_first=True
        )
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.gloss_head = nn.Linear(hidden2, num_gloss)
        self.category_head = nn.Linear(hidden2, num_cat)

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
        
        This method processes raw video frames through the InceptionV3 backbone
        to extract high-level visual features for each frame in the sequence.

        Args:
            frames (Tensor): Raw frames of shape (B, T, 3, H, W).
                           Should be ImageNet-normalized.

        Returns:
            Tensor: Feature tensor of shape (B, T, 2048).
        
        Raises:
            ValueError: If input tensor doesn't have the expected shape.
        """
        # ===== INPUT VALIDATION =====
        # Check tensor dimensions
        if len(frames.shape) != 5:
            raise ValueError(
                f"Expected frames with 5 dimensions [B, T, C, H, W], got shape {frames.shape}"
            )
        
        B, T, C, H, W = frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 color channels, got {C}")
        
        # ===== FEATURE EXTRACTION =====
        # Reshape frames for batch processing: (B, T, 3, H, W) → (B*T, 3, H, W)
        x = frames.reshape(B * T, C, H, W)
        
        # Extract features for all frames: (B*T, 3, H, W) → (B*T, 2048)
        feats = self.feat_extractor(x)
        
        # Reshape back to sequence format: (B*T, 2048) → (B, T, 2048)
        feats = feats.reshape(B, T, -1)
        
        return feats

    def forward(
        self,
        frames_or_feats: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        features_already: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        if features_already:
            seq = frames_or_feats
            if seq.shape[-1] != 2048:
                raise ValueError(
                    f"Expected features with 2048 dimensions, got {seq.shape[-1]}"
                )
        else:
            if len(frames_or_feats.shape) != 5:
                raise ValueError(
                    f"Expected raw frames with shape [B, T, 3, H, W], got {frames_or_feats.shape}"
                )
            seq = self.extract_features(frames_or_feats)

        if lengths is not None:
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > seq.shape[1]:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {seq.shape[1]}"
                )
            
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            y1, h1 = self.gru1(packed)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            y2, h2 = self.gru2(y1)
            h = h2[-1]
            
        else:
            y1, h1 = self.gru1(seq)
            y1 = self.do1(y1)
            
            y2, h2 = self.gru2(y1)
            h = h2[-1]

        h = self.do2(h)
        
        gloss_logits = self.gloss_head(h)
        cat_logits = self.category_head(h)
        
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
        """Return probability outputs instead of logits."""
        return self.forward(
            frames_or_feats,
            lengths=lengths,
            return_probs=True,
            features_already=features_already,
        )


class InceptionV3GRUCtc(nn.Module):
    """InceptionV3-GRU model with CTC for continuous sign language recognition."""
    
    def __init__(
        self,
        num_ctc_classes: int = 106,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        num_cat: Optional[int] = None,
    ):
        """Initialize the InceptionV3-GRU-CTC model."""
        super().__init__()
        
        self.num_ctc_classes = num_ctc_classes
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.num_cat = num_cat
        
        self.feat_extractor = InceptionV3FeatureExtractor(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )
        self.input_dim = self.feat_extractor.out_dim
        
        self.gru1 = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        effective_hidden1 = hidden1 * 2
        
        self.gru2 = nn.GRU(
            input_size=effective_hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        effective_hidden2 = hidden2 * 2
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.ln1 = nn.LayerNorm(effective_hidden1)
        self.ln2 = nn.LayerNorm(effective_hidden2)
        
        self.ctc_head = nn.Linear(effective_hidden2, num_ctc_classes)
        
        if num_cat is not None:
            self.category_head = nn.Linear(effective_hidden2, num_cat)
        else:
            self.category_head = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize GRU weights for stable CTC training."""
        for gru in (self.gru1, self.gru2):
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        
        nn.init.xavier_uniform_(self.ctc_head.weight, gain=0.5)
        nn.init.zeros_(self.ctc_head.bias)
        
        if self.category_head is not None:
            nn.init.xavier_uniform_(self.category_head.weight, gain=0.5)
            nn.init.zeros_(self.category_head.bias)
    
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame 2048-D features using InceptionV3.
        
        This method processes raw video frames through the InceptionV3 backbone
        to extract high-level visual features for each frame in the sequence.

        Args:
            frames (Tensor): Raw frames of shape (B, T, 3, H, W).
                           Should be ImageNet-normalized.

        Returns:
            Tensor: Feature tensor of shape (B, T, 2048).
        
        Raises:
            ValueError: If input tensor doesn't have the expected shape.
        """
        # ===== INPUT VALIDATION =====
        # Check tensor dimensions
        if len(frames.shape) != 5:
            raise ValueError(
                f"Expected frames with 5 dimensions [B, T, C, H, W], got shape {frames.shape}"
            )
        
        B, T, C, H, W = frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 color channels, got {C}")
        
        # ===== FEATURE EXTRACTION =====
        # Reshape frames for batch processing: (B, T, 3, H, W) → (B*T, 3, H, W)
        x = frames.reshape(B * T, C, H, W)
        
        # Extract features for all frames: (B*T, 3, H, W) → (B*T, 2048)
        feats = self.feat_extractor(x)
        
        # Reshape back to sequence format: (B*T, 2048) → (B, T, 2048)
        feats = feats.reshape(B, T, -1)
        
        return feats
    
    def forward(
        self,
        frames_or_feats: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        features_already: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        if features_already:
            seq = frames_or_feats
            if seq.shape[-1] != 2048:
                raise ValueError(
                    f"Expected features with 2048 dimensions, got {seq.shape[-1]}"
                )
        else:
            if len(frames_or_feats.shape) != 5:
                raise ValueError(
                    f"Expected raw frames with shape [B, T, 3, H, W], got {frames_or_feats.shape}"
                )
            seq = self.extract_features(frames_or_feats)
        
        if lengths is not None:
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > seq.shape[1]:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {seq.shape[1]}"
                )
            
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            y1, _ = self.gru1(packed)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            y2, _ = self.gru2(y1)
            
            y2, _ = nn.utils.rnn.pad_packed_sequence(y2, batch_first=True)
            
            y2 = self.do2(y2)
            
        else:
            y1, _ = self.gru1(seq)
            y1 = self.do1(y1)
            y1 = self.ln1(y1)
            
            y2, _ = self.gru2(y1)
        
        if lengths is None:
            y2 = self.do2(y2)
            y2 = self.ln2(y2)
        
        ctc_logits = self.ctc_head(y2)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2)
        
        if self.category_head is not None:
            if len(y2.shape) != 3:
                raise ValueError(
                    f"Expected y2 to have 3 dimensions [B, T, hidden2*2], got shape {y2.shape}. "
                    f"This indicates a problem with GRU sequence processing."
                )
            
            cat_logits = self.category_head(y2)
            
            return ctc_log_probs, cat_logits
        else:
            return ctc_log_probs
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'InceptionV3GRUCtc',
            'input_dim': self.input_dim,
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'num_ctc_classes': self.num_ctc_classes,
            'dropout': self.dropout_p,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }