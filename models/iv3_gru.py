"""
InceptionV3 + GRU model for sign language recognition from video sequences.

This module implements a hybrid CNN-RNN architecture specifically designed for
sign language recognition using video sequences. The model combines the powerful
visual feature extraction capabilities of InceptionV3 with the temporal modeling
strength of GRU networks.

Architecture Overview:
Input: [B, T, 3, 299, 299] raw frames (ImageNet-normalized)
  ↓
InceptionV3 Feature Extraction: [B, T, 3, 299, 299] → [B, T, 2048]
  ↓
First GRU Layer: [B, T, 2048] → [B, T, hidden1] + Dropout
  ↓
Second GRU Layer: [B, T, hidden1] → [B, T, hidden2] + Dropout
  ↓
Final Hidden State: [B, hidden2]
  ↓
Dual Classification Heads:
  • Gloss Head: [B, hidden2] → [B, num_gloss]
  • Category Head: [B, hidden2] → [B, num_cat]

Key Features:
- Pretrained InceptionV3 backbone for robust visual feature extraction
- Two-layer GRU network for temporal sequence modeling
- Support for both raw frames and precomputed features
- Variable-length sequence handling with packed sequences
- Dual classification heads for hierarchical prediction
- Optional backbone freezing for transfer learning

Usage:
    from models import InceptionV3GRU
    
    # Initialize model
    model = InceptionV3GRU(num_gloss=105, num_cat=10)
    
    # Forward pass with raw frames
    gloss_logits, cat_logits = model(frames, features_already=False)
    
    # Forward pass with precomputed features
    gloss_logits, cat_logits = model(features, features_already=True)
    
    # Get probabilities instead of logits
    gloss_probs, cat_probs = model.predict_proba(frames)

Training Notes:
- Use CrossEntropyLoss on logits for both gloss and category predictions
- Apply ImageNet normalization to raw frames: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Start with frozen backbone, unfreeze for fine-tuning after GRU head stabilizes
- GRU weights are initialized with Xavier/orthogonal initialization for stability
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 feature extractor for frame-level visual feature extraction.
    
    This module wraps a pretrained InceptionV3 model and modifies it to extract
    2048-dimensional feature vectors from individual frames. The final classification
    layer is replaced with an identity function to return raw features.
    
    The extractor can operate in two modes:
    - Frozen mode: Backbone weights are frozen for transfer learning
    - Trainable mode: Backbone weights can be fine-tuned
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """
        Initialize InceptionV3 feature extractor.
        
        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights.
            freeze (bool): Whether to freeze backbone parameters for transfer learning.
        """
        super().__init__()
        
        # Load pretrained InceptionV3 model
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = inception_v3(weights=weights)
        
        # Disable auxiliary logits (not needed for feature extraction)
        self.backbone.aux_logits = False
        
        # Replace final classification layer with identity to get raw features
        self.backbone.fc = nn.Identity()  # Output: (N, 2048)
        self.out_dim = 2048

        # Freeze backbone parameters if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Set to eval mode to keep BatchNorm statistics stable when frozen
            self.backbone.eval()

    @torch.no_grad()
    def _forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass when backbone is frozen (more efficient).
        
        Args:
            x (Tensor): Input tensor of shape (N, 3, H, W).
                       Ideally H=W=299 and normalized to ImageNet statistics.
        Returns:
            Tensor: Feature tensor of shape (N, 2048).
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x (Tensor): Input tensor of shape (N, 3, H, W).
        Returns:
            Tensor: Feature tensor of shape (N, 2048).
        """
        # Use frozen forward pass if no parameters require gradients
        if not any(param.requires_grad for param in self.backbone.parameters()):
            return self._forward_frozen(x)
        return self.backbone(x)


def _dropout_packed(packed_seq, p: float, training: bool):
    """
    Apply dropout to a PackedSequence by operating on its underlying data.
    
    This utility function applies dropout to packed sequences, which is necessary
    because standard dropout layers don't work directly with PackedSequence objects.
    The dropout is applied to the packed data tensor while preserving the sequence
    structure information.

    Args:
        packed_seq: PackedSequence to apply dropout to.
        p (float): Dropout probability.
        training (bool): Whether the model is in training mode.

    Returns:
        PackedSequence: New PackedSequence with dropout applied to underlying data.
    """
    # Skip dropout if probability is 0 or negative
    if p <= 0.0:
        return packed_seq
        
    # Apply dropout to the packed data tensor
    data = F.dropout(packed_seq.data, p=p, training=training)
    
    # Create new PackedSequence with dropped-out data
    return nn.utils.rnn.PackedSequence(
        data, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices
    )


class InceptionV3GRU(nn.Module):
    """
    InceptionV3-GRU hybrid model for sign language recognition.
    
    This model combines the visual feature extraction power of InceptionV3 with
    the temporal modeling capabilities of GRU networks. It processes video sequences
    to predict both specific sign words (gloss) and semantic categories.
    
    The architecture consists of:
    1. InceptionV3 backbone for frame-level feature extraction
    2. Two-layer GRU network for temporal sequence modeling
    3. Dual classification heads for hierarchical prediction
    
    The model supports both raw video frames and precomputed features, making it
    flexible for different preprocessing pipelines.

    Args:
        num_gloss (int): Number of gloss classes (specific sign words).
        num_cat (int): Number of category classes (semantic groups).
        hidden1 (int): Hidden units for first GRU layer (default: 16).
        hidden2 (int): Hidden units for second GRU layer (default: 12).
        dropout (float): Dropout rate applied after GRU layers (default: 0.3).
        pretrained_backbone (bool): Load ImageNet weights for InceptionV3.
        freeze_backbone (bool): Freeze CNN weights (recommended for transfer learning).

    Forward inputs:
        frames_or_feats (Tensor):
            - If features_already=False: Raw frames (B, T, 3, H, W)
            - If features_already=True: Precomputed features (B, T, 2048)
        lengths (Tensor, optional): True sequence lengths (B,) for packed sequences.
        return_probs (bool): If True, return probabilities; otherwise logits.
        features_already (bool): Set True when passing precomputed 2048-D features.

    Returns:
        Tuple[Tensor, Tensor]: (gloss_logits, category_logits) of shapes (B, num_gloss) and (B, num_cat).
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
        """
        Initialize the InceptionV3-GRU model.
        
        Args:
            num_gloss (int): Number of gloss classes.
            num_cat (int): Number of category classes.
            hidden1 (int): Hidden units for first GRU layer.
            hidden2 (int): Hidden units for second GRU layer.
            dropout (float): Dropout rate applied after GRU layers.
            pretrained_backbone (bool): Load ImageNet weights for InceptionV3.
            freeze_backbone (bool): Freeze CNN weights.
        """
        super().__init__()
        
        # ===== FEATURE EXTRACTION =====
        # Initialize InceptionV3 feature extractor
        self.feat_extractor = InceptionV3FeatureExtractor(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )
        self.input_dim = self.feat_extractor.out_dim  # 2048
        
        # ===== TEMPORAL MODELING =====
        # Two-layer GRU network for temporal sequence modeling
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
        
        # ===== REGULARIZATION =====
        # Dropout layers for regularization
        self.do1 = nn.Dropout(dropout)  # After first GRU
        self.do2 = nn.Dropout(dropout)  # After second GRU
        
        # ===== CLASSIFICATION HEADS =====
        # Dual classification heads
        self.gloss_head = nn.Linear(hidden2, num_gloss)      # Gloss prediction
        self.category_head = nn.Linear(hidden2, num_cat)      # Category prediction

        # ===== WEIGHT INITIALIZATION =====
        # Xavier/orthogonal initialization for GRU stability with small hidden sizes
        for gru in (self.gru1, self.gru2):
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    # Input-to-hidden weights: Xavier uniform initialization
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    # Hidden-to-hidden weights: Orthogonal initialization
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    # Bias terms: Zero initialization
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
        """
        Forward pass through the complete InceptionV3-GRU model.

        Args:
            frames_or_feats (Tensor): Input data of shape (B, T, 3, H, W) if features_already=False,
                                    or (B, T, 2048) if features_already=True.
            lengths (Tensor, optional): True sequence lengths (B,) for packed-sequence processing.
            return_probs (bool): If True, return softmax probabilities instead of logits.
            features_already (bool): Whether frames_or_feats contains precomputed 2048-D features.

        Returns:
            Tuple[Tensor, Tensor]: (gloss_logits, category_logits) of shapes (B, num_gloss) and (B, num_cat).
                                  If return_probs=True, returns probabilities instead of logits.
        
        Raises:
            ValueError: If input dimensions are invalid or sequence lengths are invalid.
        """
        # ===== INPUT PROCESSING =====
        # Build feature sequence (B, T, 2048)
        if features_already:
            # Use precomputed features directly
            seq = frames_or_feats  # (B, T, 2048)
            if seq.shape[-1] != 2048:
                raise ValueError(
                    f"Expected features with 2048 dimensions, got {seq.shape[-1]}"
                )
        else:
            # Extract features from raw frames
            if len(frames_or_feats.shape) != 5:
                raise ValueError(
                    f"Expected raw frames with shape [B, T, 3, H, W], got {frames_or_feats.shape}"
                )
            seq = self.extract_features(frames_or_feats)  # (B, T, 2048)

        # ===== TEMPORAL MODELING =====
        # Process sequence through GRU layers
        if lengths is not None:
            # ===== PACKED SEQUENCE PROCESSING =====
            # Validate lengths tensor
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > seq.shape[1]:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {seq.shape[1]}"
                )
            
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            # Pack sequences for efficient processing
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            # First GRU layer
            y1, h1 = self.gru1(packed)  # y1: PackedSequence, h1: (1, B, hidden1)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            # Second GRU layer
            y2, h2 = self.gru2(y1)      # h2: (1, B, hidden2)
            h = h2[-1]                  # Extract final hidden state: (B, hidden2)
            
        else:
            # ===== REGULAR SEQUENCE PROCESSING =====
            # First GRU layer
            y1, h1 = self.gru1(seq)     # y1: (B, T, hidden1)
            y1 = self.do1(y1)           # Apply dropout
            
            # Second GRU layer
            y2, h2 = self.gru2(y1)     # h2: (1, B, hidden2)
            h = h2[-1]                  # Extract final hidden state: (B, hidden2)

        # ===== FINAL PROCESSING =====
        # Apply final dropout to hidden state
        h = self.do2(h)  # (B, hidden2)
        
        # ===== CLASSIFICATION =====
        # Generate predictions from both heads
        gloss_logits = self.gloss_head(h)    # (B, num_gloss)
        cat_logits = self.category_head(h)   # (B, num_cat)
        
        # Return probabilities if requested
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
        Convenience method to return probability outputs.
        
        This method is a wrapper around the forward method that automatically
        applies softmax to the logits to return probability distributions.

        Args:
            frames_or_feats (Tensor): See forward method documentation.
            lengths (Tensor, optional): See forward method documentation.
            features_already (bool): See forward method documentation.

        Returns:
            Tuple[Tensor, Tensor]: (gloss_probs, cat_probs) of shapes (B, num_gloss) and (B, num_cat).
                                  Both tensors contain softmax probabilities.
        """
        return self.forward(
            frames_or_feats,
            lengths=lengths,
            return_probs=True,
            features_already=features_already,
        )