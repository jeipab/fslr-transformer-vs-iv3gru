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


class InceptionV3GRUCtc(nn.Module):
    """
    InceptionV3-GRU model with CTC for continuous sign language recognition.
    
    This model extends the InceptionV3-GRU architecture for CTC-based continuous
    sign language recognition. Unlike the classification-based InceptionV3GRU,
    this model outputs frame-level predictions suitable for CTC decoding.
    
    Architecture:
    Input: [B, T, 3, 299, 299] raw frames OR [B, T, 2048] precomputed features
      ↓
    InceptionV3 Feature Extraction (if raw frames): [B, T, 3, 299, 299] → [B, T, 2048]
      ↓
    Bidirectional GRU Layer 1: [B, T, 2048] → [B, T, hidden1*2]
      ↓
    Dropout
      ↓
    Bidirectional GRU Layer 2: [B, T, hidden1*2] → [B, T, hidden2*2]
      ↓
    Dropout
      ↓
    CTC Head: [B, T, hidden2*2] → [B, T, num_ctc_classes]
      ↓
    LogSoftmax: [B, T, num_ctc_classes] → log probabilities
    
    Key Features:
    - Bidirectional GRU for capturing past and future context
    - Support for both raw frames and precomputed InceptionV3 features
    - Frame-level predictions for CTC decoding
    - Optional backbone freezing for transfer learning
    - Offline-only model (not suitable for real-time production due to size)
    
    Comparison with Other CTC Models:
    - vs SignTransformerCtc: Uses CNN features instead of keypoints, heavier model
    - vs MediaPipeGRUCtc: 50x larger, uses visual features instead of keypoints
    - vs InceptionV3GRU: Sequence-to-sequence instead of classification
    
    Args:
        num_ctc_classes (int): Number of CTC classes including blank token (default: 106).
        hidden1 (int): Hidden units for first GRU layer (default: 256).
        hidden2 (int): Hidden units for second GRU layer (default: 128).
        dropout (float): Dropout rate applied after GRU layers (default: 0.3).
        pretrained_backbone (bool): Load ImageNet weights for InceptionV3 (default: True).
        freeze_backbone (bool): Freeze CNN weights (default: True, recommended).
    
    Forward inputs:
        frames_or_feats (Tensor):
            - If features_already=False: Raw frames (B, T, 3, H, W)
            - If features_already=True: Precomputed features (B, T, 2048)
        lengths (Tensor, optional): True sequence lengths (B,) for packed sequences
        features_already (bool): Set True when passing precomputed 2048-D features
    
    Returns:
        Tensor: Log probabilities of shape (B, T, num_ctc_classes).
               Use .permute(1, 0, 2) for CTCLoss which expects [T, B, C].
    
    Usage:
        # With raw frames
        model = InceptionV3GRUCtc(num_ctc_classes=106)
        log_probs = model(frames)  # frames: [B, T, 3, 299, 299]
        
        # With precomputed features (faster)
        log_probs = model(features, features_already=True)  # features: [B, T, 2048]
        
        # For CTC loss, permute to [T, B, C]
        log_probs = log_probs.permute(1, 0, 2)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    """
    
    def __init__(
        self,
        num_ctc_classes: int = 106,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        num_cat: Optional[int] = None,
    ):
        """
        Initialize the InceptionV3-GRU-CTC model with optional category head.
        
        Args:
            num_ctc_classes (int): Number of CTC classes including blank.
            hidden1 (int): Hidden units for first GRU layer.
            hidden2 (int): Hidden units for second GRU layer.
            dropout (float): Dropout rate applied after GRU layers.
            pretrained_backbone (bool): Load ImageNet weights for InceptionV3.
            freeze_backbone (bool): Freeze CNN weights.
            num_cat (int, optional): Number of category classes. If None, CTC-only mode.
        """
        super().__init__()
        
        self.num_ctc_classes = num_ctc_classes
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.num_cat = num_cat
        
        # ===== FEATURE EXTRACTION =====
        # Initialize InceptionV3 feature extractor
        self.feat_extractor = InceptionV3FeatureExtractor(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )
        self.input_dim = self.feat_extractor.out_dim  # 2048
        
        # ===== TEMPORAL MODELING =====
        # Two-layer bidirectional GRU network for temporal sequence modeling
        self.gru1 = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Calculate effective hidden size after first bidirectional GRU
        effective_hidden1 = hidden1 * 2  # *2 for bidirectional
        
        self.gru2 = nn.GRU(
            input_size=effective_hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Calculate effective hidden size after second bidirectional GRU
        effective_hidden2 = hidden2 * 2  # *2 for bidirectional
        
        # ===== REGULARIZATION =====
        # Dropout layers for regularization
        self.do1 = nn.Dropout(dropout)  # After first GRU
        self.do2 = nn.Dropout(dropout)  # After second GRU
        
        # ===== DUAL OUTPUT HEADS =====
        # CTC head for gloss sequence prediction (per-frame)
        self.ctc_head = nn.Linear(effective_hidden2, num_ctc_classes)
        
        # Optional category head for auxiliary category classification (per-sequence)
        if num_cat is not None:
            self.category_head = nn.Linear(effective_hidden2, num_cat)
        else:
            self.category_head = None
        
        # ===== WEIGHT INITIALIZATION =====
        # Xavier/orthogonal initialization for GRU stability
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize GRU weights for stable training.
        
        Uses Xavier uniform initialization for input-to-hidden weights and
        orthogonal initialization for hidden-to-hidden weights.
        """
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
        
        # Initialize CTC head
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.zeros_(self.ctc_head.bias)
    
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
        """
        Forward pass through the InceptionV3-GRU-CTC model with optional category prediction.
        
        Args:
            frames_or_feats (Tensor): Input data of shape (B, T, 3, H, W) if features_already=False,
                                    or (B, T, 2048) if features_already=True.
            lengths (Tensor, optional): True sequence lengths (B,) for packed-sequence processing.
            features_already (bool): Whether frames_or_feats contains precomputed 2048-D features.
        
        Returns:
            If num_cat is None (CTC-only mode):
                Tensor: CTC log probabilities of shape (B, T, num_ctc_classes).
            
            If num_cat is provided (dual-task mode):
                Tuple[Tensor, Tensor]: (ctc_log_probs, cat_logits)
                    - ctc_log_probs: (B, T, num_ctc_classes) for CTC loss
                    - cat_logits: (B, T, num_cat) for per-frame category classification
        
        Raises:
            ValueError: If input dimensions are invalid.
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
        # Process sequence through bidirectional GRU layers
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
            y1, _ = self.gru1(packed)  # y1: PackedSequence
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            # Second GRU layer
            y2, _ = self.gru2(y1)  # y2: PackedSequence
            
            # Unpack sequence back to padded format
            y2, _ = nn.utils.rnn.pad_packed_sequence(y2, batch_first=True)
            
        else:
            # ===== REGULAR SEQUENCE PROCESSING =====
            # First GRU layer
            y1, _ = self.gru1(seq)  # y1: (B, T, hidden1*2)
            y1 = self.do1(y1)        # Apply dropout
            
            # Second GRU layer
            y2, _ = self.gru2(y1)   # y2: (B, T, hidden2*2)
        
        # ===== FINAL DROPOUT =====
        # Apply dropout to the GRU output sequence
        y2 = self.do2(y2)  # (B, T, hidden2*2)
        
        # Get batch size and sequence length
        B, T = y2.shape[0], y2.shape[1]
        
        # ===== CTC HEAD (PER-FRAME PREDICTION) =====
        # Project to CTC vocabulary size
        # [B, T, hidden2*2] → [B, T, num_ctc_classes]
        ctc_logits = self.ctc_head(y2)
        
        # Apply log softmax for CTC loss
        # CTCLoss expects log probabilities, not raw logits
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2)
        
        # ===== CATEGORY HEAD (PER-FRAME PREDICTION) =====
        if self.category_head is not None:
            # Category prediction per frame: [B, T, hidden2*2] → [B, T, num_cat]
            cat_logits = self.category_head(y2)
            
            return ctc_log_probs, cat_logits
        else:
            # CTC-only mode (backward compatibility)
            return ctc_log_probs
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information for logging and debugging.
        
        Returns:
            dict: Dictionary containing model architecture details.
        """
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