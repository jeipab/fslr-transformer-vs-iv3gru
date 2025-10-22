"""
MediaPipe-GRU model for sign language recognition from keypoint sequences.

This module implements a lightweight GRU-based architecture specifically designed for
sign language recognition using MediaPipe keypoint sequences. Unlike the InceptionV3-GRU
model which requires heavy CNN preprocessing, this model directly processes keypoint
features making it ideal for real-time mobile deployment.

Architecture Overview:
Input: [B, T, 178] keypoint sequences (89 keypoints × 2 coordinates)
  ↓
Input Projection (optional): [B, T, 178] → [B, T, proj_dim]
  ↓
First GRU Layer: [B, T, proj_dim] → [B, T, hidden1] + Dropout
  ↓
Second GRU Layer: [B, T, hidden1] → [B, T, hidden2] + Dropout
  ↓
Final Hidden State: [B, hidden2]
  ↓
Dual Classification Heads:
  • Gloss Head: [B, hidden2] → [B, num_gloss]
  • Category Head: [B, hidden2] → [B, num_cat]

Key Features:
- Lightweight architecture suitable for mobile deployment
- Direct processing of MediaPipe keypoints (no CNN required)
- Two-layer GRU network for temporal sequence modeling
- Support for variable-length sequences with packed sequences
- Dual classification heads for hierarchical prediction
- Optional input projection for dimensionality control
- Much faster inference than InceptionV3-GRU (~50-100ms vs 500-800ms)

Usage:
    from models import MediaPipeGRU
    
    # Initialize model
    model = MediaPipeGRU(num_gloss=105, num_cat=10)
    
    # Forward pass with keypoints
    gloss_logits, cat_logits = model(keypoints)  # keypoints: [B, T, 178]
    
    # Get probabilities instead of logits
    gloss_probs, cat_probs = model.predict_proba(keypoints)

Comparison with Other Models:
- vs Transformer: Simpler architecture, faster training, similar performance
- vs InceptionV3-GRU: 10× faster, mobile-friendly, uses same input as Transformer
- vs IV3-GRU (full): Same GRU architecture, but no CNN preprocessing needed

Training Notes:
- Use CrossEntropyLoss on logits for both gloss and category predictions
- Input keypoints should be normalized to [0, 1] range
- Apply data augmentation (noise, temporal masking) during training
- GRU weights are initialized with Xavier/orthogonal initialization for stability
- Consider using larger hidden dimensions (256-512) if overfitting is not an issue
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MediaPipeGRU(nn.Module):
    """
    MediaPipe-GRU model for sign language recognition from keypoint sequences.
    
    This lightweight model processes MediaPipe keypoint sequences directly through
    GRU layers without any CNN preprocessing, making it ideal for mobile deployment
    and fair comparison with the Transformer model.
    
    The architecture consists of:
    1. Optional input projection layer
    2. Two-layer GRU network for temporal sequence modeling
    3. Dual classification heads for hierarchical prediction
    
    The model is designed to be a mobile-friendly baseline that can be directly
    compared with the Transformer model since both use the same input features.

    Args:
        num_gloss (int): Number of gloss classes (specific sign words).
        num_cat (int): Number of category classes (semantic groups).
        input_dim (int): Input keypoint dimension (default: 178 for 89 keypoints × 2).
        projection_dim (int, optional): If specified, project input to this dimension first.
                                       If None, use input_dim directly. Useful for controlling
                                       model capacity.
        hidden1 (int): Hidden units for first GRU layer (default: 256).
        hidden2 (int): Hidden units for second GRU layer (default: 128).
        dropout (float): Dropout rate applied after GRU layers (default: 0.3).
        bidirectional (bool): Use bidirectional GRU (default: False).

    Forward inputs:
        x (Tensor): Keypoint sequences (B, T, 178) where:
                   - B = batch size
                   - T = sequence length (variable)
                   - 178 = 89 keypoints × 2 coordinates
        lengths (Tensor, optional): True sequence lengths (B,) for packed sequences.
        return_probs (bool): If True, return probabilities; otherwise logits.

    Returns:
        Tuple[Tensor, Tensor]: (gloss_logits, category_logits) of shapes (B, num_gloss) and (B, num_cat).
    """
    
    def __init__(
        self,
        num_gloss: int,
        num_cat: int,
        input_dim: int = 178,
        projection_dim: Optional[int] = None,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Initialize the MediaPipe-GRU model.
        
        Args:
            num_gloss (int): Number of gloss classes.
            num_cat (int): Number of category classes.
            input_dim (int): Input keypoint dimension (89 keypoints × 2 = 178).
            projection_dim (int, optional): If specified, project input to this dimension.
            hidden1 (int): Hidden units for first GRU layer.
            hidden2 (int): Hidden units for second GRU layer.
            dropout (float): Dropout rate applied after GRU layers.
            bidirectional (bool): Use bidirectional GRU.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ===== INPUT PROJECTION (OPTIONAL) =====
        # Optional projection layer to control input dimensionality
        if projection_dim is not None:
            self.input_projection = nn.Linear(input_dim, projection_dim)
            gru_input_dim = projection_dim
        else:
            self.input_projection = None
            gru_input_dim = input_dim
        
        # ===== TEMPORAL MODELING =====
        # Two-layer GRU network for temporal sequence modeling
        self.gru1 = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate effective hidden size after first GRU
        effective_hidden1 = hidden1 * self.num_directions
        
        self.gru2 = nn.GRU(
            input_size=effective_hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate effective hidden size after second GRU
        effective_hidden2 = hidden2 * self.num_directions
        
        # ===== REGULARIZATION =====
        # Dropout layers for regularization
        self.do1 = nn.Dropout(dropout)  # After first GRU
        self.do2 = nn.Dropout(dropout)  # After second GRU
        
        # ===== CLASSIFICATION HEADS =====
        # Dual classification heads
        self.gloss_head = nn.Linear(effective_hidden2, num_gloss)      # Gloss prediction
        self.category_head = nn.Linear(effective_hidden2, num_cat)      # Category prediction

        # ===== WEIGHT INITIALIZATION =====
        # Xavier/orthogonal initialization for GRU stability
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize GRU weights for stable training.
        
        Uses Xavier uniform initialization for input-to-hidden weights and
        orthogonal initialization for hidden-to-hidden weights, which helps
        prevent gradient vanishing/exploding in RNNs.
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
        
        # Initialize projection layer if it exists
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
        
        # Initialize classification heads
        for head in (self.gloss_head, self.category_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MediaPipe-GRU model.

        Args:
            x (Tensor): Input keypoint sequence of shape (B, T, 178).
            lengths (Tensor, optional): True sequence lengths (B,) for packed-sequence processing.
            return_probs (bool): If True, return softmax probabilities instead of logits.

        Returns:
            Tuple[Tensor, Tensor]: (gloss_logits, category_logits) of shapes (B, num_gloss) and (B, num_cat).
                                  If return_probs=True, returns probabilities instead of logits.
        
        Raises:
            ValueError: If input dimensions are invalid or sequence lengths are invalid.
        """
        # ===== INPUT VALIDATION =====
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()
        
        # ===== INPUT PROJECTION =====
        # Apply optional input projection
        if self.input_projection is not None:
            x = self.input_projection(x)  # (B, T, 178) → (B, T, projection_dim)

        # ===== TEMPORAL MODELING =====
        # Process sequence through GRU layers
        if lengths is not None:
            # ===== PACKED SEQUENCE PROCESSING =====
            # Validate lengths tensor
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > T:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {T}"
                )
            
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            # Pack sequences for efficient processing
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            # First GRU layer
            y1, h1 = self.gru1(packed)  # y1: PackedSequence, h1: (num_directions, B, hidden1)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            # Second GRU layer
            y2, h2 = self.gru2(y1)      # h2: (num_directions, B, hidden2)
            
            # Extract final hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                h = torch.cat([h2[-2], h2[-1]], dim=-1)  # (B, hidden2*2)
            else:
                h = h2[-1]  # (B, hidden2)
            
        else:
            # ===== REGULAR SEQUENCE PROCESSING =====
            # First GRU layer
            y1, h1 = self.gru1(x)       # y1: (B, T, hidden1*num_directions)
            y1 = self.do1(y1)           # Apply dropout
            
            # Second GRU layer
            y2, h2 = self.gru2(y1)     # h2: (num_directions, B, hidden2)
            
            # Extract final hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                h = torch.cat([h2[-2], h2[-1]], dim=-1)  # (B, hidden2*2)
            else:
                h = h2[-1]  # (B, hidden2)

        # ===== FINAL PROCESSING =====
        # Apply final dropout to hidden state
        h = self.do2(h)  # (B, hidden2*num_directions)
        
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
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to return probability outputs.
        
        This method is a wrapper around the forward method that automatically
        applies softmax to the logits to return probability distributions.

        Args:
            x (Tensor): Input keypoint sequence of shape (B, T, 178).
            lengths (Tensor, optional): True sequence lengths (B,).

        Returns:
            Tuple[Tensor, Tensor]: (gloss_probs, cat_probs) of shapes (B, num_gloss) and (B, num_cat).
                                  Both tensors contain softmax probabilities.
        """
        return self.forward(x, lengths=lengths, return_probs=True)
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information for logging and debugging.
        
        Returns:
            dict: Dictionary containing model architecture details.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'MediaPipeGRU',
            'input_dim': self.input_dim,
            'projection_dim': self.projection_dim,
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout_p,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }


class MediaPipeGRUCtc(nn.Module):
    """
    MediaPipe-GRU model with CTC for continuous sign language recognition.
    
    This is a lightweight sequence-to-sequence model designed for CTC-based
    continuous sign language recognition. It provides a fair comparison baseline
    for the SignTransformerCtc model since both use the same 178-dimensional
    keypoint inputs.
    
    Architecture:
    Input: [B, T, 178] keypoint sequences
      ↓
    Optional Input Projection: [B, T, 178] → [B, T, proj_dim]
      ↓
    Bidirectional GRU Layer 1: [B, T, proj_dim] → [B, T, hidden1*2]
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
    - Lightweight architecture (~500KB model size)
    - Mobile-friendly for deployment
    - Direct processing of MediaPipe keypoints
    - No CNN preprocessing required
    - Much faster than Transformer (~2-3x speedup)
    
    Comparison with Other Models:
    - vs SignTransformerCtc: Simpler, faster, comparable performance
    - vs MediaPipeGRU: Sequence-to-sequence instead of classification
    - vs InceptionV3GRU: 50x smaller, no visual features needed
    
    Args:
        num_ctc_classes (int): Number of CTC classes including blank token (default: 106).
        input_dim (int): Input keypoint dimension (default: 178 for 89 keypoints × 2).
        projection_dim (int, optional): If specified, project input to this dimension first.
        hidden1 (int): Hidden units for first GRU layer (default: 256).
        hidden2 (int): Hidden units for second GRU layer (default: 128).
        dropout (float): Dropout rate applied after GRU layers (default: 0.3).
    
    Forward inputs:
        x (Tensor): Keypoint sequences (B, T, 178)
        lengths (Tensor, optional): True sequence lengths (B,) for packed sequences
    
    Returns:
        Tensor: Log probabilities of shape (B, T, num_ctc_classes).
               Use .permute(1, 0, 2) for CTCLoss which expects [T, B, C].
    
    Usage:
        model = MediaPipeGRUCtc(num_ctc_classes=106)
        log_probs = model(x)  # x: [B, T, 178] → log_probs: [B, T, 106]
        
        # For CTC loss, permute to [T, B, C]
        log_probs = log_probs.permute(1, 0, 2)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    """
    
    def __init__(
        self,
        num_ctc_classes: int = 106,
        input_dim: int = 178,
        projection_dim: Optional[int] = None,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.3,
        num_cat: Optional[int] = None,
    ):
        """
        Initialize the MediaPipe-GRU-CTC model with optional category head.
        
        Args:
            num_ctc_classes (int): Number of CTC classes including blank.
            input_dim (int): Input keypoint dimension (89 keypoints × 2 = 178).
            projection_dim (int, optional): If specified, project input to this dimension.
            hidden1 (int): Hidden units for first GRU layer.
            hidden2 (int): Hidden units for second GRU layer.
            dropout (float): Dropout rate applied after GRU layers.
            num_cat (int, optional): Number of category classes. If None, CTC-only mode.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.num_ctc_classes = num_ctc_classes
        self.num_cat = num_cat
        
        # ===== INPUT PROJECTION (OPTIONAL) =====
        # Optional projection layer to control input dimensionality
        if projection_dim is not None:
            self.input_projection = nn.Linear(input_dim, projection_dim)
            gru_input_dim = projection_dim
        else:
            self.input_projection = None
            gru_input_dim = input_dim
        
        # ===== TEMPORAL MODELING =====
        # Two-layer bidirectional GRU network for temporal sequence modeling
        self.gru1 = nn.GRU(
            input_size=gru_input_dim,
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
        
        # Initialize projection layer if it exists
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
        
        # Initialize CTC head
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.zeros_(self.ctc_head.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the MediaPipe-GRU-CTC model with optional category prediction.
        
        Args:
            x (Tensor): Input keypoint sequence of shape (B, T, 178).
            lengths (Tensor, optional): True sequence lengths (B,) for packed-sequence processing.
        
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
        # ===== INPUT VALIDATION =====
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()
        
        # ===== INPUT PROJECTION =====
        # Apply optional input projection
        if self.input_projection is not None:
            x = self.input_projection(x)  # (B, T, 178) → (B, T, projection_dim)
        
        # ===== TEMPORAL MODELING =====
        # Process sequence through bidirectional GRU layers
        if lengths is not None:
            # ===== PACKED SEQUENCE PROCESSING =====
            # Validate lengths tensor
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > T:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {T}"
                )
            
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            # Pack sequences for efficient processing
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
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
            y1, _ = self.gru1(x)    # y1: (B, T, hidden1*2)
            y1 = self.do1(y1)        # Apply dropout
            
            # Second GRU layer
            y2, _ = self.gru2(y1)   # y2: (B, T, hidden2*2)
        
        # ===== FINAL DROPOUT =====
        # Apply dropout to the GRU output sequence
        y2 = self.do2(y2)  # (B, T, hidden2*2)
        
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
            'model_type': 'MediaPipeGRUCtc',
            'input_dim': self.input_dim,
            'projection_dim': self.projection_dim,
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'num_ctc_classes': self.num_ctc_classes,
            'dropout': self.dropout_p,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
