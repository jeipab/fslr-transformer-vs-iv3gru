"""
MediaPipe-GRU model for sign language recognition from keypoint sequences.

Note: This model was used as a prototype/testing model for InceptionV3GRU development.
It is not part of the main model comparison and is kept for reference only.

Lightweight GRU-based architecture for processing MediaPipe keypoint sequences.
Suitable for real-time mobile deployment.
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dropout_packed(packed_seq, p: float, training: bool):
    """Apply dropout to a PackedSequence."""
    if p <= 0.0:
        return packed_seq
        
    data = F.dropout(packed_seq.data, p=p, training=training)
    return nn.utils.rnn.PackedSequence(
        data, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices
    )


class MediaPipeGRU(nn.Module):
    """
    MediaPipe-GRU model for sign language recognition.
    
    Lightweight model that processes MediaPipe keypoint sequences directly through GRU layers.
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
        """Initialize the MediaPipe-GRU model."""
        super().__init__()
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        if projection_dim is not None:
            self.input_projection = nn.Linear(input_dim, projection_dim)
            gru_input_dim = projection_dim
        else:
            self.input_projection = None
            gru_input_dim = input_dim
        
        self.gru1 = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        effective_hidden1 = hidden1 * self.num_directions
        
        self.gru2 = nn.GRU(
            input_size=effective_hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        effective_hidden2 = hidden2 * self.num_directions
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.gloss_head = nn.Linear(effective_hidden2, num_gloss)
        self.category_head = nn.Linear(effective_hidden2, num_cat)

        self._init_weights()
    
    def _init_weights(self):
        """Initialize GRU weights for stable training."""
        for gru in (self.gru1, self.gru2):
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
        
        for head in (self.gloss_head, self.category_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()
        
        if self.input_projection is not None:
            x = self.input_projection(x)

        if lengths is not None:
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > T:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {T}"
                )
            
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            y1, h1 = self.gru1(packed)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            y2, h2 = self.gru2(y1)
            
            if self.bidirectional:
                h = torch.cat([h2[-2], h2[-1]], dim=-1)
            else:
                h = h2[-1]
            
        else:
            y1, h1 = self.gru1(x)
            y1 = self.do1(y1)
            
            y2, h2 = self.gru2(y1)
            
            if self.bidirectional:
                h = torch.cat([h2[-2], h2[-1]], dim=-1)
            else:
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
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return probability outputs instead of logits."""
        return self.forward(x, lengths=lengths, return_probs=True)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
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
    """MediaPipe-GRU model with CTC for continuous sign language recognition."""
    
    def __init__(
        self,
        num_ctc_classes: int = 106,
        input_dim: int = 178,
        projection_dim: Optional[int] = None,
        hidden1: int = 512,
        hidden2: int = 512,
        dropout: float = 0.3,
        num_cat: Optional[int] = None,
    ):
        """Initialize the MediaPipe-GRU-CTC model."""
        super().__init__()
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.num_ctc_classes = num_ctc_classes
        self.num_cat = num_cat
        
        if projection_dim is not None:
            self.input_projection = nn.Linear(input_dim, projection_dim)
            gru_input_dim = projection_dim
        else:
            self.input_projection = None
            gru_input_dim = input_dim
        
        self.gru1 = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        effective_hidden1 = hidden1
        
        self.gru2 = nn.GRU(
            input_size=effective_hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        effective_hidden2 = hidden2
        
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
        
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight, gain=0.5)
            nn.init.zeros_(self.input_projection.bias)
        
        nn.init.xavier_uniform_(self.ctc_head.weight, gain=0.5)
        nn.init.zeros_(self.ctc_head.bias)
        
        if self.category_head is not None:
            nn.init.xavier_uniform_(self.category_head.weight, gain=0.5)
            nn.init.zeros_(self.category_head.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, T, features], got shape {x.shape}"
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.shape[-1]}"
            )
        
        B, T, _ = x.size()
        
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        if lengths is not None:
            if lengths.min() < 1:
                raise ValueError("All sequence lengths must be positive")
            if lengths.max() > T:
                raise ValueError(
                    f"Maximum length {lengths.max()} exceeds sequence length {T}"
                )
            
            lengths_cpu = lengths if lengths.device.type == 'cpu' else lengths.to("cpu")
            
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            y1, _ = self.gru1(packed)
            y1 = _dropout_packed(y1, self.do1.p, training=self.training)
            
            y2, _ = self.gru2(y1)
            
            y2, _ = nn.utils.rnn.pad_packed_sequence(y2, batch_first=True)
            
            y2 = self.do2(y2)
            
        else:
            y1, _ = self.gru1(x)
            y1 = self.do1(y1)
            y1 = self.ln1(y1)
            
            y2, _ = self.gru2(y1)
        
        if lengths is None:
            y2 = self.do2(y2)
            y2 = self.ln2(y2)
        
        ctc_logits = self.ctc_head(y2)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2)
        
        if self.category_head is not None:
            cat_logits = self.category_head(y2)
            
            return ctc_log_probs, cat_logits
        else:
            return ctc_log_probs
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
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
