"""
CTC Prediction Module for Continuous Sign Language Recognition

This module provides prediction functions for CTC-based continuous sign language
recognition models. It implements sliding window inference for processing long
sequences and various decoding strategies.

Key Components:
- Greedy CTC decoding
- Beam search CTC decoding
- Sliding window inference for continuous sequences
- Window stitching and overlap handling

Usage:
    from evaluation.prediction.predict_ctc import predict_continuous, CTCPredictor
    
    # Single sequence prediction
    predictor = CTCPredictor(model_path, blank_id=105)
    glosses = predictor.predict_from_npz('sequence.npz')
    
    # Continuous prediction with sliding window
    glosses = predictor.predict_continuous(keypoint_sequence, window_size=60, stride=15)
"""

# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Add project root to path for local imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from models import SignTransformerCtc, MediaPipeGRUCtc
from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder
from streamlit_app.core.config import CTC_CONFIG


class CTCPredictor:
    """
    Unified predictor for CTC-based sign language recognition models.
    
    This class handles the complete CTC prediction pipeline:
    1. Loads trained CTC model architecture and weights
    2. Processes input data (NPZ files or raw keypoint sequences)
    3. Applies CTC decoding (greedy or beam search)
    4. Supports sliding window inference for long sequences
    5. Manages resources and cleanup
    
    Example:
        predictor = CTCPredictor('transformer_ctc', 'path/to/checkpoint.pt')
        results = predictor.predict_from_npz('data.npz')
        predictor.cleanup()
    """
    
    def __init__(self, model_type, checkpoint_path, blank_id=None, device=None):
        """
        Initialize the CTC predictor with a trained model.
        
        Args:
            model_type (str): Model type - 'transformer_ctc' or 'mediapipe_gru_ctc'
            checkpoint_path (str): Path to the model checkpoint (.pt file)
            blank_id (int, optional): Blank token ID (default: from CTC_CONFIG)
            device (torch.device, optional): Device to use. Auto-detected if None.
            
        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If checkpoint file doesn't exist
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.blank_id = blank_id if blank_id is not None else CTC_CONFIG['blank_token_id']
        
        # Load model architecture and weights
        self.model, self.input_dim = self._load_model()
        self._load_checkpoint()
        
        print(f"✓ CTC Predictor initialized")
        print(f"  Model: {self.model_type}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Blank ID: {self.blank_id}")
        print(f"  Device: {self.device}")
    
    def _load_model(self):
        """
        Load the appropriate CTC model architecture.
        
        Returns:
            tuple: (model, input_dim)
        """
        if self.model_type == 'transformer_ctc':
            # Auto-detect input dimensions from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Extract input_dim from embedding layer
                if 'embedding.weight' in state_dict:
                    input_dim = state_dict['embedding.weight'].shape[1]
                else:
                    input_dim = 156  # Default
            except Exception:
                input_dim = 156  # Default
            
            # Create model
            model = SignTransformerCtc(
                input_dim=input_dim,
                num_ctc_classes=CTC_CONFIG['num_ctc_classes'],
            )
            
        elif self.model_type == 'mediapipe_gru_ctc':
            # MediaPipeGRUCtc always uses 156-dimensional keypoints
            input_dim = 156
            
            # Auto-detect hidden sizes from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Extract GRU hidden sizes
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3 // 2  # // 2 for bidirectional
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3 // 2
                else:
                    gru1_hidden = 256  # Default
                    gru2_hidden = 128  # Default
            except Exception:
                gru1_hidden = 256
                gru2_hidden = 128
            
            model = MediaPipeGRUCtc(
                input_dim=input_dim,
                num_ctc_classes=CTC_CONFIG['num_ctc_classes'],
                hidden1=gru1_hidden,
                hidden2=gru2_hidden,
            )
        else:
            raise ValueError(f"Unknown CTC model type: {self.model_type}")
        
        return model.to(self.device), input_dim
    
    def _load_checkpoint(self):
        """Load model checkpoint and apply weights."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights and set to eval mode
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def predict_from_npz(self, npz_path, decode_method='greedy', beam_width=10):
        """
        Make CTC prediction from preprocessed NPZ file.
        
        Args:
            npz_path (str): Path to NPZ file containing preprocessed data
            decode_method (str): Decoding method - 'greedy' or 'beam_search'
            beam_width (int): Beam width for beam search decoding
            
        Returns:
            dict: Prediction results containing:
                - predicted_glosses: List of predicted gloss IDs
                - decoded_sequence: Human-readable gloss sequence (if mapping available)
                - confidence: Average confidence score
                - decode_method: Method used for decoding
        """
        # Load NPZ data
        data = np.load(npz_path)
        
        # Extract appropriate features based on input dimension
        if self.input_dim == 2048:
            if 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X2048' key")
            X = torch.from_numpy(data['X2048']).float().unsqueeze(0)
        elif self.input_dim == 156:
            if 'X' not in data:
                raise ValueError(f"NPZ file missing 'X' key")
            X = torch.from_numpy(data['X']).float().unsqueeze(0)
        elif self.input_dim == 2204:
            if 'X' not in data or 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X' or 'X2048' key")
            X_kp = torch.from_numpy(data['X']).float()
            X_feat = torch.from_numpy(data['X2048']).float()
            X = torch.cat([X_kp, X_feat], dim=1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input dimension {self.input_dim}")
        
        # Handle sequence length truncation
        if X.shape[1] > 300:
            X = X[:, :300, :]
        
        # Move to device
        X = X.to(self.device)
        input_length = torch.tensor([X.shape[1]], dtype=torch.long).to(self.device)
        
        # Model inference
        with torch.no_grad():
            log_probs = self.model(X)  # [B, T, C]
        
        # Decode using specified method
        if decode_method == 'greedy':
            decoded_sequences = greedy_ctc_decoder(log_probs, self.blank_id, input_length)
            predicted_glosses = decoded_sequences[0]  # First (and only) sequence
            confidence = self._calculate_confidence(log_probs[0], predicted_glosses)
        elif decode_method == 'beam_search':
            results = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_length)
            predicted_glosses, log_prob = results[0]
            confidence = np.exp(log_prob / max(len(predicted_glosses), 1))
        else:
            raise ValueError(f"Unknown decode_method: {decode_method}")
        
        return {
            'predicted_glosses': predicted_glosses,
            'num_glosses': len(predicted_glosses),
            'confidence': float(confidence),
            'decode_method': decode_method,
            'input_frames': int(X.shape[1]),
        }
    
    def predict_continuous(
        self,
        keypoint_sequence,
        window_size=60,
        stride=15,
        decode_method='greedy',
        beam_width=10
    ):
        """
        Predict gloss sequence from continuous keypoint stream using sliding window.
        
        This function implements sliding window inference for processing long
        continuous sign language sequences. It:
        1. Splits the sequence into overlapping windows
        2. Processes each window independently
        3. Stitches predictions together
        
        Args:
            keypoint_sequence (np.ndarray): Keypoint sequence [T, 156]
            window_size (int): Number of frames per window
            stride (int): Stride between windows
            decode_method (str): 'greedy' or 'beam_search'
            beam_width (int): Beam width for beam search
            
        Returns:
            dict: Prediction results containing:
                - predicted_glosses: List of predicted gloss IDs
                - window_predictions: List of predictions per window
                - num_windows: Number of windows processed
                - confidence: Average confidence across all windows
        """
        # Validate input
        if not isinstance(keypoint_sequence, np.ndarray):
            keypoint_sequence = np.array(keypoint_sequence)
        
        if keypoint_sequence.ndim != 2 or keypoint_sequence.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected keypoint sequence of shape [T, {self.input_dim}], "
                f"got {keypoint_sequence.shape}"
            )
        
        seq_length = len(keypoint_sequence)
        
        # Handle short sequences (shorter than window size)
        if seq_length <= window_size:
            # Process entire sequence as one window
            X = torch.from_numpy(keypoint_sequence).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                log_probs = self.model(X)
            
            if decode_method == 'greedy':
                decoded = greedy_ctc_decoder(log_probs, self.blank_id)[0]
            else:
                decoded, _ = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width)[0]
            
            return {
                'predicted_glosses': decoded,
                'window_predictions': [decoded],
                'num_windows': 1,
                'confidence': 1.0,
            }
        
        # ================================================================
        # SLIDING WINDOW INFERENCE
        # ================================================================
        
        window_predictions = []
        window_confidences = []
        
        # Process each window
        for start_idx in range(0, seq_length - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            # Extract window
            window = keypoint_sequence[start_idx:end_idx]
            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                log_probs = self.model(window_tensor)  # [1, T, C]
            
            # Decode window
            if decode_method == 'greedy':
                decoded = greedy_ctc_decoder(log_probs, self.blank_id)[0]
                confidence = self._calculate_confidence(log_probs[0], decoded)
            else:
                decoded, log_prob = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width)[0]
                confidence = np.exp(log_prob / max(len(decoded), 1))
            
            window_predictions.append({
                'start_frame': start_idx,
                'end_frame': end_idx,
                'glosses': decoded,
                'confidence': confidence
            })
            window_confidences.append(confidence)
        
        # ================================================================
        # STITCH WINDOW PREDICTIONS
        # ================================================================
        
        # Simple stitching strategy: concatenate all predictions and remove duplicates
        # More advanced strategies could use voting or weighted averaging
        all_glosses = []
        for pred in window_predictions:
            all_glosses.extend(pred['glosses'])
        
        # Remove consecutive duplicates from stitched sequence
        stitched_glosses = []
        for gloss in all_glosses:
            if len(stitched_glosses) == 0 or gloss != stitched_glosses[-1]:
                stitched_glosses.append(gloss)
        
        avg_confidence = np.mean(window_confidences) if window_confidences else 0.0
        
        return {
            'predicted_glosses': stitched_glosses,
            'window_predictions': window_predictions,
            'num_windows': len(window_predictions),
            'confidence': float(avg_confidence),
            'window_size': window_size,
            'stride': stride,
        }
    
    def predict_from_video(self, video_path, window_size=60, stride=15, decode_method='greedy'):
        """
        Make CTC prediction from raw video file.
        
        This method:
        1. Extracts keypoints from video frames
        2. Applies sliding window inference
        3. Returns predicted gloss sequence
        
        Note: Requires preprocessing modules (MediaPipe, OpenCV)
        
        Args:
            video_path (str): Path to video file
            window_size (int): Frames per window
            stride (int): Stride between windows
            decode_method (str): 'greedy' or 'beam_search'
            
        Returns:
            dict: Prediction results
        """
        # This would require preprocessing integration
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Video processing for CTC not yet implemented. "
            "Please preprocess video to NPZ first and use predict_from_npz()"
        )
    
    def _calculate_confidence(self, log_probs, predicted_glosses):
        """
        Calculate average confidence for predicted glosses.
        
        Args:
            log_probs: Log probabilities tensor [T, C]
            predicted_glosses: List of predicted gloss IDs
            
        Returns:
            float: Average confidence score
        """
        if len(predicted_glosses) == 0:
            return 0.0
        
        # Convert log probs to probabilities
        probs = torch.exp(log_probs)
        
        # Get max probability at each timestep
        max_probs = probs.max(dim=1)[0]
        
        # Return mean probability as confidence
        return max_probs.mean().item()
    
    def cleanup(self):
        """Clean up resources."""
        pass


def predict_continuous(
    model,
    keypoint_sequence,
    blank_id,
    window_size=60,
    stride=15,
    decode_method='greedy',
    beam_width=10,
    device=None
):
    """
    Standalone function for continuous CTC prediction with sliding windows.
    
    This is a convenience function that doesn't require instantiating CTCPredictor.
    It's useful for quick predictions or when you already have a loaded model.
    
    Args:
        model: Trained CTC model (SignTransformerCtc or MediaPipeGRUCtc)
        keypoint_sequence (np.ndarray or torch.Tensor): Keypoint sequence [T, 156]
        blank_id (int): Blank token ID for CTC decoding
        window_size (int): Number of frames per window
        stride (int): Stride between consecutive windows
        decode_method (str): 'greedy' or 'beam_search'
        beam_width (int): Beam width for beam search decoding
        device (torch.device, optional): Device to use
        
    Returns:
        List[int]: Predicted sequence of gloss IDs
        
    Example:
        >>> from models import SignTransformerCtc
        >>> model = SignTransformerCtc()
        >>> model.load_state_dict(torch.load('model.pt'))
        >>> keypoints = np.random.randn(200, 156)
        >>> glosses = predict_continuous(model, keypoints, blank_id=105)
        >>> print(glosses)
        [4, 17, 23, 56, 89]
    """
    model.eval()
    
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    # Convert to numpy if needed
    if isinstance(keypoint_sequence, torch.Tensor):
        keypoint_sequence = keypoint_sequence.cpu().numpy()
    
    seq_length = len(keypoint_sequence)
    all_log_probs = []
    
    # ================================================================
    # SLIDING WINDOW PROCESSING
    # ================================================================
    
    # Process sequence with sliding windows
    for start_idx in range(0, max(1, seq_length - window_size + 1), stride):
        end_idx = min(start_idx + window_size, seq_length)
        
        # Extract window
        window = keypoint_sequence[start_idx:end_idx]
        window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
        
        # Model inference
        with torch.no_grad():
            log_probs = model(window_tensor)  # [1, T, C]
            all_log_probs.append(log_probs.squeeze(0))  # [T, C]
    
    if not all_log_probs:
        return []
    
    # ================================================================
    # CONCATENATE AND DECODE
    # ================================================================
    
    # Concatenate all window predictions
    # This is a simple approach - more sophisticated methods could use voting
    full_log_probs = torch.cat(all_log_probs, dim=0).unsqueeze(0)  # [1, T_full, C]
    
    # Decode using specified method
    if decode_method == 'greedy':
        decoded_sequences = greedy_ctc_decoder(full_log_probs, blank_id)
        return decoded_sequences[0]
    elif decode_method == 'beam_search':
        results = beam_search_ctc_decoder(full_log_probs, blank_id, beam_width)
        return results[0][0]  # Return sequence only, not score
    else:
        raise ValueError(f"Unknown decode_method: {decode_method}")


def predict_from_file(
    model_path,
    input_path,
    model_type='transformer_ctc',
    blank_id=None,
    window_size=60,
    stride=15,
    decode_method='greedy',
    beam_width=10
):
    """
    High-level prediction function for CTC models.
    
    This is the simplest way to make predictions - just provide paths
    and get results.
    
    Args:
        model_path (str): Path to model checkpoint
        input_path (str): Path to input NPZ file
        model_type (str): 'transformer_ctc' or 'mediapipe_gru_ctc'
        blank_id (int, optional): Blank token ID
        window_size (int): Frames per window for sliding window
        stride (int): Stride between windows
        decode_method (str): 'greedy' or 'beam_search'
        beam_width (int): Beam width for beam search
        
    Returns:
        dict: Prediction results
        
    Example:
        >>> results = predict_from_file(
        ...     'trained_models/transformer_ctc/best.pt',
        ...     'data/test_clip.npz',
        ...     model_type='transformer_ctc'
        ... )
        >>> print(f"Predicted: {results['predicted_glosses']}")
    """
    # Initialize predictor
    predictor = CTCPredictor(model_type, model_path, blank_id)
    
    # Make prediction
    results = predictor.predict_from_npz(input_path, decode_method, beam_width)
    
    # Cleanup
    predictor.cleanup()
    
    return results


if __name__ == "__main__":
    """
    Command-line interface for CTC prediction.
    
    Usage:
        python predict_ctc.py --model transformer_ctc \\
            --checkpoint path/to/model.pt \\
            --input data.npz \\
            --decode-method greedy
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="CTC Sign Language Recognition Prediction")
    parser.add_argument('--model', choices=['transformer_ctc', 'mediapipe_gru_ctc'],
                       required=True, help='CTC model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input NPZ file with keypoint data')
    parser.add_argument('--blank-id', type=int, default=None,
                       help='Blank token ID (default: from config)')
    parser.add_argument('--window-size', type=int, default=60,
                       help='Window size for sliding window inference')
    parser.add_argument('--stride', type=int, default=15,
                       help='Stride between windows')
    parser.add_argument('--decode-method', choices=['greedy', 'beam_search'], default='greedy',
                       help='CTC decoding method')
    parser.add_argument('--beam-width', type=int, default=10,
                       help='Beam width for beam search decoding')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}\n")
    
    # Make prediction
    try:
        results = predict_from_file(
            model_path=args.checkpoint,
            input_path=args.input,
            model_type=args.model,
            blank_id=args.blank_id,
            window_size=args.window_size,
            stride=args.stride,
            decode_method=args.decode_method,
            beam_width=args.beam_width
        )
        
        # Display results
        print("="*60)
        print("CTC PREDICTION RESULTS")
        print("="*60)
        print(f"Input file: {args.input}")
        print(f"Input frames: {results['input_frames']}")
        print(f"Decode method: {results['decode_method']}")
        print(f"Predicted glosses: {results['predicted_glosses']}")
        print(f"Number of glosses: {results['num_glosses']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

