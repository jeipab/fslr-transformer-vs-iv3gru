#!/usr/bin/env python3
"""
Sign Language Recognition Prediction Script

This script provides a command-line interface for making predictions using trained
Sign Language Recognition models. It supports both Transformer and IV3-GRU models
and can process either preprocessed NPZ files or raw video files.

COMMAND LINE USAGE:
    # List all available model checkpoints
    python predict.py --list-models

    # Predict from NPZ file using Transformer model
    python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input data.npz

    # Predict from video file using IV3-GRU model
    python predict.py --model iv3_gru --checkpoint iv3_gru/model.pt --input video.mp4

    # Save results to JSON file
    python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input data.npz --output results.json

    # Force CPU usage
    python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input data.npz --device cpu

REQUIRED ARGUMENTS:
    --model: Model type ('transformer' or 'iv3_gru')
    --checkpoint: Path to model checkpoint (.pt file)
    --input: Input file (NPZ or video file)

OPTIONAL ARGUMENTS:
    --device: Device to use ('cpu', 'cuda', or 'auto')
    --fps: Target FPS for video processing (default: 30)
    --image-size: Image size for video processing (default: 256)
    --output: Save results to JSON file
    --list-models: List available model checkpoints

MODEL COMPATIBILITY:
    Both models support both NPZ files and raw video files:
    - Transformer: NPZ with 'X' key (keypoints) OR video files (auto-extracts keypoints)
    - IV3-GRU: NPZ with 'X2048' key (IV3 features) OR video files (auto-extracts IV3 features)
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import cv2
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SignTransformer, InceptionV3GRU
from data import format_prediction_results, print_prediction_summary
try:
    from preprocessing import (
        create_models, close_models, extract_keypoints_from_frame, 
        interpolate_gaps, extract_iv3_features
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Preprocessing modules not available: {e}")
    print("Video processing will not be available. NPZ processing should still work.")
    PREPROCESSING_AVAILABLE = False


class ModelPredictor:
    """
    Unified predictor for both Transformer and IV3-GRU models.
    
    This class handles loading trained models and making predictions on either
    NPZ files (preprocessed data) or video files (raw videos).
    
    Example:
        predictor = ModelPredictor('transformer', 'path/to/checkpoint.pt')
        results = predictor.predict_from_npz('data.npz')
        predictor.cleanup()
    """
    
    def __init__(self, model_type, checkpoint_path, device=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_type (str): Model type - 'transformer' or 'iv3_gru'
            checkpoint_path (str): Path to the model checkpoint (.pt file)
            device (torch.device, optional): Device to use. Auto-detected if None.
            
        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If checkpoint file doesn't exist
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and checkpoint
        self.model = self._load_model()
        self._load_checkpoint()
        
        # Initialize preprocessing models if needed
        self.mp_models = None
        
    def _load_model(self):
        """
        Load the appropriate model architecture based on model_type.
        
        Returns:
            torch.nn.Module: Initialized model (not yet loaded with weights)
            
        Raises:
            ValueError: If model_type is not supported
        """
        if self.model_type == 'transformer':
            # Default parameters from training log (105 gloss, 10 categories)
            model = SignTransformer(
                input_dim=156,
                emb_dim=256,
                n_heads=8,
                n_layers=4,
                num_gloss=105,
                num_cat=10,
                dropout=0.1,
                max_len=300,
                pooling_method='mean'
            )
        elif self.model_type == 'iv3_gru':
            # Default parameters for IV3-GRU
            model = InceptionV3GRU(
                num_gloss=105,
                num_cat=10,
                hidden1=16,
                hidden2=12,
                dropout=0.3,
                pretrained_backbone=True,
                freeze_backbone=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def _load_checkpoint(self):
        """
        Load the model checkpoint and apply weights to the model.
        
        Handles different checkpoint formats:
        - Training checkpoints with 'model' key
        - Standard checkpoints with 'model_state_dict' or 'state_dict' keys
        - Direct state dict checkpoints
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint format is incompatible
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            # Training checkpoint with model state dict
            self.model.load_state_dict(checkpoint['model'])
        else:
            # Assume the checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Loaded {self.model_type} model from {self.checkpoint_path}")
    
    def predict_from_npz(self, npz_path):
        """
        Make prediction from preprocessed NPZ file.
        
        Args:
            npz_path (str): Path to NPZ file containing preprocessed data
            
        Returns:
            dict: Prediction results containing:
                - gloss_prediction: Predicted gloss class ID
                - category_prediction: Predicted category class ID
                - gloss_probability: Confidence for gloss prediction
                - category_probability: Confidence for category prediction
                - gloss_top5: Top 5 gloss predictions with probabilities
                - category_top3: Top 3 category predictions with probabilities
                
        Raises:
            ValueError: If NPZ file doesn't contain required keys
            FileNotFoundError: If NPZ file doesn't exist
        """
        # Load NPZ data
        data = np.load(npz_path)
        
        if self.model_type == 'transformer':
            if 'X' not in data:
                raise ValueError("NPZ file must contain 'X' key for transformer model")
            
            X = torch.from_numpy(data['X']).float().unsqueeze(0)
            
            if 'mask' in data:
                mask_data = data['mask']
                seq_mask = torch.from_numpy(mask_data.any(axis=1)).bool().unsqueeze(0)
            else:
                seq_mask = None
            
            if X.shape[1] > 300:
                print(f"Warning: Sequence length {X.shape[1]} exceeds max_len=300, truncating...")
                X = X[:, :300, :]
                if seq_mask is not None:
                    seq_mask = seq_mask[:, :300]
            
            X = X.to(self.device)
            if seq_mask is not None:
                seq_mask = seq_mask.to(self.device)
            
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, seq_mask)
                
        elif self.model_type == 'iv3_gru':
            if 'X2048' not in data:
                raise ValueError("NPZ file must contain 'X2048' key for IV3-GRU model")
            
            X2048 = torch.from_numpy(data['X2048']).float().unsqueeze(0)
            lengths = torch.tensor([X2048.shape[1]], dtype=torch.long)
            
            X2048 = X2048.to(self.device)
            lengths = lengths.to(self.device)
            
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X2048, lengths, features_already=True)
        
        gloss_pred = torch.argmax(gloss_logits, dim=-1).item()
        cat_pred = torch.argmax(cat_logits, dim=-1).item()
        
        gloss_probs = torch.softmax(gloss_logits, dim=-1).squeeze(0)
        cat_probs = torch.softmax(cat_logits, dim=-1).squeeze(0)
        
        return {
            'gloss_prediction': int(gloss_pred),
            'category_prediction': int(cat_pred),
            'gloss_probability': float(gloss_probs[gloss_pred].item()),
            'category_probability': float(cat_probs[cat_pred].item()),
            'gloss_top5': [(int(i), float(gloss_probs[i].item())) for i in torch.topk(gloss_probs, 5).indices],
            'category_top3': [(int(i), float(cat_probs[i].item())) for i in torch.topk(cat_probs, 3).indices]
        }
    
    def predict_from_video(self, video_path, target_fps=30, image_size=256):
        """
        Make prediction from raw video file.
        
        This method extracts features from the video and makes predictions.
        Requires preprocessing modules (mediapipe, opencv-python) to be installed.
        
        Args:
            video_path (str): Path to video file
            target_fps (int): Target FPS for frame extraction (default: 30)
            image_size (int): Size to resize frames for processing (default: 256)
            
        Returns:
            dict: Prediction results containing:
                - gloss_prediction: Predicted gloss class ID
                - category_prediction: Predicted category class ID
                - gloss_probability: Confidence for gloss prediction
                - category_probability: Confidence for category prediction
                - gloss_top5: Top 5 gloss predictions with probabilities
                - category_top3: Top 3 category predictions with probabilities
                - frames_extracted: Number of frames processed
                
        Raises:
            ImportError: If preprocessing modules are not available
            ValueError: If video processing fails
            FileNotFoundError: If video file doesn't exist
        """
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Video processing requires preprocessing modules. Please install mediapipe and opencv-python.")
        
        if self.mp_models is None:
            self.mp_models = create_models()
        
        frames, keypoints, iv3_features = self._extract_video_features(
            video_path, target_fps, image_size
        )
        
        if self.model_type == 'transformer':
            if keypoints is None or len(keypoints) == 0:
                raise ValueError("Could not extract keypoints from video")
            
            X = np.stack(keypoints, axis=0)
            X = torch.from_numpy(X).float().unsqueeze(0)
            
            if X.shape[1] > 300:
                print(f"Warning: Sequence length {X.shape[1]} exceeds max_len=300, truncating...")
                X = X[:, :300, :]
            
            X = X.to(self.device)
            
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X)
                
        elif self.model_type == 'iv3_gru':
            if iv3_features is None or len(iv3_features) == 0:
                raise ValueError("Could not extract IV3 features from video")
            
            X2048 = np.stack(iv3_features, axis=0)
            X2048 = torch.from_numpy(X2048).float().unsqueeze(0)
            lengths = torch.tensor([X2048.shape[1]], dtype=torch.long)
            
            X2048 = X2048.to(self.device)
            lengths = lengths.to(self.device)
            
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X2048, lengths, features_already=True)
        
        gloss_pred = torch.argmax(gloss_logits, dim=-1).item()
        cat_pred = torch.argmax(cat_logits, dim=-1).item()
        
        gloss_probs = torch.softmax(gloss_logits, dim=-1).squeeze(0)
        cat_probs = torch.softmax(cat_logits, dim=-1).squeeze(0)
        
        return {
            'gloss_prediction': int(gloss_pred),
            'category_prediction': int(cat_pred),
            'gloss_probability': float(gloss_probs[gloss_pred].item()),
            'category_probability': float(cat_probs[cat_pred].item()),
            'gloss_top5': [(int(i), float(gloss_probs[i].item())) for i in torch.topk(gloss_probs, 5).indices],
            'category_top3': [(int(i), float(cat_probs[i].item())) for i in torch.topk(cat_probs, 3).indices],
            'frames_extracted': int(len(frames))
        }
    
    def _extract_video_features(self, video_path, target_fps, image_size):
        """Extract keypoints and IV3 features from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps < 1:
            src_fps = 30.0
        
        step_s = 1.0 / target_fps
        next_t = 0.0
        
        frames = []
        keypoints = []
        iv3_features = []
        
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ms < next_t * 1000.0:
                    continue
                
                # Resize frame
                frame_bgr_resized = cv2.resize(frame_bgr, (image_size, image_size))
                frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
                
                frames.append(frame_rgb)
                
                # Extract keypoints
                try:
                    vec156, mask78 = extract_keypoints_from_frame(frame_rgb, self.mp_models)
                    keypoints.append(vec156)
                except Exception as e:
                    print(f"Warning: Could not extract keypoints from frame: {e}")
                    keypoints.append(np.zeros(156, dtype=np.float32))
                
                # Extract IV3 features
                try:
                    iv3_feat = extract_iv3_features(frame_bgr_resized, device=self.device)
                    iv3_features.append(iv3_feat)
                except Exception as e:
                    print(f"Warning: Could not extract IV3 features from frame: {e}")
                    iv3_features.append(np.zeros(2048, dtype=np.float32))
                
                next_t += step_s
                
        finally:
            cap.release()
        
        return frames, keypoints, iv3_features
    
    def cleanup(self):
        """
        Clean up resources and close any open models.
        
        Call this method when done with the predictor to free up resources,
        especially MediaPipe models used for video processing.
        """
        if self.mp_models is not None:
            close_models(self.mp_models)
            self.mp_models = None


def list_available_models():
    """
    List all available model checkpoints in the trained_models directory.
    
    Scans the trained_models directory for .pt files and displays them
    organized by model type.
    """
    trained_models_dir = Path(__file__).parent
    
    print("Available model checkpoints:")
    print("=" * 50)
    
    for model_dir in trained_models_dir.iterdir():
        if model_dir.is_dir() and model_dir.name != '__pycache__':
            print(f"\n{model_dir.name.upper()} Models:")
            for checkpoint_file in model_dir.rglob("*.pt"):
                relative_path = checkpoint_file.relative_to(trained_models_dir)
                print(f"  - {relative_path}")


def main():
    """
    Main function for command-line interface.
    
    Parses command-line arguments and runs the prediction pipeline.
    Returns exit code 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Sign Language Recognition Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python predict.py --list-models
  
  # Predict from NPZ file with Transformer
  python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input data.npz
  
  # Predict from video file with IV3-GRU
  python predict.py --model iv3_gru --checkpoint iv3_gru/model.pt --input video.mp4
  
  # Save results to JSON file
  python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input data.npz --output results.json
        """
    )
    parser.add_argument('--model', choices=['transformer', 'iv3_gru'], 
                       help='Model type to use')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--input', type=str, 
                       help='Input file (NPZ or video)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for video processing')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size for video processing')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if not args.model or not args.checkpoint or not args.input:
        parser.error("--model, --checkpoint, and --input are required")
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize predictor
    try:
        predictor = ModelPredictor(args.model, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Make prediction
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input file not found: {args.input}")
            return 1
        
        if input_path.suffix.lower() == '.npz':
            print(f"Predicting from NPZ file: {args.input}")
            results = predictor.predict_from_npz(args.input)
        else:
            print(f"Predicting from video file: {args.input}")
            results = predictor.predict_from_video(args.input, args.fps, args.image_size)
        
        # Print results with human-readable labels
        print_prediction_summary(results)
        
        # Save results if requested
        if args.output:
            # Save both raw and formatted results
            formatted_results = format_prediction_results(results)
            
            with open(args.output, 'w') as f:
                json.dump(formatted_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1
    finally:
        predictor.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
