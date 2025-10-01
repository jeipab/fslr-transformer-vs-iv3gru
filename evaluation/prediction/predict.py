"""
Sign Language Recognition Prediction Script

This script provides a command-line interface for making predictions using trained
Sign Language Recognition models. It supports both Transformer and IV3-GRU models
and can process either preprocessed NPZ files or raw video files.

Usage Examples:
    python predict.py --list-models
    python predict.py --model transformer --checkpoint path/to/model.pt --input data.npz
    python predict.py --model iv3_gru --checkpoint path/to/model.pt --input video.mp4
"""

# Standard library imports
import argparse, json, os, sys
from pathlib import Path

# Third-party imports
import cv2, numpy as np, torch

# Add project root to path for local imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from models import SignTransformer, InceptionV3GRU
from data import format_prediction_results, print_prediction_summary

# Optional preprocessing imports for video processing
try:
    from preprocessing import (
        create_models, close_models, extract_keypoints_from_frame, 
        extract_iv3_features
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Preprocessing modules not available: {e}")
    print("Video processing will not be available. NPZ processing should still work.")
    PREPROCESSING_AVAILABLE = False


class ModelPredictor:
    """
    Unified predictor for both Transformer and IV3-GRU models.
    
    This class handles the complete prediction pipeline:
    1. Loads trained model architecture and weights
    2. Processes input data (NPZ files or raw videos)
    3. Makes predictions and returns formatted results
    4. Manages resources and cleanup
    
    Example:
        predictor = ModelPredictor('transformer', 'path/to/checkpoint.pt')
        results = predictor.predict_from_npz('data.npz')
        predictor.cleanup()
    """
    
    def __init__(self, model_type, checkpoint_path, device=None):
        """
        Initialize the predictor with a trained model.
        
        Steps performed:
        1. Validate model type and set device
        2. Load model architecture based on type
        3. Load trained weights from checkpoint
        4. Set model to evaluation mode
        
        Args:
            model_type (str): Model type - 'transformer' or 'iv3_gru'
            checkpoint_path (str): Path to the model checkpoint (.pt file)
            device (torch.device, optional): Device to use. Auto-detected if None.
            
        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If checkpoint file doesn't exist
        """
        # Step 1: Validate inputs and set device
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Step 2: Load model architecture and detect input dimensions
        self.model, self.input_dim = self._load_model()
        
        # Step 3: Load trained weights
        self._load_checkpoint()
        
        # Step 4: Initialize preprocessing models for video processing
        self.mp_models = None
        
    def _load_model(self):
        """
        Load the appropriate model architecture based on model_type.
        
        Process:
        1. Determine model parameters from checkpoint (if possible)
        2. Create model instance with detected/default parameters
        3. Move model to specified device
        
        Returns:
            tuple: (model, input_dim) where:
                - model: Initialized model (not yet loaded with weights)
                - input_dim: Detected input dimension for the model
                
        Raises:
            ValueError: If model_type is not supported
        """
        if self.model_type == 'transformer':
            # Step 1: Auto-detect input dimensions from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Extract input_dim from embedding layer shape
                if 'embedding.weight' in state_dict:
                    embedding_shape = state_dict['embedding.weight'].shape
                    input_dim = embedding_shape[1]  # embedding.weight is [emb_dim, input_dim]
                else:
                    input_dim = 156  # Default fallback
            except Exception as e:
                input_dim = 156  # Default fallback
            
            # Step 2: Create Transformer model with detected/default parameters
            model = SignTransformer(
                input_dim=input_dim,      # Input feature dimension (156 for keypoints, 2048 for IV3 features)
                emb_dim=256,              # Embedding dimension
                n_heads=8,                # Number of attention heads
                n_layers=4,               # Number of transformer layers
                num_gloss=105,            # Number of gloss classes
                num_cat=10,               # Number of category classes
                dropout=0.1,              # Dropout rate
                max_len=300,              # Maximum sequence length
                pooling_method='mean'     # Pooling method for sequence aggregation
            )
            
        elif self.model_type == 'iv3_gru':
            # IV3-GRU always uses 2048-dimensional features
            input_dim = 2048
            
            # Step 1: Auto-detect GRU hidden sizes from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Extract GRU hidden sizes from weight shapes
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    # GRU weight_hh has shape [3*hidden, hidden] for each layer
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3
                else:
                    gru1_hidden = 16  # Default fallback
                    gru2_hidden = 12  # Default fallback
            except Exception as e:
                gru1_hidden = 16  # Default fallback
                gru2_hidden = 12  # Default fallback
            
            # Step 2: Create IV3-GRU model with detected/default parameters
            model = InceptionV3GRU(
                num_gloss=105,            # Number of gloss classes
                num_cat=10,               # Number of category classes
                hidden1=gru1_hidden,     # First GRU hidden size
                hidden2=gru2_hidden,      # Second GRU hidden size
                dropout=0.3,              # Dropout rate
                pretrained_backbone=True, # Use pretrained InceptionV3
                freeze_backbone=True      # Freeze InceptionV3 weights
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Step 3: Move model to specified device
        return model.to(self.device), input_dim
    
    def _load_checkpoint(self):
        """
        Load the model checkpoint and apply weights to the model.
        
        Process:
        1. Verify checkpoint file exists
        2. Load checkpoint data from file
        3. Extract state dict from various checkpoint formats
        4. Apply weights to model
        5. Set model to evaluation mode
        
        Handles different checkpoint formats:
        - Training checkpoints with 'model' key
        - Standard checkpoints with 'model_state_dict' or 'state_dict' keys
        - Direct state dict checkpoints
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint format is incompatible
        """
        # Step 1: Verify checkpoint file exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Step 2: Load checkpoint data from file
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Step 3: Extract state dict from various checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Standard PyTorch checkpoint format
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # Alternative checkpoint format
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            # Training checkpoint format
            state_dict = checkpoint['model']
        else:
            # Direct state dict checkpoint
            state_dict = checkpoint
        
        # Step 4: Apply weights to model
        self.model.load_state_dict(state_dict)
        
        # Step 5: Set model to evaluation mode (disable dropout, batch norm updates)
        self.model.eval()
    
    def predict_from_npz(self, npz_path):
        """
        Make prediction from preprocessed NPZ file.
        
        Process:
        1. Load NPZ data from file
        2. Extract appropriate features based on model type
        3. Prepare input tensors and masks
        4. Run model inference
        5. Process outputs and return formatted results
        
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
        # Step 1: Load NPZ data from file
        data = np.load(npz_path)
        
        if self.model_type == 'transformer':
            # Step 2: Extract features for Transformer model based on detected input dimension
            if self.input_dim == 2048:
                # Use IV3 features (2048-dimensional)
                if 'X2048' not in data:
                    raise ValueError(f"NPZ file missing 'X2048' key for 2048-D transformer model")
                X = torch.from_numpy(data['X2048']).float().unsqueeze(0)
            elif self.input_dim == 156:
                # Use keypoint features (156-dimensional)
                if 'X' not in data:
                    raise ValueError(f"NPZ file missing 'X' key for 156-D transformer model")
                X = torch.from_numpy(data['X']).float().unsqueeze(0)
            elif self.input_dim == 2204:
                # Use combined features (156 keypoints + 2048 features = 2204-dimensional)
                if 'X' not in data or 'X2048' not in data:
                    raise ValueError(f"NPZ file missing 'X' or 'X2048' key for combined transformer model")
                
                # Load both keypoints and features
                X_keypoints = torch.from_numpy(data['X']).float()  # [T, 156]
                X_features = torch.from_numpy(data['X2048']).float()  # [T, 2048]
                
                # Ensure both have the same sequence length
                if X_keypoints.shape[0] != X_features.shape[0]:
                    raise ValueError(f"Sequence length mismatch: keypoints {X_keypoints.shape[0]} vs features {X_features.shape[0]}")
                
                # Concatenate along feature dimension: [T, 156] + [T, 2048] = [T, 2204]
                X = torch.cat([X_keypoints, X_features], dim=1).unsqueeze(0)  # [1, T, 2204]
            else:
                raise ValueError(f"Unsupported input dimension {self.input_dim} for transformer model")
            
            # Step 3: Prepare attention mask if available
            if 'mask' in data:
                mask_data = data['mask']
                # Convert per-keypoint mask to per-frame mask
                seq_mask = torch.from_numpy(mask_data.any(axis=1)).bool().unsqueeze(0)
            else:
                seq_mask = None
            
            # Step 4: Handle sequence length truncation (max_len=300)
            if X.shape[1] > 300:
                X = X[:, :300, :]
                if seq_mask is not None:
                    seq_mask = seq_mask[:, :300]
            
            # Move tensors to device
            X = X.to(self.device)
            if seq_mask is not None:
                seq_mask = seq_mask.to(self.device)
            
            # Step 5: Run model inference
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, seq_mask)
                
        elif self.model_type == 'iv3_gru':
            # Step 2: Extract features for IV3-GRU model
            if 'X2048' not in data:
                raise ValueError("NPZ file must contain 'X2048' key for IV3-GRU model")
            
            # Prepare input tensors
            X2048 = torch.from_numpy(data['X2048']).float().unsqueeze(0)
            lengths = torch.tensor([X2048.shape[1]], dtype=torch.long)
            
            # Move tensors to device
            X2048 = X2048.to(self.device)
            lengths = lengths.to(self.device)
            
            # Step 5: Run model inference
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X2048, lengths, features_already=True)
        
        # Step 6: Process model outputs
        # Get predicted class indices
        gloss_pred = torch.argmax(gloss_logits, dim=-1).item()
        cat_pred = torch.argmax(cat_logits, dim=-1).item()
        
        # Convert logits to probabilities
        gloss_probs = torch.softmax(gloss_logits, dim=-1).squeeze(0)
        cat_probs = torch.softmax(cat_logits, dim=-1).squeeze(0)
        
        # Step 7: Return formatted results
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
        
        Process:
        1. Check preprocessing dependencies are available
        2. Initialize MediaPipe models for feature extraction
        3. Extract features from video frames
        4. Prepare input tensors based on model type
        5. Run model inference
        6. Process outputs and return formatted results
        
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
        # Step 1: Check preprocessing dependencies are available
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Video processing requires preprocessing modules. Please install mediapipe and opencv-python.")
        
        # Step 2: Initialize MediaPipe models for feature extraction
        if self.mp_models is None:
            self.mp_models = create_models()
        
        # Step 3: Extract features from video frames
        frames, keypoints, iv3_features = self._extract_video_features(
            video_path, target_fps, image_size
        )
        
        if self.model_type == 'transformer':
            # Step 4: Prepare input tensors for Transformer model based on input dimension
            if self.input_dim == 2048:
                # Use IV3 features (2048-dimensional)
                if iv3_features is None or len(iv3_features) == 0:
                    raise ValueError("Could not extract IV3 features from video")
                
                # Stack features into sequence tensor
                X = np.stack(iv3_features, axis=0)
                X = torch.from_numpy(X).float().unsqueeze(0)
            elif self.input_dim == 156:
                # Use keypoint features (156-dimensional)
                if keypoints is None or len(keypoints) == 0:
                    raise ValueError("Could not extract keypoints from video")
                
                # Stack keypoints into sequence tensor
                X = np.stack(keypoints, axis=0)
                X = torch.from_numpy(X).float().unsqueeze(0)
            elif self.input_dim == 2204:
                # Use combined features (156 keypoints + 2048 features = 2204-dimensional)
                if keypoints is None or len(keypoints) == 0:
                    raise ValueError("Could not extract keypoints from video")
                if iv3_features is None or len(iv3_features) == 0:
                    raise ValueError("Could not extract IV3 features from video")
                
                # Ensure both have the same sequence length
                if len(keypoints) != len(iv3_features):
                    raise ValueError(f"Sequence length mismatch: keypoints {len(keypoints)} vs features {len(iv3_features)}")
                
                # Stack both keypoints and features
                X_keypoints = np.stack(keypoints, axis=0)  # [T, 156]
                X_features = np.stack(iv3_features, axis=0)  # [T, 2048]
                
                # Concatenate along feature dimension: [T, 156] + [T, 2048] = [T, 2204]
                X_combined = np.concatenate([X_keypoints, X_features], axis=1)  # [T, 2204]
                X = torch.from_numpy(X_combined).float().unsqueeze(0)  # [1, T, 2204]
            else:
                raise ValueError(f"Unsupported input dimension {self.input_dim} for transformer model")
            
            # Handle sequence length truncation (max_len=300)
            if X.shape[1] > 300:
                X = X[:, :300, :]
            
            # Move tensor to device
            X = X.to(self.device)
            
            # Step 5: Run model inference
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X)
                
        elif self.model_type == 'iv3_gru':
            # Step 4: Prepare input tensors for IV3-GRU model
            if iv3_features is None or len(iv3_features) == 0:
                raise ValueError("Could not extract IV3 features from video")
            
            # Stack features into sequence tensor
            X2048 = np.stack(iv3_features, axis=0)
            X2048 = torch.from_numpy(X2048).float().unsqueeze(0)
            lengths = torch.tensor([X2048.shape[1]], dtype=torch.long)
            
            # Move tensors to device
            X2048 = X2048.to(self.device)
            lengths = lengths.to(self.device)
            
            # Step 5: Run model inference
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X2048, lengths, features_already=True)
        
        # Step 6: Process model outputs
        # Get predicted class indices
        gloss_pred = torch.argmax(gloss_logits, dim=-1).item()
        cat_pred = torch.argmax(cat_logits, dim=-1).item()
        
        # Convert logits to probabilities
        gloss_probs = torch.softmax(gloss_logits, dim=-1).squeeze(0)
        cat_probs = torch.softmax(cat_logits, dim=-1).squeeze(0)
        
        # Step 7: Return formatted results
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
        """
        Extract keypoints and IV3 features from video.
        
        Process:
        1. Open video file and get source FPS
        2. Calculate frame sampling interval for target FPS
        3. Iterate through video frames at target FPS
        4. For each frame: resize, convert color, extract features
        5. Return extracted features and frames
        
        Args:
            video_path (str): Path to video file
            target_fps (int): Target FPS for frame extraction
            image_size (int): Size to resize frames
            
        Returns:
            tuple: (frames, keypoints, iv3_features)
                - frames: List of RGB frames
                - keypoints: List of 156-dimensional keypoint vectors
                - iv3_features: List of 2048-dimensional IV3 feature vectors
        """
        # Step 1: Open video file and get source FPS
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps < 1:
            src_fps = 30.0
        
        # Step 2: Calculate frame sampling interval for target FPS
        step_s = 1.0 / target_fps  # Time interval between frames in seconds
        next_t = 0.0  # Next frame timestamp
        
        # Initialize storage lists
        frames = []
        keypoints = []
        iv3_features = []
        
        try:
            # Step 3: Iterate through video frames at target FPS
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                # Check if we should process this frame based on target FPS
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ms < next_t * 1000.0:
                    continue
                
                # Step 4: Process frame
                # Resize frame to target size
                frame_bgr_resized = cv2.resize(frame_bgr, (image_size, image_size))
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
                
                frames.append(frame_rgb)
                
                # Extract keypoints using MediaPipe
                try:
                    vec156, mask78 = extract_keypoints_from_frame(frame_rgb, self.mp_models)
                    keypoints.append(vec156)
                except Exception as e:
                    print(f"Warning: Could not extract keypoints from frame: {e}")
                    keypoints.append(np.zeros(156, dtype=np.float32))
                
                # Extract IV3 features using InceptionV3
                try:
                    iv3_feat = extract_iv3_features(frame_bgr_resized, device=self.device)
                    iv3_features.append(iv3_feat)
                except Exception as e:
                    print(f"Warning: Could not extract IV3 features from frame: {e}")
                    iv3_features.append(np.zeros(2048, dtype=np.float32))
                
                # Update next frame timestamp
                next_t += step_s
                
        finally:
            # Step 5: Clean up video capture
            cap.release()
        
        return frames, keypoints, iv3_features
    
    def cleanup(self):
        """
        Clean up resources and close any open models.
        
        Process:
        1. Close MediaPipe models if initialized
        2. Free up memory resources
        
        Call this method when done with the predictor to free up resources,
        especially MediaPipe models used for video processing.
        """
        if self.mp_models is not None:
            close_models(self.mp_models)
            self.mp_models = None


def list_available_models():
    """
    List all available model checkpoints in the trained_models directory.
    
    Process:
    1. Scan trained_models directory for subdirectories
    2. Find all .pt files in each subdirectory
    3. Display organized list by model type
    
    Scans the trained_models directory for .pt files and displays them
    organized by model type.
    """
    trained_models_dir = Path(__file__).parent.parent.parent / "trained_models"
    
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
    
    Process:
    1. Parse command-line arguments
    2. Handle special commands (list models)
    3. Validate required arguments
    4. Determine device (CPU/GPU)
    5. Initialize model predictor
    6. Make prediction based on input type
    7. Display and optionally save results
    8. Clean up resources
    
    Returns exit code 0 on success, 1 on error.
    """
    # Step 1: Parse command-line arguments
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
    
    # Step 2: Handle special commands
    if args.list_models:
        list_available_models()
        return 0
    
    # Step 3: Validate required arguments
    if not args.model or not args.checkpoint or not args.input:
        parser.error("--model, --checkpoint, and --input are required")
    
    # Step 4: Determine device (CPU/GPU)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Step 5: Initialize model predictor
    try:
        predictor = ModelPredictor(args.model, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Step 6: Make prediction based on input type
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input file not found: {args.input}")
            return 1
        
        if input_path.suffix.lower() == '.npz':
            # Predict from preprocessed NPZ file
            print(f"Predicting from NPZ file: {args.input}")
            results = predictor.predict_from_npz(args.input)
        else:
            # Predict from raw video file
            print(f"Predicting from video file: {args.input}")
            results = predictor.predict_from_video(args.input, args.fps, args.image_size)
        
        # Step 7: Display results with human-readable labels
        print_prediction_summary(results)
        
        # Step 8: Save results if requested
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
        # Clean up resources
        predictor.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
