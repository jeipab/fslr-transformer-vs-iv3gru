"""
InceptionV3 feature extraction for Filipino sign language recognition.

This module provides:
- Single-frame InceptionV3 feature extraction (2048D vectors)
- Batched InceptionV3 processing for GPU-optimized parallel processing
- Video processing with both keypoints and CNN features
- ImageNet pretrained weights for robust feature extraction

Classes:
- BatchedInceptionV3Processor: GPU-optimized batch processing

Functions:
- extract_iv3_features: Single-frame feature extraction

Input: OpenCV BGR image(s)
Output: 2048-dimensional feature vector(s) (float32)

The InceptionV3 model uses ImageNet pretrained weights and global average pooling
to produce consistent feature representations suitable for temporal modeling.
"""
# Standard library imports
import argparse  # Command-line interface
import os  # File system operations
import json  # JSON serialization for metadata

# Computer vision and numerical computing
import cv2  # OpenCV for video processing and image operations
import numpy as np  # Numerical arrays and mathematical operations
import pandas as pd  # Data manipulation and CSV handling

# Deep learning framework
import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules
from torchvision.models import inception_v3, Inception_V3_Weights  # Pre-trained InceptionV3 model

# Project-specific imports
from ..extractors.keypoints_features import (
    extract_keypoints_from_frame,  # Main keypoint extraction function
    interpolate_gaps,     # Fill missing keypoints using interpolation
    POSE_UPPER_25,        # Upper body pose keypoint indices (25 points)
    FACEMESH_11,          # Face mesh keypoint indices (11 key facial points)
    create_models,        # Initialize MediaPipe models
    close_models,         # Clean up MediaPipe models
)

# ----------------------------
# Model Configuration
# ----------------------------
# Define model weights to use (weights are cached by PyTorch after first download)
# Each worker creates its own model instance for proper GPU isolation
_iv3_weights = Inception_V3_Weights.IMAGENET1K_V1

# ImageNet normalization constants (RGB channel means and standard deviations)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ----------------------------
# Shared Preprocessing Utilities
# ----------------------------

def _preprocess_frame_to_tensor(frame_bgr, image_size=(299, 299), device=None):
    """Preprocess a BGR frame to InceptionV3 input tensor.
    
    Args:
        frame_bgr: OpenCV BGR image (H, W, 3) in [0, 255] pixel range
        image_size: Target image size for InceptionV3 (default: 299x299)
        device: PyTorch device for computation (default: CPU)
        
    Returns:
        Preprocessed tensor [1, 3, 299, 299] ready for InceptionV3
    """
    if device is None:
        device = torch.device("cpu")
    
    # STEP 1: Convert BGR to RGB and resize
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(frame_rgb, image_size)
    
    # STEP 2: Convert to PyTorch tensor and normalize pixel values to [0, 1]
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension: [1, 3, 299, 299]
    
    # STEP 3: Apply ImageNet normalization
    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)
    tensor = (tensor - mean) / std
    
    return tensor

# ----------------------------
# Batched InceptionV3 Processing for GPU Efficiency
# ----------------------------

class BatchedInceptionV3Processor:
    """Batched InceptionV3 feature extraction for efficient GPU processing.
    
    This class provides GPU-optimized batch processing of video frames through InceptionV3,
    significantly improving throughput compared to single-frame processing. Key features:
    - Batch processing reduces GPU kernel launch overhead
    - Automatic device management for multi-GPU systems
    - ImageNet-pretrained features for robust visual representation
    - Memory-efficient processing with configurable batch sizes
    """
    
    def __init__(self, device=None, batch_size=32):
        """Initialize the batched InceptionV3 processor.
        
        Args:
            device: PyTorch device (auto-detects GPU if available)
            batch_size: Number of frames to process simultaneously (32 is optimal for most GPUs)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._init_model()
    
    def _init_model(self):
        """Initialize a NEW InceptionV3 model instance for this worker."""
        # Create independent model instance per worker (weights are cached by PyTorch)
        model = inception_v3(weights=_iv3_weights)
        model.aux_logits = False
        model.fc = nn.Identity()
        model.eval()
        
        # Move to target device and keep it there
        self.model = model.to(self.device)
        
        # Freeze parameters for inference
        for p in self.model.parameters():
            p.requires_grad = False
    
    def preprocess_frame(self, frame_bgr, image_size=(299, 299)):
        """Preprocess a single BGR frame for InceptionV3 input."""
        tensor = _preprocess_frame_to_tensor(frame_bgr, image_size, self.device)
        return tensor.squeeze(0)  # Remove batch dimension for individual frame processing
    
    def extract_batch_features(self, frames_bgr, image_size=(299, 299)):
        """Extract InceptionV3 features for a batch of frames efficiently.
        
        Args:
            frames_bgr: List of BGR frames to process
            image_size: Target size for InceptionV3 (299, 299)
            
        Returns:
            np.ndarray: Feature vectors [batch_size, 2048] or empty array if no frames
        """
        if not frames_bgr:
            return np.array([])
        
        # Batch preprocessing
        tensors = [self.preprocess_frame(frame, image_size) for frame in frames_bgr]
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        return features.cpu().numpy()
    
    def extract_single_features(self, frame_bgr, image_size=(299, 299)):
        """Extract InceptionV3 features from a single frame (compatible with extract_iv3_features).
        
        Args:
            frame_bgr: Single BGR frame to process
            image_size: Target size for InceptionV3 (299, 299)
            
        Returns:
            np.ndarray: Feature vector [2048] as float32 numpy array
        """
        batch_features = self.extract_batch_features([frame_bgr], image_size)
        return batch_features[0] if len(batch_features) > 0 else np.zeros(2048, dtype=np.float32)

def extract_iv3_features(frame_bgr, image_size=(299, 299), device=None):
    """Extract InceptionV3 features from a single BGR frame.
    
    This function performs the complete pipeline for CNN feature extraction:
    1. Color space conversion (BGR to RGB)
    2. Image resizing to InceptionV3 input size
    3. Normalization using ImageNet statistics
    4. Forward pass through pre-trained InceptionV3
    5. Return 2048-dimensional feature vector
    
    Note: This function creates a new model instance for thread safety.
    For batch processing, use BatchedInceptionV3Processor instead.
    
    Args:
        frame_bgr: OpenCV BGR image (H, W, 3) in [0, 255] pixel range
        image_size: Target image size for InceptionV3 (default: 299x299)
        device: PyTorch device for computation (default: CUDA if available)
        
    Returns:
        Feature vector [2048] as float32 numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance for thread safety (weights cached by PyTorch)
    model = inception_v3(weights=_iv3_weights)
    model.aux_logits = False
    model.fc = nn.Identity()
    model.eval()
    model = model.to(device)
    
    for p in model.parameters():
        p.requires_grad = False

    # STEP 1-3: Preprocess frame using shared utility
    tensor = _preprocess_frame_to_tensor(frame_bgr, image_size, device)

    # STEP 4: Forward pass through InceptionV3 (no gradient computation needed)
    with torch.no_grad():
        feats = model(tensor)  # Shape: [1, 2048]
    
    # STEP 5: Return as numpy array on CPU
    return feats.squeeze(0).cpu().numpy()

# ----------------------------
# Utility Functions
# ----------------------------

def ensure_dir(p):
    """Create directory if it doesn't exist.
    
    Args:
        p (str): Directory path to create
    """
    os.makedirs(p, exist_ok=True)

def to_npz(out_path, X, X2048, mask, timestamps_ms, meta, also_parquet=True):
    """Save processed video data (keypoints + CNN features) to compressed .npz file.
    
    This function saves the complete output of video processing: keypoint coordinates,
    InceptionV3 features, visibility masks, timestamps, and metadata. The .npz format
    is used for efficient storage and fast loading during training.
    
    Args:
        out_path: Base path for output files (without extension)
        X: Keypoint coordinates [T, 156] as float32 - flattened x,y coords for 78 keypoints
        X2048: InceptionV3 features [T, 2048] as float32 - CNN feature vectors
        mask: Keypoint visibility mask [T, 78] as bool - True if keypoint is visible/confident
        timestamps_ms: Frame timestamps [T] as int64 - milliseconds from video start
        meta: Metadata dictionary (converted to JSON string) - processing parameters
        also_parquet: If True, also create .parquet file for inspection in spreadsheet tools
    """
    # Save primary .npz file with all data compressed (keypoints + CNN features)
    np.savez_compressed(out_path + ".npz", X=X, X2048=X2048, mask=mask, 
                       timestamps_ms=timestamps_ms, meta=json.dumps(meta))
    
    # Optionally create human-readable parquet file for data inspection
    if also_parquet:
        try:
            # Convert keypoint coordinates to DataFrame (each column = one coordinate)
            df = pd.DataFrame(X)
            # Add timestamp column for temporal reference
            df["t_ms"] = timestamps_ms
            # Convert visibility mask to compact binary string for easy inspection
            df["mask_bits"] = ["".join("1" if b else "0" for b in row) for row in mask]
            df.to_parquet(out_path + ".parquet")
        except Exception as e:
            print(f"[WARN] Could not save parquet file: {e}")
            print("[INFO] Install pyarrow or fastparquet for parquet support: pip install pyarrow")

# ----------------------------
# Labels CSV Management
# ----------------------------

def read_or_create_labels_csv(label_file):
    """Read existing labels CSV or create new file with headers.
    
    Args:
        label_file: Path to labels CSV file
        
    Returns:
        pd.DataFrame: Labels dataframe with columns [file, gloss, cat]
    """
    if os.path.exists(label_file):
        return pd.read_csv(label_file)
    else:
        # Create empty dataframe with required columns and save to file
        df = pd.DataFrame(columns=["file", "gloss", "cat"])
        df.to_csv(label_file, index=False)
        return df

def update_labels_csv(label_file, video_file, gloss, cat):
    """Add a new labeled data entry to the labels CSV file.
    
    This function appends a single row to the labels CSV, mapping a processed video file
    to its classification labels. Used for building training datasets.
    
    Args:
        label_file: Path to labels CSV file
        video_file: Processed video filename (e.g., 'clip.npz')
        gloss: Sign language gloss class ID
        cat: Category class ID
    """
    # Read existing labels or create new file
    df = read_or_create_labels_csv(label_file)
    # Create new row with video-to-label mapping
    new_row = pd.DataFrame({"file": [video_file], "gloss": [gloss], "cat": [cat]})
    # Append to existing data and save
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(label_file, index=False)

# ----------------------------
# Main Video Processing Function
# ----------------------------

def process_video(video_path, out_dir, label_file=None, target_fps=30, out_size=256, conf_thresh=0.5, max_gap=5, write_keypoints=True, write_iv3_features=True, feature_key='X2048', gloss=None, cat=None):
    """Process a single video file and extract multi-modal features.
    
    This function performs the complete processing pipeline for a single video:
    1. Video loading and frame sampling at target FPS
    2. MediaPipe keypoint extraction (pose, hands, face)
    3. InceptionV3 CNN feature extraction
    4. Gap interpolation for missing keypoints
    5. Data saving in compressed .npz format
    6. Labels CSV updates for training data
    
    Args:
        video_path: Path to input video file (.mp4, .mov, .avi, .mkv)
        out_dir: Output directory for processed .npz files
        label_file: Path to labels CSV file (default: out_dir/labels.csv)
        target_fps: Target frame sampling rate (downsamples high FPS videos)
        out_size: Image resize dimension for keypoint extraction (256x256)
        conf_thresh: Confidence threshold for keypoint detection (0.0-1.0)
        max_gap: Maximum gap size for keypoint interpolation (frames)
        write_keypoints: Extract MediaPipe keypoints (156D vectors per frame)
        write_iv3_features: Extract InceptionV3 features (2048D vectors per frame)
        feature_key: Unused parameter (kept for compatibility)
        gloss: Sign language gloss class ID for labeling
        cat: Category class ID for labeling
    """
    # STEP 1: Setup output paths and directories
    basename = os.path.splitext(os.path.basename(video_path))[0]  # Extract filename without extension
    output_npz_folder = os.path.join(out_dir, '0')  # Output to '0' subfolder (dataset convention)
    ensure_dir(output_npz_folder)  # Create output directory if it doesn't exist
    npz_out_path = os.path.join(output_npz_folder, basename)  # Base path for output files
    
    # Set default labels CSV path if not specified
    if label_file is None:
        label_file = os.path.join(out_dir, "labels.csv")

    # STEP 2: Initialize video capture and processing parameters
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    # Get source video frame rate with fallback for corrupted metadata
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1:
        src_fps = 30.0  # fallback for videos with invalid FPS metadata

    # Calculate frame sampling parameters for target FPS
    step_s = 1.0 / target_fps  # Time interval between sampled frames (seconds)
    next_t = 0.0  # Next target timestamp for frame sampling

    # Set device for InceptionV3 feature extraction (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize MediaPipe models for keypoint detection (only if needed)
    models = None
    if write_keypoints:
        models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)

    # STEP 3: Initialize data containers for collected features
    X_frames = []      # Keypoint coordinates [frame_idx] -> [156] (78 keypoints * 2 coords)
    M_frames = []      # Keypoint visibility masks [frame_idx] -> [78] (boolean visibility)
    X2048_frames = []  # InceptionV3 CNN features [frame_idx] -> [2048] (deep features)
    T_ms = []          # Frame timestamps [frame_idx] -> timestamp_ms

    # Get video metadata for processing
    t0 = cap.get(cv2.CAP_PROP_POS_MSEC)  # Starting timestamp (usually 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in video

    # STEP 4: Main video processing loop - extract features from sampled frames
    try:
        while True:
            # Read next frame from video
            ret, frame_bgr = cap.read()
            if not ret:  # End of video reached
                break
            
            # Check if this frame should be sampled based on target FPS
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # Current timestamp in milliseconds
            if ms < next_t * 1000.0:  # Skip frame if not at target sampling time
                continue

            # STEP 4a: Resize frame for consistent processing
            frame_bgr_resized = cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

            # STEP 4b: Prepare RGB frame for MediaPipe keypoint extraction
            frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)

            # STEP 4c: Extract MediaPipe keypoints (if requested)
            if write_keypoints:
                # Returns 156D vector (78 keypoints * 2 coords) and 78D visibility mask
                vec156, mask78 = extract_keypoints_from_frame(frame_rgb, models, conf_thresh=conf_thresh)
                X_frames.append(vec156)    # Store keypoint coordinates
                M_frames.append(mask78)    # Store visibility flags

            # STEP 4d: Extract InceptionV3 CNN features (if requested)
            if write_iv3_features:
                # Extract 2048-D InceptionV3 features from the original BGR frame
                # Uses original frame (not resized) to preserve image quality for CNN
                iv3_features = extract_iv3_features(frame_bgr, image_size=(299, 299), device=device)
                X2048_frames.append(iv3_features)

            # STEP 4e: Record frame timestamp and advance to next sampling time
            T_ms.append(ms)
            next_t += step_s  # Advance to next target sampling time
    finally:
        # STEP 5: Cleanup resources
        cap.release()      # Release video capture object
        if models is not None:
            close_models(models)  # Clean up MediaPipe models

    # STEP 6: Validate and post-process extracted features
    if len(X_frames) == 0:
        print(f"[WARN] No frames written for {video_path}")
        return

    # Convert lists to numpy arrays for efficient processing
    X = np.stack(X_frames, axis=0)  # Shape: [T, 156] - keypoint coordinates over time
    M = np.stack(M_frames, axis=0)  # Shape: [T, 78] - visibility masks over time
    X2048 = np.stack(X2048_frames, axis=0) if X2048_frames else np.array([])  # Shape: [T, 2048] - CNN features
    T_ms = np.array(T_ms, dtype=np.int64)  # Convert timestamps to numpy array

    # Validate temporal consistency between keypoints and CNN features
    if write_iv3_features and X2048.size > 0:
        assert X.shape[0] == X2048.shape[0], f"Mismatch in T (frames) between X and X2048: {X.shape[0]} vs {X2048.shape[0]}"
    
    # Handle case where no CNN features were extracted
    if write_iv3_features and len(X2048_frames) == 0:
        print("[WARN] No IV3 features extracted.")

    # Fill gaps in keypoint sequences using interpolation
    X_filled, M_filled = interpolate_gaps(X, M, max_gap=max_gap)
    # Ensure keypoint coordinates stay within valid bounds [0, 1]
    X_filled = np.clip(X_filled, 0.0, 1.0).astype(np.float32)
    # Note: Do not interpolate CNN features - keep raw temporal values
    X2048_filled = X2048

    # STEP 7: Prepare comprehensive metadata for reproducibility
    meta = dict(
        video=os.path.basename(video_path),      # Original video filename
        target_fps=target_fps,                   # Frame sampling rate used
        out_size=out_size,                       # Image resize dimension
        dims_per_frame=156,                      # Keypoint vector dimension (78 points * 2 coords)
        keypoints_total=78,                      # Total number of keypoints tracked
        order="pose25,left_hand21,right_hand21,face11",  # Keypoint ordering in the 156D vector
        pose_indices=POSE_UPPER_25,              # Which pose keypoints are used
        face_indices=FACEMESH_11,                # Which face keypoints are used
        conf_thresh=conf_thresh,                 # Confidence threshold used for detection
        interpolation_max_gap=max_gap,           # Maximum gap size for interpolation
        gloss=gloss,                             # Sign language gloss class ID
        cat=cat                                  # Category class ID
    )

    # STEP 8: Update training labels CSV with processed file information
    if gloss and cat:
        update_labels_csv(label_file, basename, gloss, cat)

    # STEP 9: Save final processed data with all features
    to_npz(npz_out_path, X_filled, X2048_filled, M_filled, T_ms, meta, also_parquet=True)

    # Report successful processing
    print(f"[OK] {basename}: frames={len(X_frames)} saved: {npz_out_path}.npz (+ .parquet)")

# ----------------------------
# Command-line Interface
# ----------------------------

if __name__ == "__main__":
    # COMMAND-LINE INTERFACE: Setup argument parser for single video processing
    parser = argparse.ArgumentParser(description="Single video preprocessing with keypoints and InceptionV3 features")
    
    # Required arguments
    parser.add_argument('video_path', type=str, help='Path to the video file to process')
    parser.add_argument('out_dir', type=str, help='Directory to save the processed output')
    
    # Feature extraction controls
    parser.add_argument('--write-keypoints', action='store_true', help='Extract and save MediaPipe keypoints (156D vectors)')
    parser.add_argument('--write-iv3-features', action='store_true', help='Extract and save InceptionV3 CNN features (2048D vectors)')
    
    # Processing parameters
    parser.add_argument('--fps', type=int, default=30, help='Target frames per second for sampling (default: 30)')
    parser.add_argument('--image-size', type=int, default=256, help='Output image size for keypoint extraction (default: 256)')
    parser.add_argument('--feature-key', type=str, default='X2048', help='Feature key name for compatibility (default: X2048)')
    
    # Labeling controls
    parser.add_argument('--gloss', type=str, help='Sign language gloss class ID for labeling')
    parser.add_argument('--cat', type=str, help='Category class ID for labeling')
    parser.add_argument('--label-file', type=str, help='Path to labels CSV file (default: output_dir/labels.csv)')
    
    args = parser.parse_args()
    
    # SETUP: Set default label file path if not specified
    if args.label_file is None:
        args.label_file = os.path.join(args.out_dir, "labels.csv")

    # MAIN PROCESSING: Process the single video file
    process_video(
        video_path=args.video_path,           # Input video file
        out_dir=args.out_dir,                 # Output directory
        label_file=args.label_file,           # Labels CSV file
        target_fps=args.fps,                  # Frame sampling rate
        out_size=args.image_size,             # Image resize dimension
        write_keypoints=args.write_keypoints, # Extract MediaPipe keypoints
        write_iv3_features=args.write_iv3_features,  # Extract CNN features
        feature_key=args.feature_key,         # Feature naming (compatibility)
        gloss=args.gloss,                     # Sign language class ID
        cat=args.cat                          # Category class ID
    )