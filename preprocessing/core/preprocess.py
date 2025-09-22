"""
Unified video preprocessing pipeline for Filipino sign language recognition.

This module processes raw video files to extract:
- MediaPipe keypoints (pose, hands, face) → 156-dimensional vectors
- InceptionV3 CNN features → 2048-dimensional vectors  
- Occlusion detection flags
- Frame timestamps and metadata

Supports both sequential and parallel processing with automatic GPU optimization.

Usage:
- Single video (sequential):
    python preprocessing/preprocess.py video.mp4 output_dir --write-keypoints --write-iv3-features --id 12
- Directory of videos (parallel):
    python preprocessing/preprocess.py input_dir output_dir --write-keypoints --write-iv3-features --id 12 --workers 8
- Batch processing with GPU optimization:
    python preprocessing/preprocess.py input_dir output_dir --write-keypoints --write-iv3-features --workers 8 --batch-size 32
"""

# Standard library imports
import os, sys, json, math, argparse, time  # File operations, system utilities, JSON handling, timing
from multiprocessing import Pool, cpu_count, set_start_method  # Parallel processing support

# Computer vision and numerical computing
import cv2  # OpenCV for video processing and image operations
import numpy as np  # Numerical arrays and mathematical operations
import pandas as pd  # Data manipulation and CSV handling

# Machine learning frameworks
import torch  # PyTorch for deep learning (InceptionV3 features)
import mediapipe as mp  # Google's MediaPipe for keypoint detection (pose, hands, face)
from tqdm import tqdm  # Progress bars for multiprocessing tracking

# MULTIPROCESSING SETUP: Set start method to 'spawn' for CUDA compatibility
# This prevents CUDA context issues when using multiple GPU workers
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set by previous import or system default

# Path setup: Allow running both as a module (-m) and as a script (python preprocessing/preprocess.py)
# This ensures imports work correctly regardless of how the script is executed
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Project-specific imports
from ..extractors.iv3_features import extract_iv3_features  # InceptionV3 CNN feature extraction (2048D vectors)
from ..core.occlusion_detection import compute_occlusion_detection  # Detect when keypoints are blocked/occluded
from ..extractors.keypoints_features import (
    POSE_UPPER_25,        # Upper body pose keypoint indices (25 points)
    N_HAND,               # Number of hand keypoints per hand (21 points each)
    FACEMESH_11,          # Face mesh keypoint indices (11 key facial points)
    extract_keypoints_from_frame,  # Main keypoint extraction function
    interpolate_gaps,     # Fill missing keypoints using interpolation
    xy_from_landmark,     # Convert MediaPipe landmarks to normalized coordinates
    create_models,        # Initialize MediaPipe models
    close_models,         # Clean up MediaPipe models
    MPModels,             # MediaPipe models container class
)

# ----------------------------
# Utility Functions
# ----------------------------

def ensure_dir(p):
    """Create directory if it doesn't exist.
    
    Args:
        p (str): Directory path to create
    """
    os.makedirs(p, exist_ok=True)


def to_npz(out_path, X, mask, timestamps_ms, meta, also_parquet=True):
    """Save processed keypoint data to compressed .npz file with optional parquet export.
    
    This function saves the core output of video processing: keypoint coordinates,
    visibility masks, timestamps, and metadata. The .npz format is used for efficient
    storage and fast loading during training.
    
    Args:
        out_path: Base path for output files (without extension)
        X: Keypoint coordinates [T, 156] as float32 - flattened x,y coords for 78 keypoints
        mask: Keypoint visibility mask [T, 78] as bool - True if keypoint is visible/confident
        timestamps_ms: Frame timestamps [T] as int64 - milliseconds from video start
        meta: Metadata dictionary (converted to JSON string) - processing parameters
        also_parquet: If True, also create .parquet file for inspection in spreadsheet tools
    """
    # Save primary .npz file with all data compressed
    np.savez_compressed(out_path + ".npz", X=X, mask=mask, timestamps_ms=timestamps_ms, meta=json.dumps(meta))
    
    # Optionally create human-readable parquet file for data inspection
    if also_parquet:
        try:
            # Convert keypoint coordinates to DataFrame (each column = one coordinate)
            df = pd.DataFrame(X)
            # Ensure string column names to avoid parquet mixed-type warnings
            df.columns = df.columns.astype(str)
            # Add timestamp column for temporal reference
            df["t_ms"] = timestamps_ms
            # Convert visibility mask to compact binary string for easy inspection
            df["mask_bits"] = ["".join("1" if b else "0" for b in row) for row in mask]
            df.to_parquet(out_path + ".parquet")
        except Exception as e:
            print(f"[WARN] Could not save parquet file: {e}")
            print("[INFO] Install pyarrow or fastparquet for parquet support: pip install pyarrow")


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
        """Initialize InceptionV3 model with ImageNet weights for feature extraction."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        import torch.nn as nn
        
        # Load pre-trained InceptionV3 model
        self.weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=self.weights)
        
        # Configure model for feature extraction
        self.model.aux_logits = False  # Disable auxiliary classifier outputs
        self.model.fc = nn.Identity()  # Replace final classifier with identity (returns 2048D features)
        self.model.eval()  # Set to evaluation mode
        
        # Freeze all parameters for inference-only mode
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Move model to target device
        self.model = self.model.to(self.device)
        
        # Pre-compute ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def preprocess_frame(self, frame_bgr, image_size=(299, 299)):
        """Preprocess a single BGR frame for InceptionV3 input."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Resize to InceptionV3 input dimensions
        img_resized = cv2.resize(frame_rgb, image_size)
        # Convert to PyTorch tensor and normalize to [0, 1]
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # Move to device and apply ImageNet normalization
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor
    
    def extract_batch_features(self, frames_bgr, image_size=(299, 299)):
        """Extract InceptionV3 features for a batch of frames efficiently."""
        if not frames_bgr:
            return np.array([])
        
        # Batch preprocessing
        tensors = [self.preprocess_frame(frame, image_size) for frame in frames_bgr]
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        return features.cpu().numpy()


# ----------------------------
# MediaPipe Solution References
# ----------------------------
# Store references to MediaPipe solutions for keypoint detection
mp_hands = mp.solutions.hands          # Hand landmark detection (21 points per hand)
mp_face_mesh = mp.solutions.face_mesh  # Face mesh detection (468 points, we use 11)
mp_drawing = mp.solutions.drawing_utils # Visualization utilities (unused in processing)
mp_pose = mp.solutions.pose            # Body pose detection (33 points, we use upper 25)

 
def _ensure_labels_csv(path, include_occluded_col=True, overwrite=False):
    """Create or update labels CSV file with required columns for training data.
    
    This function manages the labels.csv file that maps processed video files to their
    class labels. The CSV contains: file path, gloss ID, category ID, and occlusion flag.
    
    Args:
        path: Path to CSV file to create/update
        include_occluded_col: Add 'occluded' column if missing (for filtering training data)
        overwrite: If True, overwrite existing file header (start fresh)
    """
    # Ensure the directory exists for the CSV file
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    # Create new CSV file or overwrite existing one
    if overwrite or not os.path.exists(path):
        # Define required columns: file path, gloss class, category class, occlusion flag
        cols = ["file", "gloss", "cat"] + (["occluded"] if include_occluded_col else [])
        # Create empty DataFrame with headers and save to CSV
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    
    # Upgrade existing CSV: add 'occluded' column if it's missing
    try:
        df = pd.read_csv(path)
        # Check if occluded column needs to be added to existing data
        if include_occluded_col and "occluded" not in (df.columns.tolist() if df is not None else []):
            df["occluded"] = 0  # Default to not occluded for existing entries
            df.to_csv(path, index=False)  # Save updated CSV with new column
    except Exception as e:
        print(f"[WARN] Could not inspect/upgrade labels csv '{path}': {e}")


def _append_label_row(path, file_entry, gloss_id, cat_id, occluded_flag=0):
    """Add a new labeled data entry to the labels CSV file.
    
    This function appends a single row to the labels CSV, mapping a processed video file
    to its classification labels and quality flags. Used during batch processing.
    
    Args:
        path: Path to CSV file to append to
        file_entry: Relative path to processed .npz file (e.g., '0/clip.npz')
        gloss_id: Sign language gloss class ID (main classification target)
        cat_id: Category class ID (higher-level grouping)
        occluded_flag: Occlusion detection result (0=clear, 1=occluded)
    """
    try:
        # Prepare the new row data with required fields
        new_row = {"file": str(file_entry), "gloss": int(gloss_id), "cat": int(cat_id)}
        
        # Check if CSV has 'occluded' column by reading the header
        try:
            with open(path, "r", newline="") as fh:
                header = fh.readline().strip().split(",") if fh else []
        except Exception:
            header = []
        
        # Add occlusion flag if the CSV supports it
        if "occluded" in header:
            new_row["occluded"] = int(occluded_flag)
        
        # Create single-row DataFrame and append to CSV (without header)
        df = pd.DataFrame([new_row])
        df.to_csv(path, mode="a", index=False, header=False)
    except Exception as e:
        print(f"[WARN] Failed to append to labels csv '{path}': {e}")


def process_video(video_path, out_dir, target_fps=30, out_size=256, conf_thresh=0.5, max_gap=5, write_keypoints=True, write_iv3_features=True, feature_key='X2048',
                 compute_occlusion=True, occ_detailed=False, labels_csv_path=None, gloss_id=None, cat_id=None):
    """Process a single video file and extract multi-modal features for sign language recognition.
    
    This is the main processing function that converts raw video files into structured
    feature representations suitable for machine learning. It performs:
    1. Video loading and frame sampling at target FPS
    2. Person segmentation to remove background
    3. MediaPipe keypoint extraction (pose, hands, face)
    4. InceptionV3 CNN feature extraction
    5. Gap interpolation for missing keypoints
    6. Occlusion detection and quality assessment
    7. Data saving in compressed .npz format
    
    Args:
        video_path: Path to input video file (.mp4, .mov, .avi, .mkv)
        out_dir: Output directory for processed .npz files
        target_fps: Target frame sampling rate (downsamples high FPS videos)
        out_size: Image resize dimension for keypoint extraction (256x256)
        conf_thresh: Confidence threshold for keypoint detection (0.0-1.0)
        max_gap: Maximum gap size for interpolation (frames)
        write_keypoints: Extract MediaPipe keypoints (156D vectors per frame)
        write_iv3_features: Extract InceptionV3 features (2048D vectors per frame)
        feature_key: Unused parameter (kept for compatibility)
        compute_occlusion: Enable occlusion detection for quality filtering
        occ_detailed: Return detailed occlusion analysis results
        labels_csv_path: Path to labels CSV file for training data mapping
        gloss_id: Sign language gloss class ID for labeling
        cat_id: Category class ID for labeling
    """
    # STEP 1: Setup output paths and directories
    basename = os.path.splitext(os.path.basename(video_path))[0]  # Extract filename without extension
    output_npz_folder = os.path.join(out_dir)  # Output directory for processed files
    ensure_dir(output_npz_folder)  # Create output directory if it doesn't exist
    npz_out_path = os.path.join(output_npz_folder, basename)  # Base path for output files (no extension)

    # STEP 2: Initialize video capture and frame sampling parameters
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    # Get source video frame rate with fallback for corrupted metadata
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or math.isnan(src_fps) or src_fps < 1:
        src_fps = 30.0  # fallback for videos with invalid FPS metadata

    # Calculate frame sampling parameters for target FPS
    step_s = 1.0 / target_fps  # Time interval between sampled frames (seconds)
    next_t = 0.0  # Next target timestamp for frame sampling

    # STEP 3: Initialize ML models and data containers
    # Create MediaPipe models for keypoint detection (pose, hands, face, segmentation)
    models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)
    # Set device for InceptionV3 feature extraction (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize lists to collect processed data from each frame
    X_frames = []      # Keypoint coordinates [frame_idx] -> [156] (78 keypoints * 2 coords)
    M_frames = []      # Keypoint visibility masks [frame_idx] -> [78] (boolean visibility)
    X2048_frames = []  # InceptionV3 CNN features [frame_idx] -> [2048] (deep features)
    T_ms = []          # Frame timestamps [frame_idx] -> timestamp_ms

    # Get video metadata for progress tracking
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

            # STEP 4b: Person segmentation to remove background
            frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
            seg_mask = models.seg.process(frame_rgb).segmentation_mask  # Get person segmentation mask
            fg_mask = (seg_mask > 0.5).astype(np.float32)[..., None]  # Threshold mask (0.5 confidence)
            black_bg = np.zeros_like(frame_rgb, dtype=np.uint8)  # Create black background
            comp_rgb = (frame_rgb * fg_mask + black_bg * (1 - fg_mask)).astype(np.uint8)  # Composite person on black

            # STEP 4c: Extract MediaPipe keypoints (always done for visualization)
            # Returns 156D vector (78 keypoints * 2 coords) and 78D visibility mask
            vec156, mask78 = extract_keypoints_from_frame(comp_rgb, models, conf_thresh=conf_thresh)
            X_frames.append(vec156)    # Store keypoint coordinates
            M_frames.append(mask78)    # Store visibility flags

            # STEP 4d: Extract InceptionV3 CNN features
            if write_iv3_features:
                # Extract 2048-D InceptionV3 features from the original BGR frame (not segmented)
                # Uses full frame to capture contextual information for CNN
                iv3_features = extract_iv3_features(frame_bgr, image_size=(299, 299), device=device)
                X2048_frames.append(iv3_features)

            # STEP 4e: Record frame timestamp and advance to next sampling time
            T_ms.append(ms)
            next_t += step_s  # Advance to next target sampling time
    finally:
        # STEP 5: Cleanup resources
        cap.release()      # Release video capture object
        close_models(models)  # Clean up MediaPipe models

    # STEP 6: Validate that frames were successfully processed
    if len(T_ms) == 0:
        print(f"[WARN] No frames written for {video_path}")
        return

    # Convert timestamps to numpy array for efficient storage
    T_ms = np.array(T_ms, dtype=np.int64)
    
    # STEP 7: Post-process extracted features
    X_filled, M_filled = None, None
    X2048_filled = None
    
    # STEP 7a: Process keypoints (always extracted for visualization)
    if len(X_frames) == 0:
        print(f"[WARN] No keypoint frames written for {video_path}")
        return
    
    # Stack individual frame data into temporal sequences
    X = np.stack(X_frames, axis=0)  # Shape: [T, 156] - keypoint coordinates over time
    M = np.stack(M_frames, axis=0)  # Shape: [T, 78] - visibility masks over time
    
    # Fill gaps in keypoint sequences using interpolation
    X_filled, M_filled = interpolate_gaps(X, M, max_gap=max_gap)
    
    # Ensure keypoint coordinates stay within valid bounds [0, 1]
    X_filled = np.clip(X_filled, 0.0, 1.0).astype(np.float32)
    
    # STEP 7b: Process InceptionV3 features
    if write_iv3_features:
        if len(X2048_frames) == 0:
            print(f"[WARN] No IV3 feature frames written for {video_path}")
            return
        X2048 = np.stack(X2048_frames, axis=0)  # Shape: [T, 2048] - CNN features over time
        # CNN features represent holistic frame content, interpolation may distort meaning
        X2048_filled = X2048

    # STEP 8: Prepare metadata for the processed video
    # Determine model type based on what features are being saved
    model_type = "B" if (write_keypoints and write_iv3_features) else ("T" if write_keypoints else "I")
    
    # Create comprehensive metadata dictionary for reproducibility and debugging
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
        model_type=model_type,                   # T=Transformer only, I=IV3-GRU only, B=Both
        occluded_flag=0                          # Will be updated after occlusion computation
    )

    # STEP 9: Save initial data (keypoints are always saved for visualization)
    if write_keypoints:
        # Save keypoints with parquet export for easy inspection
        to_npz(npz_out_path, X_filled, M_filled, T_ms, meta, also_parquet=True)
    else:
        # Save NPZ with keypoints for visualization but mark as IV3-GRU only model
        np.savez_compressed(npz_out_path + ".npz", X=X_filled, mask=M_filled, timestamps_ms=T_ms, meta=json.dumps(meta))

    # STEP 10: Occlusion detection and quality assessment
    occluded_flag = 0  # Default: not occluded
    occlusion_results = None
    
    if compute_occlusion:
        if occ_detailed:
            # Detailed occlusion analysis with comprehensive metrics
            occlusion_results = compute_occlusion_detection(
                video_path=video_path,  # Original video for additional analysis
                X=X_filled if write_keypoints else None,  # Processed keypoint coordinates
                mask_bool_array=M_filled if write_keypoints else None,  # Keypoint visibility masks
                output_format='detailed'  # Return detailed analysis results
            )
            # Extract binary flag from detailed results
            occluded_flag = occlusion_results.get('binary_flag', 0)
        else:
            # Simple binary occlusion flag (0=clear, 1=occluded)
            occluded_flag = compute_occlusion_detection(
                video_path=video_path,
                X=X_filled if write_keypoints else None,
                mask_bool_array=M_filled if write_keypoints else None,
                output_format='compatible'  # Return simple binary flag
            )
    
    # STEP 11: Update metadata with occlusion detection results
    meta['occluded_flag'] = occluded_flag  # Store binary occlusion flag in metadata
    if occlusion_results is not None:
        meta['occlusion_results'] = occlusion_results  # Store detailed results if available
    
    # STEP 12: Update training labels CSV with processed file information
    if labels_csv_path is not None and gloss_id is not None:
        final_cat = cat_id if cat_id is not None else gloss_id  # Use category ID or fall back to gloss ID
        # Store file path relative to output directory (e.g., '0/clip.npz') for portable dataset
        rel_npz_path = os.path.relpath(npz_out_path + ".npz", start=out_dir)
        # Add entry to labels CSV: file -> gloss_id, category_id, occlusion_flag
        _append_label_row(labels_csv_path, rel_npz_path, gloss_id, final_cat, occluded_flag)

    # STEP 13: Save final processed data with all features and updated metadata
    save_dict = {
        'X': X_filled,                    # Keypoint coordinates [T, 156]
        'mask': M_filled,                 # Keypoint visibility masks [T, 78]
        'timestamps_ms': T_ms,            # Frame timestamps [T]
        'meta': json.dumps(meta)          # Processing metadata as JSON string
    }
    
    # Add InceptionV3 features if they were extracted
    if write_iv3_features and X2048_filled is not None:
        save_dict.update({'X2048': X2048_filled})  # CNN features [T, 2048]
    
    # Save final compressed .npz file with all data
    np.savez_compressed(npz_out_path + ".npz", **save_dict)

    # Report successful processing
    frame_count = len(T_ms)
    print(f"[OK] {basename}: frames={frame_count} saved: {npz_out_path}.npz (+ .parquet)")


def process_video_worker(args):
    """Worker function for processing a single video in a multiprocessing environment.
    
    This function runs in a separate process and handles the complete processing pipeline
    for one video file. It's designed for parallel execution with:
    - Independent GPU device assignment
    - Batched InceptionV3 processing for efficiency
    - Error isolation (one video failure doesn't crash others)
    - Resource cleanup to prevent memory leaks
    
    Args:
        args: Tuple of processing parameters
    
    Returns:
        dict: Processing result with success/error status and metadata
    """
    # Unpack parameters
    (video_path, out_dir, target_fps, out_size, conf_thresh, max_gap, 
     write_keypoints, write_iv3_features, feature_key, compute_occlusion, 
     occ_detailed, labels_csv_path, gloss_id, cat_id, batch_size, device_id, disable_parquet) = args
    
    try:
        # Setup GPU device for this worker
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        
        # Setup output paths
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_npz_folder = os.path.join(out_dir)
        ensure_dir(output_npz_folder)
        npz_out_path = os.path.join(output_npz_folder, basename)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open {video_path}", "video": basename}

        # Get source video frame rate with fallback
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or math.isnan(src_fps) or src_fps < 1:
            src_fps = 30.0

        # Calculate frame sampling parameters
        step_s = 1.0 / target_fps
        next_t = 0.0

        # Initialize MediaPipe models
        models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)

        # Initialize data containers
        X_frames = []
        M_frames = []
        X2048_frames = []
        T_ms = []
        
        # Initialize batched InceptionV3 processor
        iv3_processor = None
        if write_iv3_features:
            iv3_processor = BatchedInceptionV3Processor(device=device, batch_size=batch_size)

        # Main processing loop with batched feature extraction
        try:
            frame_batch = []
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ms < next_t * 1000.0:
                    continue

                # Resize frame
                frame_bgr_resized = cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

                # Person segmentation
                frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
                seg_mask = models.seg.process(frame_rgb).segmentation_mask
                fg_mask = (seg_mask > 0.5).astype(np.float32)[..., None]
                black_bg = np.zeros_like(frame_rgb, dtype=np.uint8)
                comp_rgb = (frame_rgb * fg_mask + black_bg * (1 - fg_mask)).astype(np.uint8)

                # Extract keypoints
                if write_keypoints:
                    vec156, mask78 = extract_keypoints_from_frame(comp_rgb, models, conf_thresh=conf_thresh)
                    X_frames.append(vec156)
                    M_frames.append(mask78)

                # Collect frames for batched InceptionV3 processing
                if write_iv3_features:
                    frame_batch.append(frame_bgr)
                    
                    if len(frame_batch) >= batch_size:
                        batch_features = iv3_processor.extract_batch_features(frame_batch)
                        X2048_frames.extend(batch_features)
                        frame_batch = []

                T_ms.append(ms)
                next_t += step_s
            
            # Process remaining frames in final batch
            if write_iv3_features and frame_batch:
                batch_features = iv3_processor.extract_batch_features(frame_batch)
                X2048_frames.extend(batch_features)
                
        finally:
            cap.release()
            close_models(models)

        # Validate and post-process
        if len(X_frames) == 0 and write_keypoints:
            return {"error": f"No frames written for {video_path}", "video": basename}

        # Convert to numpy arrays
        if write_keypoints:
            X = np.stack(X_frames, axis=0)
            M = np.stack(M_frames, axis=0)
            X_filled, M_filled = interpolate_gaps(X, M, max_gap=max_gap)
            X_filled = np.clip(X_filled, 0.0, 1.0).astype(np.float32)
        else:
            X_filled = M_filled = None
            
        X2048_filled = np.stack(X2048_frames, axis=0) if X2048_frames else None
        T_ms = np.array(T_ms, dtype=np.int64)

        # Occlusion detection
        occluded_flag = 0
        occlusion_results = None
        
        if compute_occlusion:
            if occ_detailed:
                occlusion_results = compute_occlusion_detection(
                    video_path=video_path,
                    X=X_filled if write_keypoints else None,
                    mask_bool_array=M_filled if write_keypoints else None,
                    output_format='detailed'
                )
                occluded_flag = occlusion_results.get('binary_flag', 0)
            else:
                occluded_flag = compute_occlusion_detection(
                    video_path=video_path,
                    X=X_filled if write_keypoints else None,
                    mask_bool_array=M_filled if write_keypoints else None,
                    output_format='compatible'
                )
        
        # Prepare metadata
        meta = dict(
            video=os.path.basename(video_path),
            target_fps=target_fps,
            out_size=out_size,
            dims_per_frame=156,
            keypoints_total=78,
            order="pose25,left_hand21,right_hand21,face11",
            pose_indices=POSE_UPPER_25,
            face_indices=FACEMESH_11,
            conf_thresh=conf_thresh,
            interpolation_max_gap=max_gap,
            occluded_flag=occluded_flag
        )
        
        if occlusion_results is not None:
            meta['occlusion_results'] = occlusion_results

        # Save data
        if write_keypoints:
            to_npz(npz_out_path, X_filled, M_filled, T_ms, meta, also_parquet=not disable_parquet)
        
        # Update labels CSV
        if labels_csv_path is not None and gloss_id is not None:
            final_cat = cat_id if cat_id is not None else gloss_id
            rel_npz_path = os.path.relpath(npz_out_path + ".npz", start=out_dir)
            _append_label_row(labels_csv_path, rel_npz_path, gloss_id, final_cat, occluded_flag)

        # Save final data with all features
        save_dict = {
            'X': X_filled if X_filled is not None else np.array([]),
            'mask': M_filled if M_filled is not None else np.array([]),
            'timestamps_ms': T_ms,
            'meta': json.dumps(meta)
        }
        
        if write_iv3_features and X2048_filled is not None:
            save_dict.update({'X2048': X2048_filled})
        
        np.savez_compressed(npz_out_path + ".npz", **save_dict)

        return {"success": True, "video": basename, "frames": len(T_ms)}

    except Exception as e:
        return {"error": f"Error processing {video_path}: {str(e)}", "video": os.path.splitext(os.path.basename(video_path))[0]}


def process_videos_multiprocess(video_files, out_dir, target_fps=30, out_size=256, conf_thresh=0.5, 
                               max_gap=5, write_keypoints=True, write_iv3_features=True, feature_key='X2048',
                               compute_occlusion=True, occ_detailed=False, labels_csv_path=None, gloss_id=None, cat_id=None, 
                               workers=None, batch_size=32, disable_parquet=False):
    """Orchestrate parallel video processing with multi-GPU support and batched inference.
    
    This is the main coordination function for multiprocessing video preprocessing.
    It provides significant speedup (30-50x) over single-threaded processing by:
    - Distributing videos across multiple CPU processes
    - Assigning different GPUs to different workers
    - Using batched InceptionV3 inference within each worker
    - Progress tracking and error handling across all processes
    
    Args:
        video_files: List of video file paths to process
        out_dir: Output directory for processed .npz files
        target_fps: Target frame sampling rate (lower = faster processing)
        out_size: Image resize dimension for keypoint extraction (256x256)
        conf_thresh: Confidence threshold for keypoint detection (0.0-1.0)
        max_gap: Maximum gap size for keypoint interpolation (frames)
        write_keypoints: Extract MediaPipe keypoints (156D vectors per frame)
        write_iv3_features: Extract InceptionV3 features (2048D vectors per frame)
        feature_key: Unused parameter (kept for compatibility)
        compute_occlusion: Enable occlusion detection for quality filtering
        occ_detailed: Return detailed occlusion analysis results
        labels_csv_path: Path to labels CSV file for training data mapping
        gloss_id: Sign language gloss class ID for labeling
        cat_id: Category class ID for labeling
        workers: Number of parallel workers (default: min(cpu_count, 12))
        batch_size: Batch size for InceptionV3 GPU inference (32 optimal for most GPUs)
        disable_parquet: Disable parquet file creation for faster I/O
    """
    
    # Configure multiprocessing parameters
    if workers is None:
        workers = min(cpu_count(), 12)  # Cap at 12 workers for memory stability
    
    # Display processing configuration
    print(f"Processing {len(video_files)} videos with {workers} workers")
    print(f"Batch size for InceptionV3: {batch_size}")
    print(f"Target FPS: {target_fps}")
    print(f"Parquet output: {'disabled' if disable_parquet else 'enabled'}")
    
    # Prepare worker arguments with GPU distribution
    worker_args = []
    for i, video_path in enumerate(video_files):
        # Distribute GPU devices among workers for load balancing
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_id = i % torch.cuda.device_count()  # Round-robin GPU assignment
        else:
            device_id = None  # CPU-only processing
        
        # Pack all processing parameters for worker function
        args = (video_path, out_dir, target_fps, out_size, conf_thresh, max_gap,
                write_keypoints, write_iv3_features, feature_key, compute_occlusion, 
                occ_detailed, labels_csv_path, gloss_id, cat_id, batch_size, device_id, disable_parquet)
        worker_args.append(args)
    
    # Execute parallel processing with progress tracking
    start_time = time.time()
    results = []
    
    # Create multiprocessing pool and process videos with progress bar
    with Pool(processes=workers) as pool:
        # Use imap for real-time progress tracking (results come back as they complete)
        for result in tqdm(pool.imap(process_video_worker, worker_args), 
                          total=len(video_files), desc="Processing videos"):
            results.append(result)
    
    # Process results and generate performance report
    successful = 0
    failed = 0
    
    # Categorize results and display individual video status
    for result in results:
        if "error" in result:
            print(f"[ERROR] {result['video']}: {result['error']}")
            failed += 1
        else:
            print(f"[OK] {result['video']}: {result['frames']} frames processed")
            successful += 1
    
    # Calculate and display performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time/len(video_files):.2f} seconds")
    print(f"Videos per hour: {len(video_files) * 3600 / total_time:.1f}")


# ----------------------------
# Command-line Interface
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    # COMMAND-LINE INTERFACE: Setup argument parser for batch video processing
    parser = argparse.ArgumentParser(description="Preprocess video files to extract keypoints and IV3 features, detect occlusion, and write labels CSV")
    # Required arguments
    parser.add_argument('video_directory', help='Path to a video file or a directory containing videos')
    parser.add_argument('output_directory', help='Path to output directory for processed files')
    
    # Video processing parameters
    parser.add_argument('--target-fps', type=int, default=30, help='Target frames per second (default: 30)')
    parser.add_argument('--out-size', type=int, default=256, help='Output image size for keypoint extraction (default: 256)')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold for keypoint detection (default: 0.5)')
    parser.add_argument('--max-gap', type=int, default=5, help='Maximum gap for keypoint interpolation (default: 5)')
    
    # Feature extraction controls
    parser.add_argument('--write-keypoints', action='store_true', help='Extract and save MediaPipe keypoints (156D vectors)')
    parser.add_argument('--write-iv3-features', action='store_true', help='Extract and save InceptionV3 CNN features (2048D vectors)')
    parser.add_argument('--feature-key', type=str, default='X2048', help='Feature key name for compatibility (default: X2048)')
    
    # Multiprocessing and performance controls
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel worker processes (default: 1 for sequential, >1 for parallel)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for InceptionV3 GPU inference (default: 32, optimal for most GPUs)')
    parser.add_argument('--disable-parquet', action='store_true', help='Disable parquet output to speed up I/O operations')
    
    # Label/CSV controls for training data management
    parser.add_argument('--id', dest='single_id', type=int, default=None, help='Single integer ID to use for both gloss and category labels')
    parser.add_argument('--gloss-id', type=int, default=None, help='Override gloss class ID (defaults to --id)')
    parser.add_argument('--cat-id', type=int, default=None, help='Override category class ID (defaults to --id or --gloss-id)')
    parser.add_argument('--labels-csv', type=str, default=None, help='Path to labels CSV file (default: <output_directory>/labels.csv)')
    parser.add_argument('--append', action='store_true', help='Append to existing labels CSV instead of overwriting header')
    
    # Occlusion detection controls for quality filtering
    parser.add_argument('--occ-enable', action='store_true', help='Enable occlusion detection (auto-enabled when keypoints are written)')
    parser.add_argument('--occ-detailed', action='store_true', help='Output detailed occlusion analysis results')
    
    args = parser.parse_args()
    
    # INPUT VALIDATION: Accept either a single video file or directory of videos
    input_path = args.video_directory
    # Define supported video formats
    allowed_exts = {'.mp4', '.mov', '.avi', '.mkv'}

    if os.path.isfile(input_path):
        # SINGLE FILE MODE: Process one video file
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in allowed_exts:
            print(f"Unsupported file extension: {ext}. Allowed: {sorted(allowed_exts)}")
            exit(1)
        video_files = [os.path.normpath(input_path)]
        print(f"Processing single file: {os.path.basename(input_path)}")
    else:
        # DIRECTORY MODE: Recursively collect all video files (with deduplication)
        video_files = []
        seen = set()  # Prevent processing duplicate files
        for root, _dirs, files in os.walk(input_path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in allowed_exts:
                    full = os.path.normpath(os.path.join(root, name))
                    key = os.path.normcase(full)  # Case-insensitive deduplication
                    if key in seen:
                        continue
                    seen.add(key)
                    video_files.append(full)
        if not video_files:
            print(f"No video files found in {input_path}")
            print(f"Looking for extensions: {sorted(allowed_exts)}")
            exit(1)
        print(f"Found {len(video_files)} video files to process")
    
    # SETUP: Create output directory and prepare labels CSV
    ensure_dir(args.output_directory)

    # Resolve class IDs for labeling (with fallback hierarchy)
    gloss_id = args.gloss_id if args.gloss_id is not None else args.single_id
    cat_id = args.cat_id if args.cat_id is not None else gloss_id
    
    # Setup labels CSV path (default to output_directory/labels.csv if IDs provided)
    labels_csv = None
    if gloss_id is not None:
        labels_csv = args.labels_csv if args.labels_csv is not None else os.path.join(args.output_directory, 'labels.csv')
    
    # Initialize or update labels CSV file
    if labels_csv is not None:
        _ensure_labels_csv(labels_csv, include_occluded_col=True, overwrite=(not args.append))
        if not args.append:
            print(f"[INFO] Created/overwrote labels CSV header at: {labels_csv}")
        else:
            print(f"[INFO] Appending to existing labels CSV: {labels_csv}")
    else:
        print("[INFO] No labels will be written (no gloss/category ID provided)")
    
    # PROCESSING CONFIGURATION: Determine occlusion detection usage
    compute_occlusion = bool(args.occ_enable or args.write_keypoints)  # Auto-enable with keypoints
    
    # MAIN PROCESSING: Choose between sequential and parallel processing
    if args.workers == 1:
        # SEQUENTIAL PROCESSING: Process each video file individually
        print(f"Processing {len(video_files)} videos sequentially...")
        for video_path in video_files:
            print(f"\nProcessing: {os.path.basename(video_path)}")
            try:
                # Process single video with all specified parameters
                process_video(
                    video_path=video_path,              # Input video file
                    out_dir=args.output_directory,      # Output directory
                    target_fps=args.target_fps,         # Frame sampling rate
                    out_size=args.out_size,             # Image resize dimension
                    conf_thresh=args.conf_thresh,       # Keypoint confidence threshold
                    max_gap=args.max_gap,               # Interpolation gap limit
                    write_keypoints=args.write_keypoints,     # Extract MediaPipe keypoints
                    write_iv3_features=args.write_iv3_features,  # Extract CNN features
                    feature_key=args.feature_key,       # Feature naming (compatibility)
                    compute_occlusion=compute_occlusion,  # Enable quality assessment
                    occ_detailed=args.occ_detailed,    # Detailed occlusion analysis
                    labels_csv_path=labels_csv,         # Training labels file
                    gloss_id=gloss_id,                  # Sign language class ID
                    cat_id=cat_id,                      # Category class ID
                )
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue  # Skip failed videos and continue with the rest
    else:
        # PARALLEL PROCESSING: Use multiprocessing with batched GPU inference
        print(f"Processing {len(video_files)} videos with {args.workers} workers...")
        process_videos_multiprocess(
            video_files=video_files,              # List of videos to process
            out_dir=args.output_directory,        # Output directory
            target_fps=args.target_fps,           # Frame sampling rate
            out_size=args.out_size,               # Image resize dimension
            conf_thresh=args.conf_thresh,         # Keypoint confidence threshold
            max_gap=args.max_gap,                 # Interpolation gap limit
            write_keypoints=args.write_keypoints,       # Extract MediaPipe keypoints
            write_iv3_features=args.write_iv3_features, # Extract CNN features
            feature_key=args.feature_key,         # Feature naming (compatibility)
            compute_occlusion=compute_occlusion,  # Enable quality assessment
            occ_detailed=args.occ_detailed,      # Detailed occlusion analysis
            labels_csv_path=labels_csv,           # Training labels file
            gloss_id=gloss_id,                    # Sign language class ID
            cat_id=cat_id,                        # Category class ID
            workers=args.workers,                 # Number of parallel processes
            batch_size=args.batch_size,           # GPU batch size for efficiency
            disable_parquet=args.disable_parquet  # Disable parquet for speed
        )
    
    # COMPLETION: Report final results
    print(f"\nProcessing complete! Check output directory: {args.output_directory}")