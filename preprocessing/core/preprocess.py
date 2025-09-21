"""
Video preprocessing pipeline for Filipino sign language recognition.

This module processes raw video files to extract:
- MediaPipe keypoints (pose, hands, face) → 156-dimensional vectors
- InceptionV3 CNN features → 2048-dimensional vectors  
- Occlusion detection flags
- Frame timestamps and metadata

Outputs are saved as compressed .npz files with optional .parquet files for inspection.

Usage:
- Single video:
    python preprocessing/preprocess.py video.mp4 output_dir --write-keypoints --write-iv3-features --id 12
- Directory of videos:
    python preprocessing/preprocess.py input_dir output_dir --write-keypoints --write-iv3-features --id 12
"""

# Standard library imports
import os, sys, glob, json, math, argparse, time  # File operations, system utilities, JSON handling
import warnings  # Warning control
from dataclasses import dataclass  # Data structure definitions

# Computer vision and numerical computing
import cv2  # OpenCV for video processing and image operations
import numpy as np  # Numerical arrays and mathematical operations
import pandas as pd  # Data manipulation and CSV handling
from tqdm import tqdm  # Progress bars for long operations

# Machine learning frameworks
import torch  # PyTorch for deep learning (InceptionV3 features)
import mediapipe as mp  # Google's MediaPipe for keypoint detection (pose, hands, face)

# Path setup: Allow running both as a module (-m) and as a script (python preprocessing/preprocess.py)
# This ensures imports work correctly regardless of how the script is executed
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Project-specific imports
from ..extractors.iv3_features import extract_iv3_features  # InceptionV3 CNN feature extraction (2048D vectors)
from ..core.occlusion_detection import compute_occlusion_flag_from_keypoints, compute_occlusion_detection  # Detect when keypoints are blocked/occluded
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
    
    # MAIN PROCESSING LOOP: Process each video file individually
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
    
    # COMPLETION: Report final results
    print(f"\nProcessing complete! Check output directory: {args.output_directory}")