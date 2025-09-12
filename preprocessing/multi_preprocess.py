import os, sys, glob, json, math, argparse, time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count, Manager, set_start_method
from functools import partial
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Allow running both as a module (-m) and as a script (python preprocessing/multi_preprocess.py)
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocessing.iv3_features import extract_iv3_features  # InceptionV3 (torchvision) feature extractor
from preprocessing.occlusion_detection import compute_occlusion_flag_from_keypoints
from preprocessing.keypoints_features import (
    POSE_UPPER_25,
    N_HAND,
    FACEMESH_11,
    extract_keypoints_from_frame,
    interpolate_gaps,
    xy_from_landmark,
    create_models,
    close_models,
    MPModels,
)

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_npz(out_path, X, mask, timestamps_ms, meta, also_parquet=True):
    """Write keys (`X`, `mask`, `timestamps_ms`, `meta`) to `<out_path>.npz`.

    - X: [T,156] float32
    - mask: [T,78] bool
    - timestamps_ms: [T] int64
    - meta: JSON string
    Optionally writes a `.parquet` for quick inspection.
    """
    np.savez_compressed(out_path + ".npz", X=X, mask=mask, timestamps_ms=timestamps_ms, meta=json.dumps(meta))
    if also_parquet:
        try:
            # flatten per-frame records for quick debugging in spreadsheets
            df = pd.DataFrame(X)
            # ensure string column names to avoid parquet mixed-type warning
            df.columns = df.columns.astype(str)
            df["t_ms"] = timestamps_ms
            # store mask as a compact string of 0/1 for quick inspection
            df["mask_bits"] = ["".join("1" if b else "0" for b in row) for row in mask]
            df.to_parquet(out_path + ".parquet")
        except Exception as e:
            print(f"[WARN] Could not save parquet file: {e}")
            print("[INFO] Install pyarrow or fastparquet for parquet support: pip install pyarrow")

# ----------------------------
# Batched InceptionV3 Processing
# ----------------------------
class BatchedInceptionV3Processor:
    """Batched InceptionV3 feature extraction for GPU optimization."""
    
    def __init__(self, device=None, batch_size=32):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._init_model()
    
    def _init_model(self):
        """Initialize the InceptionV3 model."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        import torch.nn as nn
        
        self.weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=self.weights)
        self.model.aux_logits = False
        self.model.fc = nn.Identity()  # return (N, 2048)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        
        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def preprocess_frame(self, frame_bgr, image_size=(299, 299)):
        """Preprocess a single BGR frame for InceptionV3."""
        # Convert BGR → RGB and resize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, image_size)
        
        # Convert to torch tensor in [0, 1] and normalize
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor
    
    def extract_batch_features(self, frames_bgr, image_size=(299, 299)):
        """Extract features for a batch of frames."""
        if not frames_bgr:
            return np.array([])
        
        # Preprocess all frames
        tensors = [self.preprocess_frame(frame, image_size) for frame in frames_bgr]
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)  # [batch_size, 2048]
        
        return features.cpu().numpy()

# ----------------------------
# MediaPipe wrappers
# ----------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def _ensure_labels_csv(path, include_occluded_col=True, overwrite=False):
    """Create or upgrade a labels CSV.

    If overwrite=True, rewrites header. If file exists and is missing 'occluded', add it with default 0.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if overwrite or not os.path.exists(path):
        cols = ["file", "gloss", "cat"] + (["occluded"] if include_occluded_col else [])
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    # Upgrade path: ensure occluded column exists
    try:
        df = pd.read_csv(path)
        if include_occluded_col and "occluded" not in (df.columns.tolist() if df is not None else []):
            df["occluded"] = 0
            df.to_csv(path, index=False)
    except Exception as e:
        print(f"[WARN] Could not inspect/upgrade labels csv '{path}': {e}")

def _append_label_row(path, file_entry, gloss_id, cat_id, occluded_flag=0):
    """Append one row (file,gloss,cat,occluded) into labels CSV.

    'file' can be a basename or a relative subpath and may include extension.
    """
    try:
        new_row = {"file": str(file_entry), "gloss": int(gloss_id), "cat": int(cat_id)}
        # Add occluded if the CSV header contains it
        try:
            with open(path, "r", newline="") as fh:
                header = fh.readline().strip().split(",") if fh else []
        except Exception:
            header = []
        if "occluded" in header:
            new_row["occluded"] = int(occluded_flag)
        df = pd.DataFrame([new_row])
        df.to_csv(path, mode="a", index=False, header=False)
    except Exception as e:
        print(f"[WARN] Failed to append to labels csv '{path}': {e}")

def process_video_worker(args):
    """Worker function for processing a single video in multiprocessing."""
    (video_path, out_dir, target_fps, out_size, conf_thresh, max_gap, 
     write_keypoints, write_iv3_features, feature_key, occ_vis_thresh, 
     occ_frame_prop, occ_min_run, compute_occlusion, labels_csv_path, 
     gloss_id, cat_id, batch_size, device_id) = args
    
    try:
        # Set CUDA device for this worker if specified
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_npz_folder = os.path.join(out_dir)
        ensure_dir(output_npz_folder)
        npz_out_path = os.path.join(output_npz_folder, basename)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open {video_path}", "video": basename}

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or math.isnan(src_fps) or src_fps < 1:
            src_fps = 30.0  # fallback

        step_s = 1.0 / target_fps
        next_t = 0.0

        models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)

        X_frames = []
        M_frames = []
        X2048_frames = []
        T_ms = []
        
        # Initialize batched IV3 processor if needed
        iv3_processor = None
        if write_iv3_features:
            iv3_processor = BatchedInceptionV3Processor(device=device, batch_size=batch_size)

        t0 = cap.get(cv2.CAP_PROP_POS_MSEC)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            frame_batch = []
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ms < next_t * 1000.0:
                    continue

                frame_bgr_resized = cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
                seg_mask = models.seg.process(frame_rgb).segmentation_mask
                fg_mask = (seg_mask > 0.5).astype(np.float32)[..., None]
                black_bg = np.zeros_like(frame_rgb, dtype=np.uint8)
                comp_rgb = (frame_rgb * fg_mask + black_bg * (1 - fg_mask)).astype(np.uint8)

                if write_keypoints:
                    vec156, mask78 = extract_keypoints_from_frame(comp_rgb, models, conf_thresh=conf_thresh)
                    X_frames.append(vec156)
                    M_frames.append(mask78)

                if write_iv3_features:
                    # Collect frames for batch processing
                    frame_batch.append(frame_bgr)
                    
                    # Process batch when it reaches batch_size or at the end
                    if len(frame_batch) >= batch_size:
                        batch_features = iv3_processor.extract_batch_features(frame_batch)
                        X2048_frames.extend(batch_features)
                        frame_batch = []

                T_ms.append(ms)
                next_t += step_s
            
            # Process remaining frames in the batch
            if write_iv3_features and frame_batch:
                batch_features = iv3_processor.extract_batch_features(frame_batch)
                X2048_frames.extend(batch_features)
                
        finally:
            cap.release()
            close_models(models)

        if len(X_frames) == 0:
            return {"error": f"No frames written for {video_path}", "video": basename}

        X = np.stack(X_frames, axis=0)
        M = np.stack(M_frames, axis=0)
        X2048 = np.stack(X2048_frames, axis=0) if X2048_frames else np.array([])
        T_ms = np.array(T_ms, dtype=np.int64)

        if write_iv3_features and X2048.size > 0:
            assert X.shape[0] == X2048.shape[0], f"Mismatch in T (frames) between X and X2048: {X.shape[0]} vs {X2048.shape[0]}"

        X_filled, M_filled = interpolate_gaps(X, M, max_gap=max_gap)
        # Ensure coordinate bounds
        X_filled = np.clip(X_filled, 0.0, 1.0).astype(np.float32)
        # Do not interpolate CNN features using keypoint visibility mask; keep raw temporal values
        X2048_filled = X2048

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
            interpolation_max_gap=max_gap
        )

        to_npz(npz_out_path, X_filled, M_filled, T_ms, meta, also_parquet=True)

        # Occlusion detection and CSV update
        occluded_flag = 0
        if compute_occlusion and write_keypoints:
            occluded_flag = compute_occlusion_flag_from_keypoints(
                X_filled,
                M_filled,
                frame_prop_threshold=occ_frame_prop,
                min_consecutive_occ_frames=occ_min_run,
                visibility_fallback_threshold=occ_vis_thresh,
            )
        # Append to labels CSV if requested and ids are provided
        if labels_csv_path is not None and gloss_id is not None:
            final_cat = cat_id if cat_id is not None else gloss_id
            # store file path relative to out_dir (e.g., '0/clip.npz')
            rel_npz_path = os.path.relpath(npz_out_path + ".npz", start=out_dir)
            _append_label_row(labels_csv_path, rel_npz_path, gloss_id, final_cat, occluded_flag)

        # Save X2048 features into the same .npz
        if write_iv3_features and X2048_filled.size > 0:
            with np.load(npz_out_path + ".npz", allow_pickle=True) as data:
                meta_data = data['meta']
                np.savez_compressed(npz_out_path + ".npz", X=X_filled, X2048=X2048_filled, mask=M_filled, timestamps_ms=T_ms, meta=meta_data)

        return {"success": True, "video": basename, "frames": len(X_frames)}

    except Exception as e:
        return {"error": f"Error processing {video_path}: {str(e)}", "video": os.path.splitext(os.path.basename(video_path))[0]}

def process_videos_multiprocess(video_files, out_dir, target_fps=30, out_size=256, conf_thresh=0.5, 
                               max_gap=5, write_keypoints=True, write_iv3_features=True, feature_key='X2048',
                               occ_vis_thresh=0.6, occ_frame_prop=0.4, occ_min_run=15, compute_occlusion=True, 
                               labels_csv_path=None, gloss_id=None, cat_id=None, workers=None, batch_size=32, 
                               disable_parquet=False):
    """Process multiple videos using multiprocessing with batched GPU inference."""
    
    if workers is None:
        workers = min(cpu_count(), 12)  # Cap at 12 workers for stability
    
    print(f"Processing {len(video_files)} videos with {workers} workers")
    print(f"Batch size for InceptionV3: {batch_size}")
    print(f"Target FPS: {target_fps} (recommended: 15 for faster processing)")
    print(f"Parquet output: {'disabled' if disable_parquet else 'enabled'}")
    
    # Prepare arguments for each worker
    worker_args = []
    for i, video_path in enumerate(video_files):
        # Distribute GPU devices among workers if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_id = i % torch.cuda.device_count()
        else:
            device_id = None
        
        args = (video_path, out_dir, target_fps, out_size, conf_thresh, max_gap,
                write_keypoints, write_iv3_features, feature_key, occ_vis_thresh,
                occ_frame_prop, occ_min_run, compute_occlusion, labels_csv_path,
                gloss_id, cat_id, batch_size, device_id)
        worker_args.append(args)
    
    # Process videos in parallel
    start_time = time.time()
    results = []
    
    with Pool(processes=workers) as pool:
        # Use imap for progress tracking
        for result in tqdm(pool.imap(process_video_worker, worker_args), 
                          total=len(video_files), desc="Processing videos"):
            results.append(result)
    
    # Process results
    successful = 0
    failed = 0
    
    for result in results:
        if "error" in result:
            print(f"[ERROR] {result['video']}: {result['error']}")
            failed += 1
        else:
            print(f"[OK] {result['video']}: {result['frames']} frames processed")
            successful += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time/len(video_files):.2f} seconds")
    print(f"Videos per hour: {len(video_files) * 3600 / total_time:.1f}")

# ----------------------------
# Command-line interface
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-process video preprocessing with batched GPU inference")
    parser.add_argument('video_directory', help='Path to a video file or a directory containing videos')
    parser.add_argument('output_directory', help='Path to output directory for processed files')
    parser.add_argument('--target-fps', type=int, default=15, help='Target frames per second (default: 15, recommended for speed)')
    parser.add_argument('--out-size', type=int, default=256, help='Output image size (default: 256)')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--max-gap', type=int, default=5, help='Maximum gap for interpolation (default: 5)')
    parser.add_argument('--write-keypoints', action='store_true', help='Extract and save keypoints')
    parser.add_argument('--write-iv3-features', action='store_true', help='Extract and save IV3 features')
    parser.add_argument('--feature-key', type=str, default='X2048', help='Feature key name (default: X2048)')
    
    # Multiprocessing controls
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: min(cpu_count, 12))')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for InceptionV3 GPU inference (default: 32)')
    parser.add_argument('--disable-parquet', action='store_true', help='Disable parquet output to speed up I/O')
    
    # Label/CSV controls
    parser.add_argument('--id', dest='single_id', type=int, default=None, help='Single integer id to use for both gloss and cat')
    parser.add_argument('--gloss-id', type=int, default=None, help='Override gloss id (defaults to --id)')
    parser.add_argument('--cat-id', type=int, default=None, help='Override category id (defaults to --id or --gloss-id)')
    parser.add_argument('--labels-csv', type=str, default=None, help='Path to labels CSV to write (default: <output_directory>/labels.csv)')
    parser.add_argument('--append', action='store_true', help='Append to labels CSV instead of overwriting header before this run')
    
    # Occlusion controls
    parser.add_argument('--occ-enable', action='store_true', help='Enable occlusion detection from keypoint visibility (defaults to enabled when keypoints are written)')
    parser.add_argument('--occ-vis-thresh', type=float, default=0.6, help='Frame visible fraction threshold (default: 0.6)')
    parser.add_argument('--occ-frame-prop', type=float, default=0.4, help='Clip occluded if proportion of occluded frames >= this (default: 0.4)')
    parser.add_argument('--occ-min-run', type=int, default=15, help='Clip occluded if there exists a run of occluded frames >= this (default: 15)')
    
    args = parser.parse_args()
    
    # Accept either a single file or a directory
    input_path = args.video_directory
    # Normalize allowed extensions (case-insensitive)
    allowed_exts = {'.mp4', '.mov', '.avi', '.mkv'}

    if os.path.isfile(input_path):
        # Single-file mode
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in allowed_exts:
            print(f"Unsupported file extension: {ext}. Allowed: {sorted(allowed_exts)}")
            exit(1)
        video_files = [os.path.normpath(input_path)]
        print(f"Processing single file: {os.path.basename(input_path)}")
    else:
        # Directory mode: collect all videos (deduplicated)
        video_files = []
        seen = set()
        for root, _dirs, files in os.walk(input_path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in allowed_exts:
                    full = os.path.normpath(os.path.join(root, name))
                    key = os.path.normcase(full)
                    if key in seen:
                        continue
                    seen.add(key)
                    video_files.append(full)
        if not video_files:
            print(f"No video files found in {input_path}")
            print(f"Looking for extensions: {sorted(allowed_exts)}")
            exit(1)
        print(f"Found {len(video_files)} video files to process")
    
    # Create output directory
    ensure_dir(args.output_directory)

    # Prepare labels CSV if ids are provided
    gloss_id = args.gloss_id if args.gloss_id is not None else args.single_id
    cat_id = args.cat_id if args.cat_id is not None else gloss_id
    # Default label path to the output directory if ids are provided and no path was specified
    labels_csv = None
    if gloss_id is not None:
        labels_csv = args.labels_csv if args.labels_csv is not None else os.path.join(args.output_directory, 'labels.csv')
    if labels_csv is not None:
        _ensure_labels_csv(labels_csv, include_occluded_col=True, overwrite=(not args.append))
        if not args.append:
            print(f"[INFO] Overwrote labels CSV header at: {labels_csv}")
        else:
            print(f"[INFO] Appending to labels CSV: {labels_csv}")
    else:
        print("[INFO] No labels will be written because no gloss/cat id was provided")
    
    # Determine occlusion usage
    compute_occlusion = bool(args.occ_enable or args.write_keypoints)
    
    # Process videos with multiprocessing
    process_videos_multiprocess(
        video_files=video_files,
        out_dir=args.output_directory,
        target_fps=args.target_fps,
        out_size=args.out_size,
        conf_thresh=args.conf_thresh,
        max_gap=args.max_gap,
        write_keypoints=args.write_keypoints,
        write_iv3_features=args.write_iv3_features,
        feature_key=args.feature_key,
        occ_vis_thresh=args.occ_vis_thresh,
        occ_frame_prop=args.occ_frame_prop,
        occ_min_run=args.occ_min_run,
        compute_occlusion=compute_occlusion,
        labels_csv_path=labels_csv,
        gloss_id=gloss_id,
        cat_id=cat_id,
        workers=args.workers,
        batch_size=args.batch_size,
        disable_parquet=args.disable_parquet
    )
    
    print(f"\nProcessing complete! Check output directory: {args.output_directory}")
