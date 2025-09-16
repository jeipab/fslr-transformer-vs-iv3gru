"""
InceptionV3 feature extraction utilities (PyTorch/torchvision).

What this provides:
- Single-frame feature extraction that returns a 2048-D ImageNet embedding.
- A simple video processor that can write both keypoints (`X`) and IV3 features (`X2048`).

Key facts:
- Input frame format: OpenCV BGR image.
- Output feature: NumPy array of shape (2048,) with dtype float32.
- Matches the training stack (torchvision InceptionV3, global average pooling).
"""
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
import os
import pandas as pd
import json

from ..extractors.keypoints_features import (
    extract_keypoints_from_frame,
    interpolate_gaps,
    POSE_UPPER_25,
    FACEMESH_11,
    create_models,
    close_models,
    MPModels,
)

# Initialize a single global InceptionV3 backbone (ImageNet weights).
_iv3_weights = Inception_V3_Weights.IMAGENET1K_V1
_iv3_model = inception_v3(weights=_iv3_weights)
_iv3_model.aux_logits = False
_iv3_model.fc = nn.Identity()  # return (N, 2048)
_iv3_model.eval()
for p in _iv3_model.parameters():
    p.requires_grad = False

# ImageNet normalization constants (standard values) - will be moved to device as needed
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def extract_iv3_features(frame_bgr, image_size=(299, 299), device=None):
    """Extract a 2048-D InceptionV3 feature for a single BGR frame.

    Args:
        frame_bgr: OpenCV BGR image (H, W, 3), uint8 in [0, 255].
        image_size: Spatial size used for InceptionV3 (default: 299x299).
        device: Optional torch.device; CUDA is used if available.

    Returns:
        np.ndarray of shape (2048,) and dtype float32.
    """
    if device is None:
        device = torch.device("cpu")

    # Convert BGR → RGB and resize to the expected InceptionV3 input size.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(frame_rgb, image_size)

    # Convert to torch tensor in [0, 1] and normalize using ImageNet stats.
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # [1, 3, 299, 299]
    
    # Move normalization constants to the same device
    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        # Ensure model is on the correct device
        model_on_device = _iv3_model.to(device)
        feats = model_on_device(tensor)  # [1, 2048]
    return feats.squeeze(0).cpu().numpy()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_npz(out_path, X, X2048, mask, timestamps_ms, meta, also_parquet=True):
    """Write a single clip to `.npz` (and optional `.parquet`).

    Saves keys:
    - X: [T,156] float32
    - X2048: [T,2048] float32
    - mask: [T,78] bool
    - timestamps_ms: [T] int64
    - meta: JSON string
    """
    np.savez_compressed(out_path + ".npz", X=X, X2048=X2048, mask=mask, timestamps_ms=timestamps_ms, meta=json.dumps(meta))
    if also_parquet:
        try:
            # flatten per-frame records for quick debugging in spreadsheets
            df = pd.DataFrame(X)
            df["t_ms"] = timestamps_ms
            df["mask_bits"] = ["".join("1" if b else "0" for b in row) for row in mask]
            df.to_parquet(out_path + ".parquet")
        except Exception as e:
            print(f"[WARN] Could not save parquet file: {e}")
            print("[INFO] Install pyarrow or fastparquet for parquet support: pip install pyarrow")

def read_or_create_labels_csv(label_file):
    """Read `labels.csv` if present; otherwise create an empty one with header."""
    if os.path.exists(label_file):
        return pd.read_csv(label_file)
    else:
        # Create an empty dataframe and save it
        df = pd.DataFrame(columns=["file", "gloss", "cat"])
        df.to_csv(label_file, index=False)
        return df

def update_labels_csv(label_file, video_file, gloss, cat):
    """Append or update one row in `labels.csv` for the given clip."""
    df = read_or_create_labels_csv(label_file)
    new_row = pd.DataFrame({"file": [video_file], "gloss": [gloss], "cat": [cat]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(label_file, index=False)

def process_video(video_path, out_dir, label_file=None, target_fps=30, out_size=256, conf_thresh=0.5, max_gap=5, write_keypoints=True, write_iv3_features=True, feature_key='X2048', gloss=None, cat=None):
    """Process one video into a training-ready `.npz`.

    Extracts per-frame keypoints (`X` [T,156]) and/or IV3 features (`X2048` [T,2048]),
    plus `mask` [T,78], `timestamps_ms` [T], and `meta`.

    Args:
        video_path: Path to the input video file.
        out_dir: Output root directory (files are saved under `<out_dir>/0/`).
        label_file: Optional CSV to update (`file,gloss,cat`). Defaults to `<out_dir>/labels.csv`.
        target_fps: Sampling fps for frames.
        out_size: Side length used to resize frames for keypoint extraction.
        conf_thresh: Confidence/visibility threshold for keypoints.
        max_gap: Max gap (frames) to interpolate for missing keypoints.
        write_keypoints: If True, write `X` and `mask`.
        write_iv3_features: If True, write `X2048`.
        feature_key: Unused here (kept for compatibility).
        gloss: Optional gloss id (written to labels CSV if provided).
        cat: Optional category id (written to labels CSV if provided).

    Returns:
        None. Writes `<basename>.npz` (and `.parquet` if available) to disk.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_npz_folder = os.path.join(out_dir, '0')  # Assuming input vids are in '0' subfolder
    ensure_dir(output_npz_folder)
    npz_out_path = os.path.join(output_npz_folder, basename)
    
    # Set default label_file to be in the output directory
    if label_file is None:
        label_file = os.path.join(out_dir, "labels.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1:
        src_fps = 30.0  # fallback

    step_s = 1.0 / target_fps
    next_t = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create MediaPipe models if keypoints are needed
    models = None
    if write_keypoints:
        models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)

    X_frames = []
    M_frames = []
    X2048_frames = []
    T_ms = []

    t0 = cap.get(cv2.CAP_PROP_POS_MSEC)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms < next_t * 1000.0:
                continue

            frame_bgr_resized = cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

            # Extract features
            frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)

            if write_keypoints:
                # Extract keypoints
                vec156, mask78 = extract_keypoints_from_frame(frame_rgb, models, conf_thresh=conf_thresh)
                X_frames.append(vec156)
                M_frames.append(mask78)

            if write_iv3_features:
                # Extract IV3 features
                iv3_features = extract_iv3_features(frame_bgr, image_size=(299, 299), device=device)
                X2048_frames.append(iv3_features)

            T_ms.append(ms)
            next_t += step_s
    finally:
        cap.release()
        if models is not None:
            close_models(models)

    if len(X_frames) == 0:
        print(f"[WARN] No frames written for {video_path}")
        return

    X = np.stack(X_frames, axis=0)
    M = np.stack(M_frames, axis=0)
    X2048 = np.stack(X2048_frames, axis=0)
    T_ms = np.array(T_ms, dtype=np.int64)

    # Ensure alignment
    assert X.shape[0] == X2048.shape[0], f"Mismatch in T (frames) between X and X2048: {X.shape[0]} vs {X2048.shape[0]}"
    
    # Handle missing X2048
    if len(X2048_frames) == 0:
        print("[WARN] No IV3 features extracted.")

    X_filled, M_filled = interpolate_gaps(X, M, max_gap=max_gap)
    # Ensure coordinate bounds
    X_filled = np.clip(X_filled, 0.0, 1.0).astype(np.float32)
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
        interpolation_max_gap=max_gap,
        gloss=gloss,
        cat=cat
    )

    # Update labels.csv with gloss and category information
    if gloss and cat:
        update_labels_csv(label_file, basename, gloss, cat)

    # Write the output as .npz and .parquet
    to_npz(npz_out_path, X_filled, X2048_filled, M_filled, T_ms, meta, also_parquet=True)

    print(f"[OK] {basename}: frames={len(X_frames)} saved: {npz_out_path}.npz (+ .parquet)")

# CLI interface using argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing for Video Clips")
    parser.add_argument('--write-keypoints', action='store_true', help='Write keypoints to output')
    parser.add_argument('--write-iv3-features', action='store_true', help='Write IV3 features to output')
    parser.add_argument('--fps', type=int, default=30, help='Target frames per second')
    parser.add_argument('--image-size', type=int, default=256, help='Output image size for feature extraction')
    parser.add_argument('--feature-key', type=str, default='X2048', help='Which feature key to use')
    parser.add_argument('--gloss', type=str, help='Gloss (optional) for labeling')
    parser.add_argument('--cat', type=str, help='Category (optional) for labeling')
    parser.add_argument('--label-file', type=str, help='Path to the labels.csv file (default: output_dir/labels.csv)')
    parser.add_argument('video_path', type=str, help='Path to the video file to process')
    parser.add_argument('out_dir', type=str, help='Directory to save the processed output')
    
    args = parser.parse_args()
    
    # Set default label_file to be in the output directory if not specified
    if args.label_file is None:
        args.label_file = os.path.join(args.out_dir, "labels.csv")

    process_video(args.video_path, args.out_dir, label_file=args.label_file, target_fps=args.fps, out_size=args.image_size, write_keypoints=args.write_keypoints, write_iv3_features=args.write_iv3_features, feature_key=args.feature_key, gloss=args.gloss, cat=args.cat)