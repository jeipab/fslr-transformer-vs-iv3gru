# --------------------------------------------------------
# IMPORTS: libraries and modules
# --------------------------------------------------------

import os, sys, glob, json, math, argparse, time
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# --------------------------------------------------------
# CONFIG: landmark index sets
# --------------------------------------------------------
# Define the indices for different parts of the body, hands, and face for keypoint extraction.
# Pose indices: we'll take 25 "upper-body" related indices from the 33 Pose landmarks.
# This set includes head (0..10), shoulders/arms (11..16), hand anchors (17..22), hips (23,24) = 25 total.

POSE_UPPER_25 = list(range(0, 11)) + list(range(11, 17)) + list(range(17, 23)) + [23, 24]

# Hands: 21 landmarks for each hand (MediaPipe Hands)
N_HAND = 21

# Face Mesh: choose 11 stable facial points (adjustable).
# Nose tip, eye corners (outer/inner), mouth corners, eyebrow inner points, chin.
# Sources and common choices: 1 (nose tip), 33 & 263 (outer eye corners), 133 & 362 (inner eye corners),
# 61 & 291 (mouth corners), 105 & 334 (brow inner), 199 (chin).
FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]  # last '4' is an extra nose base point for 11 total

# --------------------------------------------------------
# UTILITIES: helper functions
# --------------------------------------------------------

# ensure that directories exist, and create them if necessary
def ensure_dir(p):
     os.makedirs(p, exist_ok=True)

# for linear interpolation between two values a and b based on t
def lerp(a, b, t):
    return a + (b - a) * t

# interpolate gaps in the keypoints (for missing or occluded data)
def interpolate_gaps(X, mask, max_gap=5):
     """
     X: [T, D] float array (D=156)
     mask: [T, K] bool array (K=78) indicating which keypoints are valid (x,y pair shares a single mask)
     We interpolate per keypoint (pair) independently for gaps up to max_gap.
     """
     T = X.shape[0]
     K = mask.shape[1]
     X_out = X.copy()
     mask_out = mask.copy()

     for k in range(K):
          # columns for this keypoint
          xi = 2 * k
          yi = 2 * k + 1
          valid_idxs = np.where(mask[:, k])[0]

          if len(valid_idxs) == 0:
               continue

          # walk through consecutive valid ranges
          prev = valid_idxs[0]
          for vi in valid_idxs[1:]:
               if vi == prev + 1:
                    prev = vi
               continue
               # gap from prev+1 .. vi-1
               gap_start = prev + 1
               gap_end = vi - 1
               gap_len = gap_end - gap_start + 1
               if 1 <= gap_len <= max_gap:
                    x0, y0 = X_out[prev, xi], X_out[prev, yi]
                    x1, y1 = X_out[vi, xi], X_out     [vi, yi]
                    for t_idx, t_rel in enumerate     (range(gap_start, gap_end +   1), start=1):
                         t = t_idx / (gap_len + 1)
                         X_out[t_rel, xi] = lerp  (x0, x1, t)
                         X_out[t_rel, yi] = lerp  (y0, y1, t)
                         mask_out[t_rel, k] = True  # now filled
               prev = vi

     return X_out, mask_out

#  save processed data to npz and parquet files for quick inspection
def to_npz(out_path, X, mask, timestamps_ms, meta, also_parquet=True):
     np.savez_compressed(out_path + ".npz", X=X, mask=mask, timestamps_ms=timestamps_ms, meta=json.dumps(meta))
     if also_parquet:
          # flatten per-frame records for quick debugging in spreadsheets
          df = pd.DataFrame(X)
          df["t_ms"] = timestamps_ms
          # store mask as a compact string of 0/1 for quick inspection
          df["mask_bits"] = ["".join("1" if b else "0" for b in row) for row in mask]
          df.to_parquet(out_path + ".parquet")

# convert MediaPipe landmark coordinates to [0,1] scale
def xy_from_landmark(lm, w, h):
     # MediaPipe returns normalized x,y in [0,1]; keep in that scale but clip to [0,1].
     x = float(lm.x)
     y = float(lm.y)
     return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

# --------------------------------------------------------
# MEDIAPIPE MODELS INITIALIZATION
# --------------------------------------------------------

# create and initialize MediaPipe models for segmentation (background removal) and holistic (pose, hands, face) keypoints
mp_selfie = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic

@dataclass
class MPModels:
     seg: any
     hol: any

# Create and return MediaPipe models
def create_models(seg_model=1, detection_conf=0.5, tracking_conf=0.5):
     seg = mp_selfie.SelfieSegmentation(model_selection=seg_model)
     hol = mp_holistic.Holistic(
          model_complexity=1,
          smooth_landmarks=True,
          refine_face_landmarks=True,
          min_detection_confidence=detection_conf,
          min_tracking_confidence=tracking_conf
     )
     return MPModels(seg=seg, hol=hol)

# Close MediaPipe models
def close_models(models: MPModels):
     models.seg.close()
     models.hol.close()

# --------------------------------------------------------
# CORE KEYPOINT EXTRACTION from frame
# --------------------------------------------------------

def extract_keypoints_from_frame(img_rgb, models: MPModels, conf_thresh=0.5):
     """
     Returns:
          vec (156,), mask (78,)  -> concatenated [pose25, left21, right21, face11]
          also returns component flags for debugging
     """
     H, W, _ = img_rgb.shape

     # Run holistic (pose + hands + face landmarks)
     res = models.hol.process(img_rgb)

     # Prepare containers
     coords = []
     vis_mask = []

     # ---------------- POSE (25 keypoints) ----------------
     pose_present = res.pose_landmarks is not None
     pose_points = [ (0.0, 0.0) ] * len(POSE_UPPER_25)
     pose_mask = [False] * len(POSE_UPPER_25)

     if pose_present:
          ms = res.pose_landmarks.landmark
          for i, idx in enumerate(POSE_UPPER_25):
               lm = lms[idx]
               x, y = xy_from_landmark(lm, W, H)
               pose_points[i] = (x, y)
               # Use visibility (only Pose has per-landmark visibility)
               pose_mask[i] = (getattr(lm, "visibility", 0.0) or 0.0) >= conf_thresh

     # ---------------- HANDS (21 + 21 keypoints) ----------------
     # Holistic gives left_hand_landmarks and right_hand_landmarks distinctly
     def hand_block(hand_lms):
          pts = [ (0.0, 0.0) ] * N_HAND
          m = [False] * N_HAND
          if hand_lms is not None:
               for i, lm in enumerate(hand_lms.landmark[:N_HAND]):
                    x, y = xy_from_landmark(lm, W, H)
                    pts[i] = (x, y)
                    # No explicit visibility: treat detection as valid if present
                    m[i] = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)
          return pts, m

     left_pts, left_mask = hand_block(res.    left_hand_landmarks)
     right_pts, right_mask = hand_block(res.right_hand_landmarks)

     # ---------------- FACE (11 keypoints) ----------------
     face_points = [ (0.0, 0.0) ] * len(FACEMESH_11)
     face_mask = [False] * len(FACEMESH_11)
     if res.face_landmarks is not None:
          fl = res.face_landmarks.landmark
          for i, idx in enumerate(FACEMESH_11):
               lm = fl[idx]
               x, y = xy_from_landmark(lm, W, H)
               face_points[i] = (x, y)
               face_mask[i] = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)

     # ---------------- CONCATENATE ALL KEYPOINTS ----------------
     for block_pts, block_mask in [
          (pose_points, pose_mask),
          (left_pts, left_mask),
          (right_pts, right_mask),
          (face_points, face_mask)
     ]:
          for (x, y) in block_pts:
               coords.extend([x, y])
          vis_mask.extend(block_mask)

     return np.array(coords, dtype=np.float32), np.array(vis_mask, dtype=bool)

# --------------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------------

def process_video(video_path, out_dir, target_fps=30, out_size=256, conf_thresh=0.5, max_gap=5):
     basename = os.path.splitext(os.path.basename(video_path))[0]

     # OUTPUT folder for .npz (+ .parquet) files
     output_npz_folder = os.path.join(out_dir, '0') # !! CHANGE: assuming input vids are in '0' subfolder
     
     # Ensure the output directorie exist
     ensure_dir(output_npz_folder)

     # Set the output path for .npz and .parquet files
     npz_out_path = os.path.join(output_npz_folder, basename)

     cap = cv2.VideoCapture(video_path)
     if not cap.isOpened():
          print(f"[WARN] Cannot open {video_path}")
          return

     src_fps = cap.get(cv2.CAP_PROP_FPS)
     if not src_fps or math.isnan(src_fps) or src_fps < 1:
          src_fps = 30.0  # fallback

     # sampling step in seconds
     step_s = 1.0 / target_fps
     next_t = 0.0

     # build models
     models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)

     X_frames = []
     M_frames = []
     T_ms = []

     t0 = cap.get(cv2.CAP_PROP_POS_MSEC)
     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

     try:
          # We'll iterate sequentially and sample by timestamp
          while True:
               ret, frame_bgr = cap.read()
               if not ret:
                    break
               ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # current timestamp in ms
               if ms < next_t * 1000.0:
                    continue

               # Resize
               frame_bgr = cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

               # Background removal
               frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
               seg_mask = models.seg.process(frame_rgb).segmentation_mask
               # Threshold the mask; keep person
               fg_mask = (seg_mask > 0.5).astype(np.float32)[..., None]
               black_bg = np.zeros_like(frame_rgb, dtype=np.uint8)
               comp_rgb = (frame_rgb * fg_mask + black_bg * (1 - fg_mask)).astype(np.uint8)

               # Landmarks (we pass the composited frame)
               vec156, mask78 = extract_keypoints_from_frame(comp_rgb, models, conf_thresh=conf_thresh)

               X_frames.append(vec156)
               M_frames.append(mask78)
               T_ms.append(ms)

               next_t += step_s
     finally:
          cap.release()
          close_models(models)

     if len(X_frames) == 0:
          print(f"[WARN] No frames written for {video_path}")
          return

     X = np.stack(X_frames, axis=0)      # [T,156]
     M = np.stack(M_frames, axis=0)      # [T,78]
     T_ms = np.array(T_ms, dtype=np.int64)

     # Interpolate short gaps
     X_filled, M_filled = interpolate_gaps(X, M, max_gap=max_gap)

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

     # Save output in .npz and .parquet formats
     to_npz(npz_out_path, X_filled, M_filled, T_ms, meta, also_parquet=True)
     print(f"[OK] {basename}: frames={len(X_frames)}  saved: {npz_out_path}.npz (+ .parquet)")

# --------------------------------------------------------
# MAIN: argument parsing and batch processing
# --------------------------------------------------------

def main():
     ap = argparse.ArgumentParser()
     # Default (local) base directories for videos and output
     ap.add_argument("--in_dir", default="sign-preproc/videos/mp4/", help="Folder of input clips (.mp4)") # !! CHANGE PATH to your input folder
     ap.add_argument("--out_dir", default="sign-preproc/output", help="Where to write .npz and .parquet") # !! CHANGE PATH to your output folder
     ap.add_argument("--fps", type=int, default=30)
     ap.add_argument("--size", type=int, default=256, help="square resize (e.g., 256)")
     ap.add_argument("--conf", type=float, default=0.5, help="confidence/visibility threshold")
     ap.add_argument("--max_gap", type=int, default=5, help="max frames to interpolate")
     args = ap.parse_args()

     ensure_dir(args.out_dir)
     # Construct the final paths by appending subfolders (e.g., '0' to the base directories
     vids = sorted(glob.glob(os.path.join(args.in_dir, "0", "*.mp4").replace("\\", "/")))
     # Default base for input, plus '0' folder

     #debugging statement
     print(f"Looking for videos in: {os.path.join(args.in_dir, '0', '*.mp4')}")

     if not vids:
          print(f"No .mp4 found in {args.in_dir}")
          sys.exit(0)

     for vp in tqdm(vids, desc="Processing"):
          process_video(vp, args.out_dir, target_fps=args.fps, out_size=args.size, conf_thresh=args.conf, max_gap=args.max_gap)

if __name__ == "__main__":
     main()