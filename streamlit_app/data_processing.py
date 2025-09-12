"""Data processing functions for video and NPZ files."""

import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Video processing will use dummy data.")

# Import preprocessing functions
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from preprocessing.preprocess import process_video
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    st.warning("Preprocessing module not available. Video processing will use dummy data.")


def process_video_file(uploaded_file, target_fps: int = 30, out_size: int = 256) -> Dict[str, np.ndarray]:
    """
    Process uploaded video file to extract keypoints.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        target_fps: Target FPS for processing
        out_size: Target frame size for processing
        
    Returns:
        Dictionary with keypoint data similar to NPZ format
    """
    if not PREPROCESSING_AVAILABLE:
        # Fallback to dummy data generation
        return generate_dummy_keypoints_from_video(uploaded_file)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_video_path = tmp_file.name
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Process video using existing preprocessing function
            process_video(
                video_path=tmp_video_path,
                out_dir=tmp_dir,
                target_fps=target_fps,
                out_size=out_size,
                conf_thresh=0.5,
                max_gap=5,
                write_keypoints=True,
                write_iv3_features=False  # Skip IV3 features for demo
            )
            
            # Load the generated NPZ file
            basename = Path(uploaded_file.name).stem
            npz_path = os.path.join(tmp_dir, f"{basename}.npz")
            
            if os.path.exists(npz_path):
                data = dict(np.load(npz_path, allow_pickle=True))
                return data
            else:
                st.error(f"Failed to process video: {uploaded_file.name}")
                return generate_dummy_keypoints_from_video(uploaded_file)
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return generate_dummy_keypoints_from_video(uploaded_file)
        finally:
            # Clean up temporary video file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)


def generate_dummy_keypoints_from_video(uploaded_file) -> Dict[str, np.ndarray]:
    """
    Generate dummy keypoint data for video files when preprocessing is unavailable.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dictionary with dummy keypoint data
    """
    # Get basic video properties
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_video_path = tmp_file.name
    
    try:
        if CV2_AVAILABLE:
            cap = cv2.VideoCapture(tmp_video_path)
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_ms = (total_frames / fps) * 1000 if fps > 0 else 1000
                cap.release()
            else:
                fps, total_frames, duration_ms = 30, 60, 2000
        else:
            fps, total_frames, duration_ms = 30, 60, 2000
    except:
        fps, total_frames, duration_ms = 30, 60, 2000
    finally:
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
    
    # Generate dummy data
    num_output_frames = min(max(30, total_frames // 5), 150)  # Reasonable range
    
    np.random.seed(42)  # Reproducible dummy data
    keypoints = np.random.rand(num_output_frames, 156).astype(np.float32)
    
    # Add temporal smoothness
    for i in range(1, num_output_frames):
        keypoints[i] = 0.7 * keypoints[i-1] + 0.3 * keypoints[i]
    
    mask = np.random.rand(num_output_frames, 78) > 0.2  # 80% visibility
    timestamps_ms = np.linspace(0, duration_ms, num_output_frames).astype(np.int64)
    
    meta = {
        "source_video": uploaded_file.name,
        "original_fps": fps,
        "original_frames": total_frames,
        "processed_frames": num_output_frames,
        "processing_method": "dummy_extraction_streamlit",
        "keypoint_format": "mediapipe_holistic_156d"
    }
    
    return {
        'X': keypoints,
        'mask': mask,
        'timestamps_ms': timestamps_ms,
        'meta': json.dumps(meta)
    }
