"""Data processing functions for video and NPZ files."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Video processing may not work properly.")

# Import preprocessing functions
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from preprocessing.preprocess import process_video
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    st.warning("Preprocessing module not available. Video processing will not work.")


def process_video_file(uploaded_file, target_fps: int = 30, out_size: int = 256, 
                      write_keypoints: bool = True, write_iv3_features: bool = False) -> Dict[str, np.ndarray]:
    """
    Process uploaded video file to extract keypoints and/or features.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        target_fps: Target FPS for processing
        out_size: Target frame size for processing
        write_keypoints: Whether to extract 156-D keypoint features
        write_iv3_features: Whether to extract 2048-D InceptionV3 features
        
    Returns:
        Dictionary with keypoint/feature data similar to NPZ format
    """
    if not PREPROCESSING_AVAILABLE:
        st.error("Preprocessing module not available. Please ensure all dependencies are installed.")
        return {}
    
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
                write_keypoints=write_keypoints,
                write_iv3_features=write_iv3_features,
                compute_occlusion=True,  # Enable occlusion detection
                occ_vis_thresh=0.6,     # Default occlusion parameters
                occ_frame_prop=0.4,
                occ_min_run=15
            )
            
            # Load the generated NPZ file
            basename = Path(uploaded_file.name).stem
            npz_path = os.path.join(tmp_dir, f"{basename}.npz")
            
            if os.path.exists(npz_path):
                data = dict(np.load(npz_path, allow_pickle=True))
                return data
            else:
                st.error(f"Failed to process video: {uploaded_file.name}")
                return {}
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return {}
        finally:
            # Clean up temporary video file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)


