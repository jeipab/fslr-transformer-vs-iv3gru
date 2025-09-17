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
# Go up to project root: streamlit_app/components -> streamlit_app -> project_root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from preprocessing.core.preprocess import process_video
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    st.warning(f"Preprocessing module not available: {e}. Video processing will not work.")


def process_video_file(uploaded_file, target_fps: int = 30, out_size: int = 256, 
                      write_keypoints: bool = True, write_iv3_features: bool = True,
                      occ_detailed: bool = False) -> Dict[str, np.ndarray]:
    """
    Process uploaded video file to extract keypoints and/or features.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        target_fps: Target FPS for processing
        out_size: Target frame size for processing
        write_keypoints: Whether to extract 156-D keypoint features
        write_iv3_features: Whether to extract 2048-D InceptionV3 features
        occ_detailed: Whether to include detailed occlusion results
        
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
    
    # Create persistent output directory in temp folder
    tmp_dir = tempfile.mkdtemp()
    
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
            occ_detailed=occ_detailed  # Include detailed results if requested
        )
        
        # Load the generated NPZ file
        # Use the basename of the temporary video file (which is what preprocessing creates)
        temp_basename = Path(tmp_video_path).stem
        npz_path = os.path.join(tmp_dir, f"{temp_basename}.npz")
        
        if os.path.exists(npz_path):
            data = dict(np.load(npz_path, allow_pickle=True))
            
            # Clean up the temporary directory after loading
            import shutil
            shutil.rmtree(tmp_dir)
            
            return data
        else:
            st.error(f"Failed to process video: {uploaded_file.name}")
            # Clean up directory even if file not found
            import shutil
            shutil.rmtree(tmp_dir)
            return {}
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        # Clean up directory on error
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        return {}
    finally:
        # Clean up temporary video file
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)


