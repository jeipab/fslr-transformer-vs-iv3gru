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


def get_dynamic_resource_info():
    """Get real-time system resource information for dynamic optimization."""
    import torch
    import multiprocessing as mp
    import psutil
    
    # Basic hardware info
    cpu_count = mp.cpu_count()
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    # Real-time system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_available_gb = memory.available / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    # GPU memory info if available
    gpu_memory_info = {}
    if cuda_available:
        for i in range(gpu_count):
            try:
                gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_memory_free = gpu_memory_total - gpu_memory_allocated
                gpu_memory_info[i] = {
                    'total': gpu_memory_total,
                    'allocated': gpu_memory_allocated,
                    'free': gpu_memory_free
                }
            except:
                gpu_memory_info[i] = {'total': 0, 'allocated': 0, 'free': 0}
    
    return {
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_total_gb': memory_total_gb,
        'memory_available_gb': memory_available_gb,
        'memory_percent': memory.percent,
        'cuda_available': cuda_available,
        'gpu_count': gpu_count,
        'gpu_memory_info': gpu_memory_info
    }


def calculate_optimal_workers(resource_info, video_count, video_sizes_mb=None):
    """Calculate optimal number of workers based on real-time system resources."""
    cpu_count = resource_info['cpu_count']
    cpu_percent = resource_info['cpu_percent']
    memory_available_gb = resource_info['memory_available_gb']
    cuda_available = resource_info['cuda_available']
    gpu_count = resource_info['gpu_count']
    
    # Estimate memory requirements per video (rough estimate)
    estimated_memory_per_video_gb = 2.5  # Conservative estimate
    
    # Calculate memory-based worker limit
    memory_based_workers = max(1, int(memory_available_gb / estimated_memory_per_video_gb))
    
    # Calculate CPU-based worker limit (leave some CPU free)
    cpu_based_workers = max(1, int(cpu_count * (100 - cpu_percent) / 100))
    
    # Choose the more conservative limit
    max_workers = min(memory_based_workers, cpu_based_workers, cpu_count)
    
    # Limit to reasonable maximum
    max_workers = min(max_workers, 8)
    
    # For GPU processing, use GPU count as additional constraint
    if cuda_available and gpu_count > 0:
        gpu_workers = min(gpu_count, max_workers)
        return gpu_workers, 'gpu'
    else:
        cpu_workers = min(max_workers, video_count)
        return cpu_workers, 'cpu'


def calculate_optimal_batch_size(resource_info, processing_type):
    """Calculate optimal batch size based on available resources."""
    memory_available_gb = resource_info['memory_available_gb']
    
    if processing_type == 'gpu':
        # GPU batch sizes - more aggressive
        if memory_available_gb > 16:
            return 32
        elif memory_available_gb > 8:
            return 16
        elif memory_available_gb > 4:
            return 8
        else:
            return 4
    else:
        # CPU batch sizes - more conservative
        if memory_available_gb > 8:
            return 16
        elif memory_available_gb > 4:
            return 8
        elif memory_available_gb > 2:
            return 4
        else:
            return 2


def process_multiple_videos(uploaded_files, target_fps: int = 30, out_size: int = 256, 
                          write_keypoints: bool = True, write_iv3_features: bool = True,
                          occ_detailed: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process multiple video files using dynamic resource detection and adaptive optimization.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
        target_fps: Target FPS for processing
        out_size: Target frame size for processing
        write_keypoints: Whether to extract 156-D keypoint features
        write_iv3_features: Whether to extract 2048-D InceptionV3 features
        occ_detailed: Whether to include detailed occlusion results
        
    Returns:
        Dictionary mapping filename to processed data
    """
    if not PREPROCESSING_AVAILABLE:
        st.error("Preprocessing module not available. Please ensure all dependencies are installed.")
        return {}
    
    # Import multi-preprocessing function
    try:
        from preprocessing.core.multi_preprocess import process_videos_multiprocess
        MULTI_PREPROCESSING_AVAILABLE = True
    except ImportError:
        MULTI_PREPROCESSING_AVAILABLE = False
        st.warning("Multi-preprocessing not available. Falling back to sequential processing.")
    
    # Save uploaded files temporarily
    temp_paths = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save all files to temporary paths
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            temp_paths.append(temp_path)
        
        # Get real-time resource information
        resource_info = get_dynamic_resource_info()
        
        # Display dynamic resource information
        st.info("**Dynamic Resource Analysis:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CPU Usage", f"{resource_info['cpu_percent']:.1f}%")
            st.metric("CPU Cores", resource_info['cpu_count'])
        
        with col2:
            st.metric("Memory Usage", f"{resource_info['memory_percent']:.1f}%")
            st.metric("Available RAM", f"{resource_info['memory_available_gb']:.1f} GB")
        
        with col3:
            st.metric("CUDA Available", "Yes" if resource_info['cuda_available'] else "No")
            st.metric("GPU Count", resource_info['gpu_count'])
        
        with col4:
            if resource_info['cuda_available'] and resource_info['gpu_count'] > 0:
                gpu_mem = resource_info['gpu_memory_info'][0]
                st.metric("GPU Memory", f"{gpu_mem['free']:.1f} GB free")
            else:
                st.metric("Processing Type", "CPU Only")
        
        # Choose processing strategy based on dynamic resources
        if len(uploaded_files) == 1:
            # Single video - use regular processing
            st.info("Processing single video...")
            return _process_single_video_fallback(temp_paths[0], target_fps, out_size, 
                                                write_keypoints, write_iv3_features, occ_detailed)
        
        elif MULTI_PREPROCESSING_AVAILABLE and len(uploaded_files) > 1:
            # Multiple videos - use dynamic multi-processing
            
            # Calculate optimal parameters based on current resources
            optimal_workers, processing_type = calculate_optimal_workers(resource_info, len(uploaded_files))
            optimal_batch_size = calculate_optimal_batch_size(resource_info, processing_type)
            
            # Display optimization results
            st.info(f"**Adaptive Processing Configuration:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Workers", optimal_workers)
            with col2:
                st.metric("Batch Size", optimal_batch_size)
            with col3:
                st.metric("Processing Type", processing_type.upper())
            
            st.info(f"Processing {len(uploaded_files)} videos with {optimal_workers} workers...")
            
            # Process videos using dynamic multi-processing
            results = process_videos_multiprocess(
                video_files=temp_paths,
                out_dir=temp_dir,
                target_fps=target_fps,
                out_size=out_size,
                conf_thresh=0.5,
                max_gap=5,
                write_keypoints=write_keypoints,
                write_iv3_features=write_iv3_features,
                compute_occlusion=True,
                occ_detailed=occ_detailed,
                workers=optimal_workers,
                batch_size=optimal_batch_size,
                disable_parquet=True  # Disable parquet for Streamlit
            )
            
            # Load results
            processed_data = {}
            for temp_path in temp_paths:
                basename = Path(temp_path).stem
                npz_path = os.path.join(temp_dir, f"{basename}.npz")
                
                if os.path.exists(npz_path):
                    data = dict(np.load(npz_path, allow_pickle=True))
                    processed_data[basename] = data
                else:
                    st.warning(f"Failed to process: {basename}")
            
            return processed_data
        
        else:
            # Fallback to sequential processing
            st.info("Processing videos sequentially...")
            processed_data = {}
            for temp_path in temp_paths:
                basename = Path(temp_path).stem
                data = _process_single_video_fallback(temp_path, target_fps, out_size, 
                                                    write_keypoints, write_iv3_features, occ_detailed)
                if data:
                    processed_data[basename] = data
            
            return processed_data
            
    except Exception as e:
        st.error(f"Error in dynamic multi-processing: {str(e)}")
        return {}
    finally:
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def _process_single_video_fallback(video_path: str, target_fps: int, out_size: int, 
                                  write_keypoints: bool, write_iv3_features: bool,
                                  occ_detailed: bool) -> Dict[str, np.ndarray]:
    """Fallback function for single video processing."""
    try:
        # Create output directory
        out_dir = os.path.dirname(video_path)
        
        # Process video using existing preprocessing function
        process_video(
            video_path=video_path,
            out_dir=out_dir,
            target_fps=target_fps,
            out_size=out_size,
            conf_thresh=0.5,
            max_gap=5,
            write_keypoints=write_keypoints,
            write_iv3_features=write_iv3_features,
            compute_occlusion=True,
            occ_detailed=occ_detailed
        )
        
        # Load the generated NPZ file
        basename = Path(video_path).stem
        npz_path = os.path.join(out_dir, f"{basename}.npz")
        
        if os.path.exists(npz_path):
            data = dict(np.load(npz_path, allow_pickle=True))
            return data
        else:
            return {}
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return {}


