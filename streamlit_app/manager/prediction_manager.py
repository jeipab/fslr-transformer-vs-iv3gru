"""Prediction manager for NPZ file predictions and visualization."""

import io
import json
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path for model imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ..components.utils import (
    detect_file_type, check_npz_compatibility, create_npz_bytes, 
    extract_occlusion_flag, interpret_occlusion_flag,
    is_continuous_sequence, extract_continuous_metadata
)
from ..components.visualization import (
    render_consolidated_file_info, render_animated_keypoints, 
    render_feature_charts, render_topk_table, render_file_details_horizontal,
    render_summary_stats_horizontal
)
from ..components.components import render_predictions_section
from ..components.ctc_case_utils import (  # type: ignore[import]
    DEFAULT_CASE_PALETTE,
    build_case_maps_for_inference,
    enrich_ground_truth_timestamps,
)
from ..components.ctc_visualization import (
    render_case_legend,
    render_sequence_comparison,
    render_temporal_alignment,
    render_ctc_prediction_card,
    render_sequence_with_categories,
    render_sequence_chips,
)
from .upload_manager import remove_file_from_stage

# Import configuration from core module
from ..core.config import MODEL_CONFIG, DUMMY_DATA, is_ctc_model


def convert_to_ground_truth_format(raw_data: Dict) -> Dict:
    """
    Convert JSON data to ground truth format.
    
    Handles both metadata format (with 'segments') and ground truth format (with 'ground_truth_sequence').
    
    Args:
        raw_data: Raw JSON data loaded from file
        
    Returns:
        Ground truth dictionary in standardized format
    """
    # Check if it's already in ground truth format
    if 'ground_truth_sequence' in raw_data and 'ground_truth_labels' in raw_data:
        return raw_data
    
    # Check if it's in metadata format (with segments)
    if 'segments' in raw_data:
        segments = raw_data['segments']
        
        # Extract ground truth sequence and labels
        ground_truth_sequence = [seg['gloss'] for seg in segments]
        ground_truth_labels = [seg['gloss_label'] for seg in segments]
        
        # Transform timestamps
        ground_truth_timestamps = []
        ground_truth_occluded = []
        for seg in segments:
            ground_truth_timestamps.append({
                'index': seg['index'],
                'gloss': seg['gloss'],
                'gloss_label': seg['gloss_label'],
                'category': seg.get('category', 0),
                'category_label': seg.get('category_label', ''),
                'start_ms': seg['timestamp_start_ms'],
                'end_ms': seg['timestamp_end_ms'],
                'duration_ms': seg['timestamp_end_ms'] - seg['timestamp_start_ms']
            })
            ground_truth_occluded.append(seg.get('occluded', 0))
        
        # Extract categories if available
        ground_truth_categories = [seg.get('category', 0) for seg in segments]
        
        # Build ground truth dictionary
        ground_truth = {
            'file_name': raw_data.get('file_name', 'unknown'),
            'ground_truth_sequence': ground_truth_sequence,
            'ground_truth_labels': ground_truth_labels,
            'ground_truth_timestamps': ground_truth_timestamps,
            'ground_truth_categories': ground_truth_categories,
            'ground_truth_occluded': ground_truth_occluded,
            'signer': raw_data.get('signer', 'unknown'),
            'strategy': raw_data.get('strategy_name', raw_data.get('strategy', 'unknown')),
            'total_duration_sec': raw_data.get('total_duration_sec', 0),
            'num_segments': raw_data.get('num_segments', len(segments))
        }
        
        return ground_truth
    
    # If neither format is recognized, return as-is (might be valid ground truth)
    return raw_data


def validate_ground_truth_data(ground_truth: Dict) -> Dict:
    """
    Validate ground truth data structure.
    
    Args:
        ground_truth: Ground truth dictionary to validate
        
    Returns:
        Dictionary with 'valid' boolean and 'error' message if invalid
    """
    if not isinstance(ground_truth, dict):
        return {'valid': False, 'error': 'Ground truth must be a dictionary'}
    
    # Check if it's in metadata format (with segments)
    if 'segments' in ground_truth:
        segments = ground_truth['segments']
        if not isinstance(segments, list):
            return {'valid': False, 'error': 'segments must be a list'}
        
        if len(segments) == 0:
            return {'valid': False, 'error': 'segments list cannot be empty'}
        
        # Validate each segment
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                return {'valid': False, 'error': f'Segment {i} must be a dictionary'}
            
            required_fields = ['gloss', 'gloss_label']
            for field in required_fields:
                if field not in segment:
                    return {'valid': False, 'error': f'Segment {i} missing required field: {field}'}
            
            if not isinstance(segment['gloss'], int):
                return {'valid': False, 'error': f'Segment {i} gloss must be an integer'}
            
            if not isinstance(segment['gloss_label'], str):
                return {'valid': False, 'error': f'Segment {i} gloss_label must be a string'}
        
        return {'valid': True, 'error': None}
    
    # Check if it's in ground truth format (with ground_truth_sequence)
    elif 'ground_truth_sequence' in ground_truth and 'ground_truth_labels' in ground_truth:
        # Check required fields
        required_fields = ['ground_truth_sequence', 'ground_truth_labels']
        for field in required_fields:
            if field not in ground_truth:
                return {'valid': False, 'error': f'Missing required field: {field}'}
        
        # Check that sequences are lists
        if not isinstance(ground_truth['ground_truth_sequence'], list):
            return {'valid': False, 'error': 'ground_truth_sequence must be a list'}
        
        if not isinstance(ground_truth['ground_truth_labels'], list):
            return {'valid': False, 'error': 'ground_truth_labels must be a list'}
        
        # Check that sequences have same length
        if len(ground_truth['ground_truth_sequence']) != len(ground_truth['ground_truth_labels']):
            return {'valid': False, 'error': 'ground_truth_sequence and ground_truth_labels must have same length'}
        
        # Check that sequence is not empty
        if len(ground_truth['ground_truth_sequence']) == 0:
            return {'valid': False, 'error': 'Ground truth sequence cannot be empty'}
        
        # Check that gloss IDs are integers
        for i, gloss_id in enumerate(ground_truth['ground_truth_sequence']):
            if not isinstance(gloss_id, int):
                return {'valid': False, 'error': f'Gloss ID at position {i} must be an integer, got {type(gloss_id)}'}
        
        # Check that labels are strings
        for i, label in enumerate(ground_truth['ground_truth_labels']):
            if not isinstance(label, str):
                return {'valid': False, 'error': f'Label at position {i} must be a string, got {type(label)}'}
        
        return {'valid': True, 'error': None}
    
    else:
        return {'valid': False, 'error': 'JSON must contain either "segments" (metadata format) or "ground_truth_sequence" and "ground_truth_labels" (ground truth format)'}


def render_ground_truth_preview(ground_truth: Dict):
    """
    Render a preview of the ground truth data.
    
    Args:
        ground_truth: Ground truth dictionary
    """
    # Show sequence as chips
    st.markdown("**Ground Truth Sequence:**")
    labels = ground_truth.get('ground_truth_labels', [])
    if not labels:
        st.info("No ground truth labels available")
        return
    
    categories = ground_truth.get('ground_truth_categories', [])
    occlusion_flags = ground_truth.get('ground_truth_occluded', [])
    has_categories = bool(categories)

    if has_categories:
        render_sequence_with_categories(
            gloss_labels=labels,
            category_ids=categories,
            gloss_confidences=None,
            category_confidences=None,
            occlusion_flags=occlusion_flags,
            is_ground_truth=True,
        )
    else:
        render_sequence_chips(
            labels=labels,
            confidence_scores=None,
            color='#10b981',
            occlusion_flags=occlusion_flags,
        )
    
    # Show timestamps if available
    raw_timestamps = ground_truth.get('ground_truth_timestamps') or []
    if raw_timestamps:
        st.markdown("**Timestamps:**")
        timestamp_data = []
        for ts in raw_timestamps:
            if not isinstance(ts, dict):
                continue
            timestamp_data.append({
                'Position': ts.get('index', 0) + 1,
                'Gloss': ts.get('gloss_label', 'N/A'),
                'Start (ms)': ts.get('start_ms', 0),
                'End (ms)': ts.get('end_ms', 0),
                'Duration (ms)': ts.get('duration_ms', 0)
            })
        
        if timestamp_data:
            import pandas as pd
            df = pd.DataFrame(timestamp_data)
            st.dataframe(df, width='stretch')


class ModelManager:
    """Singleton model manager for loading and caching prediction models."""
    
    _instance = None
    _models = {}
    _label_mappings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str):
        """Get a loaded model, loading it if necessary."""
        if model_name not in self._models:
            self._load_model(model_name)
        return self._models.get(model_name)
    
    def _load_model(self, model_name: str):
        """Load a model and cache it."""
        config = MODEL_CONFIG.get(model_name)
        if not config or not config['enabled']:
            return None
            
        try:
            # Import ModelPredictor from predict.py
            import sys
            from pathlib import Path
            
            # Set up paths correctly - go up to project root
            project_root = Path(__file__).parent.parent.parent  # Go up to project root
            trained_models_path = project_root / "trained_models"
            
            # Add project root to path if not already there
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Import using the full module path
            from evaluation.prediction.predict import ModelPredictor
            from ..core.config import update_model_input_dim
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load the model
            predictor = ModelPredictor(
                model_type=config['model_type'],
                checkpoint_path=config['checkpoint_path'],
                device=device
            )
            
            # Update the model configuration with detected input dimension
            if hasattr(predictor, 'input_dim') and predictor.input_dim is not None:
                update_model_input_dim(model_name, predictor.input_dim)
            
            self._models[model_name] = predictor
            
        except Exception as e:
            st.toast(f"Failed to load {model_name.upper()} model: {str(e)}", icon="⚠️", duration=5000)
            self._models[model_name] = None
    
    def get_label_mappings(self):
        """Get label mappings, loading them if necessary."""
        if self._label_mappings is None:
            try:
                # Add project root to path for label_mapping import
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent  # Go up to project root
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from data.labels.label_mapping import load_label_mappings
                self._label_mappings = load_label_mappings()
            except Exception as e:
                st.toast(f"Could not load label mappings: {str(e)}", icon="⚠️", duration=3000)
                # Fallback to basic mappings
                self._label_mappings = ({}, {})
        return self._label_mappings
    
    def cleanup(self):
        """Clean up all loaded models."""
        for model in self._models.values():
            if model is not None:
                try:
                    model.cleanup()
                except:
                    pass
        self._models.clear()


# Global singleton instance
_model_manager_instance = None

def get_model_manager():
    """Get the singleton model manager instance."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance


def make_real_prediction(npz_data: Dict[str, np.ndarray], model_name: str) -> Dict:
    """
    Make real prediction using the specified model.
    
    Args:
        npz_data: NPZ data dictionary
        model_name: Name of the model to use (e.g., 'transformer_isolated', 'iv3_gru_isolated')
        
    Returns:
        Dictionary with prediction results
    """
    # Check if model is available
    if not MODEL_CONFIG[model_name]['enabled']:
        if model_name.startswith('iv3_gru'):
            st.toast("IV3-GRU model is not available. Using placeholder data.", icon="⚠️", duration=3000)
            return DUMMY_DATA.get('iv3_gru_isolated', {}).copy()
        else:
            st.error(f"Model {model_name} is not available.")
            return None
    
    model_manager = get_model_manager()
    
    # Get the model
    predictor = model_manager.get_model(model_name)
    if predictor is None:
        st.error(f"Failed to load {model_name} model. Please check model files.")
        return None
    
    try:
        # Create temporary NPZ file for prediction
        import tempfile
        import os
        import time
        
        # Use a more robust temporary file approach
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            np.savez_compressed(tmp_path, **npz_data)
        
        try:
            # Make prediction
            results = predictor.predict_from_npz_simple(tmp_path)
            return results
        finally:
            # Clean up temporary file with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Wait 100ms before retry
                    else:
                        # If all retries fail, just leave the file
                        pass
            
    except Exception as e:
        # Show error as toast instead of st.error
        st.toast(f"Prediction failed: {str(e)}", icon="⚠️", duration=5000)
        return None


def make_ctc_prediction(npz_data: Dict[str, np.ndarray], model_name: str, 
                       ground_truth: Optional[Dict] = None,
                       decode_method: str = 'greedy',
                       beam_width: int = 10) -> Dict:
    """
    Make CTC prediction for continuous sign sequence.
    
    Args:
        npz_data: NPZ data dictionary
        model_name: Name of CTC model ('transformer_continuous' or 'iv3_gru_continuous')
        ground_truth: Optional ground truth dictionary
        decode_method: Decoding method ('greedy' or 'beam_search')
        beam_width: Beam width for beam search
        
    Returns:
        Dictionary with CTC prediction results
    """
    # Check if model is CTC model
    if not is_ctc_model(model_name):
        st.error(f"{model_name} is not a CTC model. Use make_real_prediction for classification models.")
        return None
    
    # Check if model is available
    if not MODEL_CONFIG[model_name]['enabled']:
        st.error(f"CTC model {model_name} is not available.")
        return None
    
    try:
        # Import CTC predictor
        from evaluation.prediction.predict_ctc import CTCPredictor
        import tempfile
        import os
        
        # Get model config
        config = MODEL_CONFIG[model_name]
        
        # Initialize CTC predictor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = CTCPredictor(
            model_type=config['model_type'],
            checkpoint_path=config['checkpoint_path'],
            blank_id=config['num_gloss_classes'],  # Blank ID is num_gloss_classes
            device=device
        )
        
        # Create temporary NPZ file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            np.savez_compressed(tmp_path, **npz_data)
        
        try:
            # Make prediction using sliding window approach
            results = predictor.predict_sequence_sliding_window(
                npz_path=Path(tmp_path),
                ground_truth=ground_truth,
                window_size=120,  # Window size for isolated signs (4 seconds at 30fps)
                stride=40,       # Overlap between windows (1.3 seconds)
                decode_method=decode_method,
                beam_width=beam_width,
                fps=30,
                temporal_tolerance=500
            )
            
            # Add ground truth information to results if available
            if ground_truth:
                results['ground_truth_sequence'] = ground_truth.get('ground_truth_sequence', [])
                results['ground_truth_labels'] = ground_truth.get('ground_truth_labels', [])
                results['ground_truth_timestamps'] = ground_truth.get('ground_truth_timestamps', [])
                results['ground_truth_categories'] = ground_truth.get('ground_truth_categories', [])
                results['ground_truth_occluded'] = ground_truth.get('ground_truth_occluded', [])
                results['signer'] = ground_truth.get('signer', 'unknown')
                results['strategy'] = ground_truth.get('strategy', 'unknown')
            
            return results
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    except Exception as e:
        st.toast(f"CTC Prediction failed: {str(e)}", icon="⚠️", duration=5000)
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return None


def render_predictions_stage(cfg: Dict):
    """Render the predictions stage with NPZ files and visualization."""
    # Navigation header
    col1, col2, col3, col4 = st.columns([2, 6, 1, 1])
    with col1:
        # Determine back button text and destination
        back_destination = get_back_destination()
        back_text = f"← Back to {back_destination.title()}"
        back_help = f"Return to {back_destination} stage"
        
        if st.button(back_text, help=back_help, type="secondary"):
            navigate_back_from_predictions()
    with col2:
        st.markdown("")  # Empty space
    with col3:
        st.markdown("")  # Empty space
    with col4:
        # Show Upload New button only if coming from preprocessing (not from upload)
        if back_destination == "preprocessing":
            if st.button("Upload New", help="Upload new files", type="primary"):
                # Clear all current files and go to upload
                from .upload_manager import clear_all_files
                clear_all_files()
                st.session_state.workflow_stage = 'upload'
                st.rerun()
        else:
            st.markdown("")  # Empty space
    
    st.markdown("<div class='main-section-header'>READY FOR INFERENCE</div>", unsafe_allow_html=True)
    
    # Get all NPZ files (original + preprocessed)
    all_npz_files = get_all_npz_files()
    
    if not all_npz_files:
        st.info("No NPZ files available for predictions.")
        return
    
    # Show file management
    render_npz_files_management(all_npz_files)
    
    # Show visualization tabs
    render_visualization_tabs(cfg)


def get_back_destination() -> str:
    """Determine the appropriate back destination based on current state."""
    # Check if we have video files in preprocessing stage
    if st.session_state.video_files or st.session_state.preprocessed_files:
        return "preprocessing"
    else:
        return "upload"


def get_all_npz_files() -> List:
    """Get all NPZ files from both original uploads and preprocessed videos with object deduplication."""
    all_files = []
    seen_objects = set()
    
    # Add original NPZ files (deduplicate by object identity, not name)
    for file_obj in st.session_state.npz_files:
        if id(file_obj) not in seen_objects:
            all_files.append(file_obj)
            seen_objects.add(id(file_obj))
    
    # Add preprocessed files (deduplicate by object identity, not name)  
    for file_obj in st.session_state.preprocessed_files:
        if id(file_obj) not in seen_objects:
            all_files.append(file_obj)
            seen_objects.add(id(file_obj))
    
    return all_files


def render_npz_files_management(all_npz_files: List):
    """Render NPZ files management interface."""
    if not all_npz_files:
        return
    
    # Check for any files currently processing to prevent render conflicts
    processing_files = [f for f in all_npz_files 
                       if st.session_state.file_status.get(f.name, 'completed') == 'processing']
    
    # Add small space at top
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # Pagination controls
    files_per_page = 5
    total_pages = (len(all_npz_files) - 1) // files_per_page + 1
    
    # Initialize current page in session state if not exists
    if "current_file_page" not in st.session_state:
        st.session_state.current_file_page = 1
    
    current_page = st.session_state.current_file_page
    
    # Reset page to 1 if only one page
    if total_pages <= 1:
        current_page = 1
        if "current_file_page" in st.session_state:
            st.session_state.current_file_page = 1
    
    # Calculate file range for current page
    start_idx = (current_page - 1) * files_per_page
    end_idx = min(start_idx + files_per_page, len(all_npz_files))
    page_files = all_npz_files[start_idx:end_idx]
    
    # File rows for current page
    for i, uploaded_file in enumerate(page_files):
        # Adjust index for unique keys
        actual_index = start_idx + i
        filename = uploaded_file.name
        status = st.session_state.file_status.get(filename, 'completed')
        file_type = detect_file_type(uploaded_file)
        metadata = st.session_state.file_metadata.get(filename, {})
        file_size = metadata.get('file_size_formatted', 'Unknown')
        
        # Create unique key for this file instance using actual index and hash to handle duplicates
        import hashlib
        file_hash = hashlib.md5(f"{filename}_{actual_index}_{id(uploaded_file)}".encode()).hexdigest()[:8]
        unique_key_suffix = f"{filename}_{actual_index}_{file_hash}"
        
        # Status emojis only
        status_emoji = {
            'pending': '⏳',
            'processing': '🔄', 
            'completed': '✅',
            'error': '❌'
        }
        
        # Create compact file row
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            st.markdown(f"**Size:** {file_size}")
        with col3:
            st.markdown(f"**Status:** {status_emoji.get(status, '❓')} {status.title()}")
        
        # Action buttons with unique keys
        with col4:
            if status == 'completed':
                if st.button("View", key=f"view_{unique_key_suffix}", help="View this file", type="secondary"):
                    st.session_state.current_tab = filename
                    st.rerun()
            elif status == 'pending':
                if st.button("Process", key=f"process_{unique_key_suffix}", help="Process this file", type="primary"):
                    process_single_npz_file(uploaded_file, filename)
                    # Delay rerun to prevent button flicker during state transition
                    st.rerun()
            elif status == 'error':
                if st.button("Retry", key=f"retry_{unique_key_suffix}", help="Retry processing", type="primary"):
                    process_single_npz_file(uploaded_file, filename)
                    # Delay rerun to prevent button flicker during state transition
                    st.rerun()
            elif status == 'processing':
                # Show disabled button during processing to maintain layout stability
                st.button("Processing...", key=f"processing_{unique_key_suffix}", disabled=True, help="File is being processed", type="secondary")
        
        # Remove button with confirmation
        with col5:
            if st.button("Remove", key=f"remove_{unique_key_suffix}", help="Remove this file", type="secondary"):
                remove_specific_file_instance(uploaded_file, 'predictions')
                st.rerun()
        
        # Add separator line only if not the last file on current page
        if i < len(page_files) - 1:
            st.markdown("---")
    
    # Batch operations
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
    
    with col1:
        # Pagination controls (only show if multiple pages)
        if total_pages > 1:
            pag_col1, pag_col2, pag_col3, pag_col4, pag_col5 = st.columns([1, 1, 1, 1, 1])
            with pag_col1:
                st.markdown("")  # Empty space for centering
            with pag_col2:
                if st.button("← Previous", disabled=(current_page == 1), key="prev_page"):
                    st.session_state.current_file_page = max(1, current_page - 1)
                    # Reset file selector to Summary when changing pages
                    st.session_state.file_selector = "Summary"
                    st.rerun()
            with pag_col3:
                st.markdown(f"**Page {current_page} of {total_pages}**")
            with pag_col4:
                if st.button("Next →", disabled=(current_page == total_pages), key="next_page"):
                    st.session_state.current_file_page = min(total_pages, current_page + 1)
                    # Reset file selector to Summary when changing pages
                    st.session_state.file_selector = "Summary"
                    st.rerun()
            with pag_col5:
                st.markdown("")  # Empty space for centering
        else:
            st.markdown("")  # Empty space to align with file names
    
    with col2:
        st.markdown("")  # Empty space to align with size column
    
    with col3:
        if st.button("Reset", help="Reset processed files back to pending", type="primary"):
            reset_processed_files()
            st.rerun()
    
    with col4:
        if st.button("Process All Pending", type="primary", help="Process all pending files"):
            process_all_pending_npz_files(all_npz_files)
            st.rerun()
    
    with col5:
        if st.button("Clear All", help="Clear all files", type="primary"):
            clear_all_predictions_files()
            st.rerun()


def process_single_npz_file(uploaded_file, filename: str):
    """Process a single NPZ file and update session state."""
    try:
        st.session_state.file_status[filename] = 'processing'
        
        # Reset file pointer to beginning and load NPZ file
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        file_bytes = io.BytesIO(file_content)
        npz_data = dict(np.load(file_bytes, allow_pickle=True))
        
        # Detect if continuous sequence
        is_continuous = is_continuous_sequence(npz_data)
        continuous_meta = extract_continuous_metadata(npz_data) if is_continuous else None
        
        # Check compatibility
        compatibility = check_npz_compatibility(npz_data)
        
        if not any(compatibility.values()):
            st.session_state.file_status[filename] = 'error'
            st.toast(f"{filename}: Incompatible with any model architecture", icon="❌", duration=5000)
            return
        
        # Store processed data
        st.session_state.processed_data[filename] = npz_data
        
        # Update metadata while preserving file size
        existing_metadata = st.session_state.file_metadata.get(filename, {})
        st.session_state.file_metadata[filename] = {
            **existing_metadata,
            'compatibility': compatibility,
            'file_type': 'npz',
            'frame_count': npz_data['X'].shape[0] if 'X' in npz_data else npz_data['X2048'].shape[0] if 'X2048' in npz_data else 0,
            'is_continuous': is_continuous,
            'continuous_metadata': continuous_meta
        }
        st.session_state.file_status[filename] = 'completed'
        
        # Show appropriate message based on sequence type
        if is_continuous:
            num_segments = continuous_meta.get('num_segments', '?') if continuous_meta else '?'
            st.toast(f"✅ Continuous sequence loaded ({num_segments} segments)", icon="🎬", duration=3000)
        
    except Exception as e:
        st.session_state.file_status[filename] = 'error'
        st.toast(f"Processing failed for {filename}: {str(e)}", icon="❌", duration=5000)


def process_all_pending_npz_files(all_npz_files: List):
    """Process all pending NPZ files."""
    pending_files = [f for f in all_npz_files 
                    if st.session_state.file_status.get(f.name, 'pending') == 'pending']
    
    if not pending_files:
        st.toast("No pending NPZ files to process", icon="ℹ️", duration=5000)
        return
    
    # Process all files
    for uploaded_file in pending_files:
        process_single_npz_file(uploaded_file, uploaded_file.name)
    
    # Show consolidated summary
    completed_files = [f for f in all_npz_files 
                      if st.session_state.file_status.get(f.name) == 'completed']
    error_files = [f for f in all_npz_files 
                  if st.session_state.file_status.get(f.name) == 'error']
    
    if completed_files:
        # Count compatibility
        transformer_compatible = sum(1 for f in completed_files 
                                   if st.session_state.file_metadata[f.name]['compatibility']['transformer'])
        iv3_compatible = sum(1 for f in completed_files 
                           if st.session_state.file_metadata[f.name]['compatibility']['iv3_gru'])
        
        st.toast(f"{len(completed_files)} files have been loaded successfully", icon="✅", duration=5000)
        
        if transformer_compatible > 0 and iv3_compatible > 0:
            st.toast(f"{transformer_compatible} files are compatible with Transformer, {iv3_compatible} files are compatible with IV3-GRU", icon="🔧", duration=5000)
        elif transformer_compatible > 0:
            st.toast(f"{transformer_compatible} files are compatible with Transformer", icon="🔧", duration=5000)
        elif iv3_compatible > 0:
            st.toast(f"{iv3_compatible} files are compatible with IV3-GRU", icon="🔧", duration=5000)
    
    if error_files:
        st.toast(f"{len(error_files)} files failed to process", icon="❌", duration=5000)


def render_visualization_tabs(cfg: Dict):
    """Render visualization interface for processed files."""
    all_npz_files = get_all_npz_files()
    completed_files = [f for f in all_npz_files 
                      if st.session_state.file_status.get(f.name) == 'completed']
    
    if not completed_files:
        return
    
    # Check if we have continuous sequences
    continuous_files = []
    isolated_files = []
    for f in completed_files:
        metadata = st.session_state.file_metadata.get(f.name, {})
        if metadata.get('is_continuous', False):
            continuous_files.append(f)
        else:
            isolated_files.append(f)
    
    # Create file selection interface
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Show file type indicator if we have mixed types
    if continuous_files and isolated_files:
        st.markdown(f"<div class='main-section-header'>RESULTS ({len(isolated_files)} Isolated, {len(continuous_files)} Continuous)</div>", unsafe_allow_html=True)
    elif continuous_files:
        st.markdown("<div class='main-section-header'>RESULTS</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='main-section-header'>RESULTS</div>", unsafe_allow_html=True)
    
    # File selection dropdown
    file_options = []
    file_mapping = {}
    
    # Track filename counts to handle duplicates
    filename_counts = {}
    
    for uploaded_file in completed_files:
        filename = uploaded_file.name
        
        # Create unique display name for duplicates
        if filename in filename_counts:
            filename_counts[filename] += 1
            display_name = f"{filename} ({filename_counts[filename]})"
        else:
            filename_counts[filename] = 1
            display_name = filename
        
        file_options.append(display_name)
        file_mapping[display_name] = uploaded_file
    
    # Determine if we're in continuous-only mode
    continuous_only_mode = bool(continuous_files) and not bool(isolated_files)

    # Add summary option only if not continuous-only mode
    if not continuous_only_mode:
        file_options.append("Summary")
    
    # Handle programmatic file switching before creating selectbox
    if st.session_state.current_tab:
        # Find the file and set it as selected
        for display_name, uploaded_file in file_mapping.items():
            if uploaded_file.name == st.session_state.current_tab:
                st.session_state.file_selector = display_name
                break
        
        # Clear the current_tab after switching
        st.session_state.current_tab = None
    
    # Initialize or sanitize file_selector in session state
    if "file_selector" not in st.session_state:
        # Default to first file if continuous-only, otherwise Summary
        st.session_state.file_selector = file_options[0] if continuous_only_mode and file_options else "Summary"
    else:
        # If previous selection no longer exists (e.g., Summary removed), fallback to first option
        if st.session_state.file_selector not in file_options and file_options:
            st.session_state.file_selector = file_options[0]
    
    # Combined Results and File Details in same row
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        # File selector - let session state handle the selection
        selected_file = st.selectbox(
            "View results for:",
            options=file_options,
            key="file_selector",
            help="Choose a processed file to view its visualization results"
        )
    
    with col_right:
        # Render file details if a file is selected, or summary stats if Summary is selected
        if selected_file != "Summary":
            selected_uploaded_file = file_mapping.get(selected_file)
            if selected_uploaded_file:
                filename = selected_uploaded_file.name
                npz_data = st.session_state.processed_data[filename]
                metadata = st.session_state.file_metadata[filename]
                
                # Render file details in compact horizontal layout
                render_file_details_horizontal(filename, npz_data, metadata)
        else:
            # Render summary statistics in the right column
            render_summary_stats_horizontal(completed_files)
    
    # Auto-navigate to page containing selected file
    if selected_file != "Summary":
        selected_uploaded_file = file_mapping.get(selected_file)
        if selected_uploaded_file:
            # Calculate pagination parameters
            files_per_page = 5
            current_page = st.session_state.get("current_file_page", 1)
            
            # Find which page contains this file
            file_index = completed_files.index(selected_uploaded_file)
            target_page = (file_index // files_per_page) + 1
            
            # If file is not on current page, navigate to correct page
            if target_page != current_page:
                st.session_state.current_file_page = target_page
                st.rerun()
    
    # Render selected file or summary content
    if selected_file != "Summary":
        # Find the selected file
        selected_uploaded_file = file_mapping.get(selected_file)
        if selected_uploaded_file:
            filename = selected_uploaded_file.name
            npz_data = st.session_state.processed_data[filename]
            metadata = st.session_state.file_metadata[filename]
            
            # Create unique key suffix using file object id to handle duplicate filenames
            unique_key_suffix = f"{filename}_{id(selected_uploaded_file)}"
            
            # Check if continuous sequence
            is_continuous = metadata.get('is_continuous', False)
            
            try:
                if is_continuous:
                    # Render CTC prediction interface for continuous sequences
                    render_continuous_sequence_predictions(filename, npz_data, metadata, cfg, unique_key_suffix)
                else:
                    # Render standard classification interface for isolated signs
                    # Render consolidated file info (includes sequence overview data)
                    X_pad, mask, meta = render_consolidated_file_info(filename, npz_data, metadata, cfg["sequence_length"])
                    
                    # Generate and render predictions (with metadata and key_suffix for video preview)
                    render_predictions_section(cfg, npz_data, filename, metadata, unique_key_suffix)
                    
                    # Side-by-side layout for Keypoint Visualization and Feature Analysis
                    st.markdown('<div class="viz-side-by-side">', unsafe_allow_html=True)
                    viz_col1, viz_col2 = st.columns([1, 1])
                    
                    with viz_col1:
                        render_animated_keypoints(X_pad, mask if mask.size > 0 else None, key_suffix=unique_key_suffix, meta_dict=meta)
                    
                    with viz_col2:
                        render_feature_charts(X_pad, mask if mask.size > 0 else None, key_suffix=unique_key_suffix)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download button (for both types)
                st.markdown("### Download")
                npz_bytes = create_npz_bytes(npz_data)
                npz_filename = filename if filename.endswith('.npz') else f"{Path(filename).stem}.npz"
                st.download_button(
                    label=f"Download {npz_filename}",
                    data=npz_bytes,
                    file_name=npz_filename,
                    mime="application/octet-stream",
                    key=f"download_npz_{unique_key_suffix}"
                )
                
            except Exception as e:
                st.toast(f"Visualization error: {str(e)}", icon="⚠️", duration=5000)
    else:
        # Legacy summary branch (won't be hit in continuous-only mode)
        render_batch_summary_tab(cfg)


def render_continuous_sequence_predictions(filename: str, npz_data: Dict, metadata: Dict, cfg: Dict, unique_key_suffix: str):
    """
    Render CTC prediction interface for continuous sequences.
    
    Args:
        filename: Name of the file
        npz_data: NPZ data dictionary
        metadata: File metadata
        cfg: Configuration dictionary
        unique_key_suffix: Unique key suffix for widgets
    """
    st.markdown("### Continuous Sequence Recognition")
    
    # Continuous sequence info moved to file details row; hide here
    
    # CTC Configuration section for predictions UI
    st.markdown("---")
    st.markdown("#### CTC Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available CTC models
        from ..core.config import get_models_by_mode, MODEL_CONFIG
        ctc_models = get_models_by_mode('continuous')
        
        if not ctc_models:
            st.error("No CTC models available. Please ensure CTC model checkpoints exist.")
            return
        
        # Model selector
        ctc_model = st.selectbox(
            "CTC Model",
            options=ctc_models,
            format_func=lambda x: MODEL_CONFIG[x]['display_name'],
            key=f"ctc_model_select_{unique_key_suffix}"
        )
        
        # Clear results if model selection changed
        ctc_key = f'ctc_results_{unique_key_suffix}'
        if ctc_key in st.session_state:
            # Check if this is a new model selection (not initial load)
            if f'previous_ctc_model_{unique_key_suffix}' in st.session_state:
                if st.session_state[f'previous_ctc_model_{unique_key_suffix}'] != ctc_model:
                    del st.session_state[ctc_key]
            st.session_state[f'previous_ctc_model_{unique_key_suffix}'] = ctc_model
    
    with col2:
        # Always use greedy decoding; hide decode method controls
        decode_method = 'greedy'
        beam_width = 1
    
    # Ground truth upload (optional)
    st.markdown("---")
    st.markdown("#### Ground Truth Upload")
    
    # Enhanced file uploader with better help text
    ground_truth_file = st.file_uploader(
        "Ground Truth JSON (optional)",
        type=['json'],
        help="Upload ground truth JSON file for WER calculation. Supports both metadata format (with 'segments') and ground truth format (with 'ground_truth_sequence')",
        key=f"gt_upload_{unique_key_suffix}"
    )
    
    ground_truth = None
    if ground_truth_file:
        try:
            # Reset file pointer to beginning
            ground_truth_file.seek(0)
            raw_data = json.load(ground_truth_file)
            
            # Convert to ground truth format if needed
            ground_truth = convert_to_ground_truth_format(raw_data)
            
            # Validate the ground truth data
            validation_result = validate_ground_truth_data(ground_truth)
            
            if validation_result['valid']:
                num_glosses = len(ground_truth.get('ground_truth_sequence', []))
                st.toast(f"Ground truth loaded successfully: {num_glosses} glosses", duration=3000)
                
                # Show ground truth preview
                with st.expander("Ground Truth Preview", expanded=False):
                    render_ground_truth_preview(ground_truth)
            else:
                st.error(f"Invalid ground truth format: {validation_result['error']}")
                ground_truth = None
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            st.error(f"Failed to load ground truth: {str(e)}")
            with st.expander("Error Details"):
                st.code(str(e))
    
    # Predict button
    st.markdown("---")
    if st.button("Predict Sequence", type="primary", key=f"predict_ctc_{unique_key_suffix}"):
        # Clear any existing results for this file when making new prediction
        ctc_key = f'ctc_results_{unique_key_suffix}'
        if ctc_key in st.session_state:
            del st.session_state[ctc_key]
        
        with st.spinner("Predicting sequence..."):
            results = make_ctc_prediction(
                npz_data, 
                ctc_model,
                ground_truth=ground_truth,
                decode_method=decode_method,
                beam_width=beam_width
            )
            
            if results:
                st.session_state[f'ctc_results_{unique_key_suffix}'] = results
                st.rerun()
    
    # Display results if available
    if f'ctc_results_{unique_key_suffix}' in st.session_state:
        results = st.session_state[f'ctc_results_{unique_key_suffix}']
        
        st.markdown("---")
        
        # Side-by-side layout: Prediction Results on left, Video Preview on right
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            # Render CTC prediction card
            render_ctc_prediction_card(
                file_name=filename,
                predicted_sequence=results.get('predicted_sequence', []),
                predicted_labels=results.get('predicted_labels', []),
                confidence_scores=results.get('confidence_scores'),
                ground_truth_available=ground_truth is not None,
                predicted_categories=results.get('predicted_categories'),
                category_confidences=results.get('category_confidences'),
                category_accuracy=results.get('category_accuracy'),
                gloss_accuracy=results.get('gloss_accuracy')
            )
        
        with col_right:
            # Auto-generate and show keypoint video preview (Grid background), like isolated mode
            try:
                from ..components.components import render_inline_video_preview
                render_inline_video_preview(npz_data, metadata, filename, unique_key_suffix)
            except Exception as e:
                st.warning(f"Video preview unavailable: {str(e)}")
        
        # Sequence comparison and temporal alignment below
        if ground_truth:
            st.markdown("---")
            predicted_sequence = results.get('predicted_sequence', []) or []
            ground_truth_sequence = (
                results.get('ground_truth_sequence')
                or ground_truth.get('ground_truth_sequence', [])
                or []
            )
            confidence_scores = results.get('confidence_scores')
            predicted_categories = results.get('predicted_categories')
            category_confidences = results.get('category_confidences')
            ground_truth_categories = (
                results.get('ground_truth_categories')
                or ground_truth.get('ground_truth_categories', [])
            )
            ground_truth_occluded = (
                results.get('ground_truth_occluded')
                or ground_truth.get('ground_truth_occluded', [])
            )
            ground_truth_labels = (
                results.get('ground_truth_labels')
                or ground_truth.get('ground_truth_labels', [])
            )
            case_palette = DEFAULT_CASE_PALETTE.copy()
            (
                prediction_case_map,
                ground_truth_case_map,
                category_prediction_case_map,
                category_ground_truth_case_map,
            ) = build_case_maps_for_inference(
                metrics=results,
                predicted_sequence=predicted_sequence,
                ground_truth_sequence=ground_truth_sequence,
                confidence_scores=confidence_scores,
                predicted_categories=predicted_categories,
                ground_truth_categories=ground_truth_categories,
                category_confidences=category_confidences,
                confidence_threshold=results.get('confidence_threshold'),
            )
            render_case_legend()
            render_sequence_comparison(
                predicted_sequence=predicted_sequence,
                predicted_labels=results.get('predicted_labels', []),
                ground_truth_sequence=ground_truth_sequence,
                ground_truth_labels=ground_truth_labels,
                confidence_scores=confidence_scores,
                predicted_categories=predicted_categories,
                category_confidences=category_confidences,
                ground_truth_categories=ground_truth_categories,
                ground_truth_occluded=ground_truth_occluded,
                ground_truth_cases=ground_truth_case_map,
                prediction_cases=prediction_case_map,
                case_palette=case_palette,
                confidence_threshold=results.get('confidence_threshold', 0.5),
                category_prediction_cases=category_prediction_case_map,
                category_ground_truth_cases=category_ground_truth_case_map,
            )
            
            # Temporal alignment
            if 'predicted_timestamps' in results and 'ground_truth_timestamps' in results:
                st.markdown("---")
                # Extract mask and timestamps from npz_data for inactive period detection
                mask = npz_data.get('mask', None)
                timestamps_ms = npz_data.get('timestamps_ms', None)
                if mask is not None and getattr(mask, "dtype", None) is not None and mask.dtype != bool:
                    mask = mask.astype(bool)
                
                render_temporal_alignment(
                    predicted_timestamps=results['predicted_timestamps'],
                    ground_truth_timestamps=enrich_ground_truth_timestamps(
                        timestamps=results.get('ground_truth_timestamps'),
                        gloss_labels=ground_truth_labels,
                        gloss_sequence=ground_truth_sequence,
                        category_ids=ground_truth_categories,
                        category_labels=ground_truth.get('ground_truth_category_labels'),
                    ),
                    temporal_alignment_accuracy=results.get('temporal_alignment_accuracy'),
                    mask=mask,
                    timestamps_ms=timestamps_ms,
                    predicted_categories=predicted_categories,
                    prediction_cases=prediction_case_map,
                    ground_truth_cases=ground_truth_case_map,
                    case_palette=case_palette,
                    category_prediction_cases=category_prediction_case_map,
                    category_ground_truth_cases=category_ground_truth_case_map,
                )


def render_batch_summary_tab(cfg: Dict):
    """Render batch summary tab with statistics and predictions."""
    # Remove Summary header and reduce gap
    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
    
    all_npz_files = get_all_npz_files()
    completed_files = [f for f in all_npz_files 
                      if st.session_state.file_status.get(f.name) == 'completed']
    
    if not completed_files:
        st.info("No completed files to summarize.")
        return
    
    # Summary table with predictions
    summary_data = []
    for uploaded_file in completed_files:
        filename = uploaded_file.name
        metadata = st.session_state.file_metadata[filename]
        compatibility = metadata['compatibility']
        source_type = metadata.get('source_type', 'original')
        
        # Generate real predictions for this file
        model_name = cfg['model_choice']
        npz_data = st.session_state.processed_data[filename]
        prediction_results = make_real_prediction(npz_data, model_name)
        
        if prediction_results is None:
            top_gloss = "Prediction Failed"
            top_category = "Prediction Failed"
        else:
            # Format predictions with human-readable labels
            model_manager = get_model_manager()
            gloss_mapping, category_mapping = model_manager.get_label_mappings()
            
            gloss_id = prediction_results['gloss_prediction']
            cat_id = prediction_results['category_prediction']
            gloss_prob = prediction_results['gloss_probability']
            cat_prob = prediction_results['category_probability']
            
            # Convert tensors to ints if needed
            gloss_id_int = gloss_id.item() if hasattr(gloss_id, 'item') else int(gloss_id)
            cat_id_int = cat_id.item() if hasattr(cat_id, 'item') else int(cat_id)
            
            gloss_label = gloss_mapping.get(gloss_id_int, f'Unknown ({gloss_id_int})')
            cat_label = category_mapping.get(cat_id_int, f'Unknown ({cat_id_int})')
            
            top_gloss = f"{gloss_label} ({gloss_prob*100:.1f}%)"
            top_category = f"{cat_label} ({cat_prob*100:.1f}%)"
        
        # Extract occlusion status for summary table
        occlusion_flag = extract_occlusion_flag(npz_data)
        occlusion_status = interpret_occlusion_flag(occlusion_flag)
        
        summary_data.append({
            'File': filename,
            'Top Gloss': top_gloss,
            'Top Category': top_category,
            'Occluded': occlusion_status
        })
    
    # Interactive table with row selection
    selected_rows = st.dataframe(
        summary_data, 
        width='stretch',
        on_select="rerun",
        selection_mode="single-row",
        key="summary_table_selection"
    )
    
    # Handle row selection - navigate to selected file
    if selected_rows and hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
        selected_idx = selected_rows.selection.rows[0]
        if 0 <= selected_idx < len(summary_data):
            selected_filename = summary_data[selected_idx]['File']
            # Navigate to that file (same mechanism as View button)
            # Only set current_tab - file_selector will be updated by existing logic in render_visualization_tabs
            st.session_state.current_tab = selected_filename
            st.rerun()
    
    # Batch download
    st.markdown("### Batch Download")
    create_batch_download(summary_data)


def create_batch_download(summary_data):
    """Create and provide batch download as ZIP with NPZ files and summary CSV."""
    import zipfile
    import io
    import pandas as pd
    from pathlib import Path
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all NPZ files with unique names to handle duplicates
        used_names = set()
        file_counter = 0
        
        # Get all completed files from both preprocessing and inference
        all_npz_files = get_all_npz_files()
        completed_files = [f for f in all_npz_files 
                          if st.session_state.file_status.get(f.name) == 'completed']
        
        for uploaded_file in completed_files:
            filename = uploaded_file.name
            if filename in st.session_state.processed_data:
                npz_data = st.session_state.processed_data[filename]
                npz_bytes = create_npz_bytes(npz_data)
                
                # Ensure filename has .npz extension
                base_npz_filename = filename if filename.endswith('.npz') else f"{Path(filename).stem}.npz"
                
                # Handle duplicate names by adding (1), (2), etc.
                npz_filename = base_npz_filename
                counter = 1
                while npz_filename in used_names:
                    name_without_ext = Path(base_npz_filename).stem
                    ext = Path(base_npz_filename).suffix
                    npz_filename = f"{name_without_ext}({counter}){ext}"
                    counter += 1
                
                used_names.add(npz_filename)
                zip_file.writestr(npz_filename, npz_bytes)
                file_counter += 1
        
        # Add summary table as CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_buffer = io.StringIO()
            summary_df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            zip_file.writestr("summary_table.csv", csv_content)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="Download All as ZIP",
        data=zip_buffer.getvalue(),
        file_name="processed_files_with_summary.zip",
        mime="application/zip",
        type="primary",
        key="download_all_zip_predictions"
    )


def navigate_back_from_predictions():
    """Navigate back from predictions stage."""
    # Check if we have video files in preprocessing
    if st.session_state.video_files or st.session_state.preprocessed_files:
        st.session_state.workflow_stage = 'preprocessing'
    else:
        st.session_state.workflow_stage = 'upload'
    st.rerun()


def remove_file_from_predictions(filename: str):
    """Remove a file from predictions stage."""
    # Determine which stage the file came from
    if filename in [f.name for f in st.session_state.npz_files]:
        remove_file_from_stage(filename, 'npz')
    elif filename in [f.name for f in st.session_state.preprocessed_files]:
        remove_file_from_stage(filename, 'preprocessed')


def remove_specific_file_instance(file_obj, stage: str):
    """Remove a specific file instance from the specified stage."""
    filename = file_obj.name
    
    if stage == 'predictions':
        # Remove from all file lists by object reference
        if hasattr(st.session_state, 'npz_files'):
            st.session_state.npz_files = [f for f in st.session_state.npz_files if f is not file_obj]
        if hasattr(st.session_state, 'video_files'):
            st.session_state.video_files = [f for f in st.session_state.video_files if f is not file_obj]
        if hasattr(st.session_state, 'preprocessed_files'):
            st.session_state.preprocessed_files = [f for f in st.session_state.preprocessed_files if f is not file_obj]
        
        # Only clear status/metadata if this is the last file with this name
        remaining_files_with_same_name = []
        for file_list in [st.session_state.npz_files, st.session_state.video_files, st.session_state.preprocessed_files]:
            remaining_files_with_same_name.extend([f for f in file_list if f.name == filename])
        
        if not remaining_files_with_same_name:
            # This was the last file with this name, clear the status/metadata
            if filename in st.session_state.file_status:
                del st.session_state.file_status[filename]
            if filename in st.session_state.processed_data:
                del st.session_state.processed_data[filename]
            if filename in st.session_state.file_metadata:
                del st.session_state.file_metadata[filename]
            if filename in st.session_state.original_file_data:
                del st.session_state.original_file_data[filename]
            
            # Clear CTC prediction results and tracking keys for this file
            keys_to_remove = [key for key in st.session_state.keys() 
                              if (key.startswith("ctc_results_") or 
                                  key.startswith("previous_ctc_model_") or 
                                  key.startswith("previous_decode_method_")) and filename in key]
            for key in keys_to_remove:
                del st.session_state[key]


def reset_processed_files():
    """Reset all processed files back to pending status."""
    reset_count = 0
    
    # Initialize original_file_data if it doesn't exist
    if 'original_file_data' not in st.session_state:
        st.session_state.original_file_data = {}
    
    # Collect all file objects that need to be reset
    files_to_reset = []
    
    # Add all preprocessed files (these are video files that were processed)
    for preprocessed_file in st.session_state.preprocessed_files:
        files_to_reset.append(preprocessed_file)
    
    # Add all NPZ files that have been processed
    for npz_file in st.session_state.npz_files:
        filename = npz_file.name
        if st.session_state.file_status.get(filename) in ['completed', 'error']:
            files_to_reset.append(npz_file)
    
    # Reset each file
    for file_obj in files_to_reset:
        filename = file_obj.name
        # Check if this is a video file (from preprocessed_files)
        is_video_file = any(f.name == filename for f in st.session_state.preprocessed_files)
        
        if is_video_file:
            # Handle video files - restore original data and move back to video_files
            if filename in st.session_state.original_file_data:
                # Recreate the original file object from stored data
                from ..components.utils import TempUploadedFile
                original_data = st.session_state.original_file_data[filename]
                
                # Create a new file object with the original data
                file_obj = TempUploadedFile(
                    name=original_data['name'],
                    data=original_data['data'],
                    type=original_data['type'],
                    size=original_data['size']
                )
                
                # Always add to video_files (handles duplicates properly)
                st.session_state.video_files.append(file_obj)
                
                # Remove from preprocessed_files
                st.session_state.preprocessed_files = [f for f in st.session_state.preprocessed_files if f is not file_obj]
                
                # Reset status and clear processed data
                st.session_state.file_status[filename] = 'pending'
                if filename in st.session_state.processed_data:
                    del st.session_state.processed_data[filename]
                
                # Reset metadata to only keep file size info
                if filename in st.session_state.file_metadata:
                    metadata = st.session_state.file_metadata[filename]
                    if 'file_size' in metadata:
                        file_size = metadata['file_size']
                        file_size_formatted = metadata['file_size_formatted']
                        st.session_state.file_metadata[filename] = {
                            'file_size': file_size,
                            'file_size_formatted': file_size_formatted
                        }
                    else:
                        # If no file size info, remove the metadata entry
                        del st.session_state.file_metadata[filename]
                
                reset_count += 1
            else:
                # Fallback: try to find the file object from preprocessed_files
                file_obj = None
                for preprocessed_file in st.session_state.preprocessed_files:
                    if preprocessed_file.name == filename:
                        file_obj = preprocessed_file
                        break
                
                if file_obj:
                    # Always add to video_files (handles duplicates properly)
                    st.session_state.video_files.append(file_obj)
                    
                    # Remove from preprocessed_files
                    st.session_state.preprocessed_files = [f for f in st.session_state.preprocessed_files if f is not file_obj]
                    
                    # Reset status and clear processed data
                    st.session_state.file_status[filename] = 'pending'
                    if filename in st.session_state.processed_data:
                        del st.session_state.processed_data[filename]
                    
                    # Reset metadata to only keep file size info
                    if filename in st.session_state.file_metadata:
                        metadata = st.session_state.file_metadata[filename]
                        if 'file_size' in metadata:
                            file_size = metadata['file_size']
                            file_size_formatted = metadata['file_size_formatted']
                            st.session_state.file_metadata[filename] = {
                                'file_size': file_size,
                                'file_size_formatted': file_size_formatted
                            }
                        else:
                            # If no file size info, remove the metadata entry
                            del st.session_state.file_metadata[filename]
                    
                    reset_count += 1
        else:
            # Handle NPZ files - just reset status and clear processed data
            st.session_state.file_status[filename] = 'pending'
            if filename in st.session_state.processed_data:
                del st.session_state.processed_data[filename]
            
            # Reset metadata to only keep file size and file type
            if filename in st.session_state.file_metadata:
                if 'file_size' in st.session_state.file_metadata[filename]:
                    file_size = st.session_state.file_metadata[filename]['file_size']
                    file_size_formatted = st.session_state.file_metadata[filename]['file_size_formatted']
                    # Keep file type if available
                    file_type = st.session_state.file_metadata[filename].get('file_type', 'npz')
                    st.session_state.file_metadata[filename] = {
                        'file_size': file_size,
                        'file_size_formatted': file_size_formatted,
                        'file_type': file_type
                    }
                else:
                    # If no file size info, keep minimal metadata to avoid errors
                    st.session_state.file_metadata[filename] = {'file_type': 'npz'}
            
            reset_count += 1
    
    # Clear preprocessed files list (video files have been moved back to video_files)
    st.session_state.preprocessed_files = []
    
    # Clear current tab
    st.session_state.current_tab = None
    
    # Clear all CTC prediction results and tracking keys when resetting files
    ctc_keys_to_remove = [key for key in st.session_state.keys() 
                          if (key.startswith("ctc_results_") or 
                              key.startswith("previous_ctc_model_") or 
                              key.startswith("previous_decode_method_"))]
    for key in ctc_keys_to_remove:
        del st.session_state[key]
    
    if reset_count > 0:
        st.toast(f"Reset {reset_count} files back to pending status", icon="🔄", duration=5000)
    else:
        st.toast("No processed files to reset", icon="ℹ️", duration=5000)


def clear_all_predictions_files():
    """Clear all files from predictions stage."""
    # Clear all files from all stages
    from .upload_manager import clear_all_files
    clear_all_files()
    
    # Clean up models
    model_manager = get_model_manager()
    model_manager.cleanup()
    
    # Return to upload stage
    st.session_state.workflow_stage = 'upload'
    st.rerun()


def cleanup_on_app_exit():
    """Clean up resources when the app exits."""
    try:
        model_manager = get_model_manager()
        model_manager.cleanup()
    except:
        pass
