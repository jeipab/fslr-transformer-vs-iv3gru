"""Preprocessing manager for handling video file preprocessing workflow."""

import streamlit as st
from typing import List, Dict
from streamlit_app.utils import detect_file_type, format_file_size
from streamlit_app.data_processing import process_video_file
from streamlit_app.upload_manager import remove_file_from_stage


def render_preprocessing_stage():
    """Render the preprocessing stage with video files and preprocessing controls."""
    # Navigation header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Upload", help="Return to upload stage", type="secondary"):
            if st.session_state.get("confirm_back_preprocessing", False):
                # Clear confirmation state
                if "confirm_back_preprocessing" in st.session_state:
                    del st.session_state["confirm_back_preprocessing"]
                st.session_state.workflow_stage = 'upload'
                st.rerun()
            else:
                st.session_state["confirm_back_preprocessing"] = True
                st.toast("Click '← Back to Upload' again to confirm", icon="⚠️", duration=5000)
    with col2:
        st.markdown("")  # Empty space
    with col3:
        st.markdown("")  # Empty space
    
    st.markdown("### Ready for Preprocessing")
    
    video_files = st.session_state.video_files
    
    if not video_files:
        st.info("No video files to preprocess.")
        
        # Check if we have NPZ files to proceed to predictions
        if st.session_state.npz_files or st.session_state.preprocessed_files:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Go to Inference", type="primary", help="Proceed to inference stage"):
                    st.session_state.workflow_stage = 'predictions'
                    st.rerun()
        return
    
    # Show video files with preprocessing options
    render_video_files_list(video_files)
    
    # Batch preprocessing controls
    render_preprocessing_controls(video_files)
    
    # Show progress and completed files
    if st.session_state.preprocessed_files:
        render_preprocessed_files_summary()


def render_video_files_list(video_files: List):
    """Render list of video files with individual preprocessing options."""
    st.markdown("**Video Files Ready for Preprocessing:**")
    
    for i, uploaded_file in enumerate(video_files):
        filename = uploaded_file.name
        status = st.session_state.file_status.get(filename, 'pending')
        metadata = st.session_state.file_metadata.get(filename, {})
        file_size = metadata.get('file_size_formatted', 'Unknown')
        
        # Status emoji
        status_emoji = {
            'pending': '⏳',
            'processing': '🔄', 
            'completed': '✅',
            'error': '❌'
        }
        
        # Create compact file row
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**🎥 {filename}**")
        with col2:
            st.markdown(f"**Size:** {file_size}")
        with col3:
            st.markdown(f"**Status:** {status_emoji.get(status, '❓')} {status.title()}")
        
        # Action buttons based on status
        with col4:
            if status == 'pending':
                if st.button("Preprocess", key=f"preprocess_{filename}", help="Preprocess this video file", type="primary"):
                    preprocess_single_video(uploaded_file, filename)
                    st.rerun()
            elif status == 'completed':
                if st.button("View", key=f"view_{filename}", help="View preprocessed file", type="secondary"):
                    # Move to predictions stage and set as current tab
                    st.session_state.workflow_stage = 'predictions'
                    st.session_state.current_tab = filename
                    st.rerun()
            elif status == 'error':
                if st.button("Retry", key=f"retry_{filename}", help="Retry preprocessing", type="primary"):
                    preprocess_single_video(uploaded_file, filename)
                    st.rerun()
        
        # Remove button with confirmation
        with col5:
            if st.button("Remove", key=f"remove_{filename}", help="Remove this file", type="secondary"):
                if st.session_state.get(f"confirm_remove_{filename}", False):
                    remove_file_from_stage(filename, 'video')
                    st.rerun()
                else:
                    st.session_state[f"confirm_remove_{filename}"] = True
                    st.toast(f"Click 'Remove' again to confirm removal of {filename}", icon="⚠️", duration=5000)
        
        # Add separator line only if not the last file
        if i < len(video_files) - 1:
            st.markdown("---")


def render_preprocessing_controls(video_files: List):
    """Render batch preprocessing controls and options."""
    st.markdown("---")
    
    # Preprocessing options
    with st.expander("Preprocessing Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            target_fps = st.slider(
                "Target FPS", 
                min_value=15, max_value=60, value=30, step=5,
                help="Frames per second for processing"
            )
            out_size = st.slider(
                "Output Size", 
                min_value=128, max_value=512, value=256, step=32,
                help="Frame size for processing"
            )
        
        with col2:
            write_keypoints = st.checkbox(
                "Extract Keypoints (156-D)", 
                value=True,
                help="Extract pose keypoint features for Transformer model"
            )
            write_iv3_features = st.checkbox(
                "Extract IV3 Features (2048-D)", 
                value=True,
                help="Extract InceptionV3 features for IV3-GRU model"
            )
        
        # Store options in session state
        st.session_state.preprocessing_options = {
            'target_fps': target_fps,
            'out_size': out_size,
            'write_keypoints': write_keypoints,
            'write_iv3_features': write_iv3_features
        }
    
    # Batch operations
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    
    with col1:
        pending_count = sum(1 for f in video_files 
                          if st.session_state.file_status.get(f.name, 'pending') == 'pending')
        if pending_count > 0:
            if st.button(f"Preprocess All Pending ({pending_count})", type="primary", help="Preprocess all pending video files"):
                preprocess_all_pending_videos()
                st.rerun()
        else:
            st.info("All video files have been processed")
    
    with col2:
        st.markdown("")  # Empty space for alignment
    
    with col3:
        if st.button("Reset All", help="Reset all processed videos back to pending", type="primary"):
            reset_preprocessed_videos()
            st.rerun()
    
    with col4:
        if st.button("Clear All", help="Clear all video files", type="primary"):
            if st.session_state.get("confirm_clear_videos", False):
                clear_all_video_files()
                st.rerun()
            else:
                st.session_state["confirm_clear_videos"] = True
                st.toast("Click 'Clear All' again to confirm clearing all video files", icon="⚠️", duration=5000)


def render_preprocessed_files_summary():
    """Render summary of preprocessed files."""
    st.markdown("---")
    st.markdown("### Preprocessed Files")
    
    preprocessed_files = st.session_state.preprocessed_files
    
    if not preprocessed_files:
        return
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Preprocessed Videos", len(preprocessed_files))
    
    with col2:
        # Count compatibility
        transformer_compatible = sum(1 for f in preprocessed_files 
                                   if st.session_state.file_metadata.get(f.name, {}).get('compatibility', {}).get('transformer', False))
        st.metric("Transformer Compatible", transformer_compatible)
    
    with col3:
        iv3_compatible = sum(1 for f in preprocessed_files 
                           if st.session_state.file_metadata.get(f.name, {}).get('compatibility', {}).get('iv3_gru', False))
        st.metric("IV3-GRU Compatible", iv3_compatible)
    
    # Proceed to predictions button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
                if st.button("Go to Inference", type="primary", help="Proceed to inference with all processed files"):
                    st.session_state.workflow_stage = 'predictions'
                    st.rerun()


def preprocess_single_video(uploaded_file, filename: str):
    """Preprocess a single video file and update session state."""
    try:
        st.session_state.file_status[filename] = 'processing'
        
        # Get preprocessing options
        options = st.session_state.get('preprocessing_options', {
            'target_fps': 30,
            'out_size': 256,
            'write_keypoints': True,
            'write_iv3_features': True
        })
        
        # Process video file
        with st.spinner(f"Preprocessing {filename}..."):
            npz_data = process_video_file(
                uploaded_file,
                target_fps=options['target_fps'],
                out_size=options['out_size'],
                write_keypoints=options['write_keypoints'],
                write_iv3_features=options['write_iv3_features']
            )
        
        # Check compatibility
        from streamlit_app.utils import check_npz_compatibility
        compatibility = check_npz_compatibility(npz_data)
        
        if not any(compatibility.values()):
            st.session_state.file_status[filename] = 'error'
            st.toast(f"{filename}: Preprocessing failed - incompatible output", icon="❌", duration=5000)
            return
        
        # Store processed data
        st.session_state.processed_data[filename] = npz_data
        
        # Update metadata
        existing_metadata = st.session_state.file_metadata.get(filename, {})
        st.session_state.file_metadata[filename] = {
            **existing_metadata,
            'compatibility': compatibility,
            'file_type': 'npz',
            'frame_count': npz_data['X'].shape[0] if 'X' in npz_data else npz_data['X2048'].shape[0] if 'X2048' in npz_data else 0,
            'source_type': 'video',
            'preprocessing_options': options
        }
        
        # Move from video_files to preprocessed_files
        st.session_state.video_files = [f for f in st.session_state.video_files if f.name != filename]
        
        # Create a mock uploaded file object for the preprocessed file
        from streamlit_app.utils import TempUploadedFile
        preprocessed_file = TempUploadedFile(filename, b"")  # Empty content since data is in processed_data
        st.session_state.preprocessed_files.append(preprocessed_file)
        
        st.session_state.file_status[filename] = 'completed'
        
        # Show success message
        compatible_models = []
        if compatibility['transformer']:
            compatible_models.append("Transformer")
        if compatibility['iv3_gru']:
            compatible_models.append("IV3-GRU")
        
        st.toast(f"{filename} preprocessed successfully - compatible with {', '.join(compatible_models)}", icon="✅", duration=5000)
        
    except Exception as e:
        st.session_state.file_status[filename] = 'error'
        st.toast(f"Preprocessing failed for {filename}: {str(e)}", icon="❌", duration=5000)


def preprocess_all_pending_videos():
    """Preprocess all pending video files."""
    pending_files = [f for f in st.session_state.video_files 
                    if st.session_state.file_status.get(f.name, 'pending') == 'pending']
    
    if not pending_files:
        st.toast("No pending video files to preprocess", icon="ℹ️", duration=5000)
        return
    
    # Process all files
    for uploaded_file in pending_files:
        preprocess_single_video(uploaded_file, uploaded_file.name)
    
    # Show consolidated summary
    completed_count = len(st.session_state.preprocessed_files)
    error_files = [f for f in st.session_state.video_files 
                  if st.session_state.file_status.get(f.name) == 'error']
    
    if completed_count > 0:
        st.toast(f"{completed_count} videos preprocessed successfully", icon="✅", duration=5000)
    
    if error_files:
        st.toast(f"{len(error_files)} videos failed to preprocess", icon="❌", duration=5000)


def reset_preprocessed_videos():
    """Reset all preprocessed videos back to pending status."""
    reset_count = 0
    
    # Move preprocessed files back to video files
    for preprocessed_file in st.session_state.preprocessed_files:
        filename = preprocessed_file.name
        # Find original uploaded file
        original_file = next((f for f in st.session_state.uploaded_files if f.name == filename), None)
        if original_file:
            st.session_state.video_files.append(original_file)
            reset_count += 1
        
        # Reset status and clear processed data
        st.session_state.file_status[filename] = 'pending'
        if filename in st.session_state.processed_data:
            del st.session_state.processed_data[filename]
        
        # Reset metadata
        if filename in st.session_state.file_metadata:
            metadata = st.session_state.file_metadata[filename]
            if 'file_size' in metadata:
                file_size = metadata['file_size']
                file_size_formatted = metadata['file_size_formatted']
                st.session_state.file_metadata[filename] = {
                    'file_size': file_size,
                    'file_size_formatted': file_size_formatted
                }
    
    # Clear preprocessed files list
    st.session_state.preprocessed_files = []
    
    if reset_count > 0:
        st.toast(f"Reset {reset_count} preprocessed videos back to pending", icon="🔄", duration=5000)
    else:
        st.toast("No preprocessed videos to reset", icon="ℹ️", duration=5000)


def clear_all_video_files():
    """Clear all video files from preprocessing stage."""
    # Remove all video files from session state
    for video_file in st.session_state.video_files:
        filename = video_file.name
        remove_file_from_stage(filename, 'video')
    
    # Clear preprocessed files as well
    for preprocessed_file in st.session_state.preprocessed_files:
        filename = preprocessed_file.name
        remove_file_from_stage(filename, 'preprocessed')
    
    st.session_state.video_files = []
    st.session_state.preprocessed_files = []
    
    st.toast("All video files cleared", icon="🗑️", duration=5000)
