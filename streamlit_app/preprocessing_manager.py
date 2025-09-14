"""Preprocessing manager for handling video file preprocessing workflow."""

import streamlit as st
from typing import List, Dict
from streamlit_app.utils import detect_file_type, format_file_size
from streamlit_app.data_processing import process_video_file
from streamlit_app.upload_manager import remove_file_from_stage


def get_all_files_to_show():
    """Get all files to show in the preprocessing stage."""
    all_files_to_show = []
    video_files = st.session_state.video_files
    
    # Add video files that are still pending
    for video_file in video_files:
        filename = video_file.name
        status = st.session_state.file_status.get(filename, 'pending')
        if status == 'pending':
            all_files_to_show.append(('video', video_file, status))
        else:
            # Show completed/error files from video_files as well
            all_files_to_show.append(('video', video_file, status))
    
    # Add preprocessed files
    for preprocessed_file in st.session_state.preprocessed_files:
        filename = preprocessed_file.name
        status = st.session_state.file_status.get(filename, 'completed')
        all_files_to_show.append(('preprocessed', preprocessed_file, status))
    
    return all_files_to_show


def render_preprocessing_stage():
    """Render the preprocessing stage with video files and preprocessing controls."""
    # Navigation header
    col1, col2, col3, col4 = st.columns([2, 6, 1, 1])
    with col1:
        if st.button("← Back to Upload", help="Return to upload stage", type="secondary"):
            if st.session_state.get("confirm_back_preprocessing", False):
                # Clear confirmation state
                if "confirm_back_preprocessing" in st.session_state:
                    del st.session_state["confirm_back_preprocessing"]
                
                # Clear video files when going back to upload
                st.session_state.video_files = []
                st.session_state.preprocessed_files = []
                
                st.session_state.workflow_stage = 'upload'
                st.rerun()
            else:
                st.session_state["confirm_back_preprocessing"] = True
                st.toast("Click '← Back to Upload' again to confirm", icon="⚠️", duration=5000)
    with col2:
        st.markdown("")  # Empty space
    with col3:
        st.markdown("")  # Empty space
    with col4:
        # Check if we can show Go to Inference button
        all_files_to_show = get_all_files_to_show()
        if all_files_to_show:
            # Check if all video files are completed
            all_completed = True
            for (file_type, uploaded_file, status) in all_files_to_show:
                if status not in ['completed']:
                    all_completed = False
                    break
            
            button_disabled = not all_completed
            button_help = "Proceed to inference with all processed files" if all_completed else "Complete all video preprocessing first"
            
            if st.button("Go to Inference →", type="primary", help=button_help, disabled=button_disabled):
                st.session_state.workflow_stage = 'predictions'
                st.rerun()
        else:
            st.markdown("")  # Empty space
    
    st.markdown("### Ready for Preprocessing")
    
    video_files = st.session_state.video_files
    
    # Show all files (both pending videos and completed preprocessing)
    all_files_to_show = get_all_files_to_show()
    
    if not all_files_to_show:
        st.info("No video files to preprocess.")
        
        # Check if we have NPZ files to proceed to predictions
        if st.session_state.npz_files:
            st.markdown("---")
            col1, col2, col3 = st.columns([4, 1, 1])
            with col3:
                if st.button("Go to Inference", type="primary", help="Proceed to inference stage"):
                    st.session_state.workflow_stage = 'predictions'
                    st.rerun()
        else:
            # If no video files and no NPZ files, automatically go back to upload
            st.info("All video files have been cleared. Redirecting to upload stage...")
            st.session_state.workflow_stage = 'upload'
            st.rerun()
        return
    
    # Batch preprocessing controls (needed first to update session state)
    render_preprocessing_controls(video_files)
    
    # Show video files with preprocessing options
    render_video_files_list(all_files_to_show)
    
    # Show progress and completed files
    if st.session_state.preprocessed_files:
        render_preprocessed_files_summary(all_files_to_show)


def render_video_files_list(all_files_to_show: List):
    """Render list of video files with individual preprocessing options."""
    st.markdown("**Video Files Ready for Preprocessing:**")
    
    # Check if any files are currently being processed
    is_processing = any(st.session_state.file_status.get(f.name, 'pending') == 'processing' for f in st.session_state.video_files)
    is_processing = is_processing or any(st.session_state.file_status.get(f.name, 'completed') == 'processing' for f in st.session_state.preprocessed_files)
    
    # Get current extraction options from session state (will be updated by checkboxes)
    options = st.session_state.get('preprocessing_options', {
        'write_keypoints': True,
        'write_iv3_features': True
    })
    has_extraction_options = options.get('write_keypoints', True) or options.get('write_iv3_features', True)
    
    for i, (file_type, uploaded_file, status) in enumerate(all_files_to_show):
        filename = uploaded_file.name
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
                button_disabled = is_processing or not has_extraction_options
                button_help = "Preprocess this video file" if has_extraction_options else "Select at least one extraction option to enable preprocessing"
                
                if st.button("Preprocess", key=f"preprocess_{filename}", help=button_help, type="primary", disabled=button_disabled):
                    preprocess_single_video(uploaded_file, filename)
                    st.rerun()
            elif status == 'completed':
                if st.button("View", key=f"view_{filename}", help="View preprocessed file", type="secondary", disabled=is_processing):
                    # Move to predictions stage and set as current tab
                    st.session_state.workflow_stage = 'predictions'
                    st.session_state.current_tab = filename
                    st.rerun()
            elif status == 'error':
                button_disabled = is_processing or not has_extraction_options
                button_help = "Retry preprocessing" if has_extraction_options else "Select at least one extraction option to enable preprocessing"
                
                if st.button("Retry", key=f"retry_{filename}", help=button_help, type="primary", disabled=button_disabled):
                    preprocess_single_video(uploaded_file, filename)
                    st.rerun()
        
        # Remove button with confirmation
        with col5:
            if st.button("Remove", key=f"remove_{filename}", help="Remove this file", type="secondary", disabled=is_processing):
                if st.session_state.get(f"confirm_remove_{filename}", False):
                    if file_type == 'video':
                        remove_file_from_stage(filename, 'video')
                    else:
                        remove_file_from_stage(filename, 'preprocessed')
                    st.rerun()
                else:
                    st.session_state[f"confirm_remove_{filename}"] = True
                    st.toast(f"Click 'Remove' again to confirm removal of {filename}", icon="⚠️", duration=5000)
        
        # Add separator line only if not the last file
        if i < len(all_files_to_show) - 1:
            st.markdown("---")


def render_preprocessing_controls(video_files: List):
    """Render batch preprocessing controls and options."""
    st.markdown("---")
    
    # Check if any files are currently being processed
    is_processing = any(st.session_state.file_status.get(f.name, 'pending') == 'processing' for f in video_files)
    is_processing = is_processing or any(st.session_state.file_status.get(f.name, 'completed') == 'processing' for f in st.session_state.preprocessed_files)
    
    # Preprocessing options
    with st.expander("Preprocessing Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            target_fps = st.slider(
                "Target FPS", 
                min_value=15, max_value=60, value=30, step=5,
                help="Frames per second for processing",
                disabled=is_processing
            )
            out_size = st.slider(
                "Output Size", 
                min_value=128, max_value=512, value=256, step=32,
                help="Frame size for processing",
                disabled=is_processing
            )
        
        with col2:
            write_keypoints = st.checkbox(
                "Extract Keypoints (156-D)", 
                value=True,
                help="Extract pose keypoint features for Transformer model",
                disabled=is_processing
            )
            write_iv3_features = st.checkbox(
                "Extract IV3 Features (2048-D)", 
                value=True,
                help="Extract InceptionV3 features for IV3-GRU model",
                disabled=is_processing
            )
        
        # Store options in session state
        st.session_state.preprocessing_options = {
            'target_fps': target_fps,
            'out_size': out_size,
            'write_keypoints': write_keypoints,
            'write_iv3_features': write_iv3_features
        }
        
        # Validate that at least one extraction option is selected
        has_extraction_options = write_keypoints or write_iv3_features
        if not has_extraction_options:
            st.warning("⚠️ Please select at least one extraction option (Keypoints or IV3 Features) to enable preprocessing.")
    
    # Batch operations
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    
    with col1:
        # Count pending files from both video_files and preprocessed_files
        pending_count = 0
        for f in video_files:
            if st.session_state.file_status.get(f.name, 'pending') == 'pending':
                pending_count += 1
        
        # Also check preprocessed_files for any that might be pending (after reset)
        for f in st.session_state.preprocessed_files:
            if st.session_state.file_status.get(f.name, 'completed') == 'pending':
                pending_count += 1
        
        if pending_count > 0:
            # Check if at least one extraction option is selected
            options = st.session_state.get('preprocessing_options', {})
            has_extraction_options = options.get('write_keypoints', True) or options.get('write_iv3_features', True)
            
            button_disabled = is_processing or not has_extraction_options
            button_help = "Preprocess all pending video files" if has_extraction_options else "Select at least one extraction option to enable preprocessing"
            
            if st.button(f"Preprocess All Pending ({pending_count})", type="primary", help=button_help, disabled=button_disabled):
                preprocess_all_pending_videos()
                st.rerun()
        else:
            st.info("All video files have been processed")
    
    with col2:
        st.markdown("")  # Empty space for alignment
    
    with col3:
        if st.button("Reset All", help="Reset all processed videos back to pending", type="primary", disabled=is_processing):
            reset_preprocessed_videos()
            st.rerun()
    
    with col4:
        if st.button("Clear All", help="Clear all video files", type="primary", disabled=is_processing):
            if st.session_state.get("confirm_clear_videos", False):
                clear_all_video_files()
                st.rerun()
            else:
                st.session_state["confirm_clear_videos"] = True
                st.toast("Click 'Clear All' again to confirm clearing all video files", icon="⚠️", duration=5000)


def render_preprocessed_files_summary(all_files_to_show: List):
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
    
    # Note: Go to Inference button is now in the navigation header


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
        
        # Store original file data for reset functionality
        st.session_state.original_file_data[filename] = {
            'name': uploaded_file.name,
            'data': uploaded_file.getvalue(),
            'type': uploaded_file.type,
            'size': uploaded_file.size
        }
        
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
    pending_files = []
    
    # Get pending files from video_files
    for f in st.session_state.video_files:
        if st.session_state.file_status.get(f.name, 'pending') == 'pending':
            pending_files.append(f)
    
    # Get pending files from preprocessed_files (after reset)
    for f in st.session_state.preprocessed_files:
        if st.session_state.file_status.get(f.name, 'completed') == 'pending':
            pending_files.append(f)
    
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
    
    # Initialize original_file_data if it doesn't exist
    if 'original_file_data' not in st.session_state:
        st.session_state.original_file_data = {}
    
    # Debug logging
    st.write("🔍 **Reset Debug Info:**")
    st.write(f"- original_file_data keys: {list(st.session_state.original_file_data.keys())}")
    st.write(f"- preprocessed_files: {[f.name for f in st.session_state.preprocessed_files]}")
    st.write(f"- video_files: {[f.name for f in st.session_state.video_files]}")
    
    # Collect all filenames that need to be reset
    files_to_reset = set()
    
    # Add all preprocessed files
    for preprocessed_file in st.session_state.preprocessed_files:
        files_to_reset.add(preprocessed_file.name)
    
    # Add all video files that have been processed
    for video_file in st.session_state.video_files:
        filename = video_file.name
        if st.session_state.file_status.get(filename) in ['completed', 'error']:
            files_to_reset.add(filename)
    
    # Reset each file
    for filename in files_to_reset:
        st.write(f"🔄 **Resetting file: {filename}**")
        
        # Check if we have original file data stored
        if filename in st.session_state.original_file_data:
            st.write(f"✅ Found original data for {filename}")
            # Recreate the original file object from stored data
            from streamlit_app.utils import TempUploadedFile
            original_data = st.session_state.original_file_data[filename]
            
            st.write(f"- Original data type: {type(original_data.get('data', 'No data'))}")
            st.write(f"- Original data size: {len(original_data.get('data', b''))} bytes")
            
            # Create a new file object with the original data
            file_obj = TempUploadedFile(
                name=original_data['name'],
                data=original_data['data'],
                type=original_data['type'],
                size=original_data['size']
            )
            
            st.write(f"- Created file object: {type(file_obj)}")
            st.write(f"- File object data size: {len(file_obj.getvalue())} bytes")
            
            # Add to video_files if not already there
            if not any(f.name == filename for f in st.session_state.video_files):
                st.session_state.video_files.append(file_obj)
            
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
            st.write(f"❌ No original data found for {filename}, using fallback")
            # Fallback: try to find the file object from any of the lists
            file_obj = None
            
            # Check in preprocessed_files first
            for preprocessed_file in st.session_state.preprocessed_files:
                if preprocessed_file.name == filename:
                    file_obj = preprocessed_file
                    st.write(f"- Found in preprocessed_files: {type(file_obj)}")
                    break
            
            # If not found, check in video_files
            if file_obj is None:
                for video_file in st.session_state.video_files:
                    if video_file.name == filename:
                        file_obj = video_file
                        st.write(f"- Found in video_files: {type(file_obj)}")
                        break
            
            # If not found, check in uploaded_files
            if file_obj is None:
                for uploaded_file in st.session_state.uploaded_files:
                    if uploaded_file.name == filename:
                        file_obj = uploaded_file
                        st.write(f"- Found in uploaded_files: {type(file_obj)}")
                        break
            
            if file_obj:
                st.write(f"- Using fallback file object: {type(file_obj)}")
                st.write(f"- Fallback data size: {len(file_obj.getvalue())} bytes")
                # Add to video_files if not already there
                if not any(f.name == filename for f in st.session_state.video_files):
                    st.session_state.video_files.append(file_obj)
                
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
    
    # If no NPZ files either, automatically go back to upload
    if not st.session_state.npz_files:
        st.session_state.workflow_stage = 'upload'
        st.rerun()
