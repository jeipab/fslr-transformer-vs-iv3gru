"""Preprocessing manager for handling video file preprocessing workflow."""

import streamlit as st
from typing import List, Dict
from pathlib import Path
from ..components.utils import detect_file_type, format_file_size
from ..components.data_processing import process_videos_unified
from .upload_manager import remove_file_from_stage
from ..core.config import PROCESSING_CONFIG


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
    # Add CSS to prevent button text wrapping
    st.markdown("""
    <style>
    .stButton > button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation header
    col1, col2, col3, col4 = st.columns([2, 8.5, 5.5, 2])
    with col1:
        if st.button("← Back to Upload", help="Return to upload stage", type="secondary"):
            # Clear video files when going back to upload
            st.session_state.video_files = []
            st.session_state.preprocessed_files = []
            
            st.session_state.workflow_stage = 'upload'
            st.rerun()
    with col2:
        st.markdown("")  # Empty space
    with col3:
        st.markdown("")  # Empty space
    with col4:
        # Always show Go to Inference button - check for any available files
        all_files_to_show = get_all_files_to_show()
        
        # Check preprocessing status and available files
        has_pending_videos = False
        has_completed_preprocessing = False
        
        if all_files_to_show:
            for (file_type, uploaded_file, status) in all_files_to_show:
                if status == 'pending' or status == 'processing':
                    has_pending_videos = True
                elif status == 'completed':
                    has_completed_preprocessing = True
        
        has_npz_files = bool(st.session_state.npz_files)
        
        # Disable button if there are pending video files, even if NPZ files are available
        if has_pending_videos:
            button_disabled = True
            button_help = "Complete all video preprocessing first before proceeding to inference"
        elif has_completed_preprocessing or has_npz_files:
            button_disabled = False
            if has_completed_preprocessing and has_npz_files:
                button_help = "Proceed to inference with all available files (preprocessed + NPZ)"
            elif has_completed_preprocessing:
                button_help = "Proceed to inference with preprocessed files"
            else:
                button_help = "Proceed to inference with NPZ files"
        else:
            button_disabled = True
            button_help = "No files available for inference - complete preprocessing or upload NPZ files"
        
        if st.button("Go to Inference →", type="primary", help=button_help, disabled=button_disabled):
            st.session_state.workflow_stage = 'predictions'
            st.rerun()
    
    st.markdown("<div class='main-section-header'>READY FOR PREPROCESSING</div>", unsafe_allow_html=True)
    
    video_files = st.session_state.video_files
    
    # Show all files (both pending videos and completed preprocessing)
    all_files_to_show = get_all_files_to_show()
    
    if not all_files_to_show:
        st.info("No video files to preprocess.")
        
        # Check if we should redirect to upload (no files at all)
        if not st.session_state.npz_files:
            # If no video files and no NPZ files, automatically go back to upload
            st.info("All video files have been cleared. Redirecting to upload stage...")
            st.session_state.workflow_stage = 'upload'
            st.rerun()
        return
    
    # Preprocessing options removed - using default configuration
    
    # Show video files with preprocessing options
    render_video_files_list(all_files_to_show)
    
    # Batch operations
    render_batch_operations(video_files)
    
    # Show progress and completed files
    if st.session_state.preprocessed_files:
        render_preprocessed_files_summary(all_files_to_show)


def render_video_files_list(all_files_to_show: List):
    """Render list of video files with individual preprocessing options."""
    st.markdown("**Video Files Ready for Preprocessing:**")
    
    # Check if any files are currently being processed
    is_processing = any(st.session_state.file_status.get(f.name, 'pending') == 'processing' for f in st.session_state.video_files)
    is_processing = is_processing or any(st.session_state.file_status.get(f.name, 'completed') == 'processing' for f in st.session_state.preprocessed_files)
    
    # Use default extraction options
    options = get_default_preprocessing_options()
    has_extraction_options = options.get('write_keypoints', True) or options.get('write_iv3_features', True)
    
    for i, (file_type, uploaded_file, status) in enumerate(all_files_to_show):
        filename = uploaded_file.name
        metadata = st.session_state.file_metadata.get(filename, {})
        file_size = metadata.get('file_size_formatted', 'Unknown')
        
        # Create unique key for this file instance using index to handle duplicates
        unique_key_suffix = f"{filename}_{i}"
        
        # Status emoji
        status_emoji = {
            'pending': '⏳',
            'processing': '🔄', 
            'completed': '✅',
            'error': '❌'
        }
        
        # Create compact file row with download button
        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            st.markdown(f"**Size:** {file_size}")
        with col3:
            st.markdown(f"**Status:** {status_emoji.get(status, '❓')} {status.title()}")
        
        # Download button (for completed files)
        with col4:
            if status == 'completed':
                npz_data = st.session_state.processed_data.get(filename)
                if npz_data:
                    from ..components.utils import create_npz_bytes
                    npz_bytes = create_npz_bytes(npz_data)
                    # Create descriptive filename with timestamp
                    import datetime
                    base_name = Path(filename).stem  # Remove original extension
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"{base_name}_preprocessed_{timestamp}.npz"
                    st.download_button(
                        label="Download",
                        data=npz_bytes,
                        file_name=download_filename,
                        mime="application/octet-stream",
                        key=f"download_{unique_key_suffix}",
                        help="Download NPZ file",
                        disabled=is_processing
                    )
        
        # Action buttons based on status
        with col5:
            if status == 'pending':
                button_disabled = is_processing or not has_extraction_options
                button_help = "Preprocess this video file" if has_extraction_options else "Select at least one extraction option to enable preprocessing"
                
                if st.button("Preprocess", key=f"preprocess_{unique_key_suffix}", help=button_help, type="primary", disabled=button_disabled):
                    preprocess_single_video(uploaded_file, filename)
                    st.rerun()
            elif status == 'completed':
                if st.button("View", key=f"view_{unique_key_suffix}", help="View preprocessed file", type="secondary", disabled=is_processing):
                    # Move to predictions stage and set as current tab
                    st.session_state.workflow_stage = 'predictions'
                    st.session_state.current_tab = filename
                    st.rerun()
            elif status == 'error':
                button_disabled = is_processing or not has_extraction_options
                button_help = "Retry preprocessing" if has_extraction_options else "Select at least one extraction option to enable preprocessing"
                
                if st.button("Retry", key=f"retry_{unique_key_suffix}", help=button_help, type="primary", disabled=button_disabled):
                    preprocess_single_video(uploaded_file, filename)
                    st.rerun()
        
        # Remove button with confirmation
        with col6:
            if st.button("Remove", key=f"remove_{unique_key_suffix}", help="Remove this file", type="secondary", disabled=is_processing):
                remove_specific_file_instance_preprocessing(uploaded_file, file_type)
                st.rerun()
        
        # Add separator line only if not the last file
        if i < len(all_files_to_show) - 1:
            st.markdown("---")


def get_default_preprocessing_options():
    """Get default preprocessing options."""
    video_config = PROCESSING_CONFIG['video']
    return {
        'target_fps': video_config['target_fps'],
        'out_size': video_config['out_size'],
        'write_keypoints': video_config['write_keypoints'],
        'write_iv3_features': video_config['write_iv3_features'],
        'occ_detailed': st.session_state.get('occ_detailed_checkbox', video_config['occ_detailed'])
    }


def render_batch_operations(video_files: List):
    """Render batch operation buttons."""
    st.markdown("---")
    
    # Check if any files are currently being processed
    is_processing = any(st.session_state.file_status.get(f.name, 'pending') == 'processing' for f in video_files)
    is_processing = is_processing or any(st.session_state.file_status.get(f.name, 'completed') == 'processing' for f in st.session_state.preprocessed_files)
    
    # Batch operations
    col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
    
    with col1:
        st.markdown("")  # Empty space for alignment
    
    with col2:
        st.markdown("")  # Empty space for alignment
    
    with col3:
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
            # Use default extraction options
            options = get_default_preprocessing_options()
            has_extraction_options = options.get('write_keypoints', True) or options.get('write_iv3_features', True)
            
            button_disabled = is_processing or not has_extraction_options
            button_help = "Preprocess all pending video files" if has_extraction_options else "Select at least one extraction option to enable preprocessing"
            
            if st.button(f"Preprocess All Pending ({pending_count})", type="primary", help=button_help, disabled=button_disabled):
                preprocess_all_pending_videos()
                st.rerun()
        else:
            st.markdown("")  # Empty space when no pending files
    
    with col4:
        # Download All NPZ button - aligned with individual download buttons
        preprocessed_files = st.session_state.preprocessed_files
        if preprocessed_files:
            create_bulk_download_button_inline(preprocessed_files, is_processing)
        else:
            st.markdown("")  # Empty space when no files
    
    with col5:
        if st.button("Reset All", help="Reset all processed videos back to pending", type="primary", disabled=is_processing):
            reset_preprocessed_videos()
            st.rerun()
    
    with col6:
        if st.button("Clear All", help="Clear all video files", type="primary", disabled=is_processing):
            clear_all_video_files()
            st.rerun()


def render_preprocessed_files_summary(all_files_to_show: List):
    """Render summary for preprocessed files (download button now in main button row)."""
    preprocessed_files = st.session_state.preprocessed_files
    
    if not preprocessed_files:
        return
    
    # Download button is now in the main button row, so this function can be minimal
    # or removed entirely if no other summary is needed


def preprocess_single_video(uploaded_file, filename: str):
    """Preprocess a single video file and update session state."""
    try:
        st.session_state.file_status[filename] = 'processing'
        
        # Get default preprocessing options
        options = get_default_preprocessing_options()
        
        # Process video file using unified processing (with GPU acceleration)
        with st.spinner(f"Preprocessing {filename}..."):
            processed_results = process_videos_unified(
                [uploaded_file],  # Pass as list for unified processing
                target_fps=options['target_fps'],
                out_size=options['out_size'],
                write_keypoints=options['write_keypoints'],
                write_iv3_features=options['write_iv3_features'],
                occ_detailed=options['occ_detailed']
            )
            # Extract single result
            npz_data = processed_results.get(Path(uploaded_file.name).stem, {})
        
        # Check compatibility
        from ..components.utils import check_npz_compatibility
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
            'preprocessing_options': options,
            # Respect the user's selected mode at preprocessing time
            'is_continuous': st.session_state.get('recognition_mode', 'isolated') == 'continuous',
            'continuous_metadata': None
        }
        
        # Move from video_files to preprocessed_files
        st.session_state.video_files = [f for f in st.session_state.video_files if f.name != filename]
        
        # Create a mock uploaded file object for the preprocessed file
        from ..components.utils import TempUploadedFile
        preprocessed_file = TempUploadedFile(filename, b"")  # Empty content since data is in processed_data
        st.session_state.preprocessed_files.append(preprocessed_file)
        
        st.session_state.file_status[filename] = 'completed'
        
        # Store compatibility info (success toast will be shown in batch summary)
        compatible_models = []
        if compatibility['transformer']:
            compatible_models.append("Transformer")
        if compatibility['iv3_gru']:
            compatible_models.append("IV3-GRU")
        
    except Exception as e:
        st.session_state.file_status[filename] = 'error'
        # Error will be shown in batch summary


def preprocess_all_pending_videos():
    """Preprocess all pending video files using automatic multi-processing."""
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
    
    # Use multi-processing for multiple files, single processing for one file
    if len(pending_files) == 1:
        # Single file - use existing single processing
        preprocess_single_video(pending_files[0], pending_files[0].name)
    else:
        # Multiple files - use multi-processing
        preprocess_multiple_videos_batch(pending_files)
    
    # Show consolidated summary
    completed_count = len(st.session_state.preprocessed_files)
    error_files = [f for f in st.session_state.video_files 
                  if st.session_state.file_status.get(f.name) == 'error']
    
    if completed_count > 0:
        st.toast(f"{completed_count} videos preprocessed successfully", icon="✅", duration=5000)
    
    if error_files:
        st.toast(f"{len(error_files)} videos failed to preprocess", icon="❌", duration=5000)


def preprocess_multiple_videos_batch(uploaded_files):
    """Preprocess multiple videos using automatic device detection and worker optimization."""
    try:
        # Get default preprocessing options
        options = get_default_preprocessing_options()
        
        # Set all files to processing status
        for uploaded_file in uploaded_files:
            st.session_state.file_status[uploaded_file.name] = 'processing'
        
        # Process videos using unified processing (with GPU acceleration)
        with st.spinner(f"Preprocessing {len(uploaded_files)} videos..."):
            processed_results = process_videos_unified(
                uploaded_files,
                target_fps=options['target_fps'],
                out_size=options['out_size'],
                write_keypoints=options['write_keypoints'],
                write_iv3_features=options['write_iv3_features'],
                occ_detailed=options['occ_detailed']
            )
        
        # Process results and update session state
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            basename = Path(filename).stem
            
            if basename in processed_results and processed_results[basename]:
                npz_data = processed_results[basename]
                
                # Check compatibility
                from ..components.utils import check_npz_compatibility
                compatibility = check_npz_compatibility(npz_data)
                
                if not any(compatibility.values()):
                    st.session_state.file_status[filename] = 'error'
                    st.toast(f"{filename}: Preprocessing failed - incompatible output", icon="❌", duration=5000)
                    continue
                
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
                    'preprocessing_options': options,
                    # Respect the user's selected mode at preprocessing time
                    'is_continuous': st.session_state.get('recognition_mode', 'isolated') == 'continuous',
                    'continuous_metadata': None
                }
                
                # Move from video_files to preprocessed_files
                st.session_state.video_files = [f for f in st.session_state.video_files if f.name != filename]
                
                # Create a mock uploaded file object for the preprocessed file
                from ..components.utils import TempUploadedFile
                preprocessed_file = TempUploadedFile(filename, b"")  # Empty content since data is in processed_data
                st.session_state.preprocessed_files.append(preprocessed_file)
                
                st.session_state.file_status[filename] = 'completed'
                
                # Store compatibility info (success will be shown in batch summary)
                compatible_models = []
                if compatibility['transformer']:
                    compatible_models.append("Transformer")
                if compatibility['iv3_gru']:
                    compatible_models.append("IV3-GRU")
            else:
                st.session_state.file_status[filename] = 'error'
        
    except Exception as e:
        # Set all files to error status
        for uploaded_file in uploaded_files:
            st.session_state.file_status[uploaded_file.name] = 'error'
        st.toast(f"Multi-processing failed: {str(e)}", icon="❌", duration=5000)


def reset_preprocessed_videos():
    """Reset all preprocessed videos back to pending status."""
    reset_count = 0
    
    # Initialize original_file_data if it doesn't exist
    if 'original_file_data' not in st.session_state:
        st.session_state.original_file_data = {}
    
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
        # Check if we have original file data stored
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
            # No original data found, use fallback method
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


def create_bulk_download_button_inline(preprocessed_files: List, is_processing: bool = False):
    """Create inline bulk download button for the button row."""
    import zipfile
    import io
    from pathlib import Path
    from ..components.utils import create_npz_bytes
    
    if is_processing:
        st.button("Download All", disabled=True, help="⏳ Please wait for all files to finish processing before downloading.")
        return
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all preprocessed NPZ files with unique names to handle duplicates
        files_added = 0
        used_names = set()
        for preprocessed_file in preprocessed_files:
            filename = preprocessed_file.name
            npz_data = st.session_state.processed_data.get(filename)
            if npz_data:
                npz_bytes = create_npz_bytes(npz_data)
                # Create descriptive filename for ZIP contents
                base_name = Path(filename).stem  # Remove original extension
                base_npz_filename = f"{base_name}_preprocessed.npz"
                
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
                files_added += 1
        
        if files_added == 0:
            st.button("Download All", disabled=True, help="No NPZ files available for download")
            return
    
    zip_buffer.seek(0)
    
    # Create descriptive ZIP filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"preprocessed_files_{timestamp}.zip"
    
    # Inline download button
    st.download_button(
        label="Download All",
        data=zip_buffer.getvalue(),
        file_name=zip_filename,
        mime="application/zip",
        help=f"Download {files_added} preprocessed NPZ files as ZIP",
        type="primary"
    )


def create_bulk_download_button(preprocessed_files: List):
    """Create bulk download button for all preprocessed NPZ files."""
    import zipfile
    import io
    from ..components.utils import create_npz_bytes
    
    # Check if any files are currently being processed
    is_processing = any(st.session_state.file_status.get(f.name, 'completed') == 'processing' for f in preprocessed_files)
    
    if is_processing:
        st.info("⏳ Please wait for all files to finish processing before downloading.")
        return
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all preprocessed NPZ files with unique names to handle duplicates
        files_added = 0
        used_names = set()
        for preprocessed_file in preprocessed_files:
            filename = preprocessed_file.name
            npz_data = st.session_state.processed_data.get(filename)
            if npz_data:
                npz_bytes = create_npz_bytes(npz_data)
                # Create descriptive filename for ZIP contents
                base_name = Path(filename).stem  # Remove original extension
                base_npz_filename = f"{base_name}_preprocessed.npz"
                
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
                files_added += 1
            else:
                st.warning(f"⚠️ No NPZ data found for {filename}")
        
        if files_added == 0:
            st.error("No NPZ files available for download")
            return
    
    zip_buffer.seek(0)
    
    # Create descriptive ZIP filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"preprocessed_files_{timestamp}.zip"
    
    # Download button
    st.download_button(
        label="Download All NPZ Files as ZIP",
        data=zip_buffer.getvalue(),
        file_name=zip_filename,
        mime="application/zip",
        type="primary",
        help="Download all preprocessed NPZ files as a ZIP archive",
        disabled=is_processing,
        key="download_all_zip_preprocessing"
    )


def remove_specific_file_instance_preprocessing(file_obj, file_type: str):
    """Remove a specific file instance from preprocessing stage."""
    filename = file_obj.name
    
    # Remove from appropriate file list by object reference
    if file_type == 'video':
        st.session_state.video_files = [f for f in st.session_state.video_files if f is not file_obj]
    else:  # preprocessed
        st.session_state.preprocessed_files = [f for f in st.session_state.preprocessed_files if f is not file_obj]
    
    # Remove from general uploaded files
    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f is not file_obj]
    
    # Only clear status/metadata if this is the last file with this name
    remaining_files_with_same_name = []
    for file_list in [st.session_state.video_files, st.session_state.preprocessed_files, st.session_state.uploaded_files]:
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
