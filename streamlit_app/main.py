"""Main Streamlit application for FSLR Demo."""

import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from streamlit_app.components import (
    set_page, render_sidebar, render_welcome_screen, 
    render_file_upload, render_main_header, render_predictions_section
)
from streamlit_app.data_processing import process_video_file
from streamlit_app.utils import detect_file_type, TempUploadedFile, check_npz_compatibility, create_npz_bytes
from streamlit_app.visualization import (
    render_sequence_overview, render_animated_keypoints, 
    render_feature_charts, create_video_with_keypoints
)


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


def initialize_session_state():
    """Initialize session state for multiple file processing."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'file_status' not in st.session_state:
        st.session_state.file_status = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'file_metadata' not in st.session_state:
        st.session_state.file_metadata = {}
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = None
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'workflow_stage' not in st.session_state:
        st.session_state.workflow_stage = 'upload'  # 'upload' or 'processing'
    if 'pending_upload_files' not in st.session_state:
        st.session_state.pending_upload_files = []




def render_file_management_ui():
    """Render file management with compact layout using native Streamlit components."""
    if not st.session_state.uploaded_files:
        return
        
    st.markdown("### Uploaded Files")
    
    # Add small space at top
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # File rows with native Streamlit components
    for i, uploaded_file in enumerate(st.session_state.uploaded_files):
        filename = uploaded_file.name
        status = st.session_state.file_status.get(filename, 'pending')
        file_type = detect_file_type(uploaded_file)
        metadata = st.session_state.file_metadata.get(filename, {})
        file_size = metadata.get('file_size_formatted', 'Unknown')
        
        # Status and type emojis
        status_emoji = {
            'pending': '⏳',
            'processing': '🔄', 
            'completed': '✅',
            'error': '❌'
        }
        type_emoji = '📄' if file_type == 'npz' else '🎥'
        
        # Create compact file row using Streamlit columns
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{type_emoji} {filename}**")
        with col2:
            st.markdown(f"**Size:** {file_size}")
        with col3:
            st.markdown(f"**Status:** {status_emoji.get(status, '❓')} {status.title()}")
        
        # Action buttons based on status
        with col4:
            if status == 'pending':
                if st.button("Process", key=f"process_{filename}", help="Process this file", type="primary"):
                    process_single_file(uploaded_file, filename)
                    st.rerun()
            elif status == 'completed':
                if st.button("View", key=f"view_{filename}", help="View this file", type="primary"):
                    st.session_state.current_tab = filename
                    st.rerun()
            elif status == 'error':
                if st.button("Retry", key=f"retry_{filename}", help="Retry processing", type="primary"):
                    process_single_file(uploaded_file, filename)
                    st.rerun()
        
        # Remove button with confirmation
        with col5:
            if st.button("Remove", key=f"remove_{filename}", help="Remove this file", type="secondary"):
                if st.session_state.get(f"confirm_remove_{filename}", False):
                    remove_file(filename)
                    st.rerun()
                else:
                    st.session_state[f"confirm_remove_{filename}"] = True
                    st.toast(f"Click 'Remove' again to confirm removal of {filename}", icon="⚠️", duration=5000)
        
        # Add separator line only if not the last file
        if i < len(st.session_state.uploaded_files) - 1:
            st.markdown("---")
    
    # Batch operations
    st.markdown("---")
    st.markdown("#### Begin Processing")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Process All Pending", type="primary", help="Process all pending files"):
            process_all_pending_files()
            st.rerun()
    
    with col2:
        if st.button("Clear All", help="Clear all files", type="secondary"):
            if st.session_state.get("confirm_clear_all", False):
                clear_all_files()
                st.rerun()
            else:
                st.session_state["confirm_clear_all"] = True
                st.toast("Click 'Clear All' again to confirm clearing all files", icon="⚠️", duration=5000)
    
    with col3:
        if st.button("Reset", help="Reset confirmation"):
            if "confirm_clear_all" in st.session_state:
                del st.session_state["confirm_clear_all"]
                st.rerun()


def process_single_file(uploaded_file, filename):
    """Process a single file and update session state."""
    try:
        st.session_state.file_status[filename] = 'processing'
        
        file_type = detect_file_type(uploaded_file)
        file_content = uploaded_file.read()
        
        if file_type == 'npz':
            # Load NPZ file
            file_bytes = io.BytesIO(file_content)
            npz_data = dict(np.load(file_bytes, allow_pickle=True))
            
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
                'file_type': file_type,
                'frame_count': npz_data['X'].shape[0] if 'X' in npz_data else npz_data['X2048'].shape[0] if 'X2048' in npz_data else 0
            }
            st.session_state.file_status[filename] = 'completed'
            
            # Show success message
            compatible_models = []
            if compatibility['transformer']:
                compatible_models.append("Transformer")
            if compatibility['iv3_gru']:
                compatible_models.append("IV3-GRU")
            
            # Don't show individual toasts for each file - will show batch summary
            
        elif file_type == 'video':
            # For now, just mark as pending for manual processing
            st.session_state.file_status[filename] = 'pending'
            st.toast(f"Video file ready for preprocessing", icon="🎥", duration=5000)
            
        else:
            st.session_state.file_status[filename] = 'error'
            st.toast(f"Unsupported file type", icon="❌", duration=5000)
            
    except Exception as e:
        st.session_state.file_status[filename] = 'error'
        st.toast(f"Processing failed - {str(e)}", icon="❌", duration=5000)


def process_all_pending_files():
    """Process all pending files in the queue."""
    pending_files = [f for f in st.session_state.uploaded_files 
                    if st.session_state.file_status.get(f.name, 'pending') == 'pending']
    
    if not pending_files:
        st.toast("No pending files to process", icon="ℹ️", duration=5000)
        return
    
    # Process all files
    for uploaded_file in pending_files:
        process_single_file(uploaded_file, uploaded_file.name)
    
    # Show consolidated summary
    completed_files = [f for f in st.session_state.uploaded_files 
                      if st.session_state.file_status.get(f.name) == 'completed']
    error_files = [f for f in st.session_state.uploaded_files 
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


def remove_file(filename, show_toast=True):
    """Remove a file from the queue and clean up session state."""
    # Remove from all session state
    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f.name != filename]
    
    if filename in st.session_state.file_status:
        del st.session_state.file_status[filename]
    if filename in st.session_state.processed_data:
        del st.session_state.processed_data[filename]
    if filename in st.session_state.file_metadata:
        del st.session_state.file_metadata[filename]
    
    # Reset current tab if it was the removed file
    if st.session_state.current_tab == filename:
        st.session_state.current_tab = None
    
    # Clear any confirmation states for this file
    for key in list(st.session_state.keys()):
        if key.startswith(f"confirm_remove_{filename}"):
            del st.session_state[key]
    
    if show_toast:
        st.toast(f"Removed {filename}", icon="🗑️", duration=5000)


def clear_all_files():
    """Clear all files from the queue and reset session state."""
    st.session_state.uploaded_files = []
    st.session_state.file_status = {}
    st.session_state.processed_data = {}
    st.session_state.file_metadata = {}
    st.session_state.current_tab = None
    st.session_state.selected_files = []
    st.session_state.pending_upload_files = []
    
    # Clear all confirmation states
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith("confirm_")]
    for key in keys_to_remove:
        del st.session_state[key]
    
    st.toast("All files cleared", icon="🗑️", duration=5000)


def cancel_processing():
    """Cancel processing and return to upload stage with confirmation."""
    if st.session_state.get("confirm_cancel", False):
        # Clear everything and return to upload
        clear_all_files()
        st.session_state.workflow_stage = 'upload'
        st.rerun()
    else:
        st.session_state["confirm_cancel"] = True
        st.toast("Click 'Cancel' again to confirm clearing all files and returning to upload", icon="⚠️", duration=5000)






def render_upload_stage():
    """Render the upload stage with file uploader and proceed button."""
    st.markdown("### Upload Data")
    
    # File uploader
    uploaded_files = render_file_upload()
    
    # Handle file limit
    if uploaded_files and len(uploaded_files) > 10:
        st.error("Maximum 10 files allowed. Please select fewer files.")
        return
    
    # Store pending files
    if uploaded_files:
        st.session_state.pending_upload_files = uploaded_files
        
        # Display selected files
        st.markdown("**Selected Files:**")
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            icon = "📄" if file_type == "npz" else "🎥"
            file_size = len(uploaded_file.getvalue())
            size_mb = file_size / (1024 * 1024)
            st.markdown(f"{icon} **{uploaded_file.name}** ({size_mb:.1f} MB)")
        
        # Proceed button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Proceed to Processing", type="primary", help="Move to processing stage"):
                # Move files to uploaded_files and switch to processing stage
                st.session_state.uploaded_files = uploaded_files
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    st.session_state.file_status[filename] = 'pending'
                    file_size = len(uploaded_file.getvalue())
                    st.session_state.file_metadata[filename] = {
                        'file_size': file_size,
                        'file_size_formatted': format_file_size(file_size)
                    }
                st.session_state.workflow_stage = 'processing'
                st.rerun()
    else:
        render_welcome_screen()


def render_processing_stage(cfg):
    """Render the processing stage with uploaded files and batch operations."""
    # Navigation header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Cancel", help="Clear all files and return to upload", type="secondary"):
            cancel_processing()
    with col2:
        st.markdown("")  # Empty space
    with col3:
        st.markdown("")  # Empty space
    
    # Show file management
    render_file_management_ui()
    
    # Show visualization tabs
    render_visualization_tabs(cfg)




def render_visualization_tabs(cfg):
    """Render visualization tabs for processed files."""
    completed_files = [f for f in st.session_state.uploaded_files 
                      if st.session_state.file_status.get(f.name) == 'completed']
    
    if not completed_files:
        return
    
    # Create tabs
    tab_names = []
    for uploaded_file in completed_files:
        filename = uploaded_file.name
        file_type = detect_file_type(uploaded_file)
        icon = "📄" if file_type == 'npz' else "🎥"
        tab_names.append(f"{icon} {filename}")
    
    # Add batch summary tab
    tab_names.append("📊 Batch Summary")
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    # Handle programmatic tab switching
    if st.session_state.current_tab:
        # Find the index of the current tab
        target_tab_index = None
        for i, uploaded_file in enumerate(completed_files):
            if uploaded_file.name == st.session_state.current_tab:
                target_tab_index = i
                break
        
        if target_tab_index is not None:
            # Use JavaScript to switch to the target tab
            switch_tab_script = f"""
            <script>
                setTimeout(function() {{
                    var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    if (tabs.length > {target_tab_index}) {{
                        tabs[{target_tab_index}].click();
                    }}
                }}, 100);
            </script>
            """
            components.html(switch_tab_script, height=0)
            
            # Clear the current_tab after switching
            st.session_state.current_tab = None
    
    # Individual file tabs
    for i, uploaded_file in enumerate(completed_files):
        with tabs[i]:
            filename = uploaded_file.name
            npz_data = st.session_state.processed_data[filename]
            metadata = st.session_state.file_metadata[filename]
            
            # File info
            st.markdown(f"### {filename}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frames", metadata['frame_count'])
            with col2:
                compatibility = metadata['compatibility']
                compatible_count = sum(compatibility.values())
                st.metric("Compatible Models", compatible_count)
            with col3:
                st.metric("File Type", metadata['file_type'].upper())
            
            # Show compatibility info
            compatible_models = []
            if compatibility['transformer']:
                compatible_models.append("Transformer")
            if compatibility['iv3_gru']:
                compatible_models.append("IV3-GRU")
            
            if compatible_models:
                st.info(f"Compatible with: {', '.join(compatible_models)}")
            
            # Render visualizations
            try:
                X_pad, mask, meta = render_sequence_overview(npz_data, cfg["sequence_length"])
                render_animated_keypoints(X_pad, mask if mask.size > 0 else None, key_suffix=filename)
                render_feature_charts(X_pad, mask if mask.size > 0 else None, key_suffix=filename)
                
                # Generate and render predictions
                render_predictions_section(cfg, None, None)
                
                # Download button
                st.markdown("### Download")
                npz_bytes = create_npz_bytes(npz_data)
                st.download_button(
                    label=f"Download {filename}",
                    data=npz_bytes,
                    file_name=filename,
                    mime="application/octet-stream"
                )
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
    
    # Batch summary tab
    with tabs[-1]:
        render_batch_summary_tab(cfg)


def render_batch_summary_tab(cfg):
    """Render batch summary tab with statistics and predictions."""
    st.markdown("### Batch Processing Summary")
    
    completed_files = [f for f in st.session_state.uploaded_files 
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
        
        # Generate predictions for this file
        from streamlit_app.utils import simulate_predictions, topk_from_logits
        import numpy as np
        
        # Use filename hash as seed for consistent predictions per file
        file_seed = hash(filename) % (2**32)
        rng = np.random.RandomState(file_seed)
        gloss_logits, cat_logits = simulate_predictions(
            rng, cfg["num_gloss_classes"], cfg["num_category_classes"]
        )
        
        # Get top 1 predictions
        g_idx, g_prob = topk_from_logits(gloss_logits, 1)
        c_idx, c_prob = topk_from_logits(cat_logits, 1)
        
        # Format predictions
        top_gloss = f"Gloss {g_idx[0]} ({g_prob[0]*100:.1f}%)"
        top_category = f"Category {c_idx[0]} ({c_prob[0]*100:.1f}%)"
        
        summary_data.append({
            'File': filename,
            'Type': metadata['file_type'].upper(),
            'Frames': metadata['frame_count'],
            'Transformer': 'Yes' if compatibility['transformer'] else 'No',
            'IV3-GRU': 'Yes' if compatibility['iv3_gru'] else 'No',
            'Top Gloss': top_gloss,
            'Top Category': top_category,
            'Status': st.session_state.file_status[filename]
        })
    
    st.dataframe(summary_data, use_container_width=True)
    
    # Statistics
    st.markdown("### Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_files = len(completed_files)
        st.metric("Total Files", total_files)
    
    with col2:
        transformer_compatible = sum(1 for f in completed_files 
                                   if st.session_state.file_metadata[f.name]['compatibility']['transformer'])
        st.metric("Transformer Compatible", transformer_compatible)
    
    with col3:
        iv3_compatible = sum(1 for f in completed_files 
                           if st.session_state.file_metadata[f.name]['compatibility']['iv3_gru'])
        st.metric("IV3-GRU Compatible", iv3_compatible)
    
    # Batch download
    st.markdown("### Batch Download")
    if st.button("Download All as ZIP", type="primary"):
        create_batch_download()


def create_batch_download():
    """Create and provide batch download as ZIP."""
    import zipfile
    import io
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, npz_data in st.session_state.processed_data.items():
            npz_bytes = create_npz_bytes(npz_data)
            zip_file.writestr(filename, npz_bytes)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="Download All Files as ZIP",
        data=zip_buffer.getvalue(),
        file_name="processed_files.zip",
        mime="application/zip"
    )


def main() -> None:
    """Main application function."""
    set_page()
    cfg = render_sidebar()
    initialize_session_state()

    # Main header
    render_main_header()
    
    # Two-stage workflow
    if st.session_state.workflow_stage == 'upload':
        render_upload_stage()
    else:  # processing stage
        render_processing_stage(cfg)


if __name__ == "__main__":
    main()