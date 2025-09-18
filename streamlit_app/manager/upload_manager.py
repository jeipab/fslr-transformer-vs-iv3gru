"""Upload manager for handling file uploads and routing to appropriate stages."""

import streamlit as st
from typing import List, Tuple, Dict
from ..components.utils import detect_file_type, format_file_size
from ..components.components import render_file_upload, render_video_preview, render_video_carousel


def render_welcome_screen() -> None:
    """Render welcome screen when no file is uploaded."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("""
        **Upload files to get started (up to 10 files)**
        
        **Supported file types:**
        - **Preprocessed .npz files**: Ready-to-use keypoint/feature data
        - **Video files**: MP4, AVI, MOV, MKV, WMV, FLV, WebM
        
        **Features:**
        - File queue with status tracking and file size display
        - Individual file processing and batch operations
        - Tab-based visualization for each processed file
        - Batch summary with comparative statistics
        - Individual and batch download options
        """)


def initialize_upload_session_state():
    """Initialize session state variables for upload workflow."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'npz_files' not in st.session_state:
        st.session_state.npz_files = []
    if 'video_files' not in st.session_state:
        st.session_state.video_files = []
    if 'preprocessed_files' not in st.session_state:
        st.session_state.preprocessed_files = []
    if 'file_status' not in st.session_state:
        st.session_state.file_status = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'file_metadata' not in st.session_state:
        st.session_state.file_metadata = {}
    if 'original_file_data' not in st.session_state:
        st.session_state.original_file_data = {}
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = None
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'workflow_stage' not in st.session_state:
        st.session_state.workflow_stage = 'upload'
    if 'pending_upload_files' not in st.session_state:
        st.session_state.pending_upload_files = []


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
        
        # Display selected files with previews
        st.markdown("**Selected Files:**")
        
        # Separate NPZ and video files for different display layouts
        npz_files = [f for f in uploaded_files if detect_file_type(f) == 'npz']
        video_files = [f for f in uploaded_files if detect_file_type(f) == 'video']
        
        # Display NPZ files in compact columns (if any)
        if npz_files:
            st.markdown("**📄 NPZ Files:**")
            
            # Show NPZ count summary
            total_size = sum(len(f.getvalue()) for f in npz_files) / (1024 * 1024)
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem; padding: 0.5rem; background-color: rgba(255, 255, 255, 0.05); border-radius: 6px;">
                <p style="margin: 0; color: #a0aec0; font-size: 0.9rem;">
                    <span>{len(npz_files)} NPZ file{'s' if len(npz_files) > 1 else ''}</span>
                    <span style="margin: 0 1rem;">•</span>
                    <span>Total: {total_size:.1f} MB</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            num_npz = len(npz_files)
            if num_npz <= 3:
                num_cols = num_npz
            elif num_npz <= 6:
                num_cols = 3
            else:
                num_cols = 4
            
            cols = st.columns(num_cols, gap="medium")
            for i, uploaded_file in enumerate(npz_files):
                file_size = len(uploaded_file.getvalue())
                size_mb = file_size / (1024 * 1024)
                
                col_index = i % num_cols
                with cols[col_index]:
                    # Create consistent card styling for NPZ files
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.1);
                                border-radius: 6px;
                                padding: 0.75rem;
                                margin: 0.25rem 0;
                                text-align: center;
                                transition: all 0.3s ease;">
                        <div style="color: #ffffff; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.25rem;">
                            {uploaded_file.name}
                        </div>
                        <div style="color: #a0aec0; font-size: 0.8rem; font-weight: 500;">
                            {size_mb:.1f} MB
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display video files in carousel (if any)
        if video_files:
            st.markdown("**🎥 Video Files:**")
            
            # Show video count summary
            total_size = sum(len(f.getvalue()) for f in video_files) / (1024 * 1024)
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem; padding: 0.5rem; background-color: rgba(255, 255, 255, 0.05); border-radius: 6px;">
                <p style="margin: 0; color: #a0aec0; font-size: 0.9rem;">
                    <span>{len(video_files)} video file{'s' if len(video_files) > 1 else ''}</span>
                    <span style="margin: 0 1rem;">•</span>
                    <span>Total: {total_size:.1f} MB</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Render video carousel
            render_video_carousel(video_files)
        
        # Show file type summary
        npz_count = sum(1 for f in uploaded_files if detect_file_type(f) == 'npz')
        video_count = sum(1 for f in uploaded_files if detect_file_type(f) == 'video')
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(uploaded_files))
        with col2:
            st.metric("NPZ Files", npz_count)
        with col3:
            st.metric("Video Files", video_count)
        
        # Route files and show proceed options
        route_files_to_stages(uploaded_files)
        
        # Centered proceed button
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            # Determine button text based on file types
            if video_count > 0:
                button_text = "Proceed to Preprocessing"
                button_help = "Move to preprocessing stage for video files"
            elif npz_count > 0:
                button_text = "Proceed to Inference"
                button_help = "Move to inference stage for NPZ files"
            else:
                button_text = "Proceed to Processing"
                button_help = "Move to appropriate processing stage"
            
            if st.button(button_text, type="primary", help=button_help, use_container_width=True):
                proceed_to_next_stage()
    else:
        render_welcome_screen()


def route_files_to_stages(uploaded_files: List) -> Tuple[List, List]:
    """Route uploaded files to appropriate stages based on file type.
    
    Returns:
        Tuple of (npz_files, video_files)
    """
    npz_files = []
    video_files = []
    
    for uploaded_file in uploaded_files:
        file_type = detect_file_type(uploaded_file)
        if file_type == 'npz':
            npz_files.append(uploaded_file)
        elif file_type == 'video':
            video_files.append(uploaded_file)
    
    # Store in session state
    st.session_state.npz_files = npz_files
    st.session_state.video_files = video_files
    
    return npz_files, video_files


def proceed_to_next_stage():
    """Move files to appropriate stages and transition workflow."""
    # Move files to uploaded_files and set initial status
    st.session_state.uploaded_files = st.session_state.pending_upload_files
    
    for uploaded_file in st.session_state.uploaded_files:
        filename = uploaded_file.name
        st.session_state.file_status[filename] = 'pending'
        file_size = len(uploaded_file.getvalue())
        st.session_state.file_metadata[filename] = {
            'file_size': file_size,
            'file_size_formatted': format_file_size(file_size)
        }
    
    # Determine next stage based on file types
    npz_files = st.session_state.npz_files
    video_files = st.session_state.video_files
    
    if npz_files and not video_files:
        # Only NPZ files - go directly to predictions
        st.session_state.workflow_stage = 'predictions'
    elif video_files and not npz_files:
        # Only video files - go to preprocessing
        st.session_state.workflow_stage = 'preprocessing'
    else:
        # Mixed files - go to preprocessing first, then user can navigate
        st.session_state.workflow_stage = 'preprocessing'
    
    st.rerun()


def get_stage_navigation_info() -> Dict:
    """Get information about current stage and available navigation options."""
    npz_count = len(st.session_state.npz_files)
    video_count = len(st.session_state.video_files)
    preprocessed_count = len(st.session_state.preprocessed_files)
    
    return {
        'npz_count': npz_count,
        'video_count': video_count,
        'preprocessed_count': preprocessed_count,
        'has_npz_files': npz_count > 0,
        'has_video_files': video_count > 0,
        'has_preprocessed_files': preprocessed_count > 0,
        'current_stage': st.session_state.workflow_stage
    }


def clear_all_files():
    """Clear all files from all stages and reset session state."""
    st.session_state.uploaded_files = []
    st.session_state.npz_files = []
    st.session_state.video_files = []
    st.session_state.preprocessed_files = []
    st.session_state.file_status = {}
    st.session_state.processed_data = {}
    st.session_state.file_metadata = {}
    st.session_state.original_file_data = {}
    st.session_state.current_tab = None
    st.session_state.selected_files = []
    st.session_state.pending_upload_files = []
    
    # Clear all confirmation states
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith("confirm_")]
    for key in keys_to_remove:
        del st.session_state[key]
    
    st.toast("All files cleared", icon="🗑️", duration=5000)


def remove_file_from_stage(filename: str, stage: str):
    """Remove a file from a specific stage and clean up session state."""
    # Remove from stage-specific list
    if stage == 'npz':
        st.session_state.npz_files = [f for f in st.session_state.npz_files if f.name != filename]
    elif stage == 'video':
        st.session_state.video_files = [f for f in st.session_state.video_files if f.name != filename]
    elif stage == 'preprocessed':
        st.session_state.preprocessed_files = [f for f in st.session_state.preprocessed_files if f.name != filename]
    
    # Remove from general uploaded files
    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f.name != filename]
    
    # Clean up session state
    if filename in st.session_state.file_status:
        del st.session_state.file_status[filename]
    if filename in st.session_state.processed_data:
        del st.session_state.processed_data[filename]
    if filename in st.session_state.file_metadata:
        del st.session_state.file_metadata[filename]
    if filename in st.session_state.original_file_data:
        del st.session_state.original_file_data[filename]
    
    # Reset current tab if it was the removed file
    if st.session_state.current_tab == filename:
        st.session_state.current_tab = None
    
    # Clear any confirmation states for this file
    for key in list(st.session_state.keys()):
        if key.startswith(f"confirm_remove_{filename}"):
            del st.session_state[key]
    
    st.toast(f"Removed {filename}", icon="🗑️", duration=5000)
