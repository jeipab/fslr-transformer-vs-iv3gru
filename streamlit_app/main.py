"""Main Streamlit application for FSLR Demo."""

import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st

from components import (
    set_page, render_sidebar, render_welcome_screen, 
    render_file_upload, render_main_header, render_predictions_section
)
from data_processing import process_video_file
from utils import detect_file_type, TempUploadedFile
from visualization import (
    render_sequence_overview, render_animated_keypoints, 
    render_feature_charts, create_video_with_keypoints
)


def main() -> None:
    """Main application function."""
    set_page()
    cfg = render_sidebar()

    # Main header
    render_main_header()
    
    # File upload
    st.markdown("### Upload Data")
    uploaded_file = render_file_upload()

    if uploaded_file is None:
        render_welcome_screen()
        return

    # Detect file type and process accordingly
    file_type = detect_file_type(uploaded_file)
    
    if file_type == 'unknown':
        st.error(f"Unsupported file type: {uploaded_file.name}")
        return
    
    # Store file content for potential reuse (e.g., video overlay)
    file_content = uploaded_file.read()
    
    try:
        if file_type == 'npz':
            # Load NPZ file
            file_bytes = io.BytesIO(file_content)
            npz_data = dict(np.load(file_bytes, allow_pickle=True))
        elif file_type == 'video':
            # Process video file
            with st.spinner(f"Processing video file: {uploaded_file.name}..."):
                temp_file = TempUploadedFile(uploaded_file.name, file_content)
                npz_data = process_video_file(temp_file, target_fps=30, out_size=256)
            st.success(f"Video processed successfully: {npz_data['X'].shape[0]} frames extracted")
        else:
            st.error(f"Unsupported file type: {file_type}")
            return
            
    except Exception as exc:
        st.error(f"Failed to process {file_type} file: {exc}")
        return

    try:
        X_pad, mask, meta = render_sequence_overview(npz_data, cfg["sequence_length"])
    except Exception as exc:
        st.error(str(exc))
        return

    # Animated keypoint visualization
    render_animated_keypoints(X_pad, mask if mask.size > 0 else None)
    
    # Feature analysis
    render_feature_charts(X_pad, mask if mask.size > 0 else None)

    # Video visualization section (only for video input)
    if file_type == 'video':
        st.markdown("<div class='section-header'>Video Visualization</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            show_video_overlay = st.checkbox(
                "Create video with keypoint overlay", 
                value=False,
                help="Generate a new video with keypoints drawn on top of the original video"
            )
            
        with col2:
            if show_video_overlay:
                if st.button("Generate Video Overlay"):
                    with st.spinner("Creating video with keypoint overlay..."):
                        try:
                            # Create temporary file object for video processing
                            temp_video_file = TempUploadedFile(uploaded_file.name, file_content)
                            overlay_video_path = create_video_with_keypoints(
                                temp_video_file, 
                                X_pad,
                                f"overlay_{Path(uploaded_file.name).stem}.mp4"
                            )
                            
                            # Display the video
                            st.success("Video overlay created successfully!")
                            
                            # Show download button
                            with open(overlay_video_path, 'rb') as video_file:
                                st.download_button(
                                    label="Download Video with Keypoints",
                                    data=video_file.read(),
                                    file_name=f"overlay_{Path(uploaded_file.name).stem}.mp4",
                                    mime="video/mp4"
                                )
                                
                            # Display video inline
                            st.video(overlay_video_path)
                            
                        except Exception as e:
                            st.error(f"Failed to create video overlay: {str(e)}")

    # Predictions section
    from utils import simulate_predictions
    rng = np.random.RandomState(cfg["random_seed"])
    gloss_logits, cat_logits = simulate_predictions(
        rng, cfg["num_gloss_classes"], cfg["num_category_classes"]
    )
    render_predictions_section(cfg, gloss_logits, cat_logits)


if __name__ == "__main__":
    main()
