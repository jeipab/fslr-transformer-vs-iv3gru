"""Visualization components for keypoints and predictions."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def render_sequence_overview(npz_dict: Dict, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Show metadata and return processed sequence and related info.

    Args:
        npz_dict: NpzFile-like dict from numpy.load.
        sequence_length: Target T for padding/trimming.

    Returns:
        (X_pad, mask, meta_dict)
    """
    if "X" not in npz_dict:
        raise KeyError("Uploaded .npz must contain key 'X' with shape [T, 156]")

    X_raw = np.array(npz_dict["X"])  # [T, 156]
    mask = np.array(npz_dict.get("mask", []))
    timestamps_ms = np.array(npz_dict.get("timestamps_ms", []))
    X2048 = np.array(npz_dict.get("X2048", [])) if "X2048" in npz_dict else None

    raw_length, raw_features = X_raw.shape[0], X_raw.shape[1] if X_raw.ndim == 2 else (0, 0)
    
    # Parse metadata early so it's available for display logic
    meta_raw = npz_dict.get("meta")
    meta_parsed: Dict = {}
    if meta_raw is not None:
        try:
            if isinstance(meta_raw, (str, bytes)):
                meta_parsed = json.loads(meta_raw)
            else:
                meta_parsed = json.loads(str(meta_raw))
        except Exception:
            meta_parsed = {"info": "Unparsed meta"}

    st.markdown("<div class='section-header'>Sequence Overview</div>", unsafe_allow_html=True)
    
    # Main metrics in a more visual layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Frames", str(raw_length), delta=None)
    
    with col2:
        st.metric("Features", str(raw_features), delta=None)
        if raw_features == 156:
            st.markdown("<span class='status-good'>✓ Valid keypoints</span>", unsafe_allow_html=True)
            # Check model_type to determine what to display
            model_type = meta_parsed.get('model_type') if meta_parsed else None
            if model_type == 'I':
                # IV3-GRU only - just show valid keypoints, no transformer ready
                pass
            elif model_type in ['T', 'B'] or model_type is None:
                # Transformer or Both or legacy - show transformer ready
                st.markdown("<span class='status-good'>✓ Transformer ready</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-warning'>⚠ Unexpected shape</span>", unsafe_allow_html=True)
    
    with col3:
        if timestamps_ms.size > 0:
            duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0 if timestamps_ms.size > 1 else 0.0
            st.metric("Duration", f"{duration_s:.2f}s")
            if duration_s > 0:
                fps = raw_length / duration_s if duration_s > 0 else 0
                st.caption(f"~{fps:.1f} FPS")
        else:
            st.metric("Duration", "N/A")
    
    with col4:
        st.metric("Target Length", str(sequence_length), delta=None)
        if raw_length > sequence_length:
            st.caption("Will trim")
        elif raw_length < sequence_length:
            st.caption("Will pad")
        else:
            st.caption("Perfect fit")
    
    with col5:
        if X2048 is not None and isinstance(X2048, np.ndarray) and X2048.size > 0:
            st.metric("X2048 Features", "Present")
            st.markdown("<span class='status-good'>✓ IV3-GRU ready</span>", unsafe_allow_html=True)
        else:
            st.metric("X2048 Features", "Missing")
            st.markdown("<span class='status-warning'>⚠ Transformer only</span>", unsafe_allow_html=True)

    if meta_parsed:
        with st.expander("Metadata", expanded=False):
            st.json(meta_parsed)

    with st.expander("Data checks", expanded=False):
        issues = []
        if not (X_raw.ndim == 2 and X_raw.shape[1] == 156):
            issues.append(f"X shape expected [T,156], got {getattr(X_raw, 'shape', None)}")
        if mask.size > 0:
            if mask.ndim == 2 and mask.shape[1] == 78:
                coverage = float(mask.mean() * 100.0)
                st.caption(f"Mask coverage: {coverage:.1f}% of keypoints marked visible")
            else:
                issues.append(f"mask shape expected [T,78], got {getattr(mask, 'shape', None)}")
        if timestamps_ms.size > 1:
            mono = bool((timestamps_ms[1:] >= timestamps_ms[:-1]).all())
            if not mono:
                issues.append("timestamps_ms not monotonic nondecreasing")
        if X2048 is not None and isinstance(X2048, np.ndarray) and X2048.size > 0:
            if not (X2048.ndim == 2 and X2048.shape[1] == 2048):
                issues.append(f"X2048 shape expected [T,2048], got {getattr(X2048, 'shape', None)}")
        if issues:
            st.warning("\n".join(f"- {m}" for m in issues))

    from .utils import pad_or_trim
    X_pad = pad_or_trim(X_raw, sequence_length)
    return X_pad, mask, meta_parsed


def render_keypoint_video(sequence: np.ndarray, mask: Optional[np.ndarray] = None, key_suffix: str = "") -> None:
    """Generate and display a video with keypoint animation."""
    time_steps, feature_dim = sequence.shape
    keypoints_2d = sequence.reshape(time_steps, 78, 2)
    
    # Video settings
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        fps = st.slider(
            "FPS", 
            min_value=5, 
            max_value=30, 
            value=15, 
            step=1,
            help="Frames per second for video", 
            key=f"video_fps_{key_suffix}"
        )
    with col2:
        show_skeleton = st.checkbox("Show Skeleton", value=True, help="Display skeleton connections", key=f"video_skeleton_{key_suffix}")
    with col3:
        show_visibility = st.checkbox("Show Visibility", value=True, help="Color points by visibility", key=f"video_visibility_{key_suffix}")
    
    # Background options
    col4, col5 = st.columns(2)
    with col4:
        # Check if we have original video data available
        background_options = ["White", "Black", "Grid"]
        
        # Check if this is a processed video file with original video data
        if key_suffix and key_suffix in st.session_state.get('original_file_data', {}):
            original_data = st.session_state.original_file_data[key_suffix]
            if original_data.get('type', '').startswith('video/'):
                background_options.append("Original Video")
        
        bg_type = st.selectbox("Background", background_options, key=f"bg_type_{key_suffix}")
    with col5:
        # Show size options based on background selection
        if bg_type == "Original Video":
            # Only show Original Size when using original video background
            video_size_options = ["Original Size"]
        else:
            # Show standard size options for other backgrounds
            video_size_options = ["512x512", "768x768", "1024x1024"]
        
        video_size = st.selectbox("Video Size", video_size_options, key=f"video_size_{key_suffix}")
    
    # Handle original size option
    if video_size == "Original Size":
        # We'll determine dimensions from the original video
        width, height = None, None
    else:
        width, height = map(int, video_size.split('x'))
    
    # Ensure we have valid dimensions before proceeding
    if width is None or height is None:
        # This will be set when we load the original video
        pass
    
    # Video control buttons with Download Video right-aligned to dropdown and Generate Video with gap
    col_empty1, col_empty2, col_empty3, col_empty4, col_empty5, col_gen, col_gap, col_download = st.columns([2, 2, 1, 1, 1, 2, 0.25, 2])
    
    with col_gen:
        if st.button("Generate Video", key=f"generate_video_{key_suffix}"):
            with st.spinner("Generating keypoint video..."):
                video_path = create_keypoint_animation_video(
                    keypoints_2d, mask, fps, width, height, 
                    show_skeleton, show_visibility, bg_type, key_suffix
                )
                
                if video_path and os.path.exists(video_path):
                    # Store video path in session state
                    st.session_state[f"video_path_{key_suffix}"] = video_path
                    st.toast("Video generated successfully!", icon="✅")
                else:
                    st.toast("Failed to generate video", icon="❌")
    
    with col_download:
        # Download button (moved here to be next to generate video button)
        if f"video_path_{key_suffix}" in st.session_state:
            video_path = st.session_state[f"video_path_{key_suffix}"]
            if os.path.exists(video_path):
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name=f"keypoint_animation_{key_suffix}.mp4",
                    mime="video/mp4",
                    key=f"download_btn_{key_suffix}"
                )
    
    # Display video if it exists in session state
    if f"video_path_{key_suffix}" in st.session_state:
        video_path = st.session_state[f"video_path_{key_suffix}"]
        if os.path.exists(video_path):
            # Display video with dynamic sizing, autoplay, and autoloop
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            # Create a container with dynamic sizing
            video_container = st.container()
            with video_container:
                st.video(video_bytes, format="video/mp4", autoplay=True, loop=True)
            
            # Add custom CSS for dynamic video sizing and autoplay/loop
            st.markdown("""
            <style>
            .stVideo {
                width: 100% !important;
                max-width: 600px !important;
                margin: 0 auto !important;
                display: block !important;
            }
            
            .stVideo video {
                width: 100% !important;
                height: auto !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
            }
            
            /* Ensure autoplay and loop work */
            .stVideo video[autoplay] {
                autoplay: true !important;
            }
            
            .stVideo video[loop] {
                loop: true !important;
            }
            </style>
            
            <script>
            // Force autoplay and loop for better browser compatibility
            document.addEventListener('DOMContentLoaded', function() {
                const videos = document.querySelectorAll('.stVideo video');
                videos.forEach(video => {
                    video.autoplay = true;
                    video.loop = true;
                    video.muted = true; // Muted autoplay works better
                    video.play();
                });
            });
            </script>
            """, unsafe_allow_html=True)


def render_animated_keypoints(sequence: np.ndarray, mask: Optional[np.ndarray] = None, key_suffix: str = "", meta_dict: Optional[Dict] = None) -> None:
    """Render animated keypoint visualization with skeleton overlay."""
    st.markdown("<div class='section-header'>Keypoint Visualization</div>", unsafe_allow_html=True)
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available. Please install plotly to see animated keypoint visualization.")
        # Fallback to simple text display
        st.write("Keypoint data shape:", sequence.shape)
        st.write("Sample keypoints:", sequence[:3, :10])
        return
    
    time_steps, feature_dim = sequence.shape
    
    # Reshape keypoints to [T, 78, 2] for easier handling
    keypoints_2d = sequence.reshape(time_steps, 78, 2)
    
    # Extract frame-by-frame occlusion information from metadata
    frame_occlusion_data = None
    if meta_dict:
        occlusion_details = meta_dict.get('occlusion_results', None)
        if occlusion_details and 'detailed_results' in occlusion_details:
            # Create a mapping of frame index to occlusion status
            frame_occlusion_data = {}
            for result in occlusion_details['detailed_results']:
                frame_idx = result.get('frame_idx', -1)
                is_occluded = result.get('occlusion_detected', False)
                frame_occlusion_data[frame_idx] = is_occluded
    
    # Video generation option
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Choose visualization method:**")
    with col2:
        st.markdown("")  # Empty space for alignment
    with col3:
        use_video = st.checkbox("Generate Video", value=False, help="Create an MP4 video with keypoint animation", key=f"use_video_{key_suffix}")
    
    if use_video:
        render_keypoint_video(sequence, mask, key_suffix)
        return
    
    # Define skeleton connections for MediaPipe Holistic based on actual preprocessing layout
    # Layout: Pose(0-24), Left Hand(25-45), Right Hand(46-66), Face(67-77)
    # Face landmarks use FACEMESH_11 = [1, 33, 263, 133, 362, 61, 291, 105, 334, 199, 4]
    # Mapped to indices 0-10: [nose_tip, left_eye_outer, right_eye_outer, left_eye_inner, 
    #                        right_eye_inner, mouth_left, mouth_right, left_eyebrow, 
    #                        right_eyebrow, chin, nose_bridge]
    skeleton_connections = {
        "pose": [
            # Upper body pose connections (indices 0-24)
            # Shoulder connections
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            # Arm connections  
            (15, 17), (15, 19), (15, 21), (17, 19),
            (16, 18), (16, 20), (16, 22), (18, 20),
            # Torso connections
            (11, 23), (12, 24), (23, 24)
        ],
        "left_hand": [
            # Left hand connections (indices 25-45, relative to 0-20)
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ],
        "right_hand": [
            # Right hand connections (indices 46-66, relative to 0-20)
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ],
        "face": [
            # Face connections (indices 67-77, relative to 0-10)
            # FACEMESH_11 landmarks: [nose_tip, left_eye_outer, right_eye_outer, left_eye_inner, 
            #                        right_eye_inner, mouth_left, mouth_right, left_eyebrow, 
            #                        right_eyebrow, chin, nose_bridge]
            # Meaningful face connections based on facial anatomy
            (0, 10),  # nose_tip to nose_bridge
            (1, 3),   # left_eye_outer to left_eye_inner
            (2, 4),   # right_eye_outer to right_eye_inner
            (1, 7),   # left_eye_outer to left_eyebrow
            (2, 8),   # right_eye_outer to right_eyebrow
            (5, 6),   # mouth_left to mouth_right
            (0, 5),   # nose_tip to mouth_left
            (0, 6),   # nose_tip to mouth_right
            (9, 5),   # chin to mouth_left
            (9, 6)    # chin to mouth_right
        ]
    }
    
    # Simple frame selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        frame_idx = st.slider(
            "Frame", 
            min_value=0, 
            max_value=time_steps-1, 
            value=0, 
            step=1,
            help="Drag to see keypoints at different time steps",
            key=f"frame_slider_{key_suffix}"
        )
    
    with col2:
        show_skeleton = st.checkbox("Show Skeleton", value=True, help="Display skeleton connections", key=f"skeleton_checkbox_{key_suffix}")
    
    with col3:
        show_visibility = st.checkbox("Show Visibility", value=True, help="Color points by visibility", key=f"visibility_checkbox_{key_suffix}")
    
    # Get current frame keypoints
    current_keypoints = keypoints_2d[frame_idx]  # [78, 2]
    
    # Create the plot
    fig = go.Figure()
    
    # Define colors for different body parts
    colors = {
        "pose": "red",
        "left_hand": "blue", 
        "right_hand": "green",
        "face": "orange"
    }
    
    # Plot keypoints for each body part
    for part_name, connections in skeleton_connections.items():
        if part_name == "pose":
            start_idx, end_idx = 0, 25  # 25 pose landmarks (POSE_UPPER_25)
        elif part_name == "left_hand":
            start_idx, end_idx = 25, 46  # 21 hand landmarks
        elif part_name == "right_hand":
            start_idx, end_idx = 46, 67  # 21 hand landmarks
        elif part_name == "face":
            start_idx, end_idx = 67, 78  # 11 face landmarks (FACEMESH_11)
        
        part_keypoints = current_keypoints[start_idx:end_idx]
        
        # Filter out keypoints at (0,0) coordinates
        valid_mask = ~((part_keypoints[:, 0] == 0) & (part_keypoints[:, 1] == 0))
        valid_keypoints = part_keypoints[valid_mask]
        
        if len(valid_keypoints) > 0:  # Only plot if there are valid keypoints
            # Plot individual points
            if show_visibility and mask is not None and mask.size > 0:
                # Use visibility mask for coloring, filtered for valid points
                visibility = mask[frame_idx, start_idx:end_idx] if mask.shape[1] >= end_idx else np.ones(end_idx - start_idx, dtype=bool)
                valid_visibility = visibility[valid_mask]
                point_colors = ['rgba(255,0,0,1)' if v else 'rgba(255,0,0,0.3)' for v in valid_visibility]
            else:
                point_colors = [colors[part_name]] * len(valid_keypoints)
            
            # Create hover text with keypoint numbers
            hover_texts = []
            for i, (x, y) in enumerate(valid_keypoints):
                # Calculate the actual keypoint index in the full 78-point array
                actual_keypoint_idx = start_idx + np.where(valid_mask)[0][i]
                hover_texts.append(f"Keypoint {actual_keypoint_idx}<br>X: {x:.3f}<br>Y: {y:.3f}")
            
            fig.add_trace(go.Scatter(
                x=valid_keypoints[:, 0],
                y=valid_keypoints[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=point_colors,
                    line=dict(width=2, color='white')
                ),
                name=f"{part_name.title()} Points",
                showlegend=True,
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts
            ))
        
        # Plot skeleton connections
        if show_skeleton and len(valid_keypoints) > 0:
            # Create mapping from original indices to valid indices
            valid_indices = np.where(valid_mask)[0]
            original_to_valid = {orig_idx: valid_idx for valid_idx, orig_idx in enumerate(valid_indices)}
            
            for start_conn, end_conn in connections:
                if (start_conn < len(part_keypoints) and end_conn < len(part_keypoints) and
                    start_conn in original_to_valid and end_conn in original_to_valid):
                    # Only draw line if both endpoints are valid (not at 0,0)
                    valid_start = original_to_valid[start_conn]
                    valid_end = original_to_valid[end_conn]
                    fig.add_trace(go.Scatter(
                        x=[valid_keypoints[valid_start, 0], valid_keypoints[valid_end, 0]],
                        y=[valid_keypoints[valid_start, 1], valid_keypoints[valid_end, 1]],
                        mode='lines',
                        line=dict(color=colors[part_name], width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # Update layout with frame-specific occlusion indicator
    title_text = f"Keypoint Visualization - Frame {frame_idx}/{time_steps-1}"
    
    # Check if current frame is occluded
    current_frame_occluded = False
    if frame_occlusion_data is not None:
        current_frame_occluded = frame_occlusion_data.get(frame_idx, False)
    
    if current_frame_occluded:
        title_text += " 🔴 [OCCLUDED]"
    
    fig.update_layout(
        title=title_text,
        xaxis=dict(
            title="X Coordinate",
            scaleanchor="y",
            scaleratio=1,
            range=[0, 1]
        ),
        yaxis=dict(
            title="Y Coordinate", 
            range=[1, 0]  # Flip Y axis to match image coordinates
        ),
        width=600,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add frame information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Frame", f"{frame_idx + 1}/{time_steps}")
    with col2:
        if mask is not None and mask.size > 0:
            visible_points = np.sum(mask[frame_idx]) if frame_idx < mask.shape[0] else 0
            st.metric("Visible Points", f"{visible_points}/78")
    with col3:
        if len(current_keypoints) > 0:
            avg_x = np.mean(current_keypoints[:, 0])
            avg_y = np.mean(current_keypoints[:, 1])
            st.metric("Center", f"({avg_x:.3f}, {avg_y:.3f})")
    


def render_feature_charts(sequence: np.ndarray, mask: Optional[np.ndarray] = None, key_suffix: str = "") -> None:
    """Render interactive feature visualization."""
    st.markdown("<div class='section-header'>Feature Analysis</div>", unsafe_allow_html=True)
    
    time_steps, feature_dim = sequence.shape

    # Feature selection with better organization
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Keypoint group selection - Updated to match actual preprocessing layout
        keypoint_groups = {
            "Pose (upper body)": (0, 50),    # 25 landmarks * 2 coordinates = 50 features
            "Left hand": (50, 92),           # 21 landmarks * 2 coordinates = 42 features  
            "Right hand": (92, 134),         # 21 landmarks * 2 coordinates = 42 features
            "Face landmarks": (134, 156)     # 11 landmarks * 2 coordinates = 22 features
        }
        
        selected_group = st.selectbox(
            "Keypoint Group",
            list(keypoint_groups.keys()),
            help="Select which body part to visualize",
            key=f"keypoint_group_{key_suffix}"
        )
        
        start_idx, end_idx = keypoint_groups[selected_group]
    
    with col2:
        coord_type = st.selectbox(
            "Coordinate",
            ["Both X&Y", "X only", "Y only"],
            help="Choose coordinate type to display",
            key=f"coordinate_type_{key_suffix}"
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Heatmap"],
            help="Visualization style",
            key=f"chart_type_{key_suffix}"
        )
    
    # Prepare data based on selection
    if coord_type == "X only":
        data_slice = sequence[:, start_idx:end_idx:2]  # Every other starting from start_idx
        coord_suffix = "_x"
    elif coord_type == "Y only":
        data_slice = sequence[:, start_idx+1:end_idx:2]  # Every other starting from start_idx+1
        coord_suffix = "_y"
    else:  # Both X&Y
        data_slice = sequence[:, start_idx:end_idx]
        coord_suffix = ""
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available. Please install plotly to see interactive charts.")
        # Fallback to simple text display
        st.write("Feature data shape:", data_slice.shape)
        st.write("Sample values:", data_slice[:5, :5])
        return
    
    if chart_type == "Line":
        # Interactive line chart with Plotly
        fig = go.Figure()
        
        for i in range(min(8, data_slice.shape[1])):  # Limit to 8 lines for readability
            fig.add_trace(go.Scatter(
                x=list(range(time_steps)),
                y=data_slice[:, i],
                mode='lines',
                name=f'{selected_group.split()[0]}_{i}{coord_suffix}',
                line=dict(width=2),
                hovertemplate='Frame: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{selected_group} - {coord_type} Over Time",
            xaxis_title="Frame",
            yaxis_title="Normalized Coordinate",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Heatmap
        # Create heatmap of features over time
        fig = px.imshow(
            data_slice.T,
            aspect='auto',
            color_continuous_scale='viridis',
            labels={'x': 'Frame', 'y': 'Feature Index', 'color': 'Value'},
            title=f"{selected_group} - {coord_type} Heatmap"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics summary
    with st.expander("📊 Feature Statistics", expanded=False):
        stats_df = pd.DataFrame({
            'Mean': data_slice.mean(axis=0),
            'Std': data_slice.std(axis=0),
            'Min': data_slice.min(axis=0),
            'Max': data_slice.max(axis=0),
            'Range': data_slice.max(axis=0) - data_slice.min(axis=0)
        }).round(4)
        
        st.dataframe(stats_df, use_container_width=True)


def create_keypoint_animation_video(keypoints_2d: np.ndarray, mask: Optional[np.ndarray], 
                                   fps: int, width: int, height: int, show_skeleton: bool, 
                                   show_visibility: bool, bg_type: str, key_suffix: str) -> str:
    """Create an animated video with keypoint visualization."""
    if not CV2_AVAILABLE:
        st.error("OpenCV not available. Cannot create video.")
        return None
    
    # Create output path
    output_path = os.path.join(tempfile.gettempdir(), f"keypoint_animation_{key_suffix}.mp4")
    
    # Define skeleton connections
    skeleton_connections = {
        "pose": [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            (15, 17), (15, 19), (15, 21), (17, 19),
            (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24)
        ],
        "left_hand": [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ],
        "right_hand": [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ],
        "face": [
            # FACEMESH_11 landmarks: [nose_tip, left_eye_outer, right_eye_outer, left_eye_inner, 
            #                        right_eye_inner, mouth_left, mouth_right, left_eyebrow, 
            #                        right_eyebrow, chin, nose_bridge]
            # Meaningful face connections based on facial anatomy
            (0, 10),  # nose_tip to nose_bridge
            (1, 3),   # left_eye_outer to left_eye_inner
            (2, 4),   # right_eye_outer to right_eye_inner
            (1, 7),   # left_eye_outer to left_eyebrow
            (2, 8),   # right_eye_outer to right_eyebrow
            (5, 6),   # mouth_left to mouth_right
            (0, 5),   # nose_tip to mouth_left
            (0, 6),   # nose_tip to mouth_right
            (9, 5),   # chin to mouth_left
            (9, 6)    # chin to mouth_right
        ]
    }
    
    # Colors for different body parts
    colors = {
        "pose": (0, 0, 255),      # Red
        "left_hand": (255, 0, 0), # Blue
        "right_hand": (0, 255, 0), # Green
        "face": (0, 165, 255)     # Orange
    }
    
    # Handle original video background
    original_video_cap = None
    original_video_frames = []
    
    # If we need original video dimensions, get them first
    if bg_type == "Original Video" and key_suffix in st.session_state.get('original_file_data', {}):
        try:
            # Get original video data
            original_data = st.session_state.original_file_data[key_suffix]
            video_data = original_data.get('data')
            
            if video_data:
                # Save original video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(video_data)
                    temp_video_path = tmp_file.name
                
                # Open original video
                original_video_cap = cv2.VideoCapture(temp_video_path)
                
                if original_video_cap.isOpened():
                    # Get original video dimensions and properties
                    width = int(original_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(original_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    original_fps = original_video_cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(original_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Extract all frames from original video (no resizing needed)
                    while True:
                        ret, frame = original_video_cap.read()
                        if not ret:
                            break
                        original_video_frames.append(frame)
                    
                    original_video_cap.release()
                
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                    
        except Exception as e:
            st.warning(f"Could not load original video: {str(e)}. Using default background.")
            bg_type = "White"  # Fallback to white background
            width, height = 512, 512  # Default fallback
    
    # Final validation - ensure we have valid dimensions
    if width is None or height is None or width <= 0 or height <= 0:
        st.error(f"Invalid video dimensions: {width}x{height}. Please try again.")
        return None
    
    try:
        # Video writer setup - use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Alternative fallback if H264 doesn't work
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Final check
        if not out.isOpened():
            st.error(f"Failed to create video writer with dimensions {width}x{height} at {fps} FPS")
            return None
        frames_written = 0
        for frame_idx in range(len(keypoints_2d)):
            # Create background
            if bg_type == "Original Video" and original_video_frames:
                # Time-based synchronization
                # Calculate time position for this keypoint frame
                keypoint_time = frame_idx / 30.0  # Assuming keypoints are at 30fps
                
                # Calculate corresponding video frame based on time
                video_frame_idx = int(keypoint_time * original_fps)
                
                # Ensure we don't go out of bounds
                video_frame_idx = min(video_frame_idx, len(original_video_frames) - 1)
                video_frame_idx = max(0, video_frame_idx)  # Ensure non-negative
                
                # Use the calculated video frame
                frame = original_video_frames[video_frame_idx].copy()
                
            else:
                # Use static background
                if bg_type == "White":
                    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                elif bg_type == "Black":
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:  # Grid
                    frame = np.ones((height, width, 3), dtype=np.uint8) * 240
                    # Draw grid
                    for i in range(0, width, 50):
                        cv2.line(frame, (i, 0), (i, height), (200, 200, 200), 1)
                    for i in range(0, height, 50):
                        cv2.line(frame, (0, i), (width, i), (200, 200, 200), 1)
            
            current_keypoints = keypoints_2d[frame_idx]
            
            # Convert normalized coordinates to pixel coordinates
            pixel_points = current_keypoints.copy()
            pixel_points[:, 0] *= width
            pixel_points[:, 1] *= height
            pixel_points = pixel_points.astype(np.int32)
            
            # Filter out keypoints at (0,0)
            valid_mask = ~((pixel_points[:, 0] == 0) & (pixel_points[:, 1] == 0))
            
            # Draw keypoints and skeleton for each body part
            for part_name, connections in skeleton_connections.items():
                if part_name == "pose":
                    start_idx, end_idx = 0, 25  # 25 pose landmarks (0-24)
                elif part_name == "left_hand":
                    start_idx, end_idx = 25, 46  # 21 hand landmarks (25-45)
                elif part_name == "right_hand":
                    start_idx, end_idx = 46, 67  # 21 hand landmarks (46-66)
                elif part_name == "face":
                    start_idx, end_idx = 67, 78  # 11 face landmarks (67-77)
                
                part_keypoints = pixel_points[start_idx:end_idx]
                part_valid = valid_mask[start_idx:end_idx]
                
                # Draw skeleton connections
                if show_skeleton:
                    for start_conn, end_conn in connections:
                        if (start_conn < len(part_keypoints) and end_conn < len(part_keypoints) and
                            part_valid[start_conn] and part_valid[end_conn]):
                            # Add white outline for better visibility on video backgrounds
                            if bg_type == "Original Video":
                                cv2.line(frame, 
                                       tuple(part_keypoints[start_conn]), 
                                       tuple(part_keypoints[end_conn]), 
                                       (255, 255, 255), 3)  # Moderate white outline
                                cv2.line(frame, 
                                       tuple(part_keypoints[start_conn]), 
                                       tuple(part_keypoints[end_conn]), 
                                       colors[part_name], 2)  # Moderate colored line
                            else:
                                cv2.line(frame, 
                                       tuple(part_keypoints[start_conn]), 
                                       tuple(part_keypoints[end_conn]), 
                                       colors[part_name], 2)
                
                # Draw keypoints
                for i, (point, is_valid) in enumerate(zip(part_keypoints, part_valid)):
                    if is_valid:
                        # Determine point color based on visibility
                        if show_visibility and mask is not None and frame_idx < mask.shape[0]:
                            visibility = mask[frame_idx, start_idx + i] if start_idx + i < mask.shape[1] else True
                            point_color = colors[part_name] if visibility else tuple(c // 2 for c in colors[part_name])
                        else:
                            point_color = colors[part_name]
                        
                        # Different sizes for different body parts
                        # Add white outline for better visibility on video backgrounds
                        if bg_type == "Original Video":
                            # Moderately sized keypoints for original video background
                            if part_name == "pose":
                                cv2.circle(frame, tuple(point), 6, (255, 255, 255), -1)  # White outline
                                cv2.circle(frame, tuple(point), 4, point_color, -1)  # Colored center
                            elif part_name in ["left_hand", "right_hand"]:
                                cv2.circle(frame, tuple(point), 5, (255, 255, 255), -1)  # White outline
                                cv2.circle(frame, tuple(point), 3, point_color, -1)  # Colored center
                            else:  # face
                                cv2.circle(frame, tuple(point), 4, (255, 255, 255), -1)  # White outline
                                cv2.circle(frame, tuple(point), 2, point_color, -1)  # Colored center
                        else:
                            if part_name == "pose":
                                cv2.circle(frame, tuple(point), 6, point_color, -1)
                            elif part_name in ["left_hand", "right_hand"]:
                                cv2.circle(frame, tuple(point), 4, point_color, -1)
                            else:  # face
                                cv2.circle(frame, tuple(point), 3, point_color, -1)
            
            # Add frame number with better visibility on video backgrounds
            frame_text = f"Frame {frame_idx + 1}/{len(keypoints_2d)}"
            if bg_type == "Original Video":
                # Add white outline for better visibility
                cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        # Verify video was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error("Video file was not created properly")
            return None
        
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        return None


def create_video_with_keypoints(uploaded_video_file, keypoints: np.ndarray, 
                               output_filename: str = "video_with_keypoints.mp4") -> str:
    """
    Create a video with keypoint overlay for Streamlit display.
    
    Args:
        uploaded_video_file: Original uploaded video file
        keypoints: Keypoint data [T, 156]
        output_filename: Name for output video file
        
    Returns:
        Path to created video file
    """
    if not CV2_AVAILABLE:
        raise Exception("OpenCV not available. Cannot create video with keypoints.")
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video_file.name).suffix) as tmp_input:
        tmp_input.write(uploaded_video_file.getvalue())  # Use getvalue() for already read file
        input_video_path = tmp_input.name
    
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    
    try:
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
            
        # Get video properties  
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_VIDEO_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_VIDEO_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        keypoint_frames = keypoints.reshape(keypoints.shape[0], 78, 2)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get corresponding keypoints (handle length mismatch)
            if frame_idx < len(keypoint_frames):
                points = keypoint_frames[frame_idx]
                
                # Convert normalized coordinates to pixel coordinates
                pixel_points = points.copy()
                pixel_points[:, 0] *= width
                pixel_points[:, 1] *= height
                pixel_points = pixel_points.astype(np.int32)
                
                # Draw keypoints with different colors for different groups
                # Filter out keypoints at (0,0) coordinates
                valid_mask = ~((pixel_points[:, 0] == 0) & (pixel_points[:, 1] == 0))
                
                # Pose points (red) - 25 landmarks (0-24)
                for i in range(min(25, len(pixel_points))):
                    if valid_mask[i]:
                        cv2.circle(frame, tuple(pixel_points[i]), 4, (0, 0, 255), -1)
                        
                # Left hand (blue) - 21 landmarks (25-45)
                for i in range(25, min(46, len(pixel_points))):
                    if valid_mask[i]:
                        cv2.circle(frame, tuple(pixel_points[i]), 2, (255, 0, 0), -1)
                        
                # Right hand (green) - 21 landmarks (46-66)
                for i in range(46, min(67, len(pixel_points))):
                    if valid_mask[i]:
                        cv2.circle(frame, tuple(pixel_points[i]), 2, (0, 255, 0), -1)
                        
                # Face (orange/yellow) - 11 landmarks (67-77)
                for i in range(67, min(78, len(pixel_points))):
                    if valid_mask[i]:
                        cv2.circle(frame, tuple(pixel_points[i]), 1, (0, 165, 255), -1)
                        
            out.write(frame)
            frame_idx += 1
            
        # Clean up
        cap.release()
        out.release()
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Failed to create video with keypoints: {str(e)}")
    finally:
        # Clean up temporary input file
        if os.path.exists(input_video_path):
            os.unlink(input_video_path)


def render_topk_table(indices: np.ndarray, probs: np.ndarray, label_prefix: str, title: str) -> None:
    """Render an enhanced top-K predictions table with progress bars."""
    df = pd.DataFrame({
        "Rank": np.arange(1, len(indices) + 1, dtype=int),
        "Class ID": indices,
        "Label": [f"{label_prefix}_{i}" for i in indices],
        "Probability": np.round(probs, 4),
        "Confidence": [f"{p*100:.1f}%" for p in probs]
    })
    
    st.markdown(f"**{title}**")
    
    # Create a more visual representation
    for idx, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 1, 2])
        
        with col1:
            st.markdown(f"**#{row['Rank']}**")
        
        with col2:
            st.markdown(f"**{row['Label']}**")
            st.caption(f"ID: {row['Class ID']}")
        
        with col3:
            st.markdown(f"**{row['Confidence']}**")
        
        with col4:
            # Progress bar visualization
            progress_val = row['Probability']
            st.progress(progress_val)


def render_topk_table_with_labels(predictions: List[Tuple[str, float]], prediction_type: str, title: str) -> None:
    """Render top-k predictions table with human-readable labels."""
    st.markdown(f"**{title}**")
    
    # Create DataFrame
    data = []
    for i, (label, prob) in enumerate(predictions):
        data.append({
            "Rank": i + 1,
            "Label": label,
            "Probability": f"{prob:.3f}",
            "Confidence": f"{prob*100:.1f}%"
        })
    
    df = pd.DataFrame(data)
    
    # Create a more visual representation
    for idx, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 3, 1, 2])
        
        with col1:
            st.markdown(f"**#{row['Rank']}**")
        
        with col2:
            st.markdown(f"**{row['Label']}**")
        
        with col3:
            st.markdown(f"**{row['Confidence']}**")
        
        with col4:
            # Progress bar visualization
            progress_val = float(row['Probability'])
            st.progress(progress_val)
    
