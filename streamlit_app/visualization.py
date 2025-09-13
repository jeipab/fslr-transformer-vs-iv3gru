"""Visualization components for keypoints and predictions."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    st.markdown("<div class='section-header'>Sequence Overview</div>", unsafe_allow_html=True)
    
    # Main metrics in a more visual layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Frames", str(raw_length), delta=None)
    
    with col2:
        st.metric("Features", str(raw_features), delta=None)
        if raw_features == 156:
            st.markdown("<span class='status-good'>✓ Valid keypoints</span>", unsafe_allow_html=True)
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

    from streamlit_app.utils import pad_or_trim
    X_pad = pad_or_trim(X_raw, sequence_length)
    return X_pad, mask, meta_parsed


def render_animated_keypoints(sequence: np.ndarray, mask: Optional[np.ndarray] = None, key_suffix: str = "") -> None:
    """Render animated keypoint visualization with skeleton overlay."""
    st.markdown("<div class='section-header'>Animated Keypoint Visualization</div>", unsafe_allow_html=True)
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available. Please install plotly to see animated keypoint visualization.")
        # Fallback to simple text display
        st.write("Keypoint data shape:", sequence.shape)
        st.write("Sample keypoints:", sequence[:3, :10])
        return
    
    time_steps, feature_dim = sequence.shape
    
    # Reshape keypoints to [T, 78, 2] for easier handling
    keypoints_2d = sequence.reshape(time_steps, 78, 2)
    
    # Define skeleton connections for MediaPipe Holistic
    skeleton_connections = {
        "pose": [
            # Pose landmarks (0-32)
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ],
        "left_hand": [
            # Left hand landmarks (0-20, offset by 33)
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ],
        "right_hand": [
            # Right hand landmarks (0-20, offset by 54)
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ],
        "face": [
            # Face landmarks (0-10, offset by 75)
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 0)
        ]
    }
    
    # Animation controls
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
            start_idx, end_idx = 0, 33
        elif part_name == "left_hand":
            start_idx, end_idx = 33, 54
        elif part_name == "right_hand":
            start_idx, end_idx = 54, 75
        elif part_name == "face":
            start_idx, end_idx = 75, 78
        
        part_keypoints = current_keypoints[start_idx:end_idx]
        
        # Plot individual points
        if show_visibility and mask is not None and mask.size > 0:
            # Use visibility mask for coloring
            visibility = mask[frame_idx, start_idx:end_idx] if mask.shape[1] >= end_idx else np.ones(end_idx - start_idx, dtype=bool)
            point_colors = ['rgba(255,0,0,1)' if v else 'rgba(255,0,0,0.3)' for v in visibility]
        else:
            point_colors = [colors[part_name]] * len(part_keypoints)
        
        fig.add_trace(go.Scatter(
            x=part_keypoints[:, 0],
            y=part_keypoints[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=point_colors,
                line=dict(width=2, color='white')
            ),
            name=f"{part_name.title()} Points",
            showlegend=True
        ))
        
        # Plot skeleton connections
        if show_skeleton:
            for start_conn, end_conn in connections:
                if start_conn < len(part_keypoints) and end_conn < len(part_keypoints):
                    fig.add_trace(go.Scatter(
                        x=[part_keypoints[start_conn, 0], part_keypoints[end_conn, 0]],
                        y=[part_keypoints[start_conn, 1], part_keypoints[end_conn, 1]],
                        mode='lines',
                        line=dict(color=colors[part_name], width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # Update layout
    fig.update_layout(
        title=f"Keypoint Visualization - Frame {frame_idx}/{time_steps-1}",
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
        # Keypoint group selection
        keypoint_groups = {
            "Pose (upper body)": (0, 50),
            "Left hand": (50, 92),
            "Right hand": (92, 134),
            "Face landmarks": (134, 156)
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
            height=400
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
        fig.update_layout(height=400)
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
                # Pose points (red)
                for i in range(min(25, len(pixel_points))):
                    cv2.circle(frame, tuple(pixel_points[i]), 4, (0, 0, 255), -1)
                        
                # Left hand (blue) 
                for i in range(25, min(46, len(pixel_points))):
                    cv2.circle(frame, tuple(pixel_points[i]), 2, (255, 0, 0), -1)
                        
                # Right hand (green)
                for i in range(46, min(67, len(pixel_points))):
                    cv2.circle(frame, tuple(pixel_points[i]), 2, (0, 255, 0), -1)
                        
                # Face (orange/yellow)
                for i in range(67, min(78, len(pixel_points))):
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
    
