import io
import json
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def set_page() -> None:
    """Configure Streamlit page settings and global styles."""
    st.set_page_config(
        page_title="FSLR Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .status-good { color: #27ae60; }
    .status-warning { color: #f39c12; }
    .status-error { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)


def pad_or_trim(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or trim sequence to target_length along time axis.

    Args:
        sequence: Array shaped [T, D].
        target_length: Desired temporal length T.

    Returns:
        Array shaped [target_length, D], float32.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence [T, D], got shape {sequence.shape}")

    time_steps, feature_dim = sequence.shape
    sequence = sequence.astype(np.float32)

    if time_steps == target_length:
        return sequence
    if time_steps > target_length:
        return sequence[:target_length]

    output = np.zeros((target_length, feature_dim), dtype=np.float32)
    output[:time_steps] = sequence
    return output


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def simulate_predictions(
    random_state: np.random.RandomState,
    num_gloss_classes: int,
    num_category_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated logits for gloss and category heads."""
    gloss_logits = random_state.randn(num_gloss_classes).astype(np.float32)
    cat_logits = random_state.randn(num_category_classes).astype(np.float32)
    return gloss_logits, cat_logits


def topk_from_logits(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and probabilities from logits vector."""
    k = max(1, min(k, logits.shape[-1]))
    probs = softmax(logits)
    topk_idx = np.argpartition(-probs, kth=k - 1)[:k]
    topk_sorted = topk_idx[np.argsort(-probs[topk_idx])]
    return topk_sorted, probs[topk_sorted]


def render_sidebar() -> Dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.markdown("<h1 style='color: #1f77b4;'>FSLR Demo</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Data Input")
    st.sidebar.info("Upload a preprocessed .npz file containing keypoint sequences.")
    
    st.sidebar.markdown("### Model Configuration")
    with st.sidebar.container():
        model_choice = st.selectbox(
            "Model Architecture", 
            ["SignTransformer", "IV3_GRU"], 
            index=0,
            help="Choose between Transformer (keypoints) or IV3-GRU (features)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            num_gloss_classes = st.number_input(
                "Gloss Classes", 
                min_value=2, max_value=2000, value=105, step=1,
                help="Number of sign/word classes"
            )
        with col2:
            num_category_classes = st.number_input(
                "Category Classes", 
                min_value=2, max_value=200, value=10, step=1,
                help="Number of semantic categories"
            )
    
    st.sidebar.markdown("### Processing Options")
    with st.sidebar.container():
        sequence_length = st.slider(
            "Target Sequence Length", 
            min_value=50, max_value=300, value=150, step=10,
            help="Pad or trim sequences to this length"
        )
        topk = st.slider(
            "Top-K Predictions", 
            min_value=1, max_value=10, value=5,
            help="Show top K most likely predictions"
        )
        random_seed = st.number_input(
            "Random Seed", 
            min_value=0, max_value=1_000_000, value=42, step=1,
            help="Seed for reproducible simulation"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This demo processes Filipino Sign Language sequences and provides:
    - **Data validation** and visualization
    - **Feature analysis** over time
    - **Simulated predictions** for gloss and category classification
    """)

    return dict(
        model_choice=model_choice,
        sequence_length=int(sequence_length),
        topk=int(topk),
        num_gloss_classes=int(num_gloss_classes),
        num_category_classes=int(num_category_classes),
        random_seed=int(random_seed),
    )


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
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Frames", f"{raw_length}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Features", f"{raw_features}")
        if raw_features == 156:
            st.markdown("<span class='status-good'>✓ Valid keypoints</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-warning'>⚠ Unexpected shape</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if timestamps_ms.size > 0:
            duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0 if timestamps_ms.size > 1 else 0.0
            st.metric("Duration", f"{duration_s:.2f}s")
            if duration_s > 0:
                fps = raw_length / duration_s if duration_s > 0 else 0
                st.caption(f"~{fps:.1f} FPS")
        else:
            st.metric("Duration", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Target Length", f"{sequence_length}")
        if raw_length > sequence_length:
            st.caption("Will trim")
        elif raw_length < sequence_length:
            st.caption("Will pad")
        else:
            st.caption("Perfect fit")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if X2048 is not None and isinstance(X2048, np.ndarray) and X2048.size > 0:
            st.metric("X2048 Features", "Present")
            st.markdown("<span class='status-good'>✓ IV3-GRU ready</span>", unsafe_allow_html=True)
        else:
            st.metric("X2048 Features", "Missing")
            st.markdown("<span class='status-warning'>⚠ Transformer only</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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

    X_pad = pad_or_trim(X_raw, sequence_length)
    return X_pad, meta_parsed


def render_feature_charts(sequence: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
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
            help="Select which body part to visualize"
        )
        
        start_idx, end_idx = keypoint_groups[selected_group]
    
    with col2:
        coord_type = st.selectbox(
            "Coordinate",
            ["Both X&Y", "X only", "Y only"],
            help="Choose coordinate type to display"
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Heatmap"],
            help="Visualization style"
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
    
    # Show detailed table in expander
    with st.expander(f"Detailed {title} Table"):
        st.dataframe(df, hide_index=True, use_container_width=True)


def main() -> None:
    set_page()
    cfg = render_sidebar()

    # Main header with better styling
    st.markdown("""
    <div class='main-header'>
        Filipino Sign Language Recognition Demo
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; color: #7f8c8d;'>
        Load a preprocessed .npz file to analyze keypoint sequences and view predictions.
    </div>
    """, unsafe_allow_html=True)

    # File upload with better styling
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a preprocessed .npz file", 
        type=["npz"],
        help="Upload a file containing keypoint sequences (X), visibility mask, timestamps, and metadata"
    )

    if uploaded_file is None:
        # Better welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info("""
            **Expected .npz file structure:**
            
            - `X`: Keypoint sequences [T, 156] (required)
            - `mask`: Visibility mask [T, 78] (optional)
            - `timestamps_ms`: Frame timestamps [T] (optional)
            - `meta`: JSON metadata (optional)
            - `X2048`: IV3 features [T, 2048] (optional)
            """)
            
            st.markdown("""
            **How to create .npz files:**
            
            ```bash
            # Single video
            python preprocessing/preprocess.py --write-keypoints \
              /path/to/video.mp4 /path/to/output
            
            # Directory of videos
            python -m preprocessing.preprocess /path/to/videos /path/to/output \
              --write-keypoints --write-iv3-features
            ```
            """)
        
        return

    try:
        file_bytes = io.BytesIO(uploaded_file.read())
        npz = np.load(file_bytes, allow_pickle=True)
    except Exception as exc:
        st.error(f"Failed to read .npz: {exc}")
        return

    try:
        X_pad, mask, meta = render_sequence_overview(npz, cfg["sequence_length"])
    except Exception as exc:
        st.error(str(exc))
        return

    render_feature_charts(X_pad, mask if mask.size > 0 else None)

    # Predictions section with better layout
    st.markdown("<div class='section-header'>Predictions</div>", unsafe_allow_html=True)
    
    # Model info
    model_info_col1, model_info_col2 = st.columns(2)
    with model_info_col1:
        st.info(f"**Model**: {cfg['model_choice']}")
    with model_info_col2:
        st.info(f"**Input**: {'Keypoints (X)' if cfg['model_choice'] == 'SignTransformer' else 'Features (X2048)'}")
    
    # Generate predictions
    rng = np.random.RandomState(cfg["random_seed"])  # reproducible
    gloss_logits, cat_logits = simulate_predictions(
        rng, cfg["num_gloss_classes"], cfg["num_category_classes"]
    )

    g_idx, g_prob = topk_from_logits(gloss_logits, cfg["topk"]) 
    c_idx, c_prob = topk_from_logits(cat_logits, cfg["topk"]) 

    # Enhanced predictions display
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        render_topk_table(g_idx, g_prob, "gloss", "Top Gloss Predictions")
    
    with pred_col2:
        render_topk_table(c_idx, c_prob, "category", "Top Category Predictions")
    
    # Additional insights
    st.markdown("---")
    with st.expander("Prediction Insights", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Top Gloss Confidence", f"{g_prob[0]*100:.1f}%")
            st.metric("Gloss Entropy", f"{-np.sum(g_prob * np.log(g_prob + 1e-10)):.3f}")
        
        with col2:
            st.metric("Top Category Confidence", f"{c_prob[0]*100:.1f}%")
            st.metric("Category Entropy", f"{-np.sum(c_prob * np.log(c_prob + 1e-10)):.3f}")
        
        with col3:
            st.metric("Model", cfg['model_choice'])
            st.metric("Sequence Length", f"{cfg['sequence_length']} frames")
    
    # Disclaimer
    st.markdown("---")
    st.info("Note: This demo uses simulated predictions. To run real inference, load trained model weights and compute predictions.")


if __name__ == "__main__":
    main()