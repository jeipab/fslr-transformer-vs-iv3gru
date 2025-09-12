"""UI components for the Streamlit app."""

import streamlit as st
from typing import Dict


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


def render_sidebar() -> Dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.markdown("<h1 style='color: #1f77b4;'>FSLR Demo</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Data Input")
    st.sidebar.info("Upload a preprocessed .npz file or video file for processing.")
    
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
    - **Automatic file processing** (NPZ or video input)
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


def render_welcome_screen() -> None:
    """Render welcome screen when no file is uploaded."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("""
        **What files can I upload?**
        
        **Preprocessed data files (.npz):**
        - Ready-to-use keypoint data from previous processing
        - Instant analysis and visualization
        
        **Video files:**
        - Common formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM
        - The app will automatically extract keypoints from your video
        - Takes a bit longer to process but works with any sign language video
        """)


def render_file_upload() -> object:
    """Render file upload component."""
    return st.file_uploader(
        "Choose a .npz file or video file", 
        type=["npz", "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
        help="Upload a preprocessed .npz file or video file for keypoint extraction"
    )


def render_main_header() -> None:
    """Render main page header."""
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


def render_predictions_section(cfg: Dict, gloss_logits: object, cat_logits: object) -> None:
    """Render predictions section with enhanced layout."""
    st.markdown("<div class='section-header'>Predictions</div>", unsafe_allow_html=True)
    
    # Model info
    model_info_col1, model_info_col2 = st.columns(2)
    with model_info_col1:
        st.info(f"**Model**: {cfg['model_choice']}")
    with model_info_col2:
        st.info(f"**Input**: {'Keypoints (X)' if cfg['model_choice'] == 'SignTransformer' else 'Features (X2048)'}")
    
    # Generate predictions
    from utils import simulate_predictions, topk_from_logits
    import numpy as np
    
    rng = np.random.RandomState(cfg["random_seed"])  # reproducible
    gloss_logits, cat_logits = simulate_predictions(
        rng, cfg["num_gloss_classes"], cfg["num_category_classes"]
    )

    g_idx, g_prob = topk_from_logits(gloss_logits, cfg["topk"]) 
    c_idx, c_prob = topk_from_logits(cat_logits, cfg["topk"]) 

    # Enhanced predictions display
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        from visualization import render_topk_table
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
