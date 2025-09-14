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
    
    # Consolidated CSS for better styling and layout
    st.markdown("""
    <style>
    /* ===== HEADER STYLES ===== */
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
        color: #ffffff;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        backdrop-filter: blur(10px);
    }
    .status-good { color: #27ae60; }
    .status-warning { color: #f39c12; }
    .status-error { color: #e74c3c; }
    
    /* Hide progress bars in metrics */
    div[data-testid="metric-container"] div[style*="background-color"] {
        display: none !important;
    }
    
    /* ===== FILE UPLOADER STYLING ===== */
    /* Hide Streamlit's default file listing under upload area */
    .stFileUploader > div > div > div > div:not([data-testid="stFileUploaderDropzone"]) {
        display: none !important;
    }
    div[data-testid="stFileUploaderStatus"] {
        display: none !important;
    }
    div[data-testid="stFileUploaderDropzone"] ~ * {
        display: none !important;
    }
    .stFileUploader > div > div > div > div:not([data-testid="stFileUploaderDropzone"]):not([data-testid="stFileUploaderStatus"]) {
        display: none !important;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] + div {
        display: none !important;
    }
    .stFileUploader div:has(span[title*="."]) {
        display: none !important;
    }
    .stFileUploader > div > div > div > *:not([data-testid="stFileUploaderDropzone"]) {
        display: none !important;
    }
    
    /* ===== FILE MANAGEMENT LAYOUT ===== */
    /* Compact file management rows */
    .stContainer > div {
        margin: 0 !important;
        padding: 1px 0 !important;
    }
    .stContainer {
        margin-bottom: 0 !important;
    }
    
    /* Compact spacing for markdown containers */
    div[data-testid="stMarkdownContainer"] p {
        margin: 0.1rem 0 !important;
    }
    .stMarkdownContainer {
        margin: 0 !important;
        padding: 0 !important;
    }
    .stMarkdownContainer p {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Compact button spacing */
    .stButton > button {
        margin: 1px !important;
        padding: 0.2rem 0.4rem !important;
    }
    
    /* Compact column layout */
    .stColumns > div {
        padding: 2px 4px !important;
        margin: 0 !important;
    }
    .stColumns {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Compact separators */
    .stMarkdown hr {
        margin: 4px 0 !important;
        border: none !important;
        border-top: 1px solid #333 !important;
    }
    
    /* ===== TOOLTIP FIXES ===== */
    .stTooltip,
    div[data-testid="stTooltip"],
    [data-testid="stTooltip"] {
        z-index: 99999 !important;
        position: fixed !important;
        pointer-events: none !important;
    }
    div[title]:hover::after,
    button[title]:hover::after {
        content: attr(title);
        position: fixed !important;
        z-index: 99999 !important;
        background: rgba(0, 0, 0, 0.8) !important;
        color: white !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        pointer-events: none !important;
        white-space: nowrap !important;
    }
    .stContainer,
    .stMarkdownContainer,
    div[data-testid="stMarkdownContainer"] {
        overflow: visible !important;
    }
    .stButton,
    button {
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* ===== EMPTY CONTAINER HIDING ===== */
    .stMarkdownContainer:empty,
    div[data-testid="stMarkdownContainer"]:empty {
        display: none !important;
    }
    .stMarkdownContainer script,
    div[data-testid="stMarkdownContainer"] script {
        display: none !important;
    }
    .stAlert:empty,
    div[data-testid="stAlert"]:empty {
        display: none !important;
    }
    .stMarkdownContainer:has(> div:empty),
    div[data-testid="stMarkdownContainer"]:has(> div:empty) {
        display: none !important;
    }
    .stMarkdownContainer:not(:has(*)) {
        display: none !important;
    }
    </style>
    
    <script>
    // JavaScript fallback to hide file listing
    function hideFileListing() {
        const fileUploader = document.querySelector('div[data-testid="stFileUploader"]');
        if (fileUploader) {
            const dropzone = fileUploader.querySelector('div[data-testid="stFileUploaderDropzone"]');
            if (dropzone) {
                let nextSibling = dropzone.nextElementSibling;
                while (nextSibling) {
                    nextSibling.style.display = 'none';
                    nextSibling = nextSibling.nextElementSibling;
                }
            }
        }
    }
    
    // Run on page load and DOM changes
    window.addEventListener('load', hideFileListing);
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                hideFileListing();
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)


def render_sidebar() -> Dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.markdown("<h1 style='color: #1f77b4;'>FSLR Demo</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Data Input Section
    st.sidebar.markdown("### Data Input")
    st.sidebar.info("Upload a preprocessed .npz file or video file for processing.")
    
    # Model Configuration Section
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
    
    # Processing Options Section
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
    
    # About Section
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


def render_file_upload() -> object:
    """Render file upload component with native Streamlit design."""
    return st.file_uploader(
        "Choose .npz files or video files (max 10)", 
        type=["npz", "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
        accept_multiple_files=True,
        help="Upload preprocessed .npz files or video files for processing (up to 10 files)"
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
        Upload a preprocessed .npz file or video to analyze sign language sequences and view predictions.
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
    from streamlit_app.utils import simulate_predictions, topk_from_logits
    import numpy as np
    
    rng = np.random.RandomState(cfg["random_seed"])
    gloss_logits, cat_logits = simulate_predictions(
        rng, cfg["num_gloss_classes"], cfg["num_category_classes"]
    )

    g_idx, g_prob = topk_from_logits(gloss_logits, cfg["topk"]) 
    c_idx, c_prob = topk_from_logits(cat_logits, cfg["topk"]) 

    # Enhanced predictions display
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        from streamlit_app.visualization import render_topk_table
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