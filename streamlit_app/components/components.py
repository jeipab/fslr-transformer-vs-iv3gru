"""UI components for the Streamlit app."""

import streamlit as st
from typing import Dict


def set_page() -> None:
    """Configure Streamlit page settings and global styles."""
    from ..core.config import PAGE_CONFIG
    st.set_page_config(**PAGE_CONFIG)
    
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
    
    /* ===== SIDEBAR STYLING ===== */
    /* Clean sidebar styling */
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    
    /* Sidebar selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #3498db !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3498db !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.1) !important;
    }
    
    /* Fix dropdown text visibility - make it black */
    .stSelectbox > div > div > div {
        color: #000000 !important;
    }
    
    .stSelectbox > div > div > div > div {
        color: #000000 !important;
    }
    
    /* Fix selected option text - make it black */
    .stSelectbox [data-baseweb="select"] {
        color: #000000 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #000000 !important;
    }
    
    /* Additional selectors for dropdown text */
    .stSelectbox div[data-baseweb="select"] {
        color: #000000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    /* Sidebar label styling */
    .stSelectbox label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ===== CUSTOM RESPONSIVE LAYOUT ===== */
    /* Force side-by-side layout for visualization columns at 50%+ screen width */
    @media (min-width: 50vw) {
        .viz-side-by-side .stColumns > div {
            flex: 0 0 50% !important;
            max-width: 50% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
        }
        
        /* Ensure equal height for visualization sections */
        .viz-side-by-side .stColumns {
            align-items: stretch !important;
        }
        
        /* Make content fill available height */
        .viz-side-by-side .stColumns > div > div {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        /* Ensure minimum height for better visual balance */
        .viz-side-by-side .stColumns > div {
            min-height: 500px !important;
        }
        
        /* Add gap between sections for better visual separation */
        .viz-side-by-side .stColumns > div:first-child {
            padding-right: 2rem !important;
        }
        
        .viz-side-by-side .stColumns > div:last-child {
            padding-left: 2rem !important;
        }
    }
    
    /* ===== CUSTOM BUTTON COLORS ===== */
    /* Reset button - Warning/Orange color */
    .stButton > button[kind="primary"]:has-text("Reset") {
        background-color: #f39c12 !important;
        border-color: #e67e22 !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:has-text("Reset"):hover {
        background-color: #e67e22 !important;
        border-color: #d35400 !important;
    }
    
    /* Clear All button - Danger/Red color */
    .stButton > button[kind="primary"]:has-text("Clear All") {
        background-color: #e74c3c !important;
        border-color: #c0392b !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:has-text("Clear All"):hover {
        background-color: #c0392b !important;
        border-color: #a93226 !important;
    }
    
    /* Force stacked layout for visualization columns below 50% screen width */
    @media (max-width: 49.99vw) {
        .viz-side-by-side .stColumns > div {
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }
        
        /* Remove gap when stacked */
        .viz-side-by-side .stColumns > div:first-child {
            padding-right: 0 !important;
        }
        
        .viz-side-by-side .stColumns > div:last-child {
            padding-left: 0 !important;
        }
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
    # Clean, elegant header
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 0 0 1rem 0; border-bottom: 1px solid #4a5568; margin-bottom: 1rem; margin-top: -3rem;'>
        <h1 style='color: #1f77b4; font-size: 2.8rem; font-weight: bold; margin: 0;'>FSLR Demo</h1>
        <p style='color: #a0aec0; font-size: 1rem; margin: 0.5rem 0 0 0; font-weight: 400;'>Filipino Sign Language Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Status Section - Clean and minimal
    st.sidebar.markdown("""
    <div style='margin-bottom: 0.1rem;'>
        <h3 style='color: #e2e8f0; margin: 0 0 0.1rem 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Model Status</h3>
    </div>
    """, unsafe_allow_html=True)
    render_model_status()
    
    # Model Configuration Section - Clean and minimal
    st.sidebar.markdown("""
    <div style='margin: 1.5rem 0 0.1rem 0;'>
        <h3 style='color: #e2e8f0; margin: 0 0 0.1rem 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.container():
        # Model Architecture Selection
        st.markdown("**Model Architecture**")
        
        available_models = get_available_models()
        if len(available_models) == 2:
            # Use radio button for binary choice
            model_choice = st.radio(
                "Choose Model",
                available_models,
                index=0,
                help="Select the model architecture for predictions",
                key="model_architecture_radio"
            )
        else:
            # Fallback to selectbox for multiple options
            model_choice = st.selectbox(
                "Model Architecture", 
                available_models, 
                index=0,
                help="Choose between available model architectures",
                key="model_architecture_select"
            )
        
        # Occlusion Detection Options
        st.markdown("---")
        st.markdown("**Occlusion Detection**")
        
        # Detailed Results
        occ_detailed = st.checkbox(
            "Detailed Results",
            value=False,
            help="Include detailed per-frame occlusion analysis with region detection",
            key="occ_detailed_checkbox"
        )
    
    # About Section - Clean and minimal
    st.sidebar.markdown("""
    <div style='margin: 1.5rem 0 0.1rem 0;'>
        <h3 style='color: #e2e8f0; margin: 0 0 0.1rem 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>About</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style='background: rgba(255, 255, 255, 0.05); padding: 1.25rem; border-radius: 6px; border-left: 3px solid #3498db; margin-bottom: 1rem;'>
        <p style='color: #e2e8f0; margin: 0 0 0.75rem 0; font-size: 1rem; line-height: 1.5; font-weight: 400;'>
        This demo processes Filipino Sign Language sequences and provides:
        </p>
        <ul style='color: #a0aec0; font-size: 0.95rem; margin: 0; padding-left: 1.25rem; line-height: 1.6;'>
            <li>Automatic file processing</li>
            <li>Data validation & visualization</li>
            <li>Feature analysis over time</li>
            <li>Real model predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    return dict(
        model_choice=model_choice,
        sequence_length=150,  # Default sequence length
        topk=5,  # Default top-k predictions
    )




def render_model_status():
    """Render model availability status in sidebar."""
    from ..manager.prediction_manager import MODEL_CONFIG
    
    # Check model availability
    transformer_available = MODEL_CONFIG['transformer']['enabled']
    iv3_gru_available = MODEL_CONFIG['iv3_gru']['enabled']
    
    # Create clean, elegant status display
    if transformer_available and iv3_gru_available:
        st.sidebar.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 1rem; margin-bottom: 0.25rem;'>
            <div style='display: flex; align-items: center; color: #ffffff; font-weight: 500; margin-bottom: 0.5rem;'>
                <div style='width: 8px; height: 8px; background: #10b981; border-radius: 50%; margin-right: 0.75rem;'></div>
                <span style='font-size: 1rem;'>All Models Ready</span>
            </div>
            <div style='font-size: 0.9rem; color: #a0aec0; line-height: 1.4;'>
                SignTransformer & InceptionV3+GRU
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif transformer_available:
        st.sidebar.markdown("""
        <div style='background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin-bottom: 0.25rem;'>
            <div style='display: flex; align-items: center; color: #ffffff; font-weight: 500; margin-bottom: 0.5rem;'>
                <div style='width: 8px; height: 8px; background: #f59e0b; border-radius: 50%; margin-right: 0.75rem;'></div>
                <span style='font-size: 1rem;'>Partial Availability</span>
            </div>
            <div style='font-size: 0.9rem; color: #a0aec0; line-height: 1.4;'>
                SignTransformer model only
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif iv3_gru_available:
        st.sidebar.markdown("""
        <div style='background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin-bottom: 0.25rem;'>
            <div style='display: flex; align-items: center; color: #ffffff; font-weight: 500; margin-bottom: 0.5rem;'>
                <div style='width: 8px; height: 8px; background: #f59e0b; border-radius: 50%; margin-right: 0.75rem;'></div>
                <span style='font-size: 1rem;'>Partial Availability</span>
            </div>
            <div style='font-size: 0.9rem; color: #a0aec0; line-height: 1.4;'>
                InceptionV3+GRU model only
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 1rem; margin-bottom: 0.25rem;'>
            <div style='display: flex; align-items: center; color: #ffffff; font-weight: 500; margin-bottom: 0.5rem;'>
                <div style='width: 8px; height: 8px; background: #ef4444; border-radius: 50%; margin-right: 0.75rem;'></div>
                <span style='font-size: 1rem;'>No Models Available</span>
            </div>
            <div style='font-size: 0.9rem; color: #a0aec0; line-height: 1.4;'>
                Please check model configuration
            </div>
        </div>
        """, unsafe_allow_html=True)


def get_available_models():
    """Get list of available models for selection."""
    from ..manager.prediction_manager import MODEL_CONFIG
    
    available_models = []
    if MODEL_CONFIG['transformer']['enabled']:
        available_models.append("SignTransformer")
    if MODEL_CONFIG['iv3_gru']['enabled']:
        available_models.append("IV3_GRU")
    
    # Fallback to at least one model if none are available
    if not available_models:
        available_models = ["SignTransformer"]
    
    return available_models


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
    
    # Add spacing to move content down
    st.markdown("""
    <div style='margin-top: 3rem;'></div>
    """, unsafe_allow_html=True)


def render_predictions_section(cfg: Dict, npz_data: Dict = None, filename: str = None) -> None:
    """Render predictions section with enhanced layout."""
    st.markdown("<div class='section-header'>Predictions</div>", unsafe_allow_html=True)
    
    # Model info
    model_info_col1, model_info_col2 = st.columns(2)
    with model_info_col1:
        st.info(f"**Model**: {cfg['model_choice']}")
    with model_info_col2:
        st.info(f"**Input**: {'Keypoints (X)' if cfg['model_choice'] == 'SignTransformer' else 'Features (X2048)'}")
    
    # Generate real predictions if NPZ data is available
    if npz_data is not None:
        from ..manager.prediction_manager import make_real_prediction, get_model_manager
        
        model_name = 'transformer' if cfg['model_choice'] == 'SignTransformer' else 'iv3_gru'
        
        with st.spinner("Making prediction..."):
            prediction_results = make_real_prediction(npz_data, model_name)
        
        if prediction_results is None:
            st.error("Failed to make prediction. Please check model availability and try again.")
            return
        
        # Get label mappings
        model_manager = get_model_manager()
        gloss_mapping, category_mapping = model_manager.get_label_mappings()
        
        # Format predictions with human-readable labels
        gloss_top5 = []
        for gloss_id, prob in prediction_results['gloss_top5']:
            gloss_label = gloss_mapping.get(gloss_id, f'Unknown ({gloss_id})')
            gloss_top5.append((gloss_label, prob))
        
        category_top3 = []
        for cat_id, prob in prediction_results['category_top3']:
            cat_label = category_mapping.get(cat_id, f'Unknown ({cat_id})')
            category_top3.append((cat_label, prob))
        
        # Enhanced predictions display
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            from .visualization import render_topk_table_with_labels
            render_topk_table_with_labels(gloss_top5, "gloss", "Top Gloss Predictions")
        
        with pred_col2:
            render_topk_table_with_labels(category_top3, "category", "Top Category Predictions")
        
        # Additional insights
        st.markdown("---")
        with st.expander("Prediction Insights", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Top Gloss Confidence", f"{prediction_results['gloss_probability']*100:.1f}%")
                # Calculate entropy for gloss predictions
                import numpy as np
                gloss_probs = np.array([prob for _, prob in gloss_top5])
                gloss_entropy = -np.sum(gloss_probs * np.log(gloss_probs + 1e-10))
                st.metric("Gloss Entropy", f"{gloss_entropy:.3f}")
            
            with col2:
                st.metric("Top Category Confidence", f"{prediction_results['category_probability']*100:.1f}%")
                # Calculate entropy for category predictions
                cat_probs = np.array([prob for _, prob in category_top3])
                cat_entropy = -np.sum(cat_probs * np.log(cat_probs + 1e-10))
                st.metric("Category Entropy", f"{cat_entropy:.3f}")
            
            with col3:
                st.metric("Model", cfg['model_choice'])
                if 'frames_extracted' in prediction_results:
                    st.metric("Frames Processed", prediction_results['frames_extracted'])
                else:
                    st.metric("Sequence Length", f"{cfg['sequence_length']} frames")
        
        # Prediction completed successfully - no need to show message
        
    else:
        # No NPZ data available - show message to upload files
        st.info("Please upload an NPZ file or video to see model predictions.")