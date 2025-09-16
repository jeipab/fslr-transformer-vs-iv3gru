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
    # Enhanced header with better styling
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0; border-bottom: 2px solid #1f77b4; margin-bottom: 1.5rem;'>
        <h1 style='color: #1f77b4; font-size: 2rem; font-weight: bold; margin: 0;'>FSLR Demo</h1>
        <p style='color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Filipino Sign Language Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Input Section with enhanced styling
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>📁 Data Input</h3>
        <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>Upload NPZ files or video files for processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Status Section with enhanced styling
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>🤖 Model Status</h3>
    </div>
    """, unsafe_allow_html=True)
    render_model_status()
    
    # Model Configuration Section with enhanced styling
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>⚙️ Model Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.container():
        model_choice = st.selectbox(
            "Model Architecture", 
            get_available_models(), 
            index=0,
            help="Choose between available model architectures",
            key="model_architecture_select"
        )
    
    # Add some spacing
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # About Section with enhanced styling
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
        <h3 style='color: white; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>ℹ️ About</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 0.5rem;'>
        <p style='color: #333; margin: 0; font-size: 0.9rem; line-height: 1.4;'>
        This demo processes Filipino Sign Language sequences and provides:
        </p>
        <ul style='color: #333; font-size: 0.85rem; margin: 0.5rem 0 0 0; padding-left: 1rem;'>
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
    
    # Create styled status display
    if transformer_available and iv3_gru_available:
        st.sidebar.markdown("""
        <div style='background: rgba(46, 160, 67, 0.1); border: 1px solid #2ea043; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; align-items: center; color: #2ea043; font-weight: 500;'>
                <span style='font-size: 1.2rem; margin-right: 0.5rem;'>✅</span>
                <span>Both models available</span>
            </div>
            <div style='font-size: 0.8rem; color: #666; margin-top: 0.25rem;'>
                SignTransformer & InceptionV3+GRU
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif transformer_available:
        st.sidebar.markdown("""
        <div style='background: rgba(255, 193, 7, 0.1); border: 1px solid #ffc107; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; align-items: center; color: #856404; font-weight: 500;'>
                <span style='font-size: 1.2rem; margin-right: 0.5rem;'>⚠️</span>
                <span>Only Transformer available</span>
            </div>
            <div style='font-size: 0.8rem; color: #666; margin-top: 0.25rem;'>
                SignTransformer model only
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif iv3_gru_available:
        st.sidebar.markdown("""
        <div style='background: rgba(255, 193, 7, 0.1); border: 1px solid #ffc107; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; align-items: center; color: #856404; font-weight: 500;'>
                <span style='font-size: 1.2rem; margin-right: 0.5rem;'>⚠️</span>
                <span>Only IV3-GRU available</span>
            </div>
            <div style='font-size: 0.8rem; color: #666; margin-top: 0.25rem;'>
                InceptionV3+GRU model only
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style='background: rgba(220, 53, 69, 0.1); border: 1px solid #dc3545; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; align-items: center; color: #dc3545; font-weight: 500;'>
                <span style='font-size: 1.2rem; margin-right: 0.5rem;'>❌</span>
                <span>No models available</span>
            </div>
            <div style='font-size: 0.8rem; color: #666; margin-top: 0.25rem;'>
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