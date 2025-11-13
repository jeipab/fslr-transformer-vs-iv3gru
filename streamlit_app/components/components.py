"""UI components for the Streamlit app."""

import streamlit as st
from typing import Dict
from ..core.config import MODEL_CONFIG


def set_page() -> None:
    """Configure Streamlit page settings and global styles."""
    from ..core.config import PAGE_CONFIG
    st.set_page_config(**PAGE_CONFIG)
    
    # Consolidated CSS for styling and layout
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
    
    .main-section-header {
        font-size: 2rem;
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
    
    /* Sidebar action button styling */
    section[data-testid="stSidebar"] .stButton > button {
        margin: 0.25rem 0 !important;
        width: 100% !important;
        padding: 0.55rem 0.75rem !important;
        border-radius: 6px !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
        background: rgba(15, 23, 42, 0.25) !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
        background: rgba(15, 23, 42, 0.35) !important;
        border-color: rgba(148, 163, 184, 0.6) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] .stButton > button:active {
        outline: none !important;
        background: rgba(15, 23, 55, 0.45) !important;
        border-color: rgba(148, 163, 184, 0.8) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button:disabled {
        opacity: 0.75 !important;
        background: rgba(148, 163, 184, 0.15) !important;
        border-color: rgba(148, 163, 184, 0.3) !important;
        color: rgba(226, 232, 240, 0.7) !important;
    }
    section[data-testid="stSidebar"] .sidebar-link-button {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0.25rem 0 !important;
        padding: 0.55rem 0.75rem !important;
        border-radius: 6px !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
        background: rgba(15, 23, 42, 0.25) !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .sidebar-link-button:hover {
        background: rgba(15, 23, 42, 0.35) !important;
        border-color: rgba(148, 163, 184, 0.6) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .sidebar-link-button:focus,
    section[data-testid="stSidebar"] .sidebar-link-button:active {
        outline: none !important;
        background: rgba(15, 23, 55, 0.45) !important;
        border-color: rgba(148, 163, 184, 0.8) !important;
        color: #ffffff !important;
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
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
        height: fit-content !important;
        min-height: auto !important;
    }
    
    /* Remove excess bottom spacing from sidebar */
    .css-1d391kg .stMarkdownContainer:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Ensure sidebar content fits properly */
    .css-1d391kg .stMarkdownContainer {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Target sidebar content area specifically */
    .css-1d391kg > div {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Remove spacing from sidebar blocks */
    .css-1d391kg .stMarkdownContainer,
    .css-1d391kg .stContainer,
    .css-1d391kg .stSelectbox,
    .css-1d391kg .stRadio {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove any default Streamlit sidebar spacing */
    .css-1d391kg .element-container:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Additional sidebar cleanup - remove all bottom spacing */
    .css-1d391kg .stMarkdownContainer:last-child,
    .css-1d391kg .stContainer:last-child,
    .css-1d391kg .element-container:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Ensure sidebar ends cleanly without extra space */
    .css-1d391kg {
        overflow: hidden !important;
    }
    
    /* Additional comprehensive sidebar spacing cleanup */
    .css-1d391kg *:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove any Streamlit default spacing from sidebar */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stMarkdownContainer,
    .css-1d391kg .stContainer,
    .css-1d391kg .element-container {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Force sidebar to not have any bottom padding/margin */
    .css-1d391kg {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
        border-bottom: none !important;
    }
    
    /* Target Streamlit's sidebar container specifically */
    .css-1d391kg .stContainer,
    .css-1d391kg .stMarkdownContainer,
    .css-1d391kg .element-container {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove spacing from sidebar separators */
    .css-1d391kg hr {
        margin: 0.5rem 0 !important;
        border: none !important;
        border-top: 1px solid #4a5568 !important;
    }
    
    /* Remove spacing from sidebar containers */
    .css-1d391kg .stContainer {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure the last element in sidebar has no bottom spacing */
    .css-1d391kg > div:last-child,
    .css-1d391kg .stMarkdownContainer:last-child,
    .css-1d391kg .stContainer:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* More aggressive sidebar spacing removal */
    .css-1d391kg * {
        margin-bottom: 0 !important;
    }
    
    /* Target specific Streamlit sidebar elements */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stMarkdownContainer,
    .css-1d391kg .stContainer,
    .css-1d391kg .element-container,
    .css-1d391kg .stRadio,
    .css-1d391kg .stSelectbox,
    .css-1d391kg .stCheckbox {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove spacing from the sidebar content area */
    .css-1d391kg .block-container {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Force sidebar to end exactly at content */
    .css-1d391kg {
        height: auto !important;
        max-height: none !important;
        min-height: auto !important;
    }
    
    /* Comprehensive sidebar cleanup - remove ALL spacing */
    .css-1d391kg,
    .css-1d391kg *,
    .css-1d391kg > div,
    .css-1d391kg .block-container,
    .css-1d391kg .stMarkdownContainer,
    .css-1d391kg .stContainer,
    .css-1d391kg .element-container {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Ensure sidebar content area has no bottom spacing */
    .css-1d391kg .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove any Streamlit default spacing from sidebar */
    .css-1d391kg .stMarkdownContainer:last-child,
    .css-1d391kg .stContainer:last-child,
    .css-1d391kg .element-container:last-child,
    .css-1d391kg > div:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        border-bottom: none !important;
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
    
    /* ===== VIDEO PREVIEW STYLING ===== */
    .video-preview-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .video-preview-container video {
        border-radius: 6px;
        max-height: 120px;
        width: 100%;
    }
    
    /* Video file card styling */
    .video-file-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .video-file-card:hover {
        background-color: rgba(255, 255, 255, 0.08);
        border-color: rgba(52, 152, 219, 0.3);
    }
    
    /* Make Streamlit video elements fit container */
    .stVideo {
        max-height: 500px !important;
        height: 500px !important;
        width: 100% !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: rgba(0, 0, 0, 0.1) !important;
        border-radius: 8px !important;
    }

    .stVideo video {
        max-height: 500px !important;
        max-width: 100% !important;
        height: auto !important;
        width: auto !important;
        object-fit: contain !important;
        border-radius: 6px !important;
    }
    
    /* Video thumbnail carousel styling */
    .video-thumbnail {
        transition: all 0.3s ease !important;
        margin: 0.25rem !important;
    }
    
    .video-thumbnail:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-2px) !important;
    }
    
    .video-thumbnail.selected {
        background-color: rgba(52, 152, 219, 0.2) !important;
        border-color: #3498db !important;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* Vertical thumbnail carousel styling */
    .video-thumbnail-vertical {
        transition: all 0.3s ease !important;
        margin: 0.5rem 0 !important;
        width: 100% !important;
    }
    
    .video-thumbnail-vertical:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        transform: translateX(4px) !important;
    }
    
    .video-thumbnail-vertical.selected {
        background-color: rgba(52, 152, 219, 0.2) !important;
        border-color: #3498db !important;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3) !important;
        transform: translateX(4px) !important;
    }
    
    /* Compact thumbnail styling */
    .video-thumbnail-compact {
        transition: all 0.3s ease !important;
        margin: 0.2rem 0 !important;
        width: 100% !important;
    }
    
    .video-thumbnail-compact:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        transform: translateX(2px) !important;
    }
    
    .video-thumbnail-compact.selected {
        background-color: rgba(52, 152, 219, 0.2) !important;
        border-color: #3498db !important;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.3) !important;
        transform: translateX(2px) !important;
    }
    
    /* Compact button styling */
    .stButton > button[key*="thumb_"] {
        font-size: 0.75rem !important;
        padding: 0.25rem 0.5rem !important;
        height: auto !important;
        min-height: 2rem !important;
        display: block !important;
    }

    /* Video list select button styling - make them look like cards */
    .stButton > button[key*="video_select_"] {
        height: 56px !important;
        min-height: 56px !important;
        max-height: 56px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 0.75rem !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
        text-align: left !important;
        margin: 0.25rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
    }
    
    /* Enhanced button hover effects */
    .stButton > button[key*="video_select_"]:hover:not(:disabled) {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(52, 152, 219, 0.3) !important;
    }
    
    /* Selected button styling */
    .stButton > button[key*="video_select_"][disabled] {
        background-color: rgba(52, 152, 219, 0.2) !important;
        border-color: #3498db !important;
        color: #ffffff !important;
        opacity: 1 !important;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2) !important;
    }
    
    /* Primary button styling for selected state */
    .stButton > button[key*="video_select_"][kind="primary"] {
        background-color: rgba(52, 152, 219, 0.2) !important;
        border-color: #3498db !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Secondary button styling for unselected state */
    .stButton > button[key*="video_select_"][kind="secondary"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }
    
    
    /* Video list styling - now handled by native Streamlit container */
    .video-item-selected {
        background-color: rgba(52, 152, 219, 0.2) !important;
        border: 2px solid #3498db !important;
    }

    .video-item-unselected {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .video-item-unselected:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(52, 152, 219, 0.3) !important;
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
    
    /* ===== SIDEBAR SPACING FIXES ===== */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stExpander,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stContainer {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown + div,
    section[data-testid="stSidebar"] .stMarkdown + .stRadio,
    section[data-testid="stSidebar"] .stMarkdown + .stExpander {
        margin-top: 1rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: .09rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ===== MANUAL GAPS ABOVE VALIDATE & ANDROID BUTTONS ===== */

    /* Add gap before Validate Models button */
    section[data-testid="stSidebar"] .stButton:nth-last-of-type(2) {
        margin-top: 0.75rem !important;
    }

    /* Add gap before Android button (last button or link) */
    section[data-testid="stSidebar"] .stButton:last-of-type,
    section[data-testid="stSidebar"] a.sidebar-link-button {
        margin-top: 0.75rem !important;
        display: block !important;
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
    # Clean, elegant header with responsive sizing
    st.sidebar.markdown("""
    <style>
    .sidebar-header h1 {
        color: #1f77b4;
        font-size: clamp(1.5rem, 5vw, 2.5rem);
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    @media (max-width: 768px) {
        .sidebar-header h1 {
            font-size: 1.5rem;
            white-space: normal;
            word-break: keep-all;
            overflow-wrap: normal;
        }
    }
    </style>
    <div class='sidebar-header' style='text-align: left; padding: 0 0 0.5rem 0; border-bottom: 1px solid #4a5568; margin-bottom: 0.5rem; margin-top: -1rem;'>
        <h1>PANSINAYAN</h1>
        <p style='color: #a0aec0; font-size: 0.9rem; margin: 0.1rem 0 0 0; font-weight: 400;'>Where Every Sign Gets Attention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recognition Mode Section
    st.sidebar.markdown("""
    <div style='margin: 0.75rem 0 0;'>
        <h3 style='color: #e2e8f0; margin: 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Recognition Mode</h3>
    </div>
    """, unsafe_allow_html=True)
    
    from ..core.config import RECOGNITION_MODES
    default_mode = st.session_state.get('recognition_mode', 'continuous')
    recognition_mode = st.sidebar.radio(
        "Select mode",
        options=['isolated', 'continuous'],
        format_func=lambda x: RECOGNITION_MODES[x],
        index=0 if default_mode == 'isolated' else 1,
        help="📍 Isolated Mode: Classify single signs from video clips\n\n🎬 Continuous Mode: Recognize sequences of signs using CTC models",
        key="recognition_mode_radio"
    )
    
    # Update session state
    st.session_state.recognition_mode = recognition_mode
    
    # About Section
    st.sidebar.markdown("""
    <div style='margin-bottom: 0;'>
        <h3 style='color: #e2e8f0; margin: 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>About</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # About the Tool (Expandable)
    with st.sidebar.expander("About the Tool", expanded=False):
        st.markdown("""
        **PANSINAYAN** is a Multi-Head Attention Transformer for recognizing Filipino Sign Language (FSL).
        
        ### What This Demo Does:
        
        **Video Processing**
        - Upload FSL videos or use pre-processed keypoint data
        - Automatic extraction of hand, face, and body landmarks
        - Converts videos to structured numerical features
        
        **Data Analysis**
        - Real-time validation of input data quality
        - Interactive visualization of keypoints over time
        - Frame-by-frame feature analysis with charts
        
        **AI Recognition**
        - Predicts 1 of 105 Filipino Sign Language glosses
        - Detects occlusion (when hands/face are blocked)
        - Compare two architectures: **Transformer** vs **InceptionV3+GRU**
        
        **Model Comparison**
        - Ground-truth vs prediction sequence comparison
        - Temporal alignment over continuous timelines
        - Category insights with occlusion awareness
        """)
    
    st.sidebar.markdown("<div style='line-height: 0.5;'><br></div>", unsafe_allow_html=True)
    
    # About the Name (Expandable)
    with st.sidebar.expander("About the Name", expanded=False):
        st.markdown("""
        **PANSINAYAN** is a Filipino portmanteau with deep meaning:
        
        ### Word Formation:
        ```
        PANSIN  +  SENYAS  +  -AN
           ↓          ↓        ↓
        attention   sign    place
        ```
        
        **= "The place where signs receive attention"**
        
        ---
        
        ### Why This Name?
        
        **Technical Connection**  
        The name reflects our **Multi-Head Attention** mechanism — the core innovation that lets the AI focus on relevant sign language features simultaneously.
        
        **Cultural Respect**  
        A Filipino name honors the Filipino Deaf community and recognizes FSL as a complete language with its own grammar.
        
        **Our Mission**  
        Every sign — and every signer — deserves attention, recognition, and inclusivity.
        """)
    
    # Model Status Section
    st.sidebar.markdown("""
    <div style='margin: 0.75rem 0 0;'>
        <h3 style='color: #e2e8f0; margin: 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Model Status</h3>
    </div>
    """, unsafe_allow_html=True)
    render_model_status()
    
    # Model Configuration Section
    st.sidebar.markdown("""
    <div style='margin: 0.75rem 0 0;'>
        <h3 style='color: #e2e8f0; margin: 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Architecture Selection (mode-aware)
    st.sidebar.markdown("**Model Architecture**")
    
    available_models = get_available_models_for_mode(recognition_mode)
    
    if not available_models:
        st.sidebar.warning(f"⚠️ No models available for {RECOGNITION_MODES[recognition_mode]}")
        model_choice = None
    elif len(available_models) == 1:
        # Only one model available
        model_choice = available_models[0]
        st.sidebar.info(f"Using: **{MODEL_CONFIG[model_choice]['display_name']}**")
    elif len(available_models) == 2:
        # Create help text with model file paths
        help_text = f"Select model for {RECOGNITION_MODES[recognition_mode]}\n\n"
        
        for model_name in available_models:
            if MODEL_CONFIG[model_name]['enabled']:
                model_path = MODEL_CONFIG[model_name]['checkpoint_path']
                display_name = MODEL_CONFIG[model_name]['display_name']
                help_text += f"{display_name}:\n{model_path}\n\n"
        
        # Use radio button for binary choice
        model_choice = st.sidebar.radio(
            "Choose Model",
            available_models,
            format_func=lambda x: MODEL_CONFIG[x]['display_name'],
            index=0,
            help=help_text,
            key="model_architecture_radio"
        )
    else:
        # Create help text with model file paths
        help_text = "Choose between available model architectures\n\n"
        
        for model_name in available_models:
            if MODEL_CONFIG[model_name]['enabled']:
                model_path = MODEL_CONFIG[model_name]['checkpoint_path']
                display_name = MODEL_CONFIG[model_name]['display_name']
                help_text += f"{display_name}:\n{model_path}\n\n"
        
        # Fallback to selectbox for multiple options
        model_choice = st.sidebar.selectbox(
            "Model Architecture", 
            available_models,
            format_func=lambda x: MODEL_CONFIG[x]['display_name'],
            index=0,
            help=help_text,
            key="model_architecture_select"
        )
    
    # Resource Section
    st.sidebar.markdown("""
    <div style='margin: 0.75rem 0 0;'>
        <h3 style='color: #e2e8f0; margin: 0; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Resources</h3>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<div style='height: 0.05rem;'></div>", unsafe_allow_html=True)

    from .demo_video import render_demo_button
    render_demo_button()
    render_android_button()
    
    return dict(
        model_choice=model_choice,
        sequence_length=150,  # Default sequence length
        topk=5,  # Default top-k predictions
    )


def render_model_status():
    """Render model availability status in sidebar based on current recognition mode."""
    from ..core.config import MODEL_CONFIG, RECOGNITION_MODES
    
    # Get current recognition mode (default to 'continuous' if not set)
    recognition_mode = st.session_state.get('recognition_mode', 'continuous')
    
    # Get available models for the current mode using the helper function
    available_models = get_available_models_for_mode(recognition_mode)
    
    # Get display names for available models
    available_display_names = [
        MODEL_CONFIG.get(model_name, {}).get('display_name', model_name)
        for model_name in available_models
        if MODEL_CONFIG.get(model_name)
    ]
    
    # Determine status based on available models
    num_available = len(available_models)
    
    if num_available == 0:
        # No models available for current mode
        mode_display = RECOGNITION_MODES.get(recognition_mode, recognition_mode)
        _render_status_card(
            status_text='No Models Available',
            detail_text=f'Checkpoint files not found for {mode_display}',
            status_color='#ef4444',
            status_bg='rgba(239, 68, 68, 0.1)'
        )
    elif num_available == 1:
        # Only one model available
        _render_status_card(
            status_text='Partial Availability',
            detail_text=f'{available_display_names[0]} only',
            status_color='#f59e0b',
            status_bg='rgba(245, 158, 11, 0.1)'
        )
    else:
        # Multiple models available - determine if all expected models are present
        expected_models = (
            {'transformer_isolated', 'iv3_gru_isolated'} if recognition_mode == 'isolated'
            else {'transformer_continuous', 'iv3_gru_continuous'}
        )
        has_all_expected = expected_models.issubset(set(available_models))
        
        # Format model list: "A & B" for 2 models, "A, B & C" for 3+
        if len(available_display_names) == 2:
            model_list = ' & '.join(available_display_names)
        else:
            model_list = ', '.join(available_display_names[:-1]) + ' & ' + available_display_names[-1]
        
        if has_all_expected:
            _render_status_card(
                status_text='All Models Ready',
                detail_text=model_list,
                status_color='#10b981',
                status_bg='rgba(16, 185, 129, 0.1)'
            )
        else:
            _render_status_card(
                status_text='Partial Availability',
                detail_text=model_list,
                status_color='#f59e0b',
                status_bg='rgba(245, 158, 11, 0.1)'
            )
    
    # Add validation button below model status
    st.sidebar.markdown("<div style='height: 0.2rem;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("Validate Models", help="Access model validation mode", width='stretch'):
        st.session_state.workflow_stage = 'validation'
        st.rerun()


def _render_status_card(status_text: str, detail_text: str, status_color: str, status_bg: str):
    """Helper function to render a status card in the sidebar."""
    st.sidebar.markdown(f"""
    <div style='background: {status_bg}; border: 1px solid {status_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0;'>
        <div style='display: flex; align-items: center; color: #ffffff; font-weight: 500; margin-bottom: 0.3rem;'>
            <div style='width: 8px; height: 8px; background: {status_color}; border-radius: 50%; margin-right: 0.75rem;'></div>
            <span style='font-size: 1rem;'>{status_text}</span>
        </div>
        <div style='font-size: 0.9rem; color: #a0aec0; line-height: 1.4;'>
            {detail_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_android_button():
    """Render a styled sidebar button for downloading the Android APK."""
    apk_url = "https://drive.google.com/uc?export=download&id=1qrjWyIOJNSuDYvkzUncOynkVFoe1yzFk"

    if st.sidebar.button("Android", help="Download the Android APK", use_container_width=True):
        # Open the APK link in a new tab using JavaScript
        st.markdown(
            f"""
            <script>
            window.open('{apk_url}', '_blank');
            </script>
            """,
            unsafe_allow_html=True
        )


def get_available_models():
    """Get list of available models for selection (legacy function)."""
    return get_available_models_for_mode('isolated')


def get_available_models_for_mode(mode: str):
    """Get list of models compatible with recognition mode.
    
    Args:
        mode: Recognition mode ('isolated' or 'continuous')
        
    Returns:
        List of model names compatible with the mode
    """
    import os
    from ..core.config import get_models_by_mode, MODEL_CONFIG
    
    # Get models compatible with mode
    compatible_models = get_models_by_mode(mode)
    
    # Filter by checkpoint existence
    available_models = []
    for model_name in compatible_models:
        config = MODEL_CONFIG.get(model_name)
        if config and config['enabled']:
            checkpoint_path = config['checkpoint_path']
            if os.path.exists(checkpoint_path):
                available_models.append(model_name)
    
    return available_models


def render_file_upload() -> object:
    """Render file upload component with mobile camera upload support.
    
    Uses Base64 encoding to ensure media data passes through WebSocket connection,
    avoiding session affinity issues with load balancing as per Streamlit best practices.
    """
    
    # Enhanced mobile camera capture with Base64 encoding for robust delivery
    # This ensures files pass through WebSocket, avoiding HTTP session issues
    st.markdown("""
    <script>
    // Enhanced camera capture with Base64 encoding for WebSocket delivery
    function enhanceCameraCapture() {
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput && !fileInput.hasAttribute('data-camera-enhanced')) {
            // Mark as enhanced to avoid duplicate setup
            fileInput.setAttribute('data-camera-enhanced', 'true');
            
            // Enable camera capture for mobile devices
            fileInput.setAttribute('capture', 'environment');
            fileInput.setAttribute('accept', 'video/*,.mp4,.mov,.webm,.npz');
            
            // Store original change handler
            const originalOnChange = fileInput.onchange;
            
            // Enhanced change handler with immediate processing
            fileInput.addEventListener('change', function(e) {
                if (e.target.files && e.target.files.length > 0) {
                    console.log('File(s) selected:', e.target.files.length);
                    
                    // Force immediate Streamlit sync with multiple event triggers
                    const triggerSync = () => {
                        // Create multiple event types to ensure Streamlit catches it
                        ['input', 'change'].forEach(eventType => {
                            const event = new Event(eventType, { 
                                bubbles: true, 
                                cancelable: true 
                            });
                            e.target.dispatchEvent(event);
                        });
                        
                        // Trigger Streamlit-specific events
                        if (window.streamlit) {
                            try {
                                window.streamlit.setComponentValue(e.target.files);
                            } catch (err) {
                                console.log('Streamlit API call failed, using fallback');
                            }
                        }
                        
                        // Focus/blur cycle to trigger Streamlit reactivity
                        e.target.focus();
                        setTimeout(() => e.target.blur(), 50);
                    };
                    
                    // Immediate trigger
                    triggerSync();
                    
                    // Delayed triggers for reliability
                    setTimeout(triggerSync, 100);
                    setTimeout(triggerSync, 300);
                    
                    // Create a visual indicator for mobile users
                    const uploadIndicator = document.createElement('div');
                    uploadIndicator.id = 'upload-indicator';
                    uploadIndicator.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        background: rgba(52, 152, 219, 0.95);
                        color: white;
                        padding: 20px 40px;
                        border-radius: 8px;
                        font-size: 16px;
                        font-weight: 600;
                        z-index: 9999;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                        animation: fadeIn 0.3s ease;
                    `;
                    uploadIndicator.textContent = `📤 Uploading ${e.target.files.length} file(s)...`;
                    document.body.appendChild(uploadIndicator);
                    
                    // Remove indicator after 3 seconds
                    setTimeout(() => {
                        if (uploadIndicator.parentNode) {
                            uploadIndicator.style.animation = 'fadeOut 0.3s ease';
                            setTimeout(() => uploadIndicator.remove(), 300);
                        }
                    }, 3000);
                }
            }, { capture: true });
            
            // Add animation styles
            if (!document.getElementById('upload-animations')) {
                const style = document.createElement('style');
                style.id = 'upload-animations';
                style.textContent = `
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translate(-50%, -60%); }
                        to { opacity: 1; transform: translate(-50%, -50%); }
                    }
                    @keyframes fadeOut {
                        from { opacity: 1; }
                        to { opacity: 0; }
                    }
                `;
                document.head.appendChild(style);
            }
        }
    }
    
    // Apply enhancement on load and DOM changes
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', enhanceCameraCapture);
    } else {
        enhanceCameraCapture();
    }
    
    // Watch for Streamlit re-renders
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                enhanceCameraCapture();
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    return st.file_uploader(
        "Choose .npz files or video files (max 10)", 
        type=["npz", "mp4", "mov", "webm"],
        accept_multiple_files=True,
        help="Upload preprocessed .npz files or video files for processing (up to 10 files). Mobile users: Tap 'Browse files' to access camera directly.",
        key="main_file_uploader"
    )


def render_video_preview(uploaded_file, use_base64: bool = False) -> None:
    """Render video preview for uploaded video files.
    
    Args:
        uploaded_file: Uploaded video file
        use_base64: If True, encode video as Base64 data URI for WebSocket delivery
                   (recommended for mobile uploads and load-balanced deployments)
    """
    if not uploaded_file:
        return
    
    try:
        if use_base64:
            # Base64 encoding ensures data passes through WebSocket
            # Avoids session affinity issues in load-balanced deployments
            from .utils import encode_file_to_base64, get_mime_type_from_extension
            
            file_data = uploaded_file.getvalue()
            mime_type = get_mime_type_from_extension(uploaded_file.name)
            
            # Note: For large videos, Base64 encoding increases size by ~33%
            # Consider this approach for smaller files or use external storage for large files
            if len(file_data) > 50 * 1024 * 1024:  # 50MB threshold
                st.warning(f"⚠️ Large video file ({len(file_data)/(1024*1024):.1f} MB). Using standard preview.")
                # Use keyed container to prevent ID conflicts
                video_container = st.container(key=f"uploaded_video_large_container_{uploaded_file.name}")
                with video_container:
                    st.video(uploaded_file, format="video/mp4", start_time=0, autoplay=True, loop=True)
            else:
                data_uri = encode_file_to_base64(file_data, mime_type)
                st.markdown(f"""
                <video width="100%" height="auto" controls autoplay loop style="border-radius: 6px; max-height: 500px;">
                    <source src="{data_uri}" type="{mime_type}">
                    Your browser does not support the video tag.
                </video>
                """, unsafe_allow_html=True)
        else:
            # Standard Streamlit video display in keyed container
            video_container = st.container(key=f"uploaded_video_container_{uploaded_file.name}")
            with video_container:
                st.video(uploaded_file, format="video/mp4", start_time=0, autoplay=True, loop=True)
            
    except Exception as e:
        # Fallback: show file info if video preview fails
        st.info(f"Video preview not available for {uploaded_file.name}")
        st.write(f"File size: {len(uploaded_file.getvalue()) / (1024*1024):.1f} MB")


def render_video_carousel(video_files) -> None:
    """Render video files in a scrollable list with side-by-side layout."""
    if not video_files:
        return

    # Initialize session state for carousel
    if 'selected_video_index' not in st.session_state:
        st.session_state.selected_video_index = 0
    
    # Get upload configuration for Base64 encoding
    from ..core.config import get_upload_config
    use_base64 = get_upload_config('use_base64_preview')
    
    # Create side-by-side layout: video list on left, preview on right
    col1, col2 = st.columns([1, 3], gap="medium")

    with col1:
        # Use Streamlit's native container with fixed height for scrolling
        with st.container(height=500):
            # Create compact video list items with consistent spacing
            for i, video_file in enumerate(video_files):
                file_size = len(video_file.getvalue())
                size_mb = file_size / (1024 * 1024)

                # Check if this video is selected
                is_selected = i == st.session_state.selected_video_index

                # Create a clickable video card using button with custom styling
                button_text = video_file.name
                
                if st.button(
                    button_text,
                    key=f"video_select_{i}",
                    help=f"Click to view {video_file.name}" if not is_selected else f"Currently viewing {video_file.name}",
                    type="primary" if is_selected else "secondary",
                    disabled=False,
                    width='stretch'
                ):
                    st.session_state.selected_video_index = i
                    st.rerun()

                # Consistent spacing between items
                if i < len(video_files) - 1:  # Don't add spacing after last item
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    with col2:
        # Show video preview in a fixed container without scrolling
        if st.session_state.selected_video_index < len(video_files):
            selected_video = video_files[st.session_state.selected_video_index]
            render_video_preview(selected_video, use_base64=use_base64)
        else:
            st.info("Select a video from the list to preview it.")


def render_main_header() -> None:
    """Render main page header."""
    st.markdown("""
    <div style='text-align: center; margin-top: -2rem; margin-bottom: 1rem;'>
        <div style='font-size: 3.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.2rem;'>
            PANSINAYAN
        </div>
        <div style='color: #a0aec0; font-size: 1.2rem; font-weight: 400;'>
            Where Every Sign Gets Attention
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add spacing to move content down
    st.markdown("""
    <div style='margin-top: 1.5rem;'></div>
    """, unsafe_allow_html=True)


def render_predictions_section(cfg: Dict, npz_data: Dict = None, filename: str = None, metadata: Dict = None, key_suffix: str = "") -> None:
    """Render predictions section with enhanced layout."""
    st.markdown("<div class='section-header'>Predictions</div>", unsafe_allow_html=True)
    
    # Model info removed as requested
    
    # Generate real predictions if NPZ data is available
    if npz_data is not None:
        from ..manager.prediction_manager import make_real_prediction, get_model_manager
        
        model_name = cfg['model_choice']
        
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
            # Convert tensor to int if needed
            gloss_id_int = gloss_id.item() if hasattr(gloss_id, 'item') else int(gloss_id)
            gloss_label = gloss_mapping.get(gloss_id_int, f'Unknown ({gloss_id_int})')
            gloss_top5.append((gloss_label, prob))
        
        category_top3 = []
        for cat_id, prob in prediction_results['category_top3']:
            # Convert tensor to int if needed
            cat_id_int = cat_id.item() if hasattr(cat_id, 'item') else int(cat_id)
            cat_label = category_mapping.get(cat_id_int, f'Unknown ({cat_id_int})')
            category_top3.append((cat_label, prob))
        
        # Enhanced predictions display - 2 columns: predictions on left, video on right
        pred_col_left, pred_col_right = st.columns([1, 1], gap="large")
        
        with pred_col_left:
            # Stack Top Gloss and Top Category predictions vertically
            from .visualization import render_topk_table_with_labels
            render_topk_table_with_labels(gloss_top5, "gloss", "Top Gloss Predictions")
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            render_topk_table_with_labels(category_top3, "category", "Top Category Predictions")
        
        with pred_col_right:
            # Video preview spans the full height on the right
            render_inline_video_preview(npz_data, metadata, filename, key_suffix)
        
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


def render_inline_video_preview(npz_data: Dict, metadata: Dict, filename: str, key_suffix: str) -> None:
    """Render compact video preview inline with predictions."""
    import os
    import tempfile
    import numpy as np
    from pathlib import Path
    from .visualization import create_keypoint_animation_video
    
    st.markdown("**Video Preview**")
    
    # Determine background - always use Grid for automatic generation
    bg_type = "Grid"
    
    # Check if video already generated (cached)
    video_key = f"auto_preview_{filename}_{key_suffix}"
    
    if video_key not in st.session_state:
        # Auto-generate video on first view
        with st.spinner("Generating..."):
            try:
                # Extract keypoints
                if 'X' not in npz_data:
                    st.error("Missing keypoint data")
                    return
                
                X = npz_data['X']
                time_steps = X.shape[0]
                
                # Calculate number of keypoints based on feature dimension
                num_keypoints = X.shape[1] // 2
                
                # Validate that feature_dim is divisible by 2
                if X.shape[1] % 2 != 0:
                    st.error(f"Invalid keypoint feature dimension: {X.shape[1]}. Expected even number for (x,y) coordinates.")
                    return
                
                # Reshape to [T, num_keypoints, 2]
                keypoints_2d = X.reshape(time_steps, num_keypoints, 2)
                mask = npz_data.get('mask', None)
                
                # Video settings - use fixed size and scale original video to fit
                fps = 30
                width, height = 360, 360  # Fixed size for consistent output
                show_skeleton = True
                
                # Generate video with auto prefix to separate from manual generation
                video_path = create_keypoint_animation_video(
                    keypoints_2d, mask, fps, width, height,
                    show_skeleton, bg_type, f"auto_{filename}_{key_suffix}"
                )
                
                if video_path and os.path.exists(video_path):
                    st.session_state[video_key] = video_path
                else:
                    st.error("Video generation failed")
                    return
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
    
    # Display video player
    if video_key in st.session_state:
        video_path = st.session_state[video_key]
        if os.path.exists(video_path):
            try:
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Compact video player with autoplay and loop in keyed container
                video_container = st.container(key=f"auto_video_container_{filename}_{key_suffix}")
                with video_container:
                    st.video(video_bytes, autoplay=True, loop=True)
            except Exception as e:
                st.error(f"Playback error: {str(e)}")