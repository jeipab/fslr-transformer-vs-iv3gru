"""Main Streamlit application for FSLR Demo."""

from typing import Dict
import streamlit as st

from streamlit_app.components import set_page, render_sidebar, render_main_header
from streamlit_app.upload_manager import initialize_upload_session_state, render_upload_stage
from streamlit_app.preprocessing_manager import render_preprocessing_stage
from streamlit_app.prediction_manager import render_predictions_stage, cleanup_on_app_exit


def main() -> None:
    """Main application function."""
    set_page()
    cfg = render_sidebar()
    initialize_upload_session_state()

    # Main header
    render_main_header()
    
    # Three-stage workflow
    if st.session_state.workflow_stage == 'upload':
        render_upload_stage()
    elif st.session_state.workflow_stage == 'preprocessing':
        render_preprocessing_stage()
    else:  # predictions stage
        render_predictions_stage(cfg)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up resources when app exits
        cleanup_on_app_exit()