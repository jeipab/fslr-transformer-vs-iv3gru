"""FSLR Demo - Main Streamlit Application"""

from typing import Dict
import streamlit as st

from ..components import set_page, render_sidebar, render_main_header
from ..manager.upload_manager import initialize_upload_session_state, render_upload_stage
from ..manager.preprocessing_manager import render_preprocessing_stage
from ..manager.prediction_manager import render_predictions_stage, cleanup_on_app_exit
from ..manager.validation_manager import run_validation_from_folder, cleanup_temp_files
from ..components.validation_components import (
    render_model_selection, render_dataset_upload,
    render_validation_configuration, render_validation_results, render_validation_summary,
    render_download_results
)


def main() -> None:
    """Application entry point."""
    set_page()
    cfg = render_sidebar()
    initialize_upload_session_state()

    render_main_header()
    
    # Route to appropriate workflow stage
    if st.session_state.workflow_stage == 'upload':
        render_upload_stage()
    elif st.session_state.workflow_stage == 'preprocessing':
        render_preprocessing_stage()
    elif st.session_state.workflow_stage == 'validation':
        render_validation_stage(cfg)
    else:  # predictions stage
        render_predictions_stage(cfg)


def render_validation_stage(cfg: Dict):
    """Model validation interface."""
    
    # Navigation header
    col1, col2, col3, col4 = st.columns([2, 6, 1, 1])
    with col1:
        if st.button("← Back to Upload", help="Return to upload stage", type="secondary"):
            st.session_state.workflow_stage = 'upload'
            st.rerun()
    with col2:
        st.markdown("")  # Empty space
    with col3:
        st.markdown("")  # Empty space
    with col4:
        st.markdown("")  # Empty space
    
    st.markdown("### Model Validation")
    
    # Model selection
    selected_model = render_model_selection()
    if not selected_model:
        return
    
    # Dataset upload
    npz_folder_path, labels_csv = render_dataset_upload()
    
    if not npz_folder_path or not labels_csv:
        st.markdown("---")
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            st.info("""
            **Validation Setup Required**
            
            **Required inputs:**
            - **NPZ folder path**: Directory containing validation NPZ files
            - **Labels CSV file**: CSV with columns: file, gloss, cat, occluded
            
            **Validation process:**
            - Model evaluation on validation dataset
            - Performance metrics calculation (accuracy, precision, recall, F1-score)
            - Confusion matrix generation and analysis
            - Occlusion analysis (occluded vs non-occluded performance)
            - Per-class performance breakdown
            - Results export and download options
            """)
        return
    
    # Configuration
    batch_size, device = render_validation_configuration()
    
    # Run validation
    if st.button("Run Validation", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current_batch, total_batches):
                progress = current_batch / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {current_batch}/{total_batches}")
            
            with st.spinner("Running validation..."):
                results = run_validation_from_folder(
                    model_type=selected_model,
                    npz_folder_path=npz_folder_path,
                    labels_csv_file=labels_csv,
                    batch_size=batch_size,
                    progress_callback=progress_callback
                )
            
            st.session_state.validation_results = results
            progress_bar.empty()
            status_text.empty()
            st.toast("Validation completed successfully!", icon="✅")
            
        except Exception as e:
            st.error(f"Validation failed: {str(e)}")
            return
    
    # Display results
    if 'validation_results' in st.session_state and st.session_state.validation_results:
        results = st.session_state.validation_results
        render_validation_summary(results)
        render_validation_results(results)
        render_download_results(results)
    


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_on_app_exit()