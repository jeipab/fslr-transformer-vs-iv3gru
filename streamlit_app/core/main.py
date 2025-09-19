"""Main Streamlit application for FSLR Demo."""

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
    """Main application function."""
    set_page()
    cfg = render_sidebar()
    initialize_upload_session_state()

    # Main header
    render_main_header()
    
    # Four-stage workflow
    if st.session_state.workflow_stage == 'upload':
        render_upload_stage()
    elif st.session_state.workflow_stage == 'preprocessing':
        render_preprocessing_stage()
    elif st.session_state.workflow_stage == 'validation':
        render_validation_stage(cfg)
    else:  # predictions stage
        render_predictions_stage(cfg)


def render_validation_stage(cfg: Dict):
    """Render the validation stage."""
    
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
        # Add divider above info box
        st.markdown("---")
        
        # Create centered column layout for info box only
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
    
    # Validation button
    if st.button("Run Validation", type="primary", use_container_width=True):
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run validation with progress callback
            def progress_callback(current_batch, total_batches):
                progress = current_batch / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {current_batch}/{total_batches}")
            
            # Run validation
            with st.spinner("Running validation..."):
                results = run_validation_from_folder(
                    model_type=selected_model,
                    npz_folder_path=npz_folder_path,
                    labels_csv_file=labels_csv,
                    batch_size=batch_size,
                    progress_callback=progress_callback
                )
            
            # Store results in session state
            st.session_state.validation_results = results
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("Validation completed successfully!")
            
        except Exception as e:
            st.error(f"Validation failed: {str(e)}")
            return
    
    # Display results if available
    if 'validation_results' in st.session_state and st.session_state.validation_results:
        results = st.session_state.validation_results
        
        # Validation summary
        render_validation_summary(results)
        
        # Detailed results
        render_validation_results(results)
        
        # Download options
        render_download_results(results)
    
    # Cleanup on exit
    if st.button("Cleanup", help="Clean up temporary files", type="secondary"):
        cleanup_temp_files()
        st.success("Cleanup completed!")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up resources when app exits
        cleanup_on_app_exit()