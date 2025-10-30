"""FSLR Demo - Main Streamlit Application"""

from typing import Dict
import streamlit as st

from ..components import set_page, render_sidebar, render_main_header
from ..manager.upload_manager import initialize_upload_session_state, render_upload_stage
from ..manager.preprocessing_manager import render_preprocessing_stage
from ..manager.prediction_manager import render_predictions_stage, cleanup_on_app_exit
from ..manager.validation_manager import run_validation_from_folder, run_ctc_validation_with_sliding_window, cleanup_temp_files
from ..components.validation_components import (
    render_model_selection, render_dataset_upload, render_ctc_dataset_upload,
    render_validation_configuration, render_ctc_validation_configuration,
    render_validation_results, render_validation_summary, render_download_results,
    render_ctc_validation_results
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
    
    st.markdown("<div class='main-section-header'>MODEL VALIDATION</div>", unsafe_allow_html=True)
    
    # Model selection
    selected_model = render_model_selection()
    if not selected_model:
        return
    
    # Check if selected model is CTC model
    from ..core.config import is_ctc_model
    is_ctc = is_ctc_model(selected_model)
    
    # Dataset upload
    if is_ctc:
        npz_folder_path, labels_csv_or_gt = render_ctc_dataset_upload()
    else:
        npz_folder_path, labels_csv_or_gt = render_dataset_upload()
    
    labels_csv = labels_csv_or_gt  # For compatibility
    
    if not npz_folder_path or not labels_csv:
        st.markdown("---")
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            if is_ctc:
                st.info("""
                **CTC Validation Setup Required**
                
                **Required inputs:**
                - **NPZ folder path**: Directory containing continuous sequence NPZ files
                - **Ground Truth folder**: Directory containing ground truth JSON files (*_gt.json)
                
                **CTC Validation process:**
                - Sliding‑window sequence prediction with CTC greedy decoding
                - Requires ground‑truth JSON with timestamps for metrics
                - Detection metrics: Precision, Recall, F1‑Score, Mean IoU (TP)
                - Per‑signer and per‑strategy metric breakdowns
                - Detailed per‑sequence results and downloadable report
                """)
            else:
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
    if is_ctc:
        batch_size, device, decode_method, beam_width = render_ctc_validation_configuration()
    else:
        batch_size, device = render_validation_configuration()
        decode_method, beam_width = None, None
    
    # Run validation
    if st.button("Run Validation", type="primary", width='stretch'):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current_batch, total_batches):
                progress = current_batch / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {current_batch}/{total_batches}")
            
            with st.spinner("Running validation..."):
                if is_ctc:
                    # CTC validation with sliding window
                    results = run_ctc_validation_with_sliding_window(
                        model_type=selected_model,
                        npz_folder_path=npz_folder_path,
                        ground_truth_folder=labels_csv,  # For CTC, this is GT folder
                        decode_method=decode_method,
                        beam_width=beam_width,
                        window_size=150,  # 5 seconds at 30fps
                        stride=50,  # 75% overlap
                        progress_callback=progress_callback
                    )
                else:
                    # Standard classification validation
                    results = run_validation_from_folder(
                        model_type=selected_model,
                        npz_folder_path=npz_folder_path,
                        labels_csv_file=labels_csv,
                        batch_size=batch_size,
                        progress_callback=progress_callback
                    )
            
            st.session_state.validation_results = results
            st.session_state.validation_is_ctc = is_ctc
            progress_bar.empty()
            status_text.empty()
            st.toast("Validation completed successfully!", icon="✅")
            
        except Exception as e:
            st.error(f"Validation failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return
    
    # Display results
    if 'validation_results' in st.session_state and st.session_state.validation_results:
        results = st.session_state.validation_results
        is_ctc_results = st.session_state.get('validation_is_ctc', False)
        
        if is_ctc_results:
            render_ctc_validation_results(results)
        else:
            render_validation_summary(results)
            render_validation_results(results)
            render_download_results(results)
    


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_on_app_exit()