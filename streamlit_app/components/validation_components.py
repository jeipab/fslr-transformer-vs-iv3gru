"""Validation components for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Set
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path


def render_case_legend() -> None:
    """Render shared legend for prediction case colors."""
    st.markdown(
        (
            "<span style='color:white;font-weight:600;'>Legend:</span>"
            "<span style='color:white;font-weight:600;'> </span>"
            "<span style='color:#22c55e;font-weight:600;'>True Positive</span>"
            "<span style='color:white;font-weight:600;'> | </span>"
            "<span style='color:#ef4444;font-weight:600;'>False Positive</span>"
            "<span style='color:white;font-weight:600;'> | </span>"
            "<span style='color:#64748b;font-weight:600;'>True Negative</span>"
            "<span style='color:white;font-weight:600;'> | </span>"
            "<span style='color:#f97316;font-weight:600;'>False Negative</span>"
        ),
        unsafe_allow_html=True,
    )
def _augment_matches_with_order_categories(
    pred_categories: List[int],
    gt_categories: List[int],
    pred_indices: List[int],
    gt_indices: List[int],
) -> List[Dict[str, int]]:
    """
    Order-preserving augmentation of matches based on category equality.
    Mirrors backend logic used during prediction to keep frontend in sync.
    """
    if not pred_indices or not gt_indices:
        return []

    add_pairs: List[Dict[str, int]] = []
    p_i = 0
    g_i = 0
    while p_i < len(pred_indices) and g_i < len(gt_indices):
        pi = pred_indices[p_i]
        gi = gt_indices[g_i]
        if pi >= len(pred_categories) or gi >= len(gt_categories):
            break
        if pred_categories[pi] == gt_categories[gi]:
            add_pairs.append({'pred_idx': pi, 'gt_idx': gi})
            p_i += 1
            g_i += 1
        else:
            p_i += 1

    return add_pairs


def _derive_category_case_maps(
    pred: Dict[str, Any],
    predicted_categories: Optional[List[int]],
    ground_truth_categories: Optional[List[int]],
    matched_pairs: List[Dict[str, Any]],
    pred_len: int,
    gt_len: int,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Construct category TP/FP/FN maps for predicted and ground truth indexes.
    Prefer backend-provided indices; fall back to recomputing locally when absent.
    """
    pred_map: Dict[int, str] = {}
    gt_map: Dict[int, str] = {}

    # Prefer explicit indices provided by backend (newer exports)
    if pred.get('category_tp_pred_indices') is not None:
        for idx in pred.get('category_tp_pred_indices') or []:
            idx = int(idx)
            if 0 <= idx < pred_len:
                pred_map[idx] = 'TP'
    if pred.get('category_fp_pred_indices') is not None:
        for idx in pred.get('category_fp_pred_indices') or []:
            idx = int(idx)
            if 0 <= idx < pred_len:
                pred_map[idx] = 'FP'
    if pred.get('category_tp_gt_indices') is not None:
        for idx in pred.get('category_tp_gt_indices') or []:
            idx = int(idx)
            if 0 <= idx < gt_len:
                gt_map[idx] = 'TP'
    if pred.get('category_fn_gt_indices') is not None:
        for idx in pred.get('category_fn_gt_indices') or []:
            idx = int(idx)
            if 0 <= idx < gt_len:
                gt_map[idx] = 'FN'

    # If any maps were populated, return early
    if pred_map or gt_map:
        return pred_map, gt_map

    # Fallback: recompute using available sequence information
    if not predicted_categories or not ground_truth_categories:
        return pred_map, gt_map

    pred_len = min(pred_len, len(predicted_categories))
    gt_len = min(gt_len, len(ground_truth_categories))

    matched_pred: Set[int] = set()
    matched_gt: Set[int] = set()

    for pair in matched_pairs or []:
        pi = pair.get('pred_idx')
        gi = pair.get('gt_idx')
        if pi is None or gi is None:
            continue
        pi = int(pi)
        gi = int(gi)
        if 0 <= pi < pred_len and 0 <= gi < gt_len:
            if predicted_categories[pi] == ground_truth_categories[gi]:
                matched_pred.add(pi)
                matched_gt.add(gi)

    remaining_pred = [i for i in range(pred_len) if i not in matched_pred]
    remaining_gt = [j for j in range(gt_len) if j not in matched_gt]

    add_pairs = _augment_matches_with_order_categories(
        predicted_categories,
        ground_truth_categories,
        remaining_pred,
        remaining_gt,
    )

    for pair in add_pairs:
        pi = pair['pred_idx']
        gi = pair['gt_idx']
        if 0 <= pi < pred_len:
            matched_pred.add(pi)
        if 0 <= gi < gt_len:
            matched_gt.add(gi)

    for idx in matched_pred:
        pred_map[idx] = 'TP'
    for idx in matched_gt:
        gt_map[idx] = 'TP'

    for idx in range(pred_len):
        if idx not in matched_pred:
            pred_map[idx] = 'FP'

    for idx in range(gt_len):
        if idx not in matched_gt:
            gt_map[idx] = 'FN'

    return pred_map, gt_map
from ..core.config import MODEL_CONFIG, is_ctc_model, get_models_by_mode
from ..components.ctc_visualization import (
    render_sequence_comparison,
    render_temporal_alignment,
)
def _enrich_ground_truth_timestamps(
    timestamps: List[Dict[str, Any]],
    gloss_labels: List[str],
    gloss_sequence: List[int],
    category_ids: Optional[List[int]],
    category_labels: Optional[List[str]],
    gloss_mapping: Dict[int, str],
    category_mapping: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Ensure ground truth timestamps include gloss/category labels for visualization."""
    if not timestamps:
        return []

    enriched: List[Dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        ts_copy = dict(ts)

        # Determine gloss id & label
        gloss_id: Optional[int] = ts_copy.get('gloss')
        if gloss_id is None and gloss_sequence and idx < len(gloss_sequence):
            gloss_id = gloss_sequence[idx]
            ts_copy['gloss'] = gloss_id

        gloss_label = ts_copy.get('gloss_label')
        if not gloss_label:
            if gloss_labels and idx < len(gloss_labels):
                gloss_label = gloss_labels[idx]
            elif gloss_id is not None:
                gloss_label = gloss_mapping.get(int(gloss_id), str(gloss_id))
            else:
                gloss_label = ''
            ts_copy['gloss_label'] = gloss_label

        # Determine category id & label
        cat_id = ts_copy.get('category')
        if cat_id is None and category_ids and idx < len(category_ids):
            cat_id = category_ids[idx]
            ts_copy['category'] = cat_id

        cat_label = ts_copy.get('category_label')
        if not cat_label:
            if category_labels and idx < len(category_labels) and category_labels[idx]:
                cat_label = category_labels[idx]
            elif cat_id is not None:
                cat_label = category_mapping.get(int(cat_id), f"Cat_{cat_id}")
            else:
                cat_label = ''
            if cat_label:
                ts_copy['category_label'] = cat_label

        enriched.append(ts_copy)

    return enriched



def render_model_selection():
    """Render model selection interface."""
    import os
    
    # Get all enabled models (both classification and CTC)
    available_models = []
    for model_name, config in MODEL_CONFIG.items():
        if config['enabled'] and os.path.exists(config['checkpoint_path']):
            available_models.append((model_name, config['display_name']))
    
    if not available_models:
        st.error("No models are available for validation.")
        return None
    
    # Model selection with font styling
    st.markdown("**Choose Model to Validate**")
    model_options = [f"{name} ({model_type})" for model_type, name in available_models]
    default_index = 0
    for idx, (model_type, _) in enumerate(available_models):
        if model_type == 'transformer_ctc':
            default_index = idx
            break
    selected_option = st.selectbox(
        "Select model architecture",
        model_options,
        help="Select the model architecture for validation (includes both classification and CTC models)",
        key="model_selection_selectbox",
        index=default_index
    )
    
    # Check if model selection has changed and clear validation results
    if 'previous_selected_model' not in st.session_state:
        st.session_state.previous_selected_model = selected_option
    elif st.session_state.previous_selected_model != selected_option:
        # Model selection changed, clear validation results
        if 'validation_results' in st.session_state:
            del st.session_state.validation_results
        st.session_state.previous_selected_model = selected_option
    
    # Extract model type from selection
    selected_model_type = None
    for model_type, name in available_models:
        if f"{name} ({model_type})" == selected_option:
            selected_model_type = model_type
            break
    
    return selected_model_type


def render_dataset_upload():
    """Render dataset upload interface for classification models."""
    
    # NPZ folder selection
    st.markdown("**Validation NPZ Folder**")
    npz_folder_path = st.text_input(
        "Enter path to folder containing NPZ files (default: data\\processed\\fsl_val)",
        placeholder="e.g., data\\processed\\fsl_val",
        help="Path to directory containing NPZ files for validation"
    )
    
    # Labels CSV upload
    st.markdown("**Labels CSV File**")
    labels_csv = st.file_uploader(
        "Upload labels CSV file",
        type=["csv"],
        help="CSV file with columns: file, gloss, cat, occluded"
    )
    
    return npz_folder_path, labels_csv


def render_ctc_dataset_upload():
    """Render dataset upload interface for CTC models."""
    
    # NPZ folder selection
    st.markdown("**Continuous Sequences Folder**")
    npz_folder_path = st.text_input(
        "Enter path to folder containing continuous sequence NPZ files (default: data\\processed\\continuous_sequences)",
        placeholder="e.g., data\\processed\\continuous_sequences",
        help="Path to directory containing continuous sequence NPZ files"
    )
    
    # Ground truth folder
    st.markdown("**Ground Truth Folder**")
    gt_folder_path = st.text_input(
        "Enter path to folder containing ground truth JSON files (default: data\\processed\\continuous_sequences)",
        placeholder="e.g., data\\processed\\continuous_sequences (same as NPZ folder if JSON files are there)",
        help="Path to directory containing *_gt.json files"
    )
    
    return npz_folder_path, gt_folder_path


def render_validation_configuration():
    """Render validation configuration options."""
    # Auto-detect device and use maximum batch size
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Device: {device.upper()}")
    with col2:
        st.text("Batch Size: 64 (max)")
    return batch_size, device


def render_ctc_validation_configuration():
    """Render CTC validation configuration options."""
    # Header removed per request
    
    # Always use greedy decoding; hide decode method and beam width controls
    decode_method = 'greedy'
    beam_width = 1
    
    # Device selection (auto-detect)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.text(f"Device: {device.upper()}")
    
    batch_size = 1  # CTC processes sequences one at a time
    
    return batch_size, device, decode_method, beam_width


def render_validation_progress(progress_bar, status_text):
    """Render validation progress."""
    if progress_bar:
        progress_bar.progress(progress_bar.value)
    if status_text:
        status_text.text(f"Processing batch {progress_bar.value if progress_bar else 0}...")


def render_validation_results(results: Dict[str, Any]):
    """Render comprehensive validation results."""
    if not results:
        st.error("No validation results to display.")
        return
    
    # Model info (kept for potential future use)
    # model_info = results['model_info']
    # dataset_info = results['dataset_info']
    
    # Summary metrics section removed to reduce redundancy with VALIDATION SUMMARY
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overall Performance", "🎯 Per-Class Analysis", "🔍 Confusion Matrices", "📈 Occlusion Analysis", "📋 Detailed Predictions"])
    
    with tab1:
        render_overall_performance(results)
    
    with tab2:
        render_per_class_analysis(results)
    
    with tab3:
        render_confusion_matrices(results)
    
    with tab4:
        render_occlusion_analysis(results)
    
    with tab5:
        render_detailed_predictions(results)


def render_summary_metrics(results: Dict[str, Any]):
    """Render summary metrics cards."""
    overall = results['overall_results']
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Gloss Accuracy",
            f"{overall.get('gloss_accuracy', 0.0):.3f}" if 'error' not in overall else "N/A",
            help="Overall gloss classification accuracy"
        )
    
    with col2:
        st.metric(
            "Category Accuracy", 
            f"{overall.get('category_accuracy', 0.0):.3f}" if 'error' not in overall else "N/A",
            help="Overall category classification accuracy"
        )
    
    with col3:
        st.metric(
            "Gloss F1-Score",
            f"{overall.get('gloss_f1_score', 0.0):.3f}" if 'error' not in overall else "N/A",
            help="Gloss classification F1-score"
        )
    
    with col4:
        st.metric(
            "Category F1-Score",
            f"{overall.get('category_f1_score', 0.0):.3f}" if 'error' not in overall else "N/A",
            help="Category classification F1-score"
        )
    
    with col5:
        st.metric(
            "Total Samples",
            f"{overall['num_samples']:,}",
            help="Total number of validation samples"
        )
    
    with col6:
        dataset_info = results['dataset_info']
        occluded_pct = (dataset_info['occluded_samples'] / dataset_info['total_samples']) * 100
        st.metric(
            "Occluded Samples",
            f"{dataset_info['occluded_samples']:,} ({occluded_pct:.1f}%)",
            help="Number and percentage of occluded samples"
        )


def render_overall_performance(results: Dict[str, Any]):
    """Render overall performance analysis."""
    overall = results['overall_results']
    
    # Performance metrics comparison
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Gloss': [
            overall.get('gloss_precision', 0.0) if 'error' not in overall else 0.0,
            overall.get('gloss_recall', 0.0) if 'error' not in overall else 0.0,
            overall.get('gloss_f1_score', 0.0) if 'error' not in overall else 0.0
        ],
        'Category': [
            overall.get('category_precision', 0.0) if 'error' not in overall else 0.0,
            overall.get('category_recall', 0.0) if 'error' not in overall else 0.0,
            overall.get('category_f1_score', 0.0) if 'error' not in overall else 0.0
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Performance table
    st.markdown("#### Standard Metrics")
    st.dataframe(df, width='stretch')
    
    # Create comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Gloss',
        x=df['Metric'],
        y=df['Gloss'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Category',
        x=df['Metric'],
        y=df['Category'],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top-k Accuracy metrics
    if 'gloss_top1_accuracy' in overall:
        st.markdown("#### Top-k Accuracy")
        topk_col1, topk_col2 = st.columns(2)
        
        with topk_col1:
            st.markdown("**Gloss Recognition**")
            topk_gloss_data = {
                'Top-k': ['Top-1', 'Top-5', 'Top-10'],
                'Accuracy': [
                    overall.get('gloss_top1_accuracy', 0.0) if 'error' not in overall else 0.0,
                    overall.get('gloss_top5_accuracy', 0.0) if 'error' not in overall else 0.0,
                    overall.get('gloss_top10_accuracy', 0.0) if 'error' not in overall else 0.0,
                ]
            }
            topk_gloss_df = pd.DataFrame(topk_gloss_data)
            st.dataframe(topk_gloss_df, width='stretch', hide_index=True)
        
        with topk_col2:
            st.markdown("**Category Classification**")
            topk_cat_data = {
                'Top-k': ['Top-1', 'Top-5'],
                'Accuracy': [
                    overall.get('category_top1_accuracy', 0.0) if 'error' not in overall else 0.0,
                    overall.get('category_top5_accuracy', 0.0) if 'error' not in overall else 0.0,
                ]
            }
            topk_cat_df = pd.DataFrame(topk_cat_data)
            st.dataframe(topk_cat_df, width='stretch', hide_index=True)


def render_per_class_analysis(results: Dict[str, Any]):
    """Render per-class performance analysis."""
    per_class = results['per_class_results']
    
    # Load label mappings (same approach as Detailed Predictions)
    try:
        from data.labels.label_mapping import load_label_mappings
        gloss_mapping, category_mapping = load_label_mappings()
    except Exception as e:
        st.warning(f"Could not load label mappings: {e}. Showing numeric IDs only.")
        gloss_mapping, category_mapping = {}, {}
    
    # Extract per-class data for gloss
    gloss_per_class = per_class['gloss_per_class']
    cat_per_class = per_class['category_per_class']
    
    # Create gloss per-class dataframe
    gloss_data = []
    for class_id, metrics in gloss_per_class.items():
        if class_id.isdigit():
            # Handle both old format (support) and new format (occurrences)
            occurrences = metrics.get('occurrences', metrics.get('support', 0))
            
            # Convert numeric class ID to actual label (like Detailed Predictions)
            gloss_label = gloss_mapping.get(int(class_id), f"Unknown ({class_id})")
            class_display = f"{gloss_label} ({class_id})"
            
            gloss_data.append({
                'Class': class_display,  # Use actual label with ID
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Occurrences': occurrences  # Support both old and new format
            })
    
    gloss_df = pd.DataFrame(gloss_data).sort_values('F1-Score', ascending=False)
    
    # Create category per-class dataframe
    cat_data = []
    for class_id, metrics in cat_per_class.items():
        if class_id.isdigit():
            # Handle both old format (support) and new format (occurrences)
            occurrences = metrics.get('occurrences', metrics.get('support', 0))
            
            # Convert numeric class ID to actual label (like Detailed Predictions)
            cat_label = category_mapping.get(int(class_id), f"Unknown ({class_id})")
            class_display = f"{cat_label} ({class_id})"
            
            cat_data.append({
                'Class': class_display,  # Use actual label with ID
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Occurrences': occurrences  # Support both old and new format
            })
    
    cat_df = pd.DataFrame(cat_data).sort_values('F1-Score', ascending=False)
    
    # Display top performing classes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Gloss Classes (by F1-Score)")
        st.dataframe(gloss_df, width='stretch', height=400)
    
    with col2:
        st.markdown("#### Top Category Classes (by F1-Score)")
        st.dataframe(cat_df, width='stretch', height=400)
    
    # Performance distribution
    st.markdown("#### Performance Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gloss F1-score distribution
    ax1.hist(gloss_df['F1-Score'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('Gloss F1-Score Distribution')
    ax1.set_xlabel('F1-Score')
    ax1.set_ylabel('Number of Classes')
    ax1.axvline(gloss_df['F1-Score'].mean(), color='red', linestyle='--', label=f'Mean: {gloss_df["F1-Score"].mean():.3f}')
    ax1.legend()
    
    # Category F1-score distribution
    ax2.hist(cat_df['F1-Score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('Category F1-Score Distribution')
    ax2.set_xlabel('F1-Score')
    ax2.set_ylabel('Number of Classes')
    ax2.axvline(cat_df['F1-Score'].mean(), color='red', linestyle='--', label=f'Mean: {cat_df["F1-Score"].mean():.3f}')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)


def render_confusion_matrices(results: Dict[str, Any]):
    """Render confusion matrices."""
    confusion_matrices = results['confusion_matrices']
    
    gloss_cm = np.array(confusion_matrices['gloss_confusion_matrix'])
    cat_cm = np.array(confusion_matrices['category_confusion_matrix'])
    
    # Confusion matrix statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Gloss Confusion Matrix Statistics")
        gloss_diag_acc = np.trace(gloss_cm) / np.sum(gloss_cm)
        st.metric("Diagonal Accuracy", f"{gloss_diag_acc:.4f}")
        st.metric("Matrix Shape", f"{gloss_cm.shape[0]} × {gloss_cm.shape[1]}")
    
    with col2:
        st.markdown("#### Category Confusion Matrix Statistics")
        cat_diag_acc = np.trace(cat_cm) / np.sum(cat_cm)
        st.metric("Diagonal Accuracy", f"{cat_diag_acc:.4f}")
        st.metric("Matrix Shape", f"{cat_cm.shape[0]} × {cat_cm.shape[1]}")
    
    # Create confusion matrix plots (moved below statistics)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Gloss confusion matrix
    sns.heatmap(gloss_cm, annot=False, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Gloss Recognition Confusion Matrix')
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('True Class')
    
    # Category confusion matrix
    sns.heatmap(cat_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Category Classification Confusion Matrix')
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('True Class')
    
    plt.tight_layout()
    st.pyplot(fig)


def render_occlusion_analysis(results: Dict[str, Any]):
    """Render occlusion impact analysis."""
    occluded = results['occluded_results']
    non_occluded = results['non_occluded_results']
    
    # Occlusion comparison metrics (no accuracy)
    comparison_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Occluded': [
            occluded.get('gloss_precision', 0.0) if 'error' not in occluded else 0.0,
            occluded.get('gloss_recall', 0.0) if 'error' not in occluded else 0.0,
            occluded.get('gloss_f1_score', 0.0) if 'error' not in occluded else 0.0
        ],
        'Non-Occluded': [
            non_occluded.get('gloss_precision', 0.0) if 'error' not in non_occluded else 0.0,
            non_occluded.get('gloss_recall', 0.0) if 'error' not in non_occluded else 0.0,
            non_occluded.get('gloss_f1_score', 0.0) if 'error' not in non_occluded else 0.0
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Performance analysis table
    st.markdown("#### Performance Analysis")

    metrics_index = ['Precision', 'Recall', 'F1-Score']

    # Safely extract metrics
    def safe_get(d: Dict[str, Any], key: str) -> float:
        return d.get(key, 0.0) if 'error' not in d else 0.0

    table_values = [
        [
            safe_get(non_occluded, 'gloss_precision'),
            safe_get(non_occluded, 'category_precision'),
            safe_get(occluded, 'gloss_precision'),
            safe_get(occluded, 'category_precision'),
        ],
        [
            safe_get(non_occluded, 'gloss_recall'),
            safe_get(non_occluded, 'category_recall'),
            safe_get(occluded, 'gloss_recall'),
            safe_get(occluded, 'category_recall'),
        ],
        [
            safe_get(non_occluded, 'gloss_f1_score'),
            safe_get(non_occluded, 'category_f1_score'),
            safe_get(occluded, 'gloss_f1_score'),
            safe_get(occluded, 'category_f1_score'),
        ],
    ]

    columns = pd.MultiIndex.from_tuples([
        ('Without Occlusion', 'Gloss Recognition'),
        ('Without Occlusion', 'Category Classification'),
        ('With Occlusion', 'Gloss Recognition'),
        ('With Occlusion', 'Category Classification'),
    ])

    perf_df = pd.DataFrame(table_values, index=metrics_index, columns=columns)
    st.dataframe(perf_df, width='stretch')

    # Create comparison chart (moved below table)
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Occluded',
        x=df['Metric'],
        y=df['Occluded'],
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        name='Non-Occluded',
        x=df['Metric'],
        y=df['Non-Occluded'],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title="Occlusion Impact on Performance",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    


def render_validation_summary(results: Dict[str, Any]):
    """Render validation summary with key insights."""
    st.markdown("---")
    st.markdown("<div class='main-section-header'>VALIDATION SUMMARY</div>", unsafe_allow_html=True)
    
    model_info = results['model_info']
    overall = results['overall_results']
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", model_info['model_type'].upper())
    
    with col2:
        st.metric("Validation Time", model_info['timestamp'])
    
    with col3:
        st.metric("Total Samples", f"{overall.get('num_samples', 0):,}")
    


def render_download_results(results: Dict[str, Any]):
    """Render download options for validation results."""
    st.markdown("### Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download as JSON
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download Results (JSON)",
            data=json_str,
            file_name=f"validation_results_{results['model_info']['model_type']}_{results['model_info']['timestamp'].replace(':', '-')}.json",
            mime="application/json",
            key="download_validation_results_json"
        )
    
    with col2:
        # Download summary as CSV
        summary_data = {
            'Metric': ['Gloss Accuracy', 'Category Accuracy', 'Gloss F1-Score', 'Category F1-Score'],
            'Value': [
                results['overall_results'].get('gloss_accuracy', 0.0) if 'error' not in results['overall_results'] else 0.0,
                results['overall_results'].get('category_accuracy', 0.0) if 'error' not in results['overall_results'] else 0.0,
                results['overall_results'].get('gloss_f1_score', 0.0) if 'error' not in results['overall_results'] else 0.0,
                results['overall_results'].get('category_f1_score', 0.0) if 'error' not in results['overall_results'] else 0.0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_str = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Summary (CSV)",
            data=csv_str,
            file_name=f"validation_summary_{results['model_info']['model_type']}_{results['model_info']['timestamp'].replace(':', '-')}.csv",
            mime="text/csv",
            key="download_validation_summary_csv"
        )


def render_detailed_predictions(results: Dict[str, Any]):
    """Render detailed predictions table with file names, predictions, and correctness."""
    if 'detailed_predictions' not in results:
        st.error("Detailed predictions data not available.")
        return
    
    predictions = results['detailed_predictions']
    
    # Load label mappings
    try:
        from data.labels.label_mapping import load_label_mappings
        gloss_mapping, category_mapping = load_label_mappings()
    except Exception as e:
        st.warning(f"Could not load label mappings: {e}. Showing numeric IDs only.")
        gloss_mapping, category_mapping = {}, {}
    
    # Create DataFrame with the requested columns
    table_data = []
    for pred in predictions:
        # Determine correctness status
        gloss_correct = "Correct" if pred['gloss_pred'] == pred['gloss_gt'] else "Wrong"
        cat_correct = "Correct" if pred['cat_pred'] == pred['cat_gt'] else "Wrong"
        
        # Get human-readable labels
        pred_gloss_label = gloss_mapping.get(pred['gloss_pred'], f"Unknown ({pred['gloss_pred']})")
        actual_gloss_label = gloss_mapping.get(pred['gloss_gt'], f"Unknown ({pred['gloss_gt']})")
        pred_cat_label = category_mapping.get(pred['cat_pred'], f"Unknown ({pred['cat_pred']})")
        actual_cat_label = category_mapping.get(pred['cat_gt'], f"Unknown ({pred['cat_gt']})")
        
        # Determine occlusion status (text labels for CSV compatibility)
        occlusion_status = "Occluded" if pred.get('occluded', 0) == 1 else "Not Occluded"
        
        table_data.append({
            'File Name': pred['file'],
            'Predicted Gloss': f"{pred_gloss_label} ({pred['gloss_pred']})",
            'Actual Gloss': f"{actual_gloss_label} ({pred['gloss_gt']})",
            'Gloss Status': gloss_correct,
            'Gloss Confidence': f"{pred.get('gloss_prob', 0):.4f}",
            'Predicted Category': f"{pred_cat_label} ({pred['cat_pred']})",
            'Actual Category': f"{actual_cat_label} ({pred['cat_gt']})",
            'Category Status': cat_correct,
            'Category Confidence': f"{pred.get('cat_prob', 0):.4f}",
            'Occlusion Status': occlusion_status
        })
    
    df = pd.DataFrame(table_data)
    
    # Add filtering options
    st.markdown("#### Detailed Predictions")
    
    # Filter options - 3 columns for individual filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gloss_filter = st.selectbox(
            "Filter by Gloss", 
            ["All", "Correct Only", "Incorrect Only"],
            key="gloss_filter"
        )
    
    with col2:
        category_filter = st.selectbox(
            "Filter by Category", 
            ["All", "Correct Only", "Incorrect Only"],
            key="category_filter"
        )
    
    with col3:
        occlusion_filter = st.selectbox(
            "Filter by Occlusion", 
            ["All", "Occluded Only", "Non-Occluded Only"],
            key="occlusion_filter"
        )
    
    # Apply gloss filter
    if gloss_filter == "Correct Only":
        df = df[df['Gloss Status'] == "Correct"]
    elif gloss_filter == "Incorrect Only":
        df = df[df['Gloss Status'] == "Wrong"]
    
    # Apply category filter
    if category_filter == "Correct Only":
        df = df[df['Category Status'] == "Correct"]
    elif category_filter == "Incorrect Only":
        df = df[df['Category Status'] == "Wrong"]
    
    # Apply occlusion filter
    if occlusion_filter == "Occluded Only":
        df = df[df['Occlusion Status'] == "Occluded"]
    elif occlusion_filter == "Non-Occluded Only":
        df = df[df['Occlusion Status'] == "Not Occluded"]
    
    # Display statistics based on filtered data
    filtered_samples = len(df)
    correct_gloss_filtered = len(df[df['Gloss Status'] == "Correct"])
    correct_cat_filtered = len(df[df['Category Status'] == "Correct"])
    both_correct_filtered = len(df[(df['Gloss Status'] == "Correct") & (df['Category Status'] == "Correct")])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Samples", filtered_samples)
    with col2:
        st.metric("Correct Gloss", f"{correct_gloss_filtered} ({correct_gloss_filtered/filtered_samples*100:.1f}%)" if filtered_samples > 0 else "0 (0.0%)")
    with col3:
        st.metric("Correct Category", f"{correct_cat_filtered} ({correct_cat_filtered/filtered_samples*100:.1f}%)" if filtered_samples > 0 else "0 (0.0%)")
    with col4:
        st.metric("Both Correct", f"{both_correct_filtered} ({both_correct_filtered/filtered_samples*100:.1f}%)" if filtered_samples > 0 else "0 (0.0%)")
    
    # Display the table
    total_samples = len(table_data)
    st.markdown(f"**Showing {len(df)} of {total_samples} predictions**")
    
    st.dataframe(
        df,
        width='stretch',
        height=400
    )
    
    # Per-class TP/FP/TN/FN aggregated table with toggle
    st.markdown("#### TP/FP/TN/FN by Class")
    class_task_choice = st.selectbox(
        "Select task for per-class confusion counts",
        ["Gloss Recognition", "Category Classification"],
        key="class_confusion_toggle"
    )

    # Collect class IDs present in predictions
    if class_task_choice == "Gloss Recognition":
        class_ids = sorted(set([int(p['gloss_gt']) for p in predictions] + [int(p['gloss_pred']) for p in predictions]))
    else:
        class_ids = sorted(set([int(p['cat_gt']) for p in predictions] + [int(p['cat_pred']) for p in predictions]))

    total_n = len(predictions)
    per_class_rows = []

    for cid in class_ids:
        if class_task_choice == "Gloss Recognition":
            tp = sum(1 for p in predictions if p['gloss_pred'] == cid and p['gloss_gt'] == cid)
            fp = sum(1 for p in predictions if p['gloss_pred'] == cid and p['gloss_gt'] != cid)
            fn = sum(1 for p in predictions if p['gloss_pred'] != cid and p['gloss_gt'] == cid)
            label = gloss_mapping.get(cid, f"Unknown ({cid})")
        else:
            tp = sum(1 for p in predictions if p['cat_pred'] == cid and p['cat_gt'] == cid)
            fp = sum(1 for p in predictions if p['cat_pred'] == cid and p['cat_gt'] != cid)
            fn = sum(1 for p in predictions if p['cat_pred'] != cid and p['cat_gt'] == cid)
            label = category_mapping.get(cid, f"Unknown ({cid})")

        tn = total_n - (tp + fp + fn)

        per_class_rows.append({
            'Class': f"{label} ({cid})",
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
        })

    per_class_df = pd.DataFrame(per_class_rows)
    st.dataframe(
        per_class_df,
        width='stretch',
        height=400
    )

    # Note: Download button removed; users can download directly from the table UI.


def render_ctc_validation_results(results: Dict[str, Any]):
    """Render CTC validation results with detection metrics (TP/FP/FN, Precision/Recall/F1)."""
    
    
    summary = results.get('summary', {})
    predictions = results.get('predictions', [])
    overall_metrics = summary.get('overall_metrics', {})
    model_info = results.get('model_info', {})
    
    # Validation Summary (CTC)
    st.markdown("---")
    st.markdown("<div class='main-section-header'>VALIDATION SUMMARY</div>", unsafe_allow_html=True)
    summary_cols = st.columns(5)
    with summary_cols[0]:
        st.metric("Model", str(model_info.get('model_type', '')).upper())
    with summary_cols[1]:
        st.metric("Validation Time", model_info.get('timestamp', 'N/A'))
    with summary_cols[2]:
        st.metric("Total Sequences", summary.get('total_sequences', 0))
    with summary_cols[3]:
        total_occ = 0
        total_gt_signs = 0
        for p in predictions:
            occ = p.get('ground_truth_occluded')
            if occ:
                total_occ += int(np.sum(np.array(occ)))
                total_gt_signs += len(occ)
            else:
                if 'ground_truth_sequence' in p:
                    total_gt_signs += len(p.get('ground_truth_sequence', []))
        st.metric("Occluded Signs", total_occ)
    with summary_cols[4]:
        total_non_occ = max(total_gt_signs - total_occ, 0)
        st.metric("Non-Occluded Signs", total_non_occ)
    
    if not overall_metrics:
        st.warning("No detection metrics available. Please ensure ground truth timestamps are provided.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overall Performance",
        "🎯 Per-Class Analysis",
        "🔍 Confusion Matrices",
        "📈 Occlusion Analysis",
        "📋 Detailed Predictions"
    ])

    with tab1:
        # Overall Performance (no accuracy)
        st.markdown("#### Detailed Metrics")
        gloss_mean_p = overall_metrics.get('mean_precision', overall_metrics.get('overall_precision', 0.0))
        gloss_mean_r = overall_metrics.get('mean_recall', overall_metrics.get('overall_recall', 0.0))
        gloss_mean_f = overall_metrics.get('mean_f1_score', overall_metrics.get('overall_f1_score', 0.0))

        has_cat = summary.get('has_category_predictions', False)
        cat_mean_p = overall_metrics.get('category_mean_precision', overall_metrics.get('category_overall_precision', 0.0)) if has_cat else 0.0
        cat_mean_r = overall_metrics.get('category_mean_recall', overall_metrics.get('category_overall_recall', 0.0)) if has_cat else 0.0
        cat_mean_f = overall_metrics.get('category_mean_f1_score', overall_metrics.get('category_overall_f1_score', 0.0)) if has_cat else 0.0

        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Gloss': [gloss_mean_p, gloss_mean_r, gloss_mean_f],
        }
        if has_cat:
            metrics_data['Category'] = [cat_mean_p, cat_mean_r, cat_mean_f]
        else:
            metrics_data['Category'] = ['—', '—', '—']

        df = pd.DataFrame(metrics_data)
        st.dataframe(df, width='stretch', hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Gloss',
            x=df['Metric'],
            y=df['Gloss'],
            marker_color='lightblue'
        ))
        if has_cat:
            fig.add_trace(go.Bar(
                name='Category',
                x=df['Metric'],
                y=df['Category'],
                marker_color='lightgreen'
            ))
        fig.update_layout(
            title="Performance Metrics Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)


    with tab2:
        # Per-Class Analysis for CTC using confusion matrices
        # Build gloss confusion matrix
        all_gt_gloss = []
        all_pred_gloss = []
        for p in predictions:
            if 'ground_truth_sequence' in p and 'predicted_sequence' in p:
                gt_seq = p.get('ground_truth_sequence', [])
                pr_seq = p.get('predicted_sequence', [])
                for gt_id, pr_id in zip(gt_seq, pr_seq):
                    all_gt_gloss.append(int(gt_id))
                    all_pred_gloss.append(int(pr_id))

        if all_gt_gloss and all_pred_gloss:
            num_gloss_classes = max(max(all_gt_gloss, default=0), max(all_pred_gloss, default=0)) + 1
            gloss_cm = np.zeros((num_gloss_classes, num_gloss_classes), dtype=int)
            for gt, pr in zip(all_gt_gloss, all_pred_gloss):
                if 0 <= gt < num_gloss_classes and 0 <= pr < num_gloss_classes:
                    gloss_cm[gt, pr] += 1
        else:
            gloss_cm = np.zeros((1, 1), dtype=int)

        # Build category confusion matrix if available
        all_gt_cat = []
        all_pred_cat = []
        for p in predictions:
            gt_cats = p.get('ground_truth_categories')
            pr_cats = p.get('predicted_categories')
            if gt_cats and pr_cats:
                for gc, pc in zip(gt_cats, pr_cats):
                    all_gt_cat.append(int(gc))
                    all_pred_cat.append(int(pc))

        if all_gt_cat and all_pred_cat:
            num_cat_classes = max(max(all_gt_cat, default=0), max(all_pred_cat, default=0)) + 1
            cat_cm = np.zeros((num_cat_classes, num_cat_classes), dtype=int)
            for gt, pr in zip(all_gt_cat, all_pred_cat):
                if 0 <= gt < num_cat_classes and 0 <= pr < num_cat_classes:
                    cat_cm[gt, pr] += 1
        else:
            cat_cm = np.zeros((1, 1), dtype=int)

        # Load label mappings
        try:
            from data.labels.label_mapping import load_label_mappings
            gloss_mapping, category_mapping = load_label_mappings()
        except Exception:
            gloss_mapping, category_mapping = {}, {}

        # Compute per-class metrics from confusion matrices
        def per_class_from_cm(cm: np.ndarray, label_map: Dict[int, str]):
            per_rows = []
            num_classes = cm.shape[0]
            for cid in range(num_classes):
                tp = cm[cid, cid]
                fp = cm[:, cid].sum() - tp
                fn = cm[cid, :].sum() - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                occurrences = cm[cid, :].sum()
                label = label_map.get(cid, f"Unknown ({cid})")
                per_rows.append({
                    'Class': f"{label} ({cid})",
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Occurrences': int(occurrences),
                })
            return pd.DataFrame(per_rows).sort_values('F1-Score', ascending=False)

        gloss_df = per_class_from_cm(gloss_cm, gloss_mapping)
        cat_df = per_class_from_cm(cat_cm, category_mapping)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Top Gloss Classes (by F1-Score)")
            st.dataframe(gloss_df, width='stretch', height=400)
        with col_r:
            st.markdown("#### Top Category Classes (by F1-Score)")
            st.dataframe(cat_df, width='stretch', height=400)

        # Performance Distribution
        st.markdown("#### Performance Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(gloss_df['F1-Score'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('Gloss F1-Score Distribution')
        ax1.set_xlabel('F1-Score')
        ax1.set_ylabel('Number of Classes')
        if len(gloss_df) > 0:
            ax1.axvline(gloss_df['F1-Score'].mean(), color='red', linestyle='--', label=f"Mean: {gloss_df['F1-Score'].mean():.3f}")
            ax1.legend()

        ax2.hist(cat_df['F1-Score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Category F1-Score Distribution')
        ax2.set_xlabel('F1-Score')
        ax2.set_ylabel('Number of Classes')
        if len(cat_df) > 0:
            ax2.axvline(cat_df['F1-Score'].mean(), color='red', linestyle='--', label=f"Mean: {cat_df['F1-Score'].mean():.3f}")
            ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        # Confusion Matrices for CTC (aggregate across sequences)
        st.markdown("#### Confusion Matrices")

        # Build gloss confusion matrix
        all_gt_gloss = []
        all_pred_gloss = []
        for p in predictions:
            if 'ground_truth_sequence' in p and 'predicted_sequence' in p:
                gt_seq = p.get('ground_truth_sequence', [])
                pr_seq = p.get('predicted_sequence', [])
                for gt_id, pr_id in zip(gt_seq, pr_seq):
                    all_gt_gloss.append(int(gt_id))
                    all_pred_gloss.append(int(pr_id))

        if all_gt_gloss and all_pred_gloss:
            num_gloss_classes = max(max(all_gt_gloss, default=0), max(all_pred_gloss, default=0)) + 1
            gloss_cm = np.zeros((num_gloss_classes, num_gloss_classes), dtype=int)
            for gt, pr in zip(all_gt_gloss, all_pred_gloss):
                if 0 <= gt < num_gloss_classes and 0 <= pr < num_gloss_classes:
                    gloss_cm[gt, pr] += 1
        else:
            gloss_cm = np.zeros((1, 1), dtype=int)

        # Build category confusion matrix if available
        all_gt_cat = []
        all_pred_cat = []
        for p in predictions:
            gt_cats = p.get('ground_truth_categories')
            pr_cats = p.get('predicted_categories')
            if gt_cats and pr_cats:
                for gc, pc in zip(gt_cats, pr_cats):
                    all_gt_cat.append(int(gc))
                    all_pred_cat.append(int(pc))

        if all_gt_cat and all_pred_cat:
            num_cat_classes = max(max(all_gt_cat, default=0), max(all_pred_cat, default=0)) + 1
            cat_cm = np.zeros((num_cat_classes, num_cat_classes), dtype=int)
            for gt, pr in zip(all_gt_cat, all_pred_cat):
                if 0 <= gt < num_cat_classes and 0 <= pr < num_cat_classes:
                    cat_cm[gt, pr] += 1
        else:
            cat_cm = np.zeros((1, 1), dtype=int)

        # Statistics and plots in two columns
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Gloss Confusion Matrix Statistics")
            gloss_total = np.sum(gloss_cm)
            gloss_diag_acc = (np.trace(gloss_cm) / gloss_total) if gloss_total > 0 else 0.0
            st.metric("Diagonal Accuracy", f"{gloss_diag_acc:.4f}")
            st.metric("Matrix Shape", f"{gloss_cm.shape[0]} × {gloss_cm.shape[1]}")

            fig_g, ax_g = plt.subplots(1, 1, figsize=(7, 6))
            sns.heatmap(gloss_cm, annot=False, fmt='d', cmap='Blues', ax=ax_g)
            ax_g.set_title('Gloss Recognition Confusion Matrix')
            ax_g.set_xlabel('Predicted Class')
            ax_g.set_ylabel('True Class')
            st.pyplot(fig_g)

        with col_right:
            st.markdown("#### Category Confusion Matrix Statistics")
            cat_total = np.sum(cat_cm)
            cat_diag_acc = (np.trace(cat_cm) / cat_total) if cat_total > 0 else 0.0
            st.metric("Diagonal Accuracy", f"{cat_diag_acc:.4f}")
            st.metric("Matrix Shape", f"{cat_cm.shape[0]} × {cat_cm.shape[1]}")

            fig_c, ax_c = plt.subplots(1, 1, figsize=(7, 6))
            sns.heatmap(cat_cm, annot=True, fmt='d', cmap='Greens', ax=ax_c)
            ax_c.set_title('Category Classification Confusion Matrix')
            ax_c.set_xlabel('Predicted Class')
            ax_c.set_ylabel('True Class')
            st.pyplot(fig_c)

    with tab4:
        # Occlusion Analysis for CTC
        occlusion = overall_metrics.get('occlusion') if overall_metrics else None
        without = (occlusion or {}).get('without_occlusion', {})
        with_occ = (occlusion or {}).get('with_occlusion', {})

        # Performance Analysis (no accuracy)
        st.markdown("#### Performance Analysis")
        metrics_index = ['Precision', 'Recall', 'F1-Score']

        def safe(d, k):
            return d.get(k, 0.0)

        # Build multi-index table with Gloss and Category under Occluded/Non-Occluded
        occ_cat = overall_metrics.get('occlusion_category') if overall_metrics else None
        cat_without = (occ_cat or {}).get('without_occlusion', {})
        cat_with = (occ_cat or {}).get('with_occlusion', {})

        # Columns multiindex
        columns = pd.MultiIndex.from_tuples([
            ('Occluded', 'Gloss Recognition'),
            ('Occluded', 'Category Classification'),
            ('Non-Occluded', 'Gloss Recognition'),
            ('Non-Occluded', 'Category Classification'),
        ])

        table_values = [
            [safe(with_occ, 'precision'), safe(cat_with, 'precision'), safe(without, 'precision'), safe(cat_without, 'precision')],
            [safe(with_occ, 'recall'), safe(cat_with, 'recall'), safe(without, 'recall'), safe(cat_without, 'recall')],
            [safe(with_occ, 'f1_score'), safe(cat_with, 'f1_score'), safe(without, 'f1_score'), safe(cat_without, 'f1_score')],
        ]

        perf_df = pd.DataFrame(table_values, index=metrics_index, columns=columns)
        st.dataframe(perf_df, width='stretch')

        # Bar chart: Occlusion Impact (no accuracy)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Occluded',
            x=metrics_index,
            y=[safe(with_occ, 'precision'), safe(with_occ, 'recall'), safe(with_occ, 'f1_score')],
            marker_color='lightcoral'
        ))
        fig.add_trace(go.Bar(
            name='Non-Occluded',
            x=metrics_index,
            y=[safe(without, 'precision'), safe(without, 'recall'), safe(without, 'f1_score')],
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Occlusion Impact on Performance",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        # Detailed predictions
        st.markdown("#### Detailed Predictions")
        has_categories = summary.get('has_category_predictions', False)
        # Gloss mean metrics + optional category mean metrics (3)
        num_cols = 3 + (3 if has_categories else 0)
        cols = st.columns(num_cols)

        col_idx = 0
        with cols[col_idx]:
            mean_precision = overall_metrics.get('mean_precision', overall_metrics.get('overall_precision', 0))
            st.metric("Gloss Mean Precision", f"{mean_precision*100:.2f}%")
        col_idx += 1

        with cols[col_idx]:
            mean_recall = overall_metrics.get('mean_recall', overall_metrics.get('overall_recall', 0))
            st.metric("Gloss Mean Recall", f"{mean_recall*100:.2f}%")
        col_idx += 1

        with cols[col_idx]:
            mean_f1 = overall_metrics.get('mean_f1_score', overall_metrics.get('overall_f1_score', 0))
            st.metric("Gloss Mean F1-Score", f"{mean_f1*100:.2f}%")
        col_idx += 1

        if has_categories:
            with cols[col_idx]:
                cat_mean_prec = overall_metrics.get('category_mean_precision', overall_metrics.get('category_overall_precision', 0))
                st.metric("Category Mean Precision", f"{cat_mean_prec*100:.2f}%")
            col_idx += 1
            with cols[col_idx]:
                cat_mean_rec = overall_metrics.get('category_mean_recall', overall_metrics.get('category_overall_recall', 0))
                st.metric("Category Mean Recall", f"{cat_mean_rec*100:.2f}%")
            col_idx += 1
            with cols[col_idx]:
                cat_mean_f1 = overall_metrics.get('category_mean_f1_score', overall_metrics.get('category_overall_f1_score', 0))
                st.metric("Category Mean F1-Score", f"{cat_mean_f1*100:.2f}%")

        # Integrated detection counts
        count_cols = st.columns(6)
        with count_cols[0]:
            st.metric("True Positives (TP)", overall_metrics.get('total_tp', 0))
        with count_cols[1]:
            st.metric("False Positives (FP)", overall_metrics.get('total_fp', 0))
        with count_cols[2]:
            st.metric("False Negatives (FN)", overall_metrics.get('total_fn', 0))
        with count_cols[3]:
            st.metric("True Negatives (TN)", overall_metrics.get('total_tn', 0))
        with count_cols[4]:
            st.metric("Total GT Instances", overall_metrics.get('total_gt_instances', 0))
        with count_cols[5]:
            st.empty()

        # Detailed predictions table
        st.markdown("---")
        # Table

        # Load label mappings for fallback when labels missing
        try:
            from data.labels.label_mapping import load_label_mappings
            gloss_mapping, category_mapping = load_label_mappings()
        except Exception:
            gloss_mapping, category_mapping = {}, {}

        if predictions:
            # Metrics-only compact table
            pred_rows = []
            has_cat_metrics = any('category_f1_score' in p for p in predictions)

            for pred in predictions:
                if 'f1_score' in pred:
                    row = {
                        'File': pred['file_name'],
                        'GT Length': len(pred.get('ground_truth_sequence', [])),
                        'Pred Length': pred.get('num_predicted', 0),
                        'Precision': f"{pred.get('precision', 0)*100:.2f}%",
                        'Recall': f"{pred.get('recall', 0)*100:.2f}%",
                        'F1-Score': f"{pred.get('f1_score', 0)*100:.2f}%",
                        'TP': pred.get('num_tp', 0),
                        'FP': pred.get('num_fp', 0),
                        'FN': pred.get('num_fn', 0)
                    }
                    if has_cat_metrics and 'category_f1_score' in pred:
                        row['Cat F1-Score'] = f"{pred['category_f1_score']*100:.2f}%"
                        if 'category_precision' in pred:
                            row['Cat Precision'] = f"{pred['category_precision']*100:.2f}%"
                        if 'category_recall' in pred:
                            row['Cat Recall'] = f"{pred['category_recall']*100:.2f}%"
                        if 'category_num_tp' in pred:
                            row['Cat TP'] = pred['category_num_tp']
                        if 'category_num_fp' in pred:
                            row['Cat FP'] = pred['category_num_fp']
                        if 'category_num_fn' in pred:
                            row['Cat FN'] = pred['category_num_fn']
                    pred_rows.append(row)
                else:
                    row = {
                        'File': pred['file_name'],
                        'GT Length': len(pred.get('ground_truth_sequence', [])),
                        'Pred Length': pred.get('num_predicted', 0),
                        'Precision': 'N/A',
                        'Recall': 'N/A',
                        'F1-Score': 'N/A',
                        'TP': 'N/A',
                        'FP': 'N/A',
                        'FN': 'N/A'
                    }
                    if has_cat_metrics:
                        row.update({
                            'Cat Precision': 'N/A',
                            'Cat Recall': 'N/A',
                            'Cat F1-Score': 'N/A',
                            'Cat TP': 'N/A',
                            'Cat FP': 'N/A',
                            'Cat FN': 'N/A',
                        })
                    pred_rows.append(row)

            if pred_rows:
                # Build flat dataframe with prefixed column names to simulate grouping
                base_df = pd.DataFrame(pred_rows)
                # Rename metrics to include group labels
                rename_map = {
                    'Precision': 'Gloss: Precision',
                    'Recall': 'Gloss: Recall',
                    'F1-Score': 'Gloss: F1-Score',
                    'TP': 'Gloss: TP',
                    'FP': 'Gloss: FP',
                    'FN': 'Gloss: FN',
                    'Cat Precision': 'Category: Precision',
                    'Cat Recall': 'Category: Recall',
                    'Cat F1-Score': 'Category: F1-Score',
                    'Cat TP': 'Category: TP',
                    'Cat FP': 'Category: FP',
                    'Cat FN': 'Category: FN',
                }
                for k, v in rename_map.items():
                    if k in base_df.columns:
                        base_df.rename(columns={k: v}, inplace=True)

                # Add checkbox column for details (single-select behavior)
                selected_file = st.session_state.get('ctc_selected_file')
                base_df.insert(0, 'Details', base_df['File'] == selected_file)

                # Define column order
                ordered_cols = ['Details', 'File', 'GT Length', 'Pred Length',
                                'Gloss: Precision', 'Gloss: Recall', 'Gloss: F1-Score',
                                'Gloss: TP', 'Gloss: FP', 'Gloss: FN']
                cat_cols = ['Category: Precision', 'Category: Recall', 'Category: F1-Score', 'Category: TP', 'Category: FP', 'Category: FN']
                for c in cat_cols:
                    if c in base_df.columns:
                        ordered_cols.append(c)
                show_df = base_df[[c for c in ordered_cols if c in base_df.columns]].copy()

                editor_key = f"ctc_pred_table_{selected_file or 'none'}"
                edited_df = st.data_editor(
                    show_df,
                    width='stretch',
                    height=360,
                    hide_index=True,
                    column_config={
                        'Details': st.column_config.CheckboxColumn('', default=False, width="small")
                    },
                    column_order=[c for c in show_df.columns],
                    disabled=[c for c in show_df.columns if c != 'Details'],
                    key=editor_key
                )

                # Enforce single selection
                checked_files = edited_df[edited_df['Details'] == True]['File'].tolist()
                if len(checked_files) > 1:
                    # Prefer the one different from previous selection; if none, keep the first
                    prev = st.session_state.get('ctc_selected_file')
                    new_sel = next((f for f in checked_files if f != prev), checked_files[0])
                    st.session_state['ctc_selected_file'] = new_sel
                    st.rerun()
                elif len(checked_files) == 1:
                    if checked_files[0] != st.session_state.get('ctc_selected_file'):
                        st.session_state['ctc_selected_file'] = checked_files[0]
                        st.rerun()
                else:
                    if st.session_state.get('ctc_selected_file') is not None:
                        st.session_state['ctc_selected_file'] = None
                        st.rerun()

                selected_file = st.session_state.get('ctc_selected_file')
                if selected_file:
                    pred = next((p for p in predictions if p.get('file_name') == selected_file), None)
                    if pred:
                        st.markdown(
                            f"<div style=\"display:inline-block;padding:6px 10px;border-radius:6px;border:1px solid #1f77b4;color:#1f77b4;font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; font-weight:600; background:#0b1220;\">{selected_file}</div>",
                            unsafe_allow_html=True
                        )

                        predicted_sequence = pred.get('predicted_sequence', []) or []
                        ground_truth_sequence = pred.get('ground_truth_sequence', []) or []

                        pred_gloss_labels = pred.get('predicted_labels')
                        if not pred_gloss_labels and predicted_sequence:
                            pred_gloss_labels = [gloss_mapping.get(int(g), str(g)) for g in predicted_sequence]

                        gt_gloss_labels = pred.get('ground_truth_labels')
                        if not gt_gloss_labels and ground_truth_sequence:
                            gt_gloss_labels = [gloss_mapping.get(int(g), str(g)) for g in ground_truth_sequence]

                        predicted_categories = pred.get('predicted_categories')
                        ground_truth_categories = pred.get('ground_truth_categories')
                        category_confidences = pred.get('category_confidences')
                        confidence_scores = pred.get('confidence_scores')
                        ground_truth_occluded = pred.get('ground_truth_occluded')

                        case_palette = {
                            'TP': '#22c55e',
                            'FP': '#ef4444',
                            'FN': '#f97316',
                            'TN': '#64748b',
                        }
                        prediction_case_map: Dict[int, str] = {}
                        ground_truth_case_map: Dict[int, str] = {}
                        iou_threshold = pred.get('iou_threshold', 0.5)
                        matched_pairs = pred.get('matched_pairs') or []
                        len_pred = len(predicted_sequence)
                        len_gt = len(ground_truth_sequence)

                        for pair in matched_pairs:
                            pred_idx = pair.get('pred_idx')
                            gt_idx = pair.get('gt_idx')
                            if pred_idx is None or gt_idx is None:
                                continue
                            pred_idx = int(pred_idx)
                            gt_idx = int(gt_idx)
                            if not (0 <= pred_idx < len_pred and 0 <= gt_idx < len_gt):
                                continue
                            iou = float(pair.get('iou', 0.0))
                            if iou >= 0.0:  # any matched pair counts as TP
                                case = 'TP'
                            prediction_case_map[pred_idx] = case
                            ground_truth_case_map[gt_idx] = case

                        for idx in pred.get('tp_indices', []) or []:
                            idx = int(idx)
                            if 0 <= idx < len_pred:
                                prediction_case_map.setdefault(idx, 'TP')
                        for idx in pred.get('unmatched_predictions', []) or []:
                            idx = int(idx)
                            if 0 <= idx < len_pred:
                                prediction_case_map[idx] = 'FP'
                        for idx in pred.get('unmatched_ground_truth', []) or []:
                            idx = int(idx)
                            if 0 <= idx < len_gt:
                                ground_truth_case_map[idx] = 'FN'

                        low_conf_fn_count = 0
                        threshold = pred.get('confidence_threshold', 0.5)
                        for idx, conf in enumerate(confidence_scores or []):
                            if conf is None:
                                continue
                            if 0 <= idx < len_pred and conf < threshold:
                                if idx not in prediction_case_map:
                                    prediction_case_map[idx] = 'FN'
                                    low_conf_fn_count += 1

                        category_prediction_case_map, category_ground_truth_case_map = _derive_category_case_maps(
                            pred=pred,
                            predicted_categories=predicted_categories,
                            ground_truth_categories=ground_truth_categories,
                            matched_pairs=matched_pairs,
                            pred_len=len_pred,
                            gt_len=len_gt,
                        )

                        st.markdown("##### Sequence Metrics")

                        def _percent(value: Optional[float]) -> str:
                            return f"{value*100:.2f}%" if value is not None else "—"

                        def _count(value: Optional[int]) -> Any:
                            return value if value is not None else "—"

                        category_metrics_present = has_categories and any(
                            key in pred for key in (
                                'category_precision',
                                'category_recall',
                                'category_f1_score',
                                'category_num_tp',
                                'category_num_fp',
                                'category_num_fn',
                                'category_num_tn',
                            )
                        )

                        if category_metrics_present:
                            gloss_col, cat_col = st.columns(2, gap="large")
                        else:
                            gloss_col = st.container()
                            cat_col = None

                        with gloss_col:
                            st.markdown("###### Gloss Metrics")
                            gloss_metrics_cols = st.columns(3)
                            with gloss_metrics_cols[0]:
                                st.metric("Precision", _percent(pred.get('precision')))
                            with gloss_metrics_cols[1]:
                                st.metric("Recall", _percent(pred.get('recall')))
                            with gloss_metrics_cols[2]:
                                st.metric("F1-Score", _percent(pred.get('f1_score')))

                            gloss_counts_cols = st.columns(4)
                            with gloss_counts_cols[0]:
                                st.metric("True Positive", _count(pred.get('num_tp')))
                            with gloss_counts_cols[1]:
                                st.metric("False Positive", _count(pred.get('num_fp')))
                            with gloss_counts_cols[2]:
                                st.metric("True Negative", _count(pred.get('num_tn')))
                            with gloss_counts_cols[3]:
                                st.metric("False Negative", _count(pred.get('num_fn')))

                        if category_metrics_present and cat_col is not None:
                            with cat_col:
                                st.markdown("###### Category Metrics")
                                cat_metrics_cols = st.columns(3)
                                with cat_metrics_cols[0]:
                                    st.metric("Precision", _percent(pred.get('category_precision')))
                                with cat_metrics_cols[1]:
                                    st.metric("Recall", _percent(pred.get('category_recall')))
                                with cat_metrics_cols[2]:
                                    st.metric("F1-Score", _percent(pred.get('category_f1_score')))

                                cat_counts_cols = st.columns(4)
                                with cat_counts_cols[0]:
                                    st.metric("True Positive", _count(pred.get('category_num_tp')))
                                with cat_counts_cols[1]:
                                    st.metric("False Positive", _count(pred.get('category_num_fp')))
                                with cat_counts_cols[2]:
                                    st.metric("True Negative", _count(pred.get('category_num_tn')))
                                with cat_counts_cols[3]:
                                    st.metric("False Negative", _count(pred.get('category_num_fn')))

                        st.markdown("---")
                        render_case_legend()
                        render_sequence_comparison(
                            predicted_sequence=predicted_sequence,
                            predicted_labels=pred_gloss_labels or [],
                            ground_truth_sequence=ground_truth_sequence,
                            ground_truth_labels=gt_gloss_labels or [],
                            confidence_scores=confidence_scores,
                            predicted_categories=predicted_categories,
                            category_confidences=category_confidences,
                            ground_truth_categories=ground_truth_categories,
                            ground_truth_occluded=ground_truth_occluded,
                            prediction_cases=prediction_case_map,
                            case_palette=case_palette,
                            confidence_threshold=pred.get('confidence_threshold', 0.5),
                            category_prediction_cases=category_prediction_case_map,
                            category_ground_truth_cases=category_ground_truth_case_map,
                        )

                        predicted_timestamps = pred.get('predicted_timestamps')
                        ground_truth_timestamps = pred.get('ground_truth_timestamps')
                        if ground_truth_timestamps:
                            ground_truth_timestamps = _enrich_ground_truth_timestamps(
                                timestamps=ground_truth_timestamps,
                                gloss_labels=gt_gloss_labels or [],
                                gloss_sequence=ground_truth_sequence,
                                category_ids=ground_truth_categories,
                                category_labels=pred.get('ground_truth_category_labels'),
                                gloss_mapping=gloss_mapping,
                                category_mapping=category_mapping,
                            )
                        if predicted_timestamps and ground_truth_timestamps:
                            data_sources = results.get('data_sources') or {}
                            npz_folder = data_sources.get('npz_folder_path')
                            mask_array = None
                            timestamps_array = None

                            if npz_folder:
                                npz_path = Path(npz_folder) / selected_file
                                if npz_path.exists():
                                    try:
                                        npz_data = np.load(npz_path, allow_pickle=False)
                                        try:
                                            if 'mask' in npz_data:
                                                mask_array = np.array(npz_data['mask'])
                                                if mask_array.dtype != bool:
                                                    mask_array = mask_array.astype(bool)
                                            if 'timestamps_ms' in npz_data:
                                                timestamps_array = np.array(npz_data['timestamps_ms'])
                                            elif 'timestamps' in npz_data:
                                                timestamps_array = np.array(npz_data['timestamps'])
                                        finally:
                                            npz_data.close()
                                    except Exception as load_err:
                                        st.warning(f"Unable to load mask/timestamps for {selected_file}: {load_err}")

                            st.markdown("---")
                            render_temporal_alignment(
                                predicted_timestamps=predicted_timestamps,
                                ground_truth_timestamps=ground_truth_timestamps,
                                temporal_alignment_accuracy=pred.get('temporal_alignment_accuracy'),
                                mask=mask_array,
                                timestamps_ms=timestamps_array,
                                predicted_categories=predicted_categories,
                                prediction_cases=prediction_case_map,
                                ground_truth_cases=ground_truth_case_map,
                                case_palette=case_palette,
                                category_prediction_cases=category_prediction_case_map,
                                category_ground_truth_cases=category_ground_truth_case_map,
                            )

    # Download results shown for all tabs
    st.markdown("---")
    st.markdown("#### Download Results")
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Result",
        data=results_json,
        file_name="ctc_validation_results.json",
        mime="application/json",
        type="primary"
    )
