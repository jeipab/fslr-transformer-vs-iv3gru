"""Validation components for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from ..core.config import MODEL_CONFIG


def render_model_selection():
    """Render model selection interface."""
    
    available_models = []
    if MODEL_CONFIG['transformer']['enabled']:
        available_models.append(('transformer', 'SignTransformer'))
    if MODEL_CONFIG['iv3_gru']['enabled']:
        available_models.append(('iv3_gru', 'InceptionV3+GRU'))
    
    if not available_models:
        st.error("No models are available for validation.")
        return None
    
    # Model selection with font styling
    st.markdown("**Choose Model to Validate**")
    model_options = [f"{name} ({model_type})" for model_type, name in available_models]
    selected_option = st.selectbox(
        "Select model architecture",
        model_options,
        help="Select the model architecture for validation",
        key="model_selection_selectbox"
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
    """Render dataset upload interface."""
    
    # NPZ folder selection
    st.markdown("**Validation NPZ Folder**")
    npz_folder_path = st.text_input(
        "Enter path to folder containing NPZ files",
        placeholder="e.g., data/processed/prepro_09-18/validation",
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


def render_validation_configuration():
    """Render validation configuration options."""
    st.markdown("**Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=64,
            value=32,
            help="Batch size for validation (larger = faster, more memory)"
        )
    
    with col2:
        device = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=0,
            help="Device to use for validation"
        )
    
    return batch_size, device


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
    
    st.markdown("### Validation Results")
    
    # Model info
    model_info = results['model_info']
    dataset_info = results['dataset_info']
    
    # Summary metrics
    render_summary_metrics(results)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall Performance", "🎯 Per-Class Analysis", "🔍 Confusion Matrices", "📈 Occlusion Analysis"])
    
    with tab1:
        render_overall_performance(results)
    
    with tab2:
        render_per_class_analysis(results)
    
    with tab3:
        render_confusion_matrices(results)
    
    with tab4:
        render_occlusion_analysis(results)


def render_summary_metrics(results: Dict[str, Any]):
    """Render summary metrics cards."""
    overall = results['overall_results']
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Gloss Accuracy",
            f"{overall['gloss_accuracy']:.3f}",
            help="Overall gloss classification accuracy"
        )
    
    with col2:
        st.metric(
            "Category Accuracy", 
            f"{overall['category_accuracy']:.3f}",
            help="Overall category classification accuracy"
        )
    
    with col3:
        st.metric(
            "Gloss F1-Score",
            f"{overall['gloss_f1_score']:.3f}",
            help="Gloss classification F1-score"
        )
    
    with col4:
        st.metric(
            "Category F1-Score",
            f"{overall['category_f1_score']:.3f}",
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
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Gloss': [
            overall['gloss_accuracy'],
            overall['gloss_precision'],
            overall['gloss_recall'],
            overall['gloss_f1_score']
        ],
        'Category': [
            overall['category_accuracy'],
            overall['category_precision'],
            overall['category_recall'],
            overall['category_f1_score']
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
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
    
    # Performance table
    st.markdown("#### Detailed Metrics")
    st.dataframe(df, use_container_width=True)


def render_per_class_analysis(results: Dict[str, Any]):
    """Render per-class performance analysis."""
    per_class = results['per_class_results']
    
    # Extract per-class data for gloss
    gloss_per_class = per_class['gloss_per_class']
    cat_per_class = per_class['category_per_class']
    
    # Create gloss per-class dataframe
    gloss_data = []
    for class_id, metrics in gloss_per_class.items():
        if class_id.isdigit():
            gloss_data.append({
                'Class': int(class_id),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    gloss_df = pd.DataFrame(gloss_data).sort_values('F1-Score', ascending=False)
    
    # Create category per-class dataframe
    cat_data = []
    for class_id, metrics in cat_per_class.items():
        if class_id.isdigit():
            cat_data.append({
                'Class': int(class_id),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    cat_df = pd.DataFrame(cat_data).sort_values('F1-Score', ascending=False)
    
    # Display top performing classes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Gloss Classes (by F1-Score)")
        st.dataframe(gloss_df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("#### Top Category Classes (by F1-Score)")
        st.dataframe(cat_df.head(10), use_container_width=True)
    
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
    
    # Create confusion matrix plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Gloss confusion matrix
    sns.heatmap(gloss_cm, annot=False, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Gloss Classification Confusion Matrix')
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('True Class')
    
    # Category confusion matrix
    sns.heatmap(cat_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Category Classification Confusion Matrix')
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('True Class')
    
    plt.tight_layout()
    st.pyplot(fig)
    
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


def render_occlusion_analysis(results: Dict[str, Any]):
    """Render occlusion impact analysis."""
    occluded = results['occluded_results']
    non_occluded = results['non_occluded_results']
    
    # Occlusion comparison metrics
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Occluded': [
            occluded['gloss_accuracy'],
            occluded['gloss_precision'],
            occluded['gloss_recall'],
            occluded['gloss_f1_score']
        ],
        'Non-Occluded': [
            non_occluded['gloss_accuracy'],
            non_occluded['gloss_precision'],
            non_occluded['gloss_recall'],
            non_occluded['gloss_f1_score']
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison chart
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
    
    # Performance difference analysis
    st.markdown("#### Performance Difference Analysis")
    
    diff_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Difference': [
            non_occluded['gloss_accuracy'] - occluded['gloss_accuracy'],
            non_occluded['gloss_precision'] - occluded['gloss_precision'],
            non_occluded['gloss_recall'] - occluded['gloss_recall'],
            non_occluded['gloss_f1_score'] - occluded['gloss_f1_score']
        ]
    }
    
    diff_df = pd.DataFrame(diff_data)
    
    # Display dataframe with left-aligned text
    styled_df = diff_df.style.set_properties(**{'text-align': 'left'})
    st.dataframe(styled_df, use_container_width=True)
    


def render_validation_summary(results: Dict[str, Any]):
    """Render validation summary with key insights."""
    st.markdown("---")
    st.markdown("### Validation Summary")
    
    model_info = results['model_info']
    overall = results['overall_results']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", model_info['model_type'].upper())
    
    with col2:
        st.metric("Gloss Accuracy", f"{overall['gloss_accuracy']:.3f}")
    
    with col3:
        st.metric("Category Accuracy", f"{overall['category_accuracy']:.3f}")
    
    with col4:
        st.metric("Validation Time", model_info['timestamp'])
    


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
            mime="application/json"
        )
    
    with col2:
        # Download summary as CSV
        summary_data = {
            'Metric': ['Gloss Accuracy', 'Category Accuracy', 'Gloss F1-Score', 'Category F1-Score'],
            'Value': [
                results['overall_results']['gloss_accuracy'],
                results['overall_results']['category_accuracy'],
                results['overall_results']['gloss_f1_score'],
                results['overall_results']['category_f1_score']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_str = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Summary (CSV)",
            data=csv_str,
            file_name=f"validation_summary_{results['model_info']['model_type']}_{results['model_info']['timestamp'].replace(':', '-')}.csv",
            mime="text/csv"
        )
