"""CTC-specific visualization components for continuous sign language recognition."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def render_sequence_comparison(
    predicted_sequence: List[int],
    predicted_labels: List[str],
    ground_truth_sequence: Optional[List[int]] = None,
    ground_truth_labels: Optional[List[str]] = None,
    confidence_scores: Optional[List[float]] = None
):
    """
    Render side-by-side comparison of predicted vs ground truth sequences.
    
    Args:
        predicted_sequence: List of predicted gloss IDs
        predicted_labels: List of predicted gloss labels
        ground_truth_sequence: Optional list of ground truth gloss IDs
        ground_truth_labels: Optional list of ground truth gloss labels
        confidence_scores: Optional list of confidence scores per gloss
    """
    st.markdown("#### Sequence Comparison")
    
    if ground_truth_sequence:
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ground Truth**")
            render_sequence_chips(ground_truth_labels, None, color='#10b981')
        
        with col2:
            st.markdown("**Predicted**")
            render_sequence_chips(predicted_labels, confidence_scores, color='#3b82f6')
        
        # Alignment visualization
        render_alignment_chart(predicted_labels, ground_truth_labels)
    else:
        # Only prediction
        st.markdown("**Predicted Sequence**")
        render_sequence_chips(predicted_labels, confidence_scores, color='#3b82f6')


def render_sequence_chips(
    labels: List[str],
    confidence_scores: Optional[List[float]] = None,
    color: str = '#3b82f6'
):
    """
    Render sequence as colored chips/badges.
    
    Args:
        labels: List of gloss labels
        confidence_scores: Optional confidence scores
        color: Hex color for chips
    """
    if not labels:
        st.info("Empty sequence")
        return
    
    # Create HTML for chips
    chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">'
    
    for i, label in enumerate(labels):
        confidence_text = ""
        if confidence_scores and i < len(confidence_scores):
            confidence = confidence_scores[i]
            confidence_text = f' ({confidence*100:.1f}%)'
            # Adjust opacity based on confidence
            opacity = 0.5 + (confidence * 0.5)
        else:
            opacity = 1.0
        
        chip_style = f"""
            background-color: {color};
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            font-weight: 500;
            opacity: {opacity};
        """
        
        chips_html += f'<div style="{chip_style}">{i+1}. {label}{confidence_text}</div>'
    
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_alignment_chart(predicted: List[str], ground_truth: List[str]):
    """
    Render alignment visualization between predicted and ground truth.
    
    Args:
        predicted: List of predicted gloss labels
        ground_truth: List of ground truth gloss labels
    """
    st.markdown("**Alignment Visualization**")
    
    # Create alignment matrix
    max_len = max(len(predicted), len(ground_truth))
    
    # Pad sequences for visualization
    pred_padded = predicted + ['—'] * (max_len - len(predicted))
    gt_padded = ground_truth + ['—'] * (max_len - len(ground_truth))
    
    # Create DataFrame for display
    alignment_data = []
    for i in range(max_len):
        match = pred_padded[i] == gt_padded[i] and pred_padded[i] != '—'
        alignment_data.append({
            'Position': i + 1,
            'Ground Truth': gt_padded[i],
            'Predicted': pred_padded[i],
            'Match': '✓' if match else '✗'
        })
    
    df = pd.DataFrame(alignment_data)
    
    # Color-code the dataframe
    def highlight_matches(row):
        if row['Match'] == '✓':
            return ['background-color: rgba(16, 185, 129, 0.2)'] * len(row)
        else:
            return ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
    
    styled_df = df.style.apply(highlight_matches, axis=1)
    st.dataframe(styled_df, use_container_width=True)


def render_wer_metrics(
    wer: float,
    insertions: int,
    deletions: int,
    substitutions: int,
    sequence_correct: bool
):
    """
    Render WER metrics with breakdown.
    
    Args:
        wer: Word Error Rate
        insertions: Number of insertions
        deletions: Number of deletions
        substitutions: Number of substitutions
        sequence_correct: Whether sequence is completely correct
    """
    st.markdown("#### Word Error Rate (WER)")
    
    # Main WER metric
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        wer_color = "normal" if wer <= 0.2 else "off" if wer <= 0.5 else "inverse"
        st.metric("WER", f"{wer*100:.1f}%", delta=None, delta_color=wer_color)
    
    with col2:
        st.metric("Insertions", insertions)
    
    with col3:
        st.metric("Deletions", deletions)
    
    with col4:
        st.metric("Substitutions", substitutions)
    
    with col5:
        correct_emoji = "✅" if sequence_correct else "❌"
        st.metric("Perfect Match", correct_emoji)
    
    # WER visualization
    render_wer_breakdown_chart(insertions, deletions, substitutions)


def render_wer_breakdown_chart(insertions: int, deletions: int, substitutions: int):
    """
    Render WER error breakdown as bar chart.
    
    Args:
        insertions: Number of insertions
        deletions: Number of deletions
        substitutions: Number of substitutions
    """
    total_errors = insertions + deletions + substitutions
    
    if total_errors == 0:
        st.success("🎉 Perfect prediction! No errors.")
        return
    
    # Create bar chart
    error_data = {
        'Error Type': ['Insertions', 'Deletions', 'Substitutions'],
        'Count': [insertions, deletions, substitutions],
        'Percentage': [
            (insertions / total_errors * 100) if total_errors > 0 else 0,
            (deletions / total_errors * 100) if total_errors > 0 else 0,
            (substitutions / total_errors * 100) if total_errors > 0 else 0
        ]
    }
    
    df = pd.DataFrame(error_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Error Type'],
            y=df['Count'],
            text=df['Count'],
            textposition='auto',
            marker_color=['#f59e0b', '#ef4444', '#8b5cf6']
        )
    ])
    
    fig.update_layout(
        title="Error Breakdown",
        xaxis_title="Error Type",
        yaxis_title="Count",
        height=300,
        showlegend=False,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_temporal_alignment(
    predicted_timestamps: List[Dict],
    ground_truth_timestamps: Optional[List[Dict]] = None,
    temporal_alignment_accuracy: Optional[float] = None
):
    """
    Render temporal alignment visualization.
    
    Args:
        predicted_timestamps: List of predicted gloss timestamps
        ground_truth_timestamps: Optional ground truth timestamps
        temporal_alignment_accuracy: Optional alignment accuracy metric
    """
    st.markdown("#### Temporal Alignment")
    
    if temporal_alignment_accuracy is not None:
        st.metric("Temporal Alignment Accuracy", f"{temporal_alignment_accuracy*100:.1f}%")
    
    if not ground_truth_timestamps:
        # Show only predicted timeline
        render_timeline(predicted_timestamps, "Predicted Timeline")
        return
    
    # Create timeline comparison
    fig = go.Figure()
    
    # Ground truth timeline
    for i, ts in enumerate(ground_truth_timestamps):
        fig.add_trace(go.Bar(
            name=f"GT: {ts.get('gloss_label', ts.get('gloss'))}",
            x=[ts['duration_ms']],
            y=['Ground Truth'],
            orientation='h',
            marker=dict(color='rgba(16, 185, 129, 0.7)'),
            text=ts.get('gloss_label', str(ts.get('gloss'))),
            textposition='inside',
            hovertemplate=f"<b>{ts.get('gloss_label', '')}</b><br>" +
                         f"Start: {ts['start_ms']}ms<br>" +
                         f"End: {ts['end_ms']}ms<br>" +
                         f"Duration: {ts['duration_ms']}ms<extra></extra>",
            base=ts['start_ms']
        ))
    
    # Predicted timeline
    for i, ts in enumerate(predicted_timestamps):
        fig.add_trace(go.Bar(
            name=f"Pred: {ts.get('gloss', ts.get('index'))}",
            x=[ts['duration_ms']],
            y=['Predicted'],
            orientation='h',
            marker=dict(color='rgba(59, 130, 246, 0.7)'),
            text=str(ts.get('gloss', ts.get('index'))),
            textposition='inside',
            hovertemplate=f"<b>Gloss {ts.get('gloss', ts.get('index'))}</b><br>" +
                         f"Start: {ts['start_ms']}ms<br>" +
                         f"End: {ts['end_ms']}ms<br>" +
                         f"Duration: {ts['duration_ms']}ms<extra></extra>",
            base=ts['start_ms']
        ))
    
    fig.update_layout(
        title="Temporal Alignment Comparison",
        xaxis_title="Time (ms)",
        yaxis_title="",
        barmode='overlay',
        height=300,
        showlegend=False,
        template='plotly_dark',
        xaxis=dict(range=[0, max(
            ground_truth_timestamps[-1]['end_ms'] if ground_truth_timestamps else 0,
            predicted_timestamps[-1]['end_ms'] if predicted_timestamps else 0
        )])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_timeline(timestamps: List[Dict], title: str = "Timeline"):
    """
    Render simple timeline visualization.
    
    Args:
        timestamps: List of timestamp dictionaries
        title: Chart title
    """
    if not timestamps:
        st.info("No timestamp data available")
        return
    
    fig = go.Figure()
    
    for i, ts in enumerate(timestamps):
        gloss_label = ts.get('gloss_label', str(ts.get('gloss', i)))
        fig.add_trace(go.Bar(
            name=gloss_label,
            x=[ts['duration_ms']],
            y=['Sequence'],
            orientation='h',
            marker=dict(color=f'hsl({i * 360 / len(timestamps)}, 70%, 60%)'),
            text=gloss_label,
            textposition='inside',
            hovertemplate=f"<b>{gloss_label}</b><br>" +
                         f"Start: {ts['start_ms']}ms<br>" +
                         f"End: {ts['end_ms']}ms<br>" +
                         f"Duration: {ts['duration_ms']}ms<extra></extra>",
            base=ts['start_ms']
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="",
        barmode='stack',
        height=200,
        showlegend=False,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_ctc_prediction_card(
    file_name: str,
    predicted_sequence: List[int],
    predicted_labels: List[str],
    confidence_scores: Optional[List[float]] = None,
    wer: Optional[float] = None,
    ground_truth_available: bool = False
):
    """
    Render comprehensive CTC prediction card.
    
    Args:
        file_name: Name of the sequence file
        predicted_sequence: List of predicted gloss IDs
        predicted_labels: List of predicted gloss labels
        confidence_scores: Optional confidence scores
        wer: Optional WER if ground truth available
        ground_truth_available: Whether ground truth is available
    """
    st.markdown(f"### Prediction Results: `{file_name}`")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sequence Length", len(predicted_sequence))
    
    with col2:
        if confidence_scores:
            avg_conf = np.mean(confidence_scores)
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        else:
            st.metric("Confidence", "N/A")
    
    with col3:
        if wer is not None:
            wer_color = "normal" if wer <= 0.2 else "off"
            st.metric("WER", f"{wer*100:.1f}%", delta=None, delta_color=wer_color)
        else:
            st.metric("WER", "N/A" if not ground_truth_available else "Computing...")
    
    # Predicted sequence
    st.markdown("**Predicted Sequence:**")
    render_sequence_chips(predicted_labels, confidence_scores, color='#3b82f6')


def render_ctc_batch_summary(predictions: List[Dict]):
    """
    Render summary table for batch CTC predictions.
    
    Args:
        predictions: List of prediction result dictionaries
    """
    st.markdown("### Batch Prediction Summary")
    
    # Create summary dataframe
    summary_data = []
    for pred in predictions:
        summary_data.append({
            'File': pred['file_name'],
            'Length': pred['num_predicted'],
            'WER': f"{pred.get('wer', 0)*100:.1f}%" if 'wer' in pred else 'N/A',
            'Correct': '✓' if pred.get('correct', False) else '✗' if 'correct' in pred else '—',
            'Avg Confidence': f"{np.mean(pred.get('confidence_scores', [0]))*100:.1f}%" if pred.get('confidence_scores') else 'N/A'
        })
    
    df = pd.DataFrame(summary_data)
    
    # Interactive table
    st.dataframe(df, use_container_width=True, height=400)
    
    # Overall statistics if ground truth available
    if any('wer' in pred for pred in predictions):
        st.markdown("---")
        st.markdown("#### Overall Statistics")
        
        wers = [pred['wer'] for pred in predictions if 'wer' in pred]
        correct_count = sum(1 for pred in predictions if pred.get('correct', False))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean WER", f"{np.mean(wers)*100:.1f}%")
        
        with col2:
            st.metric("Median WER", f"{np.median(wers)*100:.1f}%")
        
        with col3:
            seq_accuracy = correct_count / len(predictions) * 100 if predictions else 0
            st.metric("Sequence Accuracy", f"{seq_accuracy:.1f}%")

