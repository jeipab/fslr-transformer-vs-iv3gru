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
    confidence_scores: Optional[List[float]] = None,
    predicted_categories: Optional[List[int]] = None,
    category_confidences: Optional[List[float]] = None,
    ground_truth_categories: Optional[List[int]] = None
):
    """
    Render side-by-side comparison of predicted vs ground truth sequences.
    
    Args:
        predicted_sequence: List of predicted gloss IDs
        predicted_labels: List of predicted gloss labels
        ground_truth_sequence: Optional list of ground truth gloss IDs
        ground_truth_labels: Optional list of ground truth gloss labels
        confidence_scores: Optional list of confidence scores per gloss
        predicted_categories: Optional list of predicted category IDs
        category_confidences: Optional category confidence scores
        ground_truth_categories: Optional list of ground truth category IDs
    """
    st.markdown("#### Sequence Comparison")
    
    has_categories = predicted_categories is not None and len(predicted_categories) > 0
    
    if ground_truth_sequence and len(ground_truth_sequence) > 0:
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ground Truth**")
            if has_categories and ground_truth_categories:
                render_sequence_with_categories(ground_truth_labels, ground_truth_categories, None, None)
            else:
                render_sequence_chips(ground_truth_labels, None, color='#10b981')
        
        with col2:
            st.markdown("**Predicted**")
            if has_categories:
                render_sequence_with_categories(predicted_labels, predicted_categories, confidence_scores, category_confidences)
            else:
                render_sequence_chips(predicted_labels, confidence_scores, color='#3b82f6')
        
    else:
        # Only prediction - show warning if ground truth was expected
        if ground_truth_sequence is not None and len(ground_truth_sequence) == 0:
            st.warning("Ground truth sequence is empty - check your JSON file format")
        
        st.markdown("**Predicted Sequence**")
        if has_categories:
            render_sequence_with_categories(predicted_labels, predicted_categories, confidence_scores, category_confidences)
        else:
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


def render_sequence_with_categories(
    gloss_labels: List[str],
    category_ids: List[int],
    gloss_confidences: Optional[List[float]] = None,
    category_confidences: Optional[List[float]] = None
):
    """
    Render sequence with both glosses and categories.
    
    Args:
        gloss_labels: List of gloss labels
        category_ids: List of category IDs
        gloss_confidences: Optional gloss confidence scores
        category_confidences: Optional category confidence scores
    """
    if not gloss_labels:
        st.info("Empty sequence")
        return
    
    # Load category mappings
    try:
        from data.labels.label_mapping import load_label_mappings
        _, category_mapping = load_label_mappings()
    except:
        category_mapping = {}
    
    # Create HTML for dual chips (gloss + category)
    chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 1rem 0;">'
    
    for i, gloss_label in enumerate(gloss_labels):
        # Gloss chip
        gloss_conf_text = ""
        if gloss_confidences and i < len(gloss_confidences):
            gloss_conf = gloss_confidences[i]
            gloss_conf_text = f' ({gloss_conf*100:.1f}%)'
            gloss_opacity = 0.5 + (gloss_conf * 0.5)
        else:
            gloss_opacity = 1.0
        
        # Category chip
        cat_label = "N/A"
        cat_conf_text = ""
        cat_opacity = 1.0
        if i < len(category_ids):
            cat_id = category_ids[i]
            cat_label = category_mapping.get(cat_id, f"Cat_{cat_id}")
            
            if category_confidences and i < len(category_confidences):
                cat_conf = category_confidences[i]
                cat_conf_text = f' ({cat_conf*100:.1f}%)'
                cat_opacity = 0.5 + (cat_conf * 0.5)
        
        # Container for this sign (vertical stack)
        chips_html += f'<div style="display: flex; flex-direction: column; gap: 0.25rem;"><div style="background-color: #3b82f6; color: white; padding: 0.4rem 0.8rem; border-radius: 1rem; font-size: 0.9rem; font-weight: 500; opacity: {gloss_opacity};">{i+1}. {gloss_label}{gloss_conf_text}</div><div style="background-color: #10b981; color: white; padding: 0.3rem 0.6rem; border-radius: 0.8rem; font-size: 0.75rem; font-weight: 500; opacity: {cat_opacity}; text-align: center;">{cat_label}{cat_conf_text}</div></div>'
    
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
    st.dataframe(styled_df, width='stretch')




def smooth_timestamps(timestamps: List[Dict]) -> List[Dict]:
    """
    Smooth timestamps by merging consecutive identical glosses.
    
    Args:
        timestamps: List of timestamp dictionaries
        
    Returns:
        List of smoothed timestamp dictionaries
    """
    if not timestamps:
        return []
    
    smoothed = []
    current_gloss = timestamps[0].get('gloss', timestamps[0].get('index'))
    start_ms = timestamps[0]['start_ms']
    end_ms = timestamps[0]['end_ms']
    
    for i in range(1, len(timestamps)):
        next_gloss = timestamps[i].get('gloss', timestamps[i].get('index'))
        
        if next_gloss == current_gloss:
            # Merge consecutive identical glosses
            end_ms = timestamps[i]['end_ms']
        else:
            # Add the current merged segment
            smoothed.append({
                'gloss': current_gloss,
                'index': timestamps[i-1].get('index', i-1),
                'start_ms': start_ms,
                'end_ms': end_ms,
                'duration_ms': end_ms - start_ms
            })
            
            # Start new segment
            current_gloss = next_gloss
            start_ms = timestamps[i]['start_ms']
            end_ms = timestamps[i]['end_ms']
    
    # Add the final segment
    smoothed.append({
        'gloss': current_gloss,
        'index': timestamps[-1].get('index', len(timestamps)-1),
        'start_ms': start_ms,
        'end_ms': end_ms,
        'duration_ms': end_ms - start_ms
    })
    
    return smoothed


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
    
    # Apply smoothing to predicted timestamps
    smoothed_predicted = smooth_timestamps(predicted_timestamps)
    
    if not ground_truth_timestamps:
        # Show only predicted timeline
        render_timeline(smoothed_predicted, "Predicted Timeline (Smoothed)")
        return
    
    # Load label mappings to convert gloss indices to labels
    try:
        from data.labels.label_mapping import load_label_mappings
        gloss_mapping, _ = load_label_mappings()
    except Exception as e:
        st.warning(f"Could not load label mappings: {e}. Showing numeric IDs only.")
        gloss_mapping = {}
    
    # Create timeline comparison
    fig = go.Figure()
    
    # Ground truth timeline
    for i, ts in enumerate(ground_truth_timestamps):
        gloss_label = ts.get('gloss_label', ts.get('gloss', ''))
        fig.add_trace(go.Bar(
            name=f"GT: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth'],
            orientation='h',
            marker=dict(color='rgba(16, 185, 129, 0.8)'),
            text=gloss_label,
            textposition='inside',
            textfont=dict(
                color='white',
                size=11,
                family='Arial Black'
            ),
            hovertemplate=f"<b>{gloss_label}</b><br>" +
                         f"Start: {ts['start_ms']}ms<br>" +
                         f"End: {ts['end_ms']}ms<br>" +
                         f"Duration: {ts['duration_ms']}ms<extra></extra>",
            base=ts['start_ms']
        ))
    
    # Predicted timeline (smoothed)
    for i, ts in enumerate(smoothed_predicted):
        gloss_index = ts.get('gloss', ts.get('index'))
        # Map gloss index to actual gloss label
        gloss_label = gloss_mapping.get(gloss_index, f"Gloss {gloss_index}")
        
        fig.add_trace(go.Bar(
            name=f"Pred: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Predicted'],
            orientation='h',
            marker=dict(color='rgba(59, 130, 246, 0.8)'),
            text=gloss_label,
            textposition='inside',
            textfont=dict(
                color='white',
                size=11,
                family='Arial Black'
            ),
            hovertemplate=f"<b>{gloss_label}</b><br>" +
                         f"Gloss ID: {gloss_index}<br>" +
                         f"Start: {ts['start_ms']}ms<br>" +
                         f"End: {ts['end_ms']}ms<br>" +
                         f"Duration: {ts['duration_ms']}ms<extra></extra>",
            base=ts['start_ms']
        ))
    
    fig.update_layout(
        title=dict(
            text="Temporal Alignment Comparison",
            font=dict(size=16, color='white')
        ),
        xaxis=dict(
            title=dict(text="Time (ms)", font=dict(size=12, color='white')),
            tickfont=dict(size=10, color='white'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            range=[0, max(
                ground_truth_timestamps[-1]['end_ms'] if ground_truth_timestamps else 0,
                smoothed_predicted[-1]['end_ms'] if smoothed_predicted else 0
            )]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color='white')
        ),
        barmode='overlay',
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=60, b=50)
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
    
    # Apply smoothing to timestamps
    smoothed_timestamps = smooth_timestamps(timestamps)
    
    # Load label mappings to convert gloss indices to labels
    try:
        from data.labels.label_mapping import load_label_mappings
        gloss_mapping, _ = load_label_mappings()
    except Exception as e:
        gloss_mapping = {}
    
    fig = go.Figure()
    
    for i, ts in enumerate(smoothed_timestamps):
        gloss_index = ts.get('gloss', i)
        # Map gloss index to actual gloss label
        gloss_label = gloss_mapping.get(gloss_index, f"Gloss {gloss_index}")
        
        fig.add_trace(go.Bar(
            name=gloss_label,
            x=[ts['duration_ms']],
            y=['Sequence'],
            orientation='h',
            marker=dict(color=f'hsl({i * 360 / len(timestamps)}, 70%, 60%)'),
            text=gloss_label,
            textposition='inside',
            textfont=dict(
                color='white',
                size=10,
                family='Arial Black'
            ),
            hovertemplate=f"<b>{gloss_label}</b><br>" +
                         f"Gloss ID: {gloss_index}<br>" +
                         f"Start: {ts['start_ms']}ms<br>" +
                         f"End: {ts['end_ms']}ms<br>" +
                         f"Duration: {ts['duration_ms']}ms<extra></extra>",
            base=ts['start_ms']
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='white')
        ),
        xaxis=dict(
            title=dict(text="Time (ms)", font=dict(size=12, color='white')),
            tickfont=dict(size=10, color='white'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color='white')
        ),
        barmode='stack',
        height=200,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_ctc_prediction_card(
    file_name: str,
    predicted_sequence: List[int],
    predicted_labels: List[str],
    confidence_scores: Optional[List[float]] = None,
    ground_truth_available: bool = False,
    predicted_categories: Optional[List[int]] = None,
    category_confidences: Optional[List[float]] = None,
    category_accuracy: Optional[float] = None
):
    """
    Render comprehensive CTC prediction card with improved UI/UX.
    
    Args:
        file_name: Name of the sequence file
        predicted_sequence: List of predicted gloss IDs
        predicted_labels: List of predicted gloss labels
        confidence_scores: Optional confidence scores
        ground_truth_available: Whether ground truth is available
        predicted_categories: Optional list of predicted category IDs
        category_confidences: Optional category confidence scores
        category_accuracy: Optional category accuracy if ground truth available
    """
    # Enhanced header with file info
    st.markdown(f"### Prediction Results: `{file_name}`")
    
    # Check if predicted_sequence is valid
    if predicted_sequence is None:
        predicted_sequence = []
    if predicted_labels is None:
        predicted_labels = []
    
    # Summary metrics with better visual design
    has_categories = predicted_categories is not None and len(predicted_categories) > 0
    cols = st.columns(4 if has_categories else 3)
    
    with cols[0]:
        st.metric(
            "Sequence Length", 
            len(predicted_sequence),
            help="Number of predicted glosses in the sequence"
        )
    
    with cols[1]:
        if confidence_scores:
            avg_conf = np.mean(confidence_scores)
            conf_color = "normal" if avg_conf >= 0.7 else "off"
            st.metric(
                "Average Gloss Confidence", 
                f"{avg_conf*100:.1f}%",
                delta=None,
                delta_color=conf_color,
                help="Average confidence across all predicted glosses"
            )
        else:
            st.metric("Average Gloss Confidence", "N/A", help="No confidence scores available")
    
    if has_categories and len(cols) > 2:
        with cols[2]:
            if category_accuracy is not None:
                cat_color = "normal" if category_accuracy >= 0.8 else "off"
                st.metric(
                    "Category Acc", 
                    f"{category_accuracy*100:.1f}%", 
                    delta_color=cat_color,
                    help="Category prediction accuracy"
                )
            elif category_confidences:
                avg_cat_conf = np.mean(category_confidences)
                st.metric(
                    "Average Category Confidence", 
                    f"{avg_cat_conf*100:.1f}%",
                    help="Average category confidence"
                )
            else:
                st.metric("Category", "N/A", help="No category predictions available")
    
    # Enhanced predicted sequence display
    st.markdown("---")
    st.markdown("#### Predicted Sequence")
    
    if not predicted_labels:
        st.warning("No predictions generated - sequence may be too short or model failed")
        return
    
    if has_categories:
        render_sequence_with_categories(predicted_labels, predicted_categories, confidence_scores, category_confidences)
    else:
        render_sequence_chips(predicted_labels, confidence_scores, color='#3b82f6')


def render_ctc_batch_summary(predictions: List[Dict]):
    """
    Render summary table for batch CTC predictions.
    
    Args:
        predictions: List of prediction result dictionaries
    """
    st.markdown("### Batch Prediction Summary")
    
    # Check if predictions have category data
    has_categories = any('predicted_categories' in p and len(p['predicted_categories']) > 0 for p in predictions)
    
    # Create summary dataframe
    summary_data = []
    for pred in predictions:
        row = {
            'File': pred['file_name'],
            'Length': pred['num_predicted'],
            'WER': f"{pred.get('wer', 0)*100:.1f}%" if 'wer' in pred else 'N/A',
            'Correct': '✓' if pred.get('correct', False) else '✗' if 'correct' in pred else '—',
            'Avg Confidence': f"{np.mean(pred.get('confidence_scores', [0]))*100:.1f}%" if pred.get('confidence_scores') else 'N/A'
        }
        
        # Add category accuracy if available
        if has_categories and 'category_accuracy' in pred:
            row['Cat Acc'] = f"{pred['category_accuracy']*100:.1f}%"
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Interactive table
    st.dataframe(df, width='stretch', height=400)
    
    # Overall statistics if ground truth available
    if any('wer' in pred for pred in predictions):
        st.markdown("---")
        st.markdown("#### Overall Statistics")
        
        wers = [pred['wer'] for pred in predictions if 'wer' in pred]
        correct_count = sum(1 for pred in predictions if pred.get('correct', False))
        
        cols = st.columns(4 if has_categories else 3)
        
        with cols[0]:
            st.metric("Mean WER", f"{np.mean(wers)*100:.1f}%")
        
        with cols[1]:
            st.metric("Median WER", f"{np.median(wers)*100:.1f}%")
        
        with cols[2]:
            seq_accuracy = correct_count / len(predictions) * 100 if predictions else 0
            st.metric("Sequence Accuracy", f"{seq_accuracy:.1f}%")
        
        if has_categories and len(cols) > 3:
            cat_accs = [pred['category_accuracy'] for pred in predictions if 'category_accuracy' in pred]
            if cat_accs:
                with cols[3]:
                    st.metric("Mean Category Acc", f"{np.mean(cat_accs)*100:.1f}%")

