"""CTC-specific visualization components for continuous sign language recognition."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def detect_inactive_hand_periods(
    mask: np.ndarray,
    timestamps_ms: np.ndarray,
    left_hand_start: int = 25,
    left_hand_end: int = 45,
    right_hand_start: int = 46,
    right_hand_end: int = 66
) -> List[Tuple[float, float]]:
    """
    Detect time periods where hands are not in frame (all hand keypoints inactive).
    
    Args:
        mask: Keypoint visibility mask [T, 89] where True = visible
        timestamps_ms: Timestamp array [T] in milliseconds
        left_hand_start: Starting index for left hand keypoints (default 25)
        left_hand_end: Ending index for left hand keypoints (default 45)
        right_hand_start: Starting index for right hand keypoints (default 46)
        right_hand_end: Ending index for right hand keypoints (default 66)
        
    Returns:
        List of (start_ms, end_ms) tuples for inactive periods
    """
    if mask is None or len(mask) == 0 or timestamps_ms is None or len(timestamps_ms) == 0:
        return []
    
    if len(mask) != len(timestamps_ms):
        return []
    
    inactive_periods = []
    in_inactive_period = False
    period_start = None
    
    for t in range(len(mask)):
        # Check if both hands are inactive (all keypoints False)
        left_hand_mask = mask[t, left_hand_start:left_hand_end+1]
        right_hand_mask = mask[t, right_hand_start:right_hand_end+1]
        
        left_inactive = not np.any(left_hand_mask)
        right_inactive = not np.any(right_hand_mask)
        both_inactive = left_inactive and right_inactive
        
        if both_inactive and not in_inactive_period:
            # Start of inactive period
            in_inactive_period = True
            period_start = timestamps_ms[t]
        elif not both_inactive and in_inactive_period:
            # End of inactive period
            in_inactive_period = False
            if period_start is not None:
                inactive_periods.append((float(period_start), float(timestamps_ms[t])))
                period_start = None
    
    # Handle case where sequence ends in inactive period
    if in_inactive_period and period_start is not None:
        inactive_periods.append((float(period_start), float(timestamps_ms[-1])))
    
    return inactive_periods


def render_sequence_comparison(
    predicted_sequence: List[int],
    predicted_labels: List[str],
    ground_truth_sequence: Optional[List[int]] = None,
    ground_truth_labels: Optional[List[str]] = None,
    confidence_scores: Optional[List[float]] = None,
    predicted_categories: Optional[List[int]] = None,
    category_confidences: Optional[List[float]] = None,
    ground_truth_categories: Optional[List[int]] = None,
    ground_truth_occluded: Optional[List[int]] = None,
    prediction_cases: Optional[Dict[int, str]] = None,
    case_palette: Optional[Dict[str, str]] = None,
    confidence_threshold: float = 0.5,
    category_prediction_cases: Optional[Dict[int, str]] = None,
    category_ground_truth_cases: Optional[Dict[int, str]] = None,
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
        ground_truth_occluded: Optional list of occlusion flags (0 or 1) for ground truth
    """
    st.markdown("#### Sequence Comparison")
    
    has_categories = predicted_categories is not None and len(predicted_categories) > 0
    case_palette = case_palette or {
        'TP': '#22c55e',
        'FP': '#ef4444',
        'FN': '#f97316',
        'TN': '#64748b',
    }
    
    if ground_truth_sequence and len(ground_truth_sequence) > 0:
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ground Truth**")
            if has_categories and ground_truth_categories:
                render_sequence_with_categories(
                    ground_truth_labels,
                    ground_truth_categories,
                    None,
                    None,
                    occlusion_flags=ground_truth_occluded,
                    is_ground_truth=True,
                    case_palette=case_palette,
                    category_case_map=category_ground_truth_cases,
                )
            else:
                render_sequence_chips(
                    ground_truth_labels,
                    None,
                    color=case_palette.get('TP', '#22c55e'),
                    occlusion_flags=ground_truth_occluded,
                    case_palette=case_palette,
                )
        
        with col2:
            st.markdown("**Predicted**")
            if has_categories:
                render_sequence_with_categories(
                    predicted_labels,
                    predicted_categories,
                    confidence_scores,
                    category_confidences,
                    case_map=prediction_cases,
                    case_palette=case_palette,
                    confidence_threshold=confidence_threshold,
                    category_case_map=category_prediction_cases,
                )
            else:
                render_sequence_chips(
                    predicted_labels,
                    confidence_scores,
                    color='#3b82f6',
                    case_map=prediction_cases,
                    case_palette=case_palette,
                    confidence_threshold=confidence_threshold,
                )
        
    else:
        # Only prediction - show warning if ground truth was expected
        if ground_truth_sequence is not None and len(ground_truth_sequence) == 0:
            st.warning("Ground truth sequence is empty - check your JSON file format")
        
        st.markdown("**Predicted Sequence**")
        if has_categories:
            render_sequence_with_categories(
                predicted_labels,
                predicted_categories,
                confidence_scores,
                category_confidences,
                case_map=prediction_cases,
                case_palette=case_palette,
                confidence_threshold=confidence_threshold,
                category_case_map=category_prediction_cases,
            )
        else:
            render_sequence_chips(
                predicted_labels,
                confidence_scores,
                color='#3b82f6',
                case_map=prediction_cases,
                case_palette=case_palette,
                confidence_threshold=confidence_threshold,
            )


def render_sequence_chips(
    labels: List[str],
    confidence_scores: Optional[List[float]] = None,
    color: str = '#3b82f6',
    occlusion_flags: Optional[List[int]] = None,
    case_map: Optional[Dict[int, str]] = None,
    case_palette: Optional[Dict[str, str]] = None,
    confidence_threshold: float = 0.5,
    low_confidence_color: str = '#f97316',
):
    """
    Render sequence as colored chips/badges.
    
    Args:
        labels: List of gloss labels
        confidence_scores: Optional confidence scores
        color: Hex color for chips
        occlusion_flags: Optional list of occlusion flags (0 or 1) - if 1, use red color
    """
    if not labels:
        st.info("Empty sequence")
        return
    
    # Create HTML for chips
    chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">'
    palette = case_palette or {}
    
    for i, label in enumerate(labels):
        confidence_text = ""
        if confidence_scores and i < len(confidence_scores):
            confidence = confidence_scores[i]
            confidence_text = f' ({confidence*100:.1f}%)'
            # Adjust opacity based on confidence
            opacity = 0.5 + (confidence * 0.5)
        else:
            opacity = 1.0
        
        # Use red color if occluded, otherwise use provided color
        chip_color = '#ef4444' if (occlusion_flags and i < len(occlusion_flags) and occlusion_flags[i] == 1 and not case_map) else color
        if case_map and i in case_map:
            chip_color = palette.get(case_map[i], chip_color)
        elif confidence_scores and i < len(confidence_scores):
            if confidence_scores[i] < confidence_threshold:
                chip_color = low_confidence_color
        
        chip_style = f"""
            background-color: {chip_color};
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            font-weight: 500;
            text-align: center;
            opacity: {opacity};
        """
        
        chips_html += f'<div style="{chip_style}">{label}{confidence_text}</div>'
    
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_sequence_with_categories(
    gloss_labels: List[str],
    category_ids: List[int],
    gloss_confidences: Optional[List[float]] = None,
    category_confidences: Optional[List[float]] = None,
    occlusion_flags: Optional[List[int]] = None,
    is_ground_truth: bool = False,
    case_map: Optional[Dict[int, str]] = None,
    case_palette: Optional[Dict[str, str]] = None,
    confidence_threshold: float = 0.5,
    low_confidence_color: str = '#f97316',
    category_case_map: Optional[Dict[int, str]] = None,
):
    """
    Render sequence with both glosses and categories.
    
    Args:
        gloss_labels: List of gloss labels
        category_ids: List of category IDs
        gloss_confidences: Optional gloss confidence scores
        category_confidences: Optional category confidence scores
        occlusion_flags: Optional list of occlusion flags (0 or 1) - if 1, use red color for ground truth
        is_ground_truth: Whether this is ground truth (affects color when occluded)
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
    
    palette = case_palette or {}
    gloss_case_map = case_map or {}
    category_case_map = category_case_map or {}

    for i, gloss_label in enumerate(gloss_labels):
        # Gloss chip
        gloss_conf_text = ""
        if gloss_confidences and i < len(gloss_confidences):
            gloss_conf = gloss_confidences[i]
            gloss_conf_text = f' ({gloss_conf*100:.1f}%)'
            gloss_opacity = 0.5 + (gloss_conf * 0.5)
        else:
            gloss_opacity = 1.0
        
        # Determine gloss chip color
        is_occluded = occlusion_flags and i < len(occlusion_flags) and occlusion_flags[i] == 1
        gloss_base = case_palette.get('TP', '#22c55e') if is_ground_truth else '#3b82f6'
        gloss_color = gloss_base
        if gloss_case_map and i in gloss_case_map:
            gloss_color = palette.get(gloss_case_map[i], gloss_color)
        elif gloss_confidences and i < len(gloss_confidences) and gloss_confidences[i] < confidence_threshold:
            gloss_color = low_confidence_color
        
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
        
        cat_border = '1px solid transparent'

        if is_ground_truth:
            if is_occluded:
                cat_bg = '#000000'
                cat_color = 'white'
            else:
                cat_bg = '#ffffff'
                cat_color = '#000000'
            cat_border = '1px solid rgba(148, 163, 184, 0.35)'
            if category_case_map and i in category_case_map:
                border_color = palette.get(category_case_map[i], '#22c55e')
                cat_border = f'2px solid {border_color}'
        else:
            cat_bg = '#10b981'
            cat_color = 'white'
            if category_case_map and i in category_case_map:
                cat_bg = palette.get(category_case_map[i], cat_bg)
            elif category_confidences and i < len(category_confidences) and category_confidences[i] < confidence_threshold:
                cat_bg = low_confidence_color
            if category_case_map and i in category_case_map:
                cat_border = f'1px solid {palette.get(category_case_map[i], "#0f172a")}'

        chips_html += (
            '<div style="display: flex; flex-direction: column; gap: 0.25rem;">'
            f'<div style="background-color: {gloss_color}; color: white; padding: 0.4rem 0.8rem; border-radius: 1rem; font-size: 0.9rem; font-weight: 500; opacity: {gloss_opacity}; text-align: center;">'
            f'{gloss_label}{gloss_conf_text}</div>'
            f'<div style="background-color: {cat_bg}; color: {cat_color}; padding: 0.3rem 0.6rem; border-radius: 0.8rem; font-size: 0.75rem; font-weight: 500; opacity: {cat_opacity}; text-align: center; border: {cat_border};">'
            f'{cat_label}{cat_conf_text}</div></div>'
        )
    
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
    # Preserve category info from first timestamp in merged segment
    current_category = timestamps[0].get('category', None)
    current_category_label = timestamps[0].get('category_label', '')
    
    for i in range(1, len(timestamps)):
        next_gloss = timestamps[i].get('gloss', timestamps[i].get('index'))
        
        if next_gloss == current_gloss:
            # Merge consecutive identical glosses
            end_ms = timestamps[i]['end_ms']
            # Keep category from first occurrence, or update if current is None
            if current_category is None:
                current_category = timestamps[i].get('category', None)
                current_category_label = timestamps[i].get('category_label', '')
        else:
            # Add the current merged segment
            smoothed_ts = {
                'gloss': current_gloss,
                'index': timestamps[i-1].get('index', i-1),
                'start_ms': start_ms,
                'end_ms': end_ms,
                'duration_ms': end_ms - start_ms
            }
            # Preserve category info if available
            if current_category is not None:
                smoothed_ts['category'] = current_category
            if current_category_label:
                smoothed_ts['category_label'] = current_category_label
            smoothed.append(smoothed_ts)
            
            # Start new segment
            current_gloss = next_gloss
            start_ms = timestamps[i]['start_ms']
            end_ms = timestamps[i]['end_ms']
            current_category = timestamps[i].get('category', None)
            current_category_label = timestamps[i].get('category_label', '')
    
    # Add the final segment
    final_ts = {
        'gloss': current_gloss,
        'index': timestamps[-1].get('index', len(timestamps)-1),
        'start_ms': start_ms,
        'end_ms': end_ms,
        'duration_ms': end_ms - start_ms
    }
    # Preserve category info if available
    if current_category is not None:
        final_ts['category'] = current_category
    if current_category_label:
        final_ts['category_label'] = current_category_label
    smoothed.append(final_ts)
    
    return smoothed


def render_temporal_alignment(
    predicted_timestamps: List[Dict],
    ground_truth_timestamps: Optional[List[Dict]] = None,
    temporal_alignment_accuracy: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    timestamps_ms: Optional[np.ndarray] = None,
    predicted_categories: Optional[List[int]] = None,
    prediction_cases: Optional[Dict[int, str]] = None,
    ground_truth_cases: Optional[Dict[int, str]] = None,
    case_palette: Optional[Dict[str, str]] = None,
    category_prediction_cases: Optional[Dict[int, str]] = None,
    category_ground_truth_cases: Optional[Dict[int, str]] = None,
):
    """
    Render temporal alignment visualization.
    
    Args:
        predicted_timestamps: List of predicted gloss timestamps
        ground_truth_timestamps: Optional ground truth timestamps
        temporal_alignment_accuracy: Optional alignment accuracy metric
        mask: Optional keypoint visibility mask [T, 89] for detecting inactive periods
        timestamps_ms: Optional timestamp array [T] in milliseconds for inactive period detection
        predicted_categories: Optional list of predicted category IDs as fallback if not in timestamps
        category_prediction_cases: Optional mapping of predicted indices to TP/FP/TN/FN cases for categories
        category_ground_truth_cases: Optional mapping of ground truth indices to TP/FN cases for categories
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
        gloss_mapping, category_mapping = load_label_mappings()
    except Exception as e:
        st.warning(f"Could not load label mappings: {e}. Showing numeric IDs only.")
        gloss_mapping = {}
        category_mapping = {}
    
    # Detect inactive hand periods if mask is available
    inactive_periods = []
    if mask is not None and timestamps_ms is not None:
        inactive_periods = detect_inactive_hand_periods(mask, timestamps_ms)
    
    # Calculate max time for x-axis range
    max_time = max(
        ground_truth_timestamps[-1]['end_ms'] if ground_truth_timestamps else 0,
        smoothed_predicted[-1]['end_ms'] if smoothed_predicted else 0
    )
    
    # Prepare common styling values
    bar_height = 0.3
    ground_truth_color = 'rgba(16, 185, 129, 0.8)'
    predicted_color = 'rgba(59, 130, 246, 0.8)'
    ground_truth_category_color = 'rgba(148, 163, 184, 0.65)'
    predicted_category_color = 'rgba(56, 189, 248, 0.7)'
    ground_truth_label_bg = 'rgba(6, 95, 70, 0.92)'

    inactive_segments = {
        'Ground Truth Gloss': {'bases': [], 'durations': []},
        'Ground Truth Category': {'bases': [], 'durations': []},
    }
    for start_ms, end_ms in inactive_periods:
        if end_ms <= start_ms:
            continue
        duration = end_ms - start_ms
        for row in ['Ground Truth Gloss', 'Ground Truth Category']:
            inactive_segments[row]['bases'].append(start_ms)
            inactive_segments[row]['durations'].append(duration)

    inactive_pattern = dict(
        shape='/',
        fgcolor='rgba(14, 17, 23, 0.9)',
        bgcolor='rgba(0, 0, 0, 0)',
        size=8,
        solidity=0.35,
        fillmode='replace'
    )

    palette = case_palette or {
        'TP': 'rgba(34, 197, 94, 0.85)',
        'FP': 'rgba(239, 68, 68, 0.9)',
        'FN': 'rgba(249, 115, 22, 0.9)',
        'TN': 'rgba(100, 116, 139, 0.85)',
    }
    category_prediction_cases = category_prediction_cases or {}
    category_ground_truth_cases = category_ground_truth_cases or {}

    case_label_map = {
        'TP': 'True Positive',
        'FP': 'False Positive',
        'FN': 'False Negative',
        'TN': 'True Negative',
    }

    def add_inactive_overlay(fig, row_label, bar_width):
        segments = inactive_segments.get(row_label)
        if not segments or not segments['bases']:
            return
        fig.add_trace(go.Bar(
            name=f"{row_label} Inactive",
            x=segments['durations'],
            y=[row_label] * len(segments['durations']),
            orientation='h',
            base=segments['bases'],
            marker=dict(
                color='rgba(0, 0, 0, 0)',
                line=dict(width=0),
                pattern=inactive_pattern
            ),
            hoverinfo='skip',
            showlegend=False,
            width=bar_width,
            opacity=1.0
        ))

    # --- Gloss Comparison chart ---
    gloss_row_labels = ['Predicted Gloss', 'Ground Truth Gloss']
    gloss_axis_labels = list(reversed(gloss_row_labels))
    y_center_gloss = {label: idx for idx, label in enumerate(gloss_axis_labels)}
    fig_gloss = go.Figure()

    for i, ts in enumerate(smoothed_predicted):
        gloss_index = ts.get('gloss', ts.get('index'))
        # Map gloss index to actual gloss label
        gloss_label = gloss_mapping.get(gloss_index, f"Gloss {gloss_index}")
        category_id = ts.get('category', None)
        category_label = ts.get('category_label', '')
        
        # Get category label from mapping if category_id is available
        if category_id is not None:
            if not category_label:
                category_label = category_mapping.get(category_id, f"Cat_{category_id}")
        
        case_index = ts.get('index', i)
        case = prediction_cases.get(case_index) if prediction_cases else None
        pred_color = palette.get(case, predicted_color) if case else predicted_color
        case_label = case_label_map.get(case)
        
        hover_text = f"<b>{gloss_label}</b><br>Gloss ID: {gloss_index}<br>"
        if category_label:
            hover_text += f"Category: {category_label}<br>"
        if case_label:
            hover_text += f"Case: {case_label}<br>"
        elif case:
            hover_text += f"Case: {case}<br>"
        hover_text += f"Start: {ts['start_ms']}ms<br>End: {ts['end_ms']}ms<br>Duration: {ts['duration_ms']}ms<extra></extra>"

        bar = go.Bar(
            name=f"Pred Gloss: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Predicted Gloss'],
            orientation='h',
            marker=dict(
                color=pred_color,
                line=dict(width=0)
            ),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.6
        )
        fig_gloss.add_trace(bar)

        if case and case.startswith('FN'):
            fig_gloss.add_shape(
                type='rect',
                x0=ts['start_ms'],
                x1=ts['end_ms'],
                y0=y_center_gloss['Predicted Gloss'] - bar_height,
                y1=y_center_gloss['Predicted Gloss'] + bar_height,
                fillcolor='rgba(249, 115, 22, 0.35)',
                line=dict(color='rgba(249, 115, 22, 0.9)', width=2),
                layer='above',
            )

        if ts['duration_ms'] > 0:
            fig_gloss.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=y_center_gloss['Predicted Gloss'] - bar_height,
                y1=y_center_gloss['Predicted Gloss'] + bar_height,
                xref='x',
                yref='y',
                line=dict(color='rgba(255,255,255,0.95)', width=3),
                layer='above',
            )

        annotation_text = gloss_label
        duration = ts['duration_ms']
        label_offset = max(duration * 0.05, 40)
        label_x = ts['end_ms'] - label_offset
        if label_x <= ts['start_ms']:
            label_x = ts['start_ms'] + duration * 0.5

        fig_gloss.add_annotation(
            x=label_x,
            y='Predicted Gloss',
            text=annotation_text,
            showarrow=False,
            xref='x',
            yref='y',
            xanchor='right',
            yanchor='middle',
            align='right',
            bgcolor='white',
            bordercolor='rgba(15, 23, 42, 0.3)',
            borderwidth=1,
            borderpad=3,
            font=dict(
                color='black',
                size=11,
                family='Arial Black'
            )
        )

        # Bar outline defines separator; no additional shape required.
    
    # Ground truth gloss timeline
    for i, ts in enumerate(ground_truth_timestamps):
        gloss_label = ts.get('gloss_label', ts.get('gloss', ''))
        category_id = ts.get('category', None)
        category_label = ts.get('category_label', '')

        if not category_label and category_id is not None:
            category_label = category_mapping.get(category_id, f"Cat_{category_id}")

        gloss_case = ground_truth_cases.get(i) if ground_truth_cases else None
        gloss_case_label = case_label_map.get(gloss_case)
        gloss_color = palette.get(gloss_case, ground_truth_color) if gloss_case else ground_truth_color

        hover_text = f"<b>{gloss_label}</b><br>"
        if category_label:
            hover_text += f"Category: {category_label}<br>"
        if gloss_case_label:
            hover_text += f"Case: {gloss_case_label}<br>"
        elif gloss_case:
            hover_text += f"Case: {gloss_case}<br>"
        hover_text += (
            f"Start: {ts['start_ms']}ms<br>"
            f"End: {ts['end_ms']}ms<br>"
            f"Duration: {ts['duration_ms']}ms<extra></extra>"
        )

        fig_gloss.add_trace(go.Bar(
            name=f"GT Gloss: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth Gloss'],
            orientation='h',
            marker=dict(
                color=gloss_color,
                line=dict(width=0)
            ),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.6
        ))

        if ts['duration_ms'] > 0:
            fig_gloss.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=y_center_gloss['Ground Truth Gloss'] - bar_height,
                y1=y_center_gloss['Ground Truth Gloss'] + bar_height,
                xref='x',
                yref='y',
                line=dict(color='rgba(255,255,255,0.95)', width=3),
                layer='above',
            )

        annotation_text = gloss_label
        duration = ts['duration_ms']
        label_offset = max(duration * 0.05, 40)
        label_x = ts['end_ms'] - label_offset
        if label_x <= ts['start_ms']:
            label_x = ts['start_ms'] + duration * 0.5

        fig_gloss.add_annotation(
            x=label_x,
            y='Ground Truth Gloss',
            text=annotation_text,
            showarrow=False,
            xref='x',
            yref='y',
            xanchor='right',
            yanchor='middle',
            align='right',
            bgcolor='white',
            bordercolor='rgba(15, 23, 42, 0.3)',
            borderwidth=1,
            borderpad=3,
            font=dict(
                color='black',
                size=11,
                family='Arial Black'
            )
        )
    
    add_inactive_overlay(fig_gloss, 'Ground Truth Gloss', bar_width=0.6)

    fig_gloss.update_layout(
        title=dict(
            text="Gloss Comparison",
            font=dict(size=15, color='white')
        ),
        xaxis=dict(
            title=dict(text="Time (ms)", font=dict(size=12, color='white')),
            tickfont=dict(size=10, color='white'),
            gridcolor='rgba(255, 255, 255, 0.12)',
            range=[0, max_time]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color='white'),
            categoryorder='array',
            categoryarray=gloss_axis_labels
        ),
        barmode='overlay',
        height=320,
        bargap=0.22,
        bargroupgap=0.05,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=45, r=45, t=50, b=40)
    )

    # --- Category Comparison chart ---
    category_row_labels = ['Predicted Category', 'Ground Truth Category']
    category_axis_labels = list(reversed(category_row_labels))
    y_center_category = {label: idx for idx, label in enumerate(category_axis_labels)}
    fig_category = go.Figure()

    for i, ts in enumerate(smoothed_predicted):
        case_index = ts.get('index', i)
        category_id = ts.get('category')
        category_label = ts.get('category_label', '')
        if category_id is None and predicted_categories and case_index < len(predicted_categories):
            category_id = predicted_categories[case_index]
        if category_id is not None and not category_label:
            category_label = category_mapping.get(category_id, f"Cat_{category_id}")

        display_label = category_label or (f"Cat_{category_id}" if category_id is not None else "Category")

        category_case = category_prediction_cases.get(case_index) if category_prediction_cases else None
        category_case_label = case_label_map.get(category_case)
        category_color = palette.get(category_case, predicted_category_color) if category_case else predicted_category_color

        hover_text = f"<b>{display_label}</b><br>"
        if category_case_label:
            hover_text += f"Case: {category_case_label}<br>"
        elif category_case:
            hover_text += f"Case: {category_case}<br>"
        hover_text += (
            f"Start: {ts['start_ms']}ms<br>"
            f"End: {ts['end_ms']}ms<br>"
            f"Duration: {ts['duration_ms']}ms<extra></extra>"
        )

        fig_category.add_trace(go.Bar(
            name=f"Pred Cat: {display_label}",
            x=[ts['duration_ms']],
            y=['Predicted Category'],
            orientation='h',
            marker=dict(
                color=category_color
            ),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.55
        ))

        if ts['duration_ms'] > 0:
            fig_category.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=y_center_category['Predicted Category'] - bar_height,
                y1=y_center_category['Predicted Category'] + bar_height,
                xref='x',
                yref='y',
                line=dict(color='rgba(255,255,255,0.9)', width=2.5),
                layer='above',
            )

        if category_label or category_id is not None:
            duration = ts['duration_ms']
            label_offset = max(duration * 0.05, 32)
            label_x = ts['end_ms'] - label_offset
            if label_x <= ts['start_ms']:
                label_x = ts['start_ms'] + duration * 0.5

            fig_category.add_annotation(
                x=label_x,
                y='Predicted Category',
                text=display_label,
                showarrow=False,
                xref='x',
                yref='y',
                xanchor='right',
                yanchor='middle',
                align='right',
                bgcolor='rgba(255, 255, 255, 0.9)',
                borderpad=3,
                font=dict(
                    color='rgba(15, 23, 42, 0.95)',
                    size=10,
                    family='Arial Black'
                )
            )

    # Ground truth categories
    for i, ts in enumerate(ground_truth_timestamps):
        category_id = ts.get('category')
        category_label = ts.get('category_label', '')
        if category_id is not None and not category_label:
            category_label = category_mapping.get(category_id, f"Cat_{category_id}")

        display_label = category_label or (f"Cat_{category_id}" if category_id is not None else "Category")

        category_case = category_ground_truth_cases.get(i) if category_ground_truth_cases else None
        category_case_label = case_label_map.get(category_case)
        category_color = palette.get(category_case, ground_truth_category_color) if category_case else ground_truth_category_color

        hover_text = f"<b>{display_label}</b><br>"
        if category_case_label:
            hover_text += f"Case: {category_case_label}<br>"
        elif category_case:
            hover_text += f"Case: {category_case}<br>"
        hover_text += (
            f"Start: {ts['start_ms']}ms<br>"
            f"End: {ts['end_ms']}ms<br>"
            f"Duration: {ts['duration_ms']}ms<extra></extra>"
        )

        fig_category.add_trace(go.Bar(
            name=f"GT Cat: {display_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth Category'],
            orientation='h',
            marker=dict(
                color=category_color,
                line=dict(width=0)
            ),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.55
        ))

        if ts['duration_ms'] > 0:
            fig_category.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=y_center_category['Ground Truth Category'] - bar_height,
                y1=y_center_category['Ground Truth Category'] + bar_height,
                xref='x',
                yref='y',
                line=dict(color='rgba(255,255,255,0.9)', width=2.5),
                layer='above',
            )

        if category_label or category_id is not None:
            duration = ts['duration_ms']
            label_offset = max(duration * 0.05, 32)
            label_x = ts['end_ms'] - label_offset
            if label_x <= ts['start_ms']:
                label_x = ts['start_ms'] + duration * 0.5

            fig_category.add_annotation(
                x=label_x,
                y='Ground Truth Category',
                text=display_label,
                showarrow=False,
                xref='x',
                yref='y',
                xanchor='right',
                yanchor='middle',
                align='right',
                bgcolor='white',
                bordercolor='rgba(15, 23, 42, 0.3)',
                borderwidth=1,
                borderpad=3,
                font=dict(
                    color='black',
                    size=10,
                    family='Arial Black'
                )
            )

    add_inactive_overlay(fig_category, 'Ground Truth Category', bar_width=0.55)

    fig_category.update_layout(
        title=dict(
            text="Category Comparison",
            font=dict(size=15, color='white')
        ),
        xaxis=dict(
            title=dict(text="Time (ms)", font=dict(size=12, color='white')),
            tickfont=dict(size=10, color='white'),
            gridcolor='rgba(255, 255, 255, 0.12)',
            range=[0, max_time]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color='white'),
            categoryorder='array',
            categoryarray=category_axis_labels
        ),
        barmode='overlay',
        height=320,
        bargap=0.22,
        bargroupgap=0.05,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=45, r=45, t=50, b=40)
    )

    st.plotly_chart(fig_gloss, use_container_width=True)
    st.plotly_chart(fig_category, use_container_width=True)


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
    category_accuracy: Optional[float] = None,
    gloss_accuracy: Optional[float] = None
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
        gloss_accuracy: Optional gloss accuracy if ground truth available
    """
    # Enhanced header with file info
    st.markdown(f"### Prediction Results")
    st.markdown(f'<div style="font-size: 1.5rem; font-weight: 600; color: #1f77b4; padding: 0.5rem 0;">{file_name}</div>', unsafe_allow_html=True)
    
    # Check if predicted_sequence is valid
    if predicted_sequence is None:
        predicted_sequence = []
    if predicted_labels is None:
        predicted_labels = []
    
    # Summary metrics with better visual design
    has_categories = predicted_categories is not None and len(predicted_categories) > 0
    has_gloss_acc = gloss_accuracy is not None
    num_cols = 3 + (1 if has_categories else 0) + (1 if has_gloss_acc else 0)
    cols = st.columns(num_cols)
    
    col_idx = 0
    with cols[col_idx]:
        st.metric(
            "Sequence Length", 
            len(predicted_sequence),
            help="Number of predicted glosses in the sequence"
        )
    col_idx += 1
    
    with cols[col_idx]:
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
    col_idx += 1
    
    if has_gloss_acc:
        with cols[col_idx]:
            gloss_acc_color = "normal" if gloss_accuracy >= 0.8 else "off"
            st.metric(
                "Gloss Accuracy", 
                f"{gloss_accuracy*100:.1f}%", 
                delta_color=gloss_acc_color,
                help="Gloss prediction accuracy"
            )
        col_idx += 1
    
    if has_categories:
        with cols[col_idx]:
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
            'Precision': f"{pred.get('precision', 0)*100:.1f}%" if 'precision' in pred else 'N/A',
            'Recall': f"{pred.get('recall', 0)*100:.1f}%" if 'recall' in pred else 'N/A',
            'F1-Score': f"{pred.get('f1_score', 0)*100:.1f}%" if 'f1_score' in pred else 'N/A',
            'TP/FP/FN': f"{pred.get('num_tp', 0)}/{pred.get('num_fp', 0)}/{pred.get('num_fn', 0)}" if 'num_tp' in pred else 'N/A',
            'Avg Confidence': f"{np.mean(pred.get('confidence_scores', [0]))*100:.1f}%" if pred.get('confidence_scores') else 'N/A'
        }
        
        # Add category F1 if available
        if has_categories and 'category_f1_score' in pred:
            row['Cat F1'] = f"{pred['category_f1_score']*100:.1f}%"
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Interactive table
    st.dataframe(df, width='stretch', height=400)
    
    # Overall statistics if ground truth available
    if any('f1_score' in pred for pred in predictions):
        st.markdown("---")
        st.markdown("#### Overall Statistics")
        
        f1_scores = [pred['f1_score'] for pred in predictions if 'f1_score' in pred]
        precisions = [pred['precision'] for pred in predictions if 'precision' in pred]
        recalls = [pred['recall'] for pred in predictions if 'recall' in pred]
        
        cols = st.columns(4 if has_categories else 3)
        
        with cols[0]:
            st.metric("Mean Precision", f"{np.mean(precisions)*100:.1f}%")
        
        with cols[1]:
            st.metric("Mean Recall", f"{np.mean(recalls)*100:.1f}%")
        
        with cols[2]:
            st.metric("Mean F1-Score", f"{np.mean(f1_scores)*100:.1f}%")
        
        if has_categories and len(cols) > 3:
            cat_f1s = [pred['category_f1_score'] for pred in predictions if 'category_f1_score' in pred]
            if cat_f1s:
                with cols[3]:
                    st.metric("Mean Cat F1", f"{np.mean(cat_f1s)*100:.1f}%")

