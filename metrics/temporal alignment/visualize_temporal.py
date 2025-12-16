"""
Visualize temporal alignment per NPZ file with recognition and classification.
Similar to Streamlit app visualization, saved as PNG files.

Usage:
    python "metrics/temporal alignment/visualize_temporal.py"

Input:
    - NPZ files: data/processed/sample/*.npz (temporarily using sample folder)
    - JSON files:
      - metrics/extract/shared_inputs/ctc_validation_results_iv3gru.json
      - metrics/extract/shared_inputs/ctc_validation_results_transformer.json

Output:
    - PNG files saved to: metrics/temporal alignment/visualizations/
    - Format: {filename}_{model_name}_temporal_alignment.png
    - Each image contains both recognition (gloss) and classification (category) 
      temporal alignment visualizations with TP/FP/FN coloring and inactive regions

Features:
    - Shows predicted vs ground truth alignment for both glosses and categories
    - Color-coded by TP (green), FP (red), FN (orange)
    - Displays inactive hand periods with striped pattern
    - Includes gloss/category labels on timeline bars
    - Combines recognition and classification in a single image
"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.labels.label_mapping import load_label_mappings
from streamlit_app.components.ctc_visualization import (
    smooth_timestamps,
    detect_inactive_hand_periods
)
from streamlit_app.components.ctc_case_utils import (
    build_case_maps,
    derive_category_case_maps,
    enrich_ground_truth_timestamps
)


def create_gloss_figure(
    smoothed_predicted: List[Dict],
    ground_truth_timestamps: List[Dict],
    gloss_mapping: Dict[int, str],
    category_mapping: Dict[int, str],
    max_time: float,
    inactive_segments: Dict[str, Dict[str, List]],
    prediction_cases: Dict[int, str],
    ground_truth_cases: Dict[int, str],
    bar_height: float = 0.3,
) -> go.Figure:
    """Create gloss comparison figure."""
    ground_truth_color = 'rgba(16, 185, 129, 0.8)'
    predicted_color = 'rgba(59, 130, 246, 0.8)'
    
    palette = {
        'TP': 'rgba(34, 197, 94, 0.85)',
        'FP': 'rgba(239, 68, 68, 0.9)',
        'FN': 'rgba(249, 115, 22, 0.9)',
    }
    
    case_label_map = {
        'TP': 'True Positive',
        'FP': 'False Positive',
        'FN': 'False Negative',
    }
    
    gloss_row_labels = ['Predicted Gloss', 'Ground Truth Gloss']
    gloss_axis_labels = list(reversed(gloss_row_labels))
    y_center_gloss = {label: idx for idx, label in enumerate(gloss_axis_labels)}
    fig = go.Figure()
    
    # Predicted glosses
    for i, ts in enumerate(smoothed_predicted):
        gloss_index = ts.get('gloss', ts.get('index'))
        gloss_label = gloss_mapping.get(gloss_index, f"Gloss {gloss_index}")
        
        case_index = ts.get('index', i)
        case = prediction_cases.get(case_index) if prediction_cases else None
        pred_color = palette.get(case, predicted_color) if case else predicted_color
        case_label = case_label_map.get(case)
        
        hover_text = (
            f"<b>{gloss_label}</b><br>"
            f"Gloss ID: {gloss_index}<br>"
        )
        if case_label:
            hover_text += f"Case: {case_label}<br>"
        elif case:
            hover_text += f"Case: {case}<br>"
        hover_text += (
            f"Start: {ts['start_ms']}ms<br>"
            f"End: {ts['end_ms']}ms<br>"
            f"Duration: {ts['duration_ms']}ms<extra></extra>"
        )
        
        fig.add_trace(go.Bar(
            name=f"Pred Gloss: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Predicted Gloss'],
            orientation='h',
            marker=dict(color=pred_color, line=dict(width=0)),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.6,
            showlegend=False
        ))
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
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
        
        fig.add_annotation(
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
            font=dict(color='black', size=11, family='Arial Black')
        )
    
    # Ground truth glosses
    for i, ts in enumerate(ground_truth_timestamps):
        gloss_label = ts.get('gloss_label', ts.get('gloss', ''))
        
        gloss_case = ground_truth_cases.get(i) if ground_truth_cases else None
        gloss_case_label = case_label_map.get(gloss_case)
        gloss_color = palette.get(gloss_case, ground_truth_color) if gloss_case else ground_truth_color
        
        hover_text = f"<b>{gloss_label}</b><br>"
        if gloss_case_label:
            hover_text += f"Case: {gloss_case_label}<br>"
        elif gloss_case:
            hover_text += f"Case: {gloss_case}<br>"
        hover_text += (
            f"Start: {ts['start_ms']}ms<br>"
            f"End: {ts['end_ms']}ms<br>"
            f"Duration: {ts['duration_ms']}ms<extra></extra>"
        )
        
        fig.add_trace(go.Bar(
            name=f"GT Gloss: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth Gloss'],
            orientation='h',
            marker=dict(color=gloss_color, line=dict(width=0)),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.6,
            showlegend=False
        ))
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
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
        
        fig.add_annotation(
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
            font=dict(color='black', size=11, family='Arial Black')
        )
    
    # Add inactive overlay
    inactive_pattern = dict(
        shape='/',
        fgcolor='rgba(14, 17, 23, 0.9)',
        bgcolor='rgba(0, 0, 0, 0)',
        size=8,
        solidity=0.35,
        fillmode='replace'
    )
    
    if inactive_segments['Ground Truth Gloss']['bases']:
        fig.add_trace(go.Bar(
            name="Inactive",
            x=inactive_segments['Ground Truth Gloss']['durations'],
            y=['Ground Truth Gloss'] * len(inactive_segments['Ground Truth Gloss']['durations']),
            orientation='h',
            base=inactive_segments['Ground Truth Gloss']['bases'],
            marker=dict(
                color='rgba(0, 0, 0, 0)',
                line=dict(width=0),
                pattern=inactive_pattern
            ),
            hoverinfo='skip',
            showlegend=False,
            width=0.6,
            opacity=1.0
        ))
    
    fig.update_layout(
        title=dict(text="Gloss Comparison", font=dict(size=15)),
        xaxis=dict(
            title=dict(text="Time (ms)", font=dict(size=12)),
            tickfont=dict(size=10),
            gridcolor='rgba(200, 200, 200, 0.3)',
            range=[0, max_time]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12),
            categoryorder='array',
            categoryarray=gloss_axis_labels
        ),
        barmode='overlay',
        height=320,
        bargap=0.22,
        bargroupgap=0.05,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=45, r=45, t=50, b=40)
    )
    
    return fig


def create_category_figure(
    smoothed_predicted: List[Dict],
    ground_truth_timestamps: List[Dict],
    category_mapping: Dict[int, str],
    predicted_categories: Optional[List[int]],
    max_time: float,
    inactive_segments: Dict[str, Dict[str, List]],
    category_prediction_cases: Dict[int, str],
    category_ground_truth_cases: Dict[int, str],
    bar_height: float = 0.3,
) -> go.Figure:
    """Create category comparison figure."""
    ground_truth_category_color = 'rgba(148, 163, 184, 0.65)'
    predicted_category_color = 'rgba(56, 189, 248, 0.7)'
    
    palette = {
        'TP': 'rgba(34, 197, 94, 0.85)',
        'FP': 'rgba(239, 68, 68, 0.9)',
        'FN': 'rgba(249, 115, 22, 0.9)',
    }
    
    case_label_map = {
        'TP': 'True Positive',
        'FP': 'False Positive',
        'FN': 'False Negative',
    }
    
    category_row_labels = ['Predicted Category', 'Ground Truth Category']
    category_axis_labels = list(reversed(category_row_labels))
    y_center_category = {label: idx for idx, label in enumerate(category_axis_labels)}
    fig = go.Figure()
    
    # Predicted categories
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
        
        fig.add_trace(go.Bar(
            name=f"Pred Cat: {display_label}",
            x=[ts['duration_ms']],
            y=['Predicted Category'],
            orientation='h',
            marker=dict(color=category_color),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.55,
            showlegend=False
        ))
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
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
        
        duration = ts['duration_ms']
        label_offset = max(duration * 0.05, 32)
        label_x = ts['end_ms'] - label_offset
        if label_x <= ts['start_ms']:
            label_x = ts['start_ms'] + duration * 0.5
        
        fig.add_annotation(
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
            font=dict(color='rgba(15, 23, 42, 0.95)', size=10, family='Arial Black')
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
        
        fig.add_trace(go.Bar(
            name=f"GT Cat: {display_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth Category'],
            orientation='h',
            marker=dict(color=category_color, line=dict(width=0)),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.55,
            showlegend=False
        ))
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
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
            
            fig.add_annotation(
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
                font=dict(color='black', size=10, family='Arial Black')
            )
    
    # Add inactive overlay
    inactive_pattern = dict(
        shape='/',
        fgcolor='rgba(14, 17, 23, 0.9)',
        bgcolor='rgba(0, 0, 0, 0)',
        size=8,
        solidity=0.35,
        fillmode='replace'
    )
    
    if inactive_segments['Ground Truth Category']['bases']:
        fig.add_trace(go.Bar(
            name="Inactive",
            x=inactive_segments['Ground Truth Category']['durations'],
            y=['Ground Truth Category'] * len(inactive_segments['Ground Truth Category']['durations']),
            orientation='h',
            base=inactive_segments['Ground Truth Category']['bases'],
            marker=dict(
                color='rgba(0, 0, 0, 0)',
                line=dict(width=0),
                pattern=inactive_pattern
            ),
            hoverinfo='skip',
            showlegend=False,
            width=0.55,
            opacity=1.0
        ))
    
    fig.update_layout(
        title=dict(text="Category Comparison", font=dict(size=15)),
        xaxis=dict(
            title=dict(text="Time (ms)", font=dict(size=12)),
            tickfont=dict(size=10),
            gridcolor='rgba(200, 200, 200, 0.3)',
            range=[0, max_time]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12),
            categoryorder='array',
            categoryarray=category_axis_labels
        ),
        barmode='overlay',
        height=320,
        bargap=0.22,
        bargroupgap=0.05,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=45, r=45, t=50, b=40)
    )
    
    return fig


def create_temporal_alignment_figure(
    predicted_timestamps: List[Dict],
    ground_truth_timestamps: List[Dict],
    mask: Optional[np.ndarray] = None,
    timestamps_ms: Optional[np.ndarray] = None,
    predicted_categories: Optional[List[int]] = None,
    prediction_cases: Optional[Dict[int, str]] = None,
    ground_truth_cases: Optional[Dict[int, str]] = None,
    category_prediction_cases: Optional[Dict[int, str]] = None,
    category_ground_truth_cases: Optional[Dict[int, str]] = None,
    temporal_alignment_accuracy: Optional[float] = None,
) -> go.Figure:
    """
    Create temporal alignment visualization figure with both recognition and classification.
    
    Returns:
        Combined plotly figure with gloss and category comparisons
    """
    # Load label mappings
    try:
        gloss_mapping, category_mapping = load_label_mappings()
    except Exception as e:
        # print(f"Warning: Could not load label mappings: {e}. Showing numeric IDs only.")
        gloss_mapping = {}
        category_mapping = {}
    
    # Apply smoothing to predicted timestamps
    smoothed_predicted = smooth_timestamps(predicted_timestamps)
    
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
    
    # Create separate figures
    fig_gloss = create_gloss_figure(
        smoothed_predicted=smoothed_predicted,
        ground_truth_timestamps=ground_truth_timestamps,
        gloss_mapping=gloss_mapping,
        category_mapping=category_mapping,
        max_time=max_time,
        inactive_segments=inactive_segments,
        prediction_cases=prediction_cases or {},
        ground_truth_cases=ground_truth_cases or {},
        bar_height=bar_height,
    )
    
    fig_category = create_category_figure(
        smoothed_predicted=smoothed_predicted,
        ground_truth_timestamps=ground_truth_timestamps,
        category_mapping=category_mapping,
        predicted_categories=predicted_categories,
        max_time=max_time,
        inactive_segments=inactive_segments,
        category_prediction_cases=category_prediction_cases or {},
        category_ground_truth_cases=category_ground_truth_cases or {},
        bar_height=bar_height,
    )
    
    # Combine into subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        subplot_titles=('Gloss Comparison', 'Category Comparison')
    )
    
    # Add traces from gloss figure
    for trace in fig_gloss.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add traces from category figure
    for trace in fig_category.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Add shapes and annotations from gloss figure
    for shape in fig_gloss.layout.shapes:
        fig.add_shape(shape, row=1, col=1)
    
    for annotation in fig_gloss.layout.annotations:
        if annotation.text != "Gloss Comparison":  # Skip title annotation
            fig.add_annotation(annotation, row=1, col=1)
    
    # Add shapes and annotations from category figure
    for shape in fig_category.layout.shapes:
        fig.add_shape(shape, row=2, col=1)
    
    for annotation in fig_category.layout.annotations:
        if annotation.text != "Category Comparison":  # Skip title annotation
            fig.add_annotation(annotation, row=2, col=1)
    
    # Update axes - manually copy properties
    fig.update_xaxes(
        title_text="Time (ms)",
        range=[0, max_time],
        tickfont=dict(size=10),
        gridcolor='rgba(200, 200, 200, 0.3)',
        row=1, col=1
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=list(reversed(['Predicted Gloss', 'Ground Truth Gloss'])),
        tickfont=dict(size=12),
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Time (ms)",
        range=[0, max_time],
        tickfont=dict(size=10),
        gridcolor='rgba(200, 200, 200, 0.3)',
        row=2, col=1
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=list(reversed(['Predicted Category', 'Ground Truth Category'])),
        tickfont=dict(size=12),
        row=2, col=1
    )
    
    # Update overall layout
    title_text = "Temporal Alignment"
    if temporal_alignment_accuracy is not None:
        title_text += f" (Accuracy: {temporal_alignment_accuracy*100:.1f}%)"
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        height=800,
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60),
        showlegend=False
    )
    
    return fig
    
    # --- Gloss Comparison chart (row 1) ---
    gloss_row_labels = ['Predicted Gloss', 'Ground Truth Gloss']
    gloss_y_centers = {'Predicted Gloss': 0, 'Ground Truth Gloss': 1}
    
    for i, ts in enumerate(smoothed_predicted):
        gloss_index = ts.get('gloss', ts.get('index'))
        gloss_label = gloss_mapping.get(gloss_index, f"Gloss {gloss_index}")
        category_id = ts.get('category', None)
        category_label = ts.get('category_label', '')
        
        if category_id is not None and not category_label:
            category_label = category_mapping.get(category_id, f"Cat_{category_id}")
        
        case_index = ts.get('index', i)
        case = prediction_cases.get(case_index) if prediction_cases else None
        pred_color = palette.get(case, predicted_color) if case else predicted_color
        case_label = case_label_map.get(case)
        
        hover_text = (
            f"<b>{gloss_label}</b><br>"
            f"Gloss ID: {gloss_index}<br>"
        )
        if category_label:
            hover_text += f"Category: {category_label}<br>"
        if case_label:
            hover_text += f"Case: {case_label}<br>"
        elif case:
            hover_text += f"Case: {case}<br>"
        hover_text += (
            f"Start: {ts['start_ms']}ms<br>"
            f"End: {ts['end_ms']}ms<br>"
            f"Duration: {ts['duration_ms']}ms<extra></extra>"
        )
        
        fig.add_trace(go.Bar(
            name=f"Pred Gloss: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Predicted Gloss'],
            orientation='h',
            marker=dict(color=pred_color, line=dict(width=0)),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.6,
            showlegend=False
        ), row=1, col=1)
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=gloss_y_centers['Predicted Gloss'] - bar_height,
                y1=gloss_y_centers['Predicted Gloss'] + bar_height,
                xref='x',
                yref='y',
                line=dict(color='rgba(255,255,255,0.95)', width=3),
                layer='above',
                row=1, col=1
            )
        
        annotation_text = gloss_label
        duration = ts['duration_ms']
        label_offset = max(duration * 0.05, 40)
        label_x = ts['end_ms'] - label_offset
        if label_x <= ts['start_ms']:
            label_x = ts['start_ms'] + duration * 0.5
        
        fig.add_annotation(
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
            font=dict(color='black', size=11, family='Arial Black'),
            row=1, col=1
        )
    
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
        
        fig.add_trace(go.Bar(
            name=f"GT Gloss: {gloss_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth Gloss'],
            orientation='h',
            marker=dict(color=gloss_color, line=dict(width=0)),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.6,
            showlegend=False
        ), row=1, col=1)
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=gloss_y_centers['Ground Truth Gloss'] - bar_height,
                y1=gloss_y_centers['Ground Truth Gloss'] + bar_height,
                xref='x',
                yref='y',
                line=dict(color='rgba(255,255,255,0.95)', width=3),
                layer='above',
                row=1, col=1
            )
        
        annotation_text = gloss_label
        duration = ts['duration_ms']
        label_offset = max(duration * 0.05, 40)
        label_x = ts['end_ms'] - label_offset
        if label_x <= ts['start_ms']:
            label_x = ts['start_ms'] + duration * 0.5
        
        fig.add_annotation(
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
            font=dict(color='black', size=11, family='Arial Black'),
            row=1, col=1
        )
    
    # Add inactive overlay for gloss
    if inactive_segments['Ground Truth Gloss']['bases']:
        fig.add_trace(go.Bar(
            name="Inactive",
            x=inactive_segments['Ground Truth Gloss']['durations'],
            y=['Ground Truth Gloss'] * len(inactive_segments['Ground Truth Gloss']['durations']),
            orientation='h',
            base=inactive_segments['Ground Truth Gloss']['bases'],
            marker=dict(
                color='rgba(0, 0, 0, 0)',
                line=dict(width=0),
                pattern=inactive_pattern
            ),
            hoverinfo='skip',
            showlegend=False,
            width=0.6,
            opacity=1.0
        ), row=1, col=1)
    
    # Update gloss subplot layout
    gloss_axis_labels = list(reversed(gloss_row_labels))
    fig.update_xaxes(
        title_text="Time (ms)",
        range=[0, max_time],
        row=1, col=1
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=gloss_axis_labels,
        row=1, col=1
    )
    
    # --- Category Comparison chart (row 2) ---
    category_row_labels = ['Predicted Category', 'Ground Truth Category']
    category_y_centers = {'Predicted Category': 0, 'Ground Truth Category': 1}
    
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
        
        fig.add_trace(go.Bar(
            name=f"Pred Cat: {display_label}",
            x=[ts['duration_ms']],
            y=['Predicted Category'],
            orientation='h',
            marker=dict(color=category_color),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.55,
            showlegend=False
        ), row=2, col=1)
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=category_y_centers['Predicted Category'] - bar_height,
                y1=category_y_centers['Predicted Category'] + bar_height,
                xref='x2',
                yref='y2',
                line=dict(color='rgba(255,255,255,0.9)', width=2.5),
                layer='above',
                row=2, col=1
            )
        
        duration = ts['duration_ms']
        label_offset = max(duration * 0.05, 32)
        label_x = ts['end_ms'] - label_offset
        if label_x <= ts['start_ms']:
            label_x = ts['start_ms'] + duration * 0.5
        
        fig.add_annotation(
            x=label_x,
            y='Predicted Category',
            text=display_label,
            showarrow=False,
            xref='x2',
            yref='y2',
            xanchor='right',
            yanchor='middle',
            align='right',
            bgcolor='rgba(255, 255, 255, 0.9)',
            borderpad=3,
            font=dict(color='rgba(15, 23, 42, 0.95)', size=10, family='Arial Black'),
            row=2, col=1
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
        
        fig.add_trace(go.Bar(
            name=f"GT Cat: {display_label}",
            x=[ts['duration_ms']],
            y=['Ground Truth Category'],
            orientation='h',
            marker=dict(color=category_color, line=dict(width=0)),
            textposition='none',
            hovertemplate=hover_text,
            base=ts['start_ms'],
            width=0.55,
            showlegend=False
        ), row=2, col=1)
        
        if ts['duration_ms'] > 0:
            fig.add_shape(
                type='line',
                x0=ts['end_ms'],
                x1=ts['end_ms'],
                y0=category_y_centers['Ground Truth Category'] - bar_height,
                y1=category_y_centers['Ground Truth Category'] + bar_height,
                xref='x2',
                yref='y2',
                line=dict(color='rgba(255,255,255,0.9)', width=2.5),
                layer='above',
                row=2, col=1
            )
        
        if category_label or category_id is not None:
            duration = ts['duration_ms']
            label_offset = max(duration * 0.05, 32)
            label_x = ts['end_ms'] - label_offset
            if label_x <= ts['start_ms']:
                label_x = ts['start_ms'] + duration * 0.5
            
            fig.add_annotation(
                x=label_x,
                y='Ground Truth Category',
                text=display_label,
                showarrow=False,
                xref='x2',
                yref='y2',
                xanchor='right',
                yanchor='middle',
                align='right',
                bgcolor='white',
                bordercolor='rgba(15, 23, 42, 0.3)',
                borderwidth=1,
                borderpad=3,
                font=dict(color='black', size=10, family='Arial Black'),
                row=2, col=1
            )
    
    # Add inactive overlay for category
    if inactive_segments['Ground Truth Category']['bases']:
        fig.add_trace(go.Bar(
            name="Inactive",
            x=inactive_segments['Ground Truth Category']['durations'],
            y=['Ground Truth Category'] * len(inactive_segments['Ground Truth Category']['durations']),
            orientation='h',
            base=inactive_segments['Ground Truth Category']['bases'],
            marker=dict(
                color='rgba(0, 0, 0, 0)',
                line=dict(width=0),
                pattern=inactive_pattern
            ),
            hoverinfo='skip',
            showlegend=False,
            width=0.55,
            opacity=1.0
        ), row=2, col=1)
    
    # Update category subplot layout
    category_axis_labels = list(reversed(category_row_labels))
    fig.update_xaxes(
        title_text="Time (ms)",
        range=[0, max_time],
        row=2, col=1
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=category_axis_labels,
        row=2, col=1
    )
    
    # Update overall layout
    title_text = "Temporal Alignment"
    if temporal_alignment_accuracy is not None:
        title_text += f" (Accuracy: {temporal_alignment_accuracy*100:.1f}%)"
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        height=800,
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def process_file(
    pred_data: Dict[str, Any],
    npz_path: Path,
    output_path: Path
) -> bool:
    """Process a single prediction and save visualization."""
    try:
        # Load NPZ file
        npz_data = np.load(npz_path, allow_pickle=False)
        mask = None
        timestamps_ms = None
        
        try:
            if 'mask' in npz_data:
                mask = np.array(npz_data['mask'])
                if mask.dtype != bool:
                    mask = mask.astype(bool)
            if 'timestamps_ms' in npz_data:
                timestamps_ms = np.array(npz_data['timestamps_ms'])
            elif 'timestamps' in npz_data:
                timestamps_ms = np.array(npz_data['timestamps'])
        finally:
            npz_data.close()
        
        # Extract data from prediction
        predicted_sequence = pred_data.get('predicted_sequence', [])
        predicted_labels = pred_data.get('predicted_labels', [])
        predicted_timestamps = pred_data.get('predicted_timestamps', [])
        predicted_categories = pred_data.get('predicted_categories', [])
        
        ground_truth_sequence = pred_data.get('ground_truth_sequence', [])
        ground_truth_labels = pred_data.get('ground_truth_labels', [])
        ground_truth_timestamps = pred_data.get('ground_truth_timestamps', [])
        ground_truth_categories = pred_data.get('ground_truth_categories', [])
        ground_truth_category_labels = pred_data.get('ground_truth_category_labels', [])
        
        if not predicted_timestamps or not ground_truth_timestamps:
            print(f"Skipping {pred_data.get('file_name')}: Missing timestamps")
            return False
        
        # Build case maps
        metrics = {
            'matched_pairs': pred_data.get('matched_pairs', []),
            'unmatched_predictions': pred_data.get('unmatched_predictions', []),
            'unmatched_ground_truth': pred_data.get('unmatched_ground_truth', []),
            'tp_indices': pred_data.get('tp_indices', []),
            'fp_indices': pred_data.get('fp_indices', []),
            'fn_indices': pred_data.get('fn_indices', []),
            'confidence_threshold': pred_data.get('confidence_threshold', 0.5),
        }
        
        confidence_scores = pred_data.get('confidence_scores', [])
        category_confidences = pred_data.get('category_confidences', [])
        
        prediction_case_map, ground_truth_case_map, category_prediction_case_map, category_ground_truth_case_map = build_case_maps(
            metrics=metrics,
            predicted_sequence=predicted_sequence,
            ground_truth_sequence=ground_truth_sequence,
            confidence_scores=confidence_scores,
            predicted_categories=predicted_categories,
            ground_truth_categories=ground_truth_categories,
            category_confidences=category_confidences,
            confidence_threshold=metrics.get('confidence_threshold'),
        )
        
        # Enrich ground truth timestamps
        enriched_ground_truth_timestamps = enrich_ground_truth_timestamps(
            timestamps=ground_truth_timestamps,
            gloss_labels=ground_truth_labels,
            gloss_sequence=ground_truth_sequence,
            category_ids=ground_truth_categories,
            category_labels=ground_truth_category_labels,
        )
        
        # Create visualization
        fig = create_temporal_alignment_figure(
            predicted_timestamps=predicted_timestamps,
            ground_truth_timestamps=enriched_ground_truth_timestamps,
            mask=mask,
            timestamps_ms=timestamps_ms,
            predicted_categories=predicted_categories,
            prediction_cases=prediction_case_map,
            ground_truth_cases=ground_truth_case_map,
            category_prediction_cases=category_prediction_case_map,
            category_ground_truth_cases=category_ground_truth_case_map,
            temporal_alignment_accuracy=pred_data.get('temporal_alignment_accuracy'),
        )
        
        # Save as PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path), width=1600, height=800, scale=2)
        print(f"Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {pred_data.get('file_name', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all files."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    npz_dir = base_dir / "data" / "processed" / "sample"
    json_files = [
        base_dir / "metrics" / "extract" / "shared_inputs" / "ctc_validation_results_iv3gru.json",
        base_dir / "metrics" / "extract" / "shared_inputs" / "ctc_validation_results_transformer.json",
    ]
    output_dir = base_dir / "metrics" / "temporal alignment" / "visualizations"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each JSON file
    for json_path in json_files:
        if not json_path.exists():
            print(f"Warning: JSON file not found: {json_path}")
            continue
        
        print(f"\nProcessing: {json_path.name}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = data.get('predictions', [])
        model_name = json_path.stem.replace('ctc_validation_results_', '')
        
        # Process each prediction
        for pred in predictions:
            file_name = pred.get('file_name')
            if not file_name:
                continue
            
            # Find NPZ file
            npz_path = npz_dir / file_name
            if not npz_path.exists():
                # print(f"Warning: NPZ file not found: {npz_path}")
                continue
            
            # Output path
            output_filename = f"{Path(file_name).stem}_{model_name}_temporal_alignment.png"
            output_path = output_dir / output_filename
            
            # Process and save
            process_file(pred, npz_path, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

