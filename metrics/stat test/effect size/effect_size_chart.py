"""
Usage:
    python "metrics\stat test\effect size\effect_size_chart.py"

Requirements:
    - matplotlib
    - pandas
    - numpy

Input Files (expected in subdirectories):
    - recognition/Recognition-StatsResults.csv
    - classification/Classification-StatsResults.csv

Output:
    Generates two PNG charts in the effect size subdirectory:
    - recognition_effect_sizes.png
    - classification_effect_sizes.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
import colorsys
from pathlib import Path

# Color palette for effect sizes based on magnitude
COLOR_LARGE_EFFECT = '#d32f2f'    # Red for large effect
COLOR_MEDIUM_EFFECT = '#f57c00'   # Orange for medium effect
COLOR_SMALL_EFFECT = '#2e7d32'    # Green for small effect

# Set academic style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3
})

# Directory for output
output_dir = os.path.dirname(os.path.abspath(__file__))
effect_size_dir = os.path.join(output_dir, 'effect size')
os.makedirs(effect_size_dir, exist_ok=True)




def parse_effect_size(effect_str):
    """Extract numeric effect size value and magnitude from formatted string."""
    if pd.isna(effect_str) or effect_str == 'N/A':
        return None, None
    
    # Match patterns like "d = 1.83 (large)" or "r = 0.42 (medium)"
    # Pattern: letter (d or r), optional whitespace, =, optional whitespace, number, optional whitespace, (magnitude)
    match = re.search(r'[dr]\s*=\s*(\d+\.?\d*)\s*\((\w+)\)', str(effect_str))
    if match:
        try:
            effect_value = float(match.group(1))
            magnitude = match.group(2).lower()  # small, medium, large
            return abs(effect_value), magnitude
        except ValueError:
            return None, None
    
    # Fallback: try to extract just the value
    match = re.search(r'[dr]\s*=\s*(\d+\.?\d*)', str(effect_str))
    if match:
        try:
            effect_value = float(match.group(1))
            # Determine magnitude based on value if not provided
            if abs(effect_value) >= 0.8:  # For both d and r, large is >= 0.8
                magnitude = 'large'
            elif abs(effect_value) >= 0.5:  # Medium is >= 0.5
                magnitude = 'medium'
            else:
                magnitude = 'small'
            return abs(effect_value), magnitude
        except ValueError:
            return None, None
    
    return None, None


def darken_color(color_hex, factor=0.35):
    """Darken a hex color by a factor (0-1)."""
    color_hex = color_hex.lstrip('#')
    r, g, b = [int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(0, v * (1 - factor))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


def lighten_color(color_hex, factor=0.25):
    """Lighten a hex color by a factor (0-1)."""
    color_hex = color_hex.lstrip('#')
    r, g, b = [int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = min(1, v + (1 - v) * factor)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


def get_color_for_effect_size(magnitude, occlusion):
    """
    Get color based on effect size magnitude with tint for occlusion.
    
    Parameters:
    magnitude: 'small', 'medium', or 'large'
    occlusion: 'Occluded' or 'Non-Occluded'
    
    Returns:
    Hex color string
    """
    # Base color based on magnitude
    if magnitude == 'large':
        base_color = COLOR_LARGE_EFFECT  # Red
    elif magnitude == 'medium':
        base_color = COLOR_MEDIUM_EFFECT  # Orange
    elif magnitude == 'small':
        base_color = COLOR_SMALL_EFFECT  # Green
    else:
        return '#757575'  # Gray for unknown
    
    # Apply tint based on occlusion - make distinction more apparent
    if occlusion == 'Occluded':
        return darken_color(base_color, 0.3)  # Much darker for occluded
    else:  # Non-Occluded
        return lighten_color(base_color, 0.2)  # Much lighter for non-occluded


def create_effect_size_chart(df, task_name, filename):
    """
    Create a grouped bar chart showing effect sizes for each metric and occlusion condition.
    
    Parameters:
    df: DataFrame with metric, effect_size columns
    task_name: Name of the task (e.g., "Recognition", "Classification")
    filename: Output filename
    """
    # Parse data
    chart_data = []
    metrics = ['Precision', 'Recall', 'F1-score']
    occlusion_types = ['Occluded', 'Nonoccluded']
    
    for occ in occlusion_types:
        for metric in metrics:
            # Find matching row
            row = df[df['metric'].str.contains(occ, case=False, na=False) & 
                     df['metric'].str.contains(metric, case=False, na=False)]
            
            if len(row) > 0:
                row = row.iloc[0]
                effect_value, magnitude = parse_effect_size(row['effect_size'])
                
                if effect_value is not None:
                    # Normalize occlusion name
                    occ_normalized = 'Occluded' if 'occluded' in occ.lower() and 'non' not in occ.lower() else 'Non-Occluded'
                    
                    # Store effect size data
                    chart_data.append({
                        'Metric': metric,
                        'Occlusion': occ_normalized,
                        'Effect Size': effect_value,
                        'Magnitude': magnitude
                    })
    
    if not chart_data:
        print(f"Warning: No effect size data found for {task_name}")
        return
    
    chart_df = pd.DataFrame(chart_data)
    
    # Prepare data for grouped bars
    x = np.arange(len(metrics))
    width = 0.35  # Width of individual bars
    gap = 0.05   # Gap between groups
    
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Create grouped bars: for each metric, show 2 bars (Occluded and Non-Occluded)
    for i, metric in enumerate(metrics):
        base_pos = i
        
        # Get data for this metric
        metric_data = chart_df[chart_df['Metric'] == metric]
        
        # Positions for two bars
        positions = [
            base_pos - width/2 - gap/2,  # Occluded
            base_pos + width/2 + gap/2   # Non-Occluded
        ]
        
        # Get values and colors
        values = []
        colors = []
        
        for occ in ['Occluded', 'Non-Occluded']:
            row = metric_data[metric_data['Occlusion'] == occ]
            if len(row) > 0:
                row = row.iloc[0]
                values.append(row['Effect Size'])
                colors.append(get_color_for_effect_size(row['Magnitude'], occ))
            else:
                values.append(0)
                colors.append('#757575')  # Gray for missing data
        
        # Plot bars
        for pos, val, color in zip(positions, values, colors):
            bar = ax.bar(pos, val, width, color=color, 
                        edgecolor='black', linewidth=0.8, zorder=2)
            
            # Add value labels
            if val > 0:
                ax.text(pos, val + 0.02, f'{val:.2f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    # Customize chart
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Effect Size (|d| or |r|)', fontweight='bold')
    ax.set_title(f'{task_name}: Effect Sizes Across Metrics', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    max_effect = max(chart_df['Effect Size']) if len(chart_df) > 0 else 1.0
    ax.set_ylim(0, max_effect * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Create custom legend
    from matplotlib.patches import Rectangle
    # Generate example colors for legend
    large_occ = get_color_for_effect_size('large', 'Occluded')
    large_non = get_color_for_effect_size('large', 'Non-Occluded')
    medium_occ = get_color_for_effect_size('medium', 'Occluded')
    medium_non = get_color_for_effect_size('medium', 'Non-Occluded')
    small_occ = get_color_for_effect_size('small', 'Occluded')
    small_non = get_color_for_effect_size('small', 'Non-Occluded')
    
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=large_occ, edgecolor='black', 
                 linewidth=0.8, label='Large Effect (Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=large_non, edgecolor='black', 
                 linewidth=0.8, label='Large Effect (Non-Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=medium_occ, edgecolor='black', 
                 linewidth=0.8, label='Medium Effect (Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=medium_non, edgecolor='black', 
                 linewidth=0.8, label='Medium Effect (Non-Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=small_occ, edgecolor='black', 
                 linewidth=0.8, label='Small Effect (Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=small_non, edgecolor='black', 
                 linewidth=0.8, label='Small Effect (Non-Occluded)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
             fancybox=True, shadow=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(effect_size_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    sys.stdout.flush()


def main():
    """Main function to generate effect size charts."""
    try:
        print("=" * 60)
        print("Creating Effect Size Charts")
        print("=" * 60)
        print(f"Working directory: {output_dir}")
        sys.stdout.flush()
        
        # Load data
        recognition_file = os.path.join(output_dir, 'recognition', 'Recognition-StatsResults.csv')
        classification_file = os.path.join(output_dir, 'classification', 'Classification-StatsResults.csv')
        
        if not os.path.exists(recognition_file):
            raise FileNotFoundError(f"Recognition file not found: {recognition_file}")
        if not os.path.exists(classification_file):
            raise FileNotFoundError(f"Classification file not found: {classification_file}")
        
        print("\nLoading statistical results...")
        sys.stdout.flush()
        df_recognition = pd.read_csv(recognition_file)
        df_classification = pd.read_csv(classification_file)
        
        print(f"✓ Loaded {len(df_recognition)} recognition results")
        print(f"✓ Loaded {len(df_classification)} classification results")
        sys.stdout.flush()
        
        # Create charts
        print("\nCreating recognition effect size chart...")
        sys.stdout.flush()
        create_effect_size_chart(
            df_recognition,
            'Recognition',
            'recognition_effect_sizes.png'
        )
        
        print("Creating classification effect size chart...")
        sys.stdout.flush()
        create_effect_size_chart(
            df_classification,
            'Classification',
            'classification_effect_sizes.png'
        )
        
        print("\n" + "=" * 60)
        print("✓ All effect size charts created successfully!")
        print("=" * 60)
        sys.stdout.flush()
        
    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()
