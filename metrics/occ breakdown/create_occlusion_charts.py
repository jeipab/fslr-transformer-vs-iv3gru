"""
Usage:
    python "metrics\occ breakdown\create_occlusion_charts.py"
    
    Or from the project root:
    python "metrics\occ breakdown\create_occlusion_charts.py"
    
Requirements:
    - matplotlib
    - pandas
    - numpy
    
Output:
    Generates two PNG charts in the same directory:
    - recognition_occlusion_comparison.png
    - classification_occlusion_comparison.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path to import shared color palette
sys.path.insert(0, str(Path(__file__).parent.parent))
from color_palette import (
    COLOR_TRANSFORMER_OCC, COLOR_TRANSFORMER_NON,
    COLOR_IV3_GRU_OCC, COLOR_IV3_GRU_NON
)

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

def create_comparison_chart(df, title, filename):
    """
    Create a grouped bar chart comparing Transformer vs IV3-GRU metrics.
    
    Parameters:
    df: DataFrame with Model, Occlusion, Precision, Recall, F1-score columns
    title: Chart title
    filename: Output filename
    """
    # Separate data by model
    transformer_data = df[df['Model'] == 'Transformer']
    iv3_gru_data = df[df['Model'] == 'IV3-GRU']
    
    # Extract metrics
    metrics = ['Precision', 'Recall', 'F1-score']
    occlusion_types = ['Occluded', 'Non-Occluded']
    
    # Prepare data for grouped bars
    x = np.arange(len(metrics))
    width = 0.2  # Width of individual bars
    gap = 0.05   # Gap between groups
    
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Create grouped bars: for each metric, show 4 bars grouped by model
    for i, metric in enumerate(metrics):
        # Calculate positions for 4 bars per metric group
        base_pos = i
        positions = [
            base_pos - 1.5*width - gap,      # Transformer Occluded
            base_pos - 0.5*width - gap/2,    # Transformer Non-Occluded
            base_pos + 0.5*width + gap/2,    # IV3-GRU Occluded
            base_pos + 1.5*width + gap       # IV3-GRU Non-Occluded
        ]
        
        # Get values
        trans_occ = transformer_data[transformer_data['Occlusion'] == 'Occluded'][metric].values[0]
        trans_non = transformer_data[transformer_data['Occlusion'] == 'Non-Occluded'][metric].values[0]
        iv3_occ = iv3_gru_data[iv3_gru_data['Occlusion'] == 'Occluded'][metric].values[0]
        iv3_non = iv3_gru_data[iv3_gru_data['Occlusion'] == 'Non-Occluded'][metric].values[0]
        
        values = [trans_occ, trans_non, iv3_occ, iv3_non]
        colors = [COLOR_TRANSFORMER_OCC, COLOR_TRANSFORMER_NON, COLOR_IV3_GRU_OCC, COLOR_IV3_GRU_NON]
        
        # Plot bars
        bars = []
        for pos, val, color in zip(positions, values, colors):
            bar = ax.bar(pos, val, width, color=color, 
                        edgecolor='black', linewidth=0.8, zorder=2)
            bars.append(bar[0])
            
            # Add value labels
            ax.text(pos, val + 0.01, f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    # Customize chart
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Create custom legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_TRANSFORMER_OCC, edgecolor='black', 
                 linewidth=0.8, label='Transformer (Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_TRANSFORMER_NON, edgecolor='black', 
                 linewidth=0.8, label='Transformer (Non-Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_IV3_GRU_OCC, edgecolor='black', 
                 linewidth=0.8, label='IV3-GRU (Occluded)'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_IV3_GRU_NON, edgecolor='black', 
                 linewidth=0.8, label='IV3-GRU (Non-Occluded)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
             fancybox=True, shadow=True, framealpha=0.95)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    sys.stdout.flush()

def main():
    try:
        # Read CSV files
        recognition_file = os.path.join(output_dir, 'Compiled Data Summary - TABLE B2 — Recognition Occlusion Breakdown.csv')
        classification_file = os.path.join(output_dir, 'Compiled Data Summary - TABLE B4 — Classification Occlusion Breakdown.csv')
        
        print(f"Reading recognition file: {recognition_file}")
        sys.stdout.flush()
        print(f"Reading classification file: {classification_file}")
        sys.stdout.flush()
        
        # Load data
        df_recognition = pd.read_csv(recognition_file)
        df_classification = pd.read_csv(classification_file)
        
        # Forward-fill empty Model values (for continuation rows)
        df_recognition['Model'] = df_recognition['Model'].ffill()
        df_classification['Model'] = df_classification['Model'].ffill()
        
        print(f"Recognition data shape: {df_recognition.shape}")
        sys.stdout.flush()
        print(f"Classification data shape: {df_classification.shape}")
        sys.stdout.flush()
        
        # Create charts
        print("\nCreating recognition chart...")
        sys.stdout.flush()
        create_comparison_chart(
            df_recognition, 
            'Recognition Performance: Transformer vs IV3-GRU',
            'recognition_occlusion_comparison.png'
        )
        
        print("Creating classification chart...")
        sys.stdout.flush()
        create_comparison_chart(
            df_classification, 
            'Classification Performance: Transformer vs IV3-GRU',
            'classification_occlusion_comparison.png'
        )
        
        print("\nCharts created successfully!")
        sys.stdout.flush()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()

