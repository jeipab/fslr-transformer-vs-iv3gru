"""
Usage:
    python "metrics\sequence breakdown\create_performance_distributions.py"
    
    Or from the project root:
    python "metrics\sequence breakdown\create_performance_distributions.py"
    
Requirements:
    - matplotlib
    - pandas
    - numpy
    - seaborn (for KDE plots)
    
Input Files (must be in same directory):
    - Compiled Data Summary - TABLE E2 — Per-Sequence Recognition Performance - T.csv
    - Compiled Data Summary - TABLE E2 — Per-Sequence Recognition Performance - I.csv
    - Compiled Data Summary - TABLE E4 — Per-Sequence Classification Performance - T.csv
    - Compiled Data Summary - TABLE E4 — Per-Sequence Classification Performance - I.csv
    
Output:
    Generates 6 PNG distribution plots in the same directory:
    - recognition_precision_distribution.png
    - recognition_recall_distribution.png
    - recognition_f1_distribution.png
    - classification_precision_distribution.png
    - classification_recall_distribution.png
    - classification_f1_distribution.png
    
    Charts show distribution comparison:
    - Side-by-side histograms/KDE for Transformer vs IV3-GRU
    - Shows overall performance distribution for each model
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import seaborn as sns

# Add parent directory to path to import shared color palette
sys.path.insert(0, str(Path(__file__).parent.parent))
from color_palette import COLOR_TRANSFORMER, COLOR_IV3_GRU

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

def load_sequence_data(transformer_file, iv3_file):
    """Load per-sequence data from both model files."""
    # Read CSV files, skipping first row (model name)
    df_t = pd.read_csv(transformer_file, skiprows=1, header=0)
    df_i = pd.read_csv(iv3_file, skiprows=1, header=0)
    
    # Clean column names
    df_t.columns = df_t.columns.str.strip()
    df_i.columns = df_i.columns.str.strip()
    
    # Convert percentage metrics to decimal (0-1)
    metric_cols = ['Precision', 'Recall', 'F1-score']
    for col in metric_cols:
        if col in df_t.columns:
            # Remove % and convert to decimal
            df_t[col] = df_t[col].astype(str).str.rstrip('%').astype(float) / 100.0
        if col in df_i.columns:
            df_i[col] = df_i[col].astype(str).str.rstrip('%').astype(float) / 100.0
    
    return df_t, df_i

def create_distribution_plot(df_t, df_i, metric, title_prefix, filename_prefix):
    """
    Create a distribution plot comparing Transformer vs IV3-GRU for a specific metric.
    
    Parameters:
    df_t: Transformer dataframe
    df_i: IV3-GRU dataframe
    metric: Metric name ('Precision', 'Recall', or 'F1-score')
    title_prefix: Prefix for chart title
    filename_prefix: Prefix for output filename
    """
    if metric not in df_t.columns or metric not in df_i.columns:
        print(f"  Warning: Missing {metric} column")
        return
    
    # Extract metric values
    values_t = df_t[metric].dropna()
    values_i = df_i[metric].dropna()
    
    if len(values_t) == 0 or len(values_i) == 0:
        print(f"  Warning: No valid data for {metric}")
        return
    
    print(f"  Plotting {len(values_t)} Transformer and {len(values_i)} IV3-GRU sequences for {metric}...")
    sys.stdout.flush()
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create histogram
    # Use transparency and overlapping histograms
    ax.hist(values_t, bins=30, alpha=0.6, color=COLOR_TRANSFORMER, 
           label='Transformer', edgecolor='black', linewidth=0.5)
    ax.hist(values_i, bins=30, alpha=0.6, color=COLOR_IV3_GRU, 
           label='IV3-GRU', edgecolor='black', linewidth=0.5)
    
    # Customize chart
    ax.set_xlabel(metric, fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{title_prefix}: {metric} Distribution Comparison', fontweight='bold', pad=20)
    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
             framealpha=0.95)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{filename_prefix}_{metric.lower().replace("-", "_")}_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

def main():
    try:
        # File paths
        rec_t_file = os.path.join(output_dir, 'Compiled Data Summary - TABLE E2 — Per-Sequence Recognition Performance - T.csv')
        rec_i_file = os.path.join(output_dir, 'Compiled Data Summary - TABLE E2 — Per-Sequence Recognition Performance - I.csv')
        cls_t_file = os.path.join(output_dir, 'Compiled Data Summary - TABLE E4 — Per-Sequence Classification Performance - T.csv')
        cls_i_file = os.path.join(output_dir, 'Compiled Data Summary - TABLE E4 — Per-Sequence Classification Performance - I.csv')
        
        print("=" * 60)
        print("Creating Performance Distribution Plots")
        print("=" * 60)
        print(f"Working directory: {output_dir}")
        sys.stdout.flush()
        
        # Load recognition data
        print("\nLoading recognition data...")
        sys.stdout.flush()
        df_rec_t, df_rec_i = load_sequence_data(rec_t_file, rec_i_file)
        print(f"✓ Loaded {len(df_rec_t)} Transformer recognition sequences")
        print(f"✓ Loaded {len(df_rec_i)} IV3-GRU recognition sequences")
        sys.stdout.flush()
        
        # Create recognition distribution plots
        print("\nCreating recognition distribution plots...")
        sys.stdout.flush()
        for metric in ['Precision', 'Recall', 'F1-score']:
            create_distribution_plot(df_rec_t, df_rec_i, metric, 
                                   'Recognition Performance', 'recognition')
        
        # Load classification data
        print("\nLoading classification data...")
        sys.stdout.flush()
        df_cls_t, df_cls_i = load_sequence_data(cls_t_file, cls_i_file)
        print(f"✓ Loaded {len(df_cls_t)} Transformer classification sequences")
        print(f"✓ Loaded {len(df_cls_i)} IV3-GRU classification sequences")
        sys.stdout.flush()
        
        # Create classification distribution plots
        print("\nCreating classification distribution plots...")
        sys.stdout.flush()
        for metric in ['Precision', 'Recall', 'F1-score']:
            create_distribution_plot(df_cls_t, df_cls_i, metric, 
                                   'Classification Performance', 'classification')
        
        print("\n" + "=" * 60)
        print("✓ All distribution plots created successfully!")
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

