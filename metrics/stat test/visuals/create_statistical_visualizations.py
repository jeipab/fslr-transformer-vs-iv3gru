"""
Statistical Analysis Visualization Generator

Usage:
    python "metrics\stat test\visuals\create_statistical_visualizations.py"

Requirements:
    - matplotlib
    - pandas
    - numpy
    - scipy

Input Files (expected in subdirectories):
    - classification/Classification-StatsResults.csv
    - recognition/Recognition-StatsResults.csv
    - classification/Classification-Breakdown.csv
    - recognition/Recognition-Breakdown.csv

Output:
    Generates four PNG visualizations in the same directory:
    - statistical_effect_sizes.png
    - statistical_results_summary.png
    - statistical_normality_assessment.png
    - statistical_normality_distributions.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import shapiro, norm
import numpy as np
import os
import sys
import re
from pathlib import Path

# Add parent directory to path to import shared color palette
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from color_palette import (
    COLOR_TRANSFORMER, COLOR_IV3_GRU,
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

# Color scheme for significance and effect sizes
COLOR_SIGNIFICANT = '#2e7d32'  # Green
COLOR_NON_SIGNIFICANT = '#757575'  # Gray
COLOR_LARGE_EFFECT = '#d32f2f'  # Red
COLOR_MEDIUM_EFFECT = '#f57c00'  # Orange
COLOR_SMALL_EFFECT = '#1976d2'  # Blue


def parse_effect_size(effect_str):
    """Extract numeric effect size value from formatted string."""
    if pd.isna(effect_str) or effect_str == 'N/A':
        return None, None
    
    # Match patterns like "d = 0.57 (medium)" or "r = 0.89 (large)"
    match = re.search(r'([dr])\s*=\s*([\d.]+)', str(effect_str))
    if match:
        effect_type = match.group(1)
        effect_value = float(match.group(2))
        return effect_value, effect_type
    return None, None


def parse_p_value(p_str):
    """Extract numeric p-value and significance status."""
    if pd.isna(p_str) or p_str == 'N/A':
        return None, None
    
    p_str = str(p_str).lower()
    
    # Check for non-significant
    is_significant = '(ns)' not in p_str
    
    # Extract numeric value
    # Handle scientific notation like "3.82×10⁻⁵"
    if '×10' in p_str or 'x10' in p_str:
        # Scientific notation
        match = re.search(r'([\d.]+)\s*[×x]\s*10\s*[⁻⁻-]?\s*(\d+)', p_str)
        if match:
            mantissa = float(match.group(1))
            exponent = int(match.group(2))
            p_value = mantissa * (10 ** -exponent)
            return p_value, is_significant
    else:
        # Regular decimal notation
        match = re.search(r'p\s*=\s*\.?(\d+)', p_str)
        if match:
            p_value = float('0.' + match.group(1))
            return p_value, is_significant
    
    return None, None


def parse_normality(normality_str):
    """Extract normality status and Shapiro-Wilk p-value from normality string."""
    if pd.isna(normality_str) or normality_str == 'N/A':
        return None, None, None
    
    norm_str = str(normality_str)
    
    # Check for "cannot be assessed"
    if 'cannot be assessed' in norm_str.lower() or 'no variance' in norm_str.lower():
        return 'N/A', None, None
    
    # Determine if normal or non-normal
    is_normal = 'normal' in norm_str.lower() and 'non-normal' not in norm_str.lower()
    status = 'Normal' if is_normal else 'Non-normal'
    
    # Extract Shapiro-Wilk p-value
    # Patterns: "p = .1580", "p = 2.24×10⁻⁸", "p < .00001"
    p_value = None
    
    # Handle scientific notation
    if '×10' in norm_str or 'x10' in norm_str:
        match = re.search(r'p\s*=\s*([\d.]+)\s*[×x]\s*10\s*[⁻⁻-]?\s*(\d+)', norm_str)
        if match:
            mantissa = float(match.group(1))
            exponent = int(match.group(2))
            p_value = mantissa * (10 ** -exponent)
    # Handle "p < .00001" type
    elif 'p <' in norm_str:
        match = re.search(r'p\s*<\s*\.?(\d+)', norm_str)
        if match:
            # Use the threshold value as upper bound
            p_value = float('0.' + match.group(1))
    # Handle regular decimal notation
    else:
        match = re.search(r'p\s*=\s*\.?(\d+)', norm_str)
        if match:
            p_value = float('0.' + match.group(1))
    
    return status, p_value, is_normal


def load_statistical_results():
    """Load and combine statistical results from both tasks."""
    # CSV files are in parent directory (stat test), not in visuals subdirectory
    stat_test_dir = os.path.dirname(output_dir)
    classification_file = os.path.join(stat_test_dir, 'classification', 'Classification-StatsResults.csv')
    recognition_file = os.path.join(stat_test_dir, 'recognition', 'Recognition-StatsResults.csv')
    
    dfs = []
    for filepath, task in [(classification_file, 'Classification'), (recognition_file, 'Recognition')]:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['task'] = task
            dfs.append(df)
        else:
            print(f"Warning: {filepath} not found")
    
    if not dfs:
        raise FileNotFoundError("No statistical results files found")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Parse effect sizes and p-values
    combined_df[['effect_value', 'effect_type']] = combined_df['effect_size'].apply(
        lambda x: pd.Series(parse_effect_size(x))
    )
    combined_df[['p_value', 'is_significant']] = combined_df['p_value'].apply(
        lambda x: pd.Series(parse_p_value(x))
    )
    
    # Parse normality information
    combined_df[['normality_status', 'shapiro_p', 'is_normal']] = combined_df['normality'].apply(
        lambda x: pd.Series(parse_normality(x))
    )
    
    # Extract metric type and occlusion from metric name
    combined_df['metric_type'] = combined_df['metric'].str.extract(r'(Precision|Recall|F1-score)')
    combined_df['occlusion'] = combined_df['metric'].str.extract(r'(Occluded|Nonoccluded)')
    
    # Determine test type from test_used column
    combined_df['test_type'] = combined_df['test_used'].apply(
        lambda x: 't-Test' if 't-Test' in str(x) or 't-test' in str(x) 
        else 'Wilcoxon' if 'Wilcoxon' in str(x) 
        else 'N/A'
    )
    
    return combined_df


def create_effect_size_plot(df):
    """Create forest plot of effect sizes."""
    # Filter out rows without effect sizes
    plot_df = df[df['effect_value'].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No effect sizes to plot")
        return
    
    # Create labels
    plot_df['label'] = plot_df.apply(
        lambda row: f"{row['task'][:4]} - {row['occlusion'][:3]} - {row['metric_type']}", 
        axis=1
    )
    
    # Sort by task, occlusion, then metric type
    plot_df = plot_df.sort_values(['task', 'occlusion', 'metric_type'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(plot_df))
    
    # Color bars by effect size magnitude
    colors = []
    for _, row in plot_df.iterrows():
        eff_val = abs(row['effect_value'])
        if row['effect_type'] == 'd':
            if eff_val >= 0.8:
                colors.append(COLOR_LARGE_EFFECT)
            elif eff_val >= 0.5:
                colors.append(COLOR_MEDIUM_EFFECT)
            else:
                colors.append(COLOR_SMALL_EFFECT)
        else:  # r
            if eff_val >= 0.5:
                colors.append(COLOR_LARGE_EFFECT)
            elif eff_val >= 0.3:
                colors.append(COLOR_MEDIUM_EFFECT)
            else:
                colors.append(COLOR_SMALL_EFFECT)
    
    # Create horizontal bars
    bars = ax.barh(y_pos, plot_df['effect_value'], color=colors, 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Add significance markers
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if row['is_significant']:
            ax.plot(row['effect_value'], i, 'k*', markersize=12, zorder=3)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['label'], fontsize=9)
    ax.set_xlabel('Effect Size', fontweight='bold')
    ax.set_title('Effect Sizes Across Metrics\n(* = significant, p < 0.05)', 
                 fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, zorder=0)
    
    # Add legend for effect size magnitudes
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_LARGE_EFFECT, edgecolor='black', 
                 linewidth=0.8, label='Large effect'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_MEDIUM_EFFECT, edgecolor='black', 
                 linewidth=0.8, label='Medium effect'),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_SMALL_EFFECT, edgecolor='black', 
                 linewidth=0.8, label='Small effect')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
             fancybox=True, shadow=True, framealpha=0.95)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'statistical_effect_sizes.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_results_summary_plot(df):
    """Create heatmap summary of statistical test results."""
    # Prepare data for heatmap
    metrics = ['Precision', 'Recall', 'F1-score']
    occlusion_types = ['Occluded', 'Nonoccluded']
    tasks = sorted(df['task'].unique())
    
    # Build data matrix
    n_rows = len(tasks) * len(occlusion_types)
    n_cols = len(metrics)
    p_array = np.zeros((n_rows, n_cols))
    sig_array = np.zeros((n_rows, n_cols), dtype=bool)
    
    row_idx = 0
    y_labels = []
    
    for task in tasks:
        for occ in occlusion_types:
            y_labels.append(f"{task[:4]}-{occ[:3]}")
            for col_idx, metric in enumerate(metrics):
                row = df[(df['task'] == task) & 
                         (df['occlusion'] == occ) & 
                         (df['metric_type'] == metric)]
                if len(row) > 0:
                    row = row.iloc[0]
                    p_val = row['p_value']
                    is_sig = row['is_significant'] if pd.notna(row['is_significant']) else False
                    
                    # Use -log10(p) for visualization, cap at 16
                    if pd.notna(p_val) and p_val > 0:
                        log_p = -np.log10(p_val)
                        log_p = min(log_p, 16)  # Cap at p = 1e-16
                    else:
                        log_p = 0
                    
                    p_array[row_idx, col_idx] = log_p
                    sig_array[row_idx, col_idx] = is_sig
            row_idx += 1
    
    if np.all(p_array == 0):
        print("Warning: No data for summary plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(p_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=16)
    
    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            text_color = 'white' if p_array[i, j] > 8 else 'black'
            sig_marker = '*' if sig_array[i, j] else 'o'
            ax.text(j, i, f'{sig_marker}\n{p_array[i, j]:.1f}', 
                   ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
    
    # Set labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Task - Occlusion', fontweight='bold')
    ax.set_title('Statistical Test Results Summary\n(-log₁₀ p-value, * = significant, o = non-significant)', 
                 fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('-log₁₀(p-value)', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'statistical_results_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_breakdown_data():
    """Load breakdown CSV files to get raw paired differences."""
    # Breakdown files are in parent directory (stat test), not in visuals subdirectory
    stat_test_dir = os.path.dirname(output_dir)
    classification_file = os.path.join(stat_test_dir, 'classification', 'Classification-Breakdown.csv')
    recognition_file = os.path.join(stat_test_dir, 'recognition', 'Recognition-Breakdown.csv')
    
    data = {}
    
    for filepath, task in [(classification_file, 'Classification'), (recognition_file, 'Recognition')]:
        if not os.path.exists(filepath):
            continue
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Clean column names
        df.columns = [col.replace('\n', ' ').strip() for col in df.columns]
        
        # Standardize column names
        column_mapping = {
            'Category ID': 'ID',
            'Gloss ID': 'ID',
            'Category Label': 'Label',
            'Gloss Label': 'Label',
            'Transformer\nPrecision': 'Transformer Precision',
            'Transformer\nRecall': 'Transformer Recall',
            'Transformer\nF1-score': 'Transformer F1-score',
            'IV3-GRU\nPrecision': 'IV3-GRU Precision',
            'IV3-GRU\nRecall': 'IV3-GRU Recall',
            'IV3-GRU\nF1-score': 'IV3-GRU F1-score',
        }
        df = df.rename(columns=column_mapping)
        
        # Extract paired differences for each metric and occlusion
        metrics = ['Precision', 'Recall', 'F1-score']
        occlusion_types = ['occluded', 'nonoccluded']
        
        for occ in occlusion_types:
            for metric in metrics:
                mask = df['Occlusion'].str.lower() == occ.lower()
                filtered_df = df[mask].copy()
                
                if len(filtered_df) == 0:
                    continue
                
                transformer_col = f'Transformer {metric}'
                gru_col = f'IV3-GRU {metric}'
                
                if transformer_col in filtered_df.columns and gru_col in filtered_df.columns:
                    transformer_scores = filtered_df[transformer_col].values
                    gru_scores = filtered_df[gru_col].values
                    
                    # Remove NaN pairs
                    valid_mask = ~(np.isnan(transformer_scores) | np.isnan(gru_scores))
                    transformer_scores = transformer_scores[valid_mask]
                    gru_scores = gru_scores[valid_mask]
                    
                    if len(transformer_scores) > 0:
                        differences = transformer_scores - gru_scores
                        key = f"{task}_{occ}_{metric}"
                        data[key] = {
                            'differences': differences,
                            'task': task,
                            'occlusion': occ,
                            'metric': metric,
                            'n': len(differences)
                        }
    
    return data


def create_normality_distributions_plot():
    """Create visualization of actual normality distributions (paired differences)."""
    # Load breakdown data
    breakdown_data = load_breakdown_data()
    
    if not breakdown_data:
        print("Warning: No breakdown data found for distribution plots")
        return
    
    # Organize by task
    tasks = sorted(set([v['task'] for v in breakdown_data.values()]))
    metrics = ['Precision', 'Recall', 'F1-score']
    occlusion_types = ['occluded', 'nonoccluded']
    
    # Create subplots: one figure per task
    for task in tasks:
        fig, axes = plt.subplots(len(occlusion_types), len(metrics), 
                                figsize=(14, 8))
        fig.suptitle(f'{task}: Distribution of Paired Differences\n(Transformer - IV3-GRU)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes for easier indexing
        if len(occlusion_types) == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        plot_idx = 0
        
        for occ_idx, occ in enumerate(occlusion_types):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[occ_idx, metric_idx]
                key = f"{task}_{occ}_{metric}"
                
                if key in breakdown_data:
                    data_info = breakdown_data[key]
                    differences = data_info['differences']
                    n = data_info['n']
                    
                    # Skip if no variance
                    if len(differences) < 2 or np.std(differences, ddof=1) == 0:
                        ax.text(0.5, 0.5, 'No variance', 
                               ha='center', va='center', 
                               transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'{occ.capitalize()} {metric}\n(N = {n})', 
                                   fontweight='bold', fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue
                    
                    # Create histogram
                    n_bins = min(15, max(5, int(np.sqrt(n))))
                    counts, bins, patches = ax.hist(differences, bins=n_bins, 
                                                   alpha=0.7, color=COLOR_TRANSFORMER,
                                                   edgecolor='black', linewidth=0.8,
                                                   density=True)
                    
                    # Overlay normal distribution
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences, ddof=1)
                    x_norm = np.linspace(differences.min(), differences.max(), 100)
                    y_norm = norm.pdf(x_norm, mean_diff, std_diff)
                    ax.plot(x_norm, y_norm, 'r-', linewidth=2, 
                           label='Normal fit', alpha=0.8)
                    
                    # Add vertical line at mean
                    ax.axvline(mean_diff, color='blue', linestyle='--', 
                              linewidth=1.5, alpha=0.7, label=f'Mean = {mean_diff:.3f}')
                    
                    # Perform Shapiro-Wilk test for annotation
                    if n >= 3:
                        sh_stat, sh_p = shapiro(differences)
                        norm_status = 'Normal' if sh_p > 0.05 else 'Non-normal'
                        if sh_p < 0.0001:
                            p_str = f"p < .0001"
                        else:
                            p_str = f"p = {sh_p:.4f}".lstrip('0')
                        ax.text(0.02, 0.98, f'Shapiro-Wilk: {norm_status}\n{p_str}', 
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    # Customize
                    ax.set_xlabel('Difference', fontweight='bold', fontsize=9)
                    ax.set_ylabel('Density', fontweight='bold', fontsize=9)
                    # Full title with occlusion and metric
                    ax.set_title(f'{occ.capitalize()} {metric}\n(N = {n})', 
                               fontweight='bold', fontsize=10, pad=6)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
                else:
                    # No data for this combination
                    ax.text(0.5, 0.5, 'No data', 
                           ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{occ.capitalize()} {metric}', 
                               fontweight='bold', fontsize=10, pad=6)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.0, w_pad=2.5)
        output_path = os.path.join(output_dir, f'statistical_normality_distributions_{task.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def create_normality_assessment_plot(df):
    """Create visualization of normality assessment and test selection."""
    metrics = ['Precision', 'Recall', 'F1-score']
    occlusion_types = ['Occluded', 'Nonoccluded']
    tasks = sorted(df['task'].unique())
    
    # Build data matrices
    n_rows = len(tasks) * len(occlusion_types)
    n_cols = len(metrics)
    
    # Matrix for normality status (1=Normal, 0=Non-normal, -1=N/A)
    normality_matrix = np.zeros((n_rows, n_cols))
    # Matrix for test type (1=t-Test, 0=Wilcoxon, -1=N/A)
    test_matrix = np.zeros((n_rows, n_cols))
    # Matrix for Shapiro-Wilk p-values (for annotation)
    p_matrix = np.full((n_rows, n_cols), np.nan)
    
    row_idx = 0
    y_labels = []
    
    for task in tasks:
        for occ in occlusion_types:
            y_labels.append(f"{task[:4]}-{occ[:3]}")
            for col_idx, metric in enumerate(metrics):
                row = df[(df['task'] == task) & 
                         (df['occlusion'] == occ) & 
                         (df['metric_type'] == metric)]
                if len(row) > 0:
                    row = row.iloc[0]
                    norm_status = row['normality_status']
                    test_type = row['test_type']
                    shapiro_p = row['shapiro_p']
                    
                    # Set normality matrix value
                    if norm_status == 'Normal':
                        normality_matrix[row_idx, col_idx] = 1
                    elif norm_status == 'Non-normal':
                        normality_matrix[row_idx, col_idx] = 0
                    else:  # N/A
                        normality_matrix[row_idx, col_idx] = -1
                    
                    # Set test matrix value
                    if test_type == 't-Test':
                        test_matrix[row_idx, col_idx] = 1
                    elif test_type == 'Wilcoxon':
                        test_matrix[row_idx, col_idx] = 0
                    else:  # N/A
                        test_matrix[row_idx, col_idx] = -1
                    
                    # Store p-value
                    if pd.notna(shapiro_p):
                        p_matrix[row_idx, col_idx] = shapiro_p
            row_idx += 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Normality Assessment and Test Selection', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Normality Status
    # Custom colormap: Red (Non-normal), Green (Normal), Gray (N/A)
    # Map: -1 (N/A) -> Gray, 0 (Non-normal) -> Red, 1 (Normal) -> Green
    norm_cmap = ListedColormap(['#9e9e9e', '#d32f2f', '#2e7d32'])  # Gray, Red, Green
    norm_bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_norm = BoundaryNorm(norm_bounds, norm_cmap.N)
    
    im1 = ax1.imshow(normality_matrix, cmap=norm_cmap, norm=norm_norm, aspect='auto')
    
    # Add annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(p_matrix[i, j]):
                p_val = p_matrix[i, j]
                # Format p-value for display
                if p_val < 0.0001:
                    p_str = f"{p_val:.2e}"
                else:
                    p_str = f"{p_val:.4f}".lstrip('0')
                # Choose text color based on background
                # Red background (non-normal) -> white text, others -> black
                text_color = 'white' if normality_matrix[i, j] == 0 else 'black'
                ax1.text(j, i, p_str, ha='center', va='center', 
                        color=text_color, fontsize=8, fontweight='bold')
            elif normality_matrix[i, j] == -1:
                # N/A case
                ax1.text(j, i, 'N/A', ha='center', va='center', 
                        color='black', fontsize=8, fontweight='bold')
    
    ax1.set_xticks(range(n_cols))
    ax1.set_xticklabels(metrics)
    ax1.set_yticks(range(n_rows))
    ax1.set_yticklabels(y_labels, fontsize=9)
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Task - Occlusion', fontweight='bold')
    ax1.set_title('Normality Status\n(Shapiro-Wilk p-values shown)', 
                 fontweight='bold', pad=15)
    
    # Add colorbar for normality
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[-1, 0, 1])
    cbar1.set_ticklabels(['N/A', 'Non-normal', 'Normal'])
    cbar1.set_label('Normality Status', fontweight='bold')
    
    # Plot 2: Test Selection
    # Custom colormap: Orange (Wilcoxon), Blue (t-Test), Gray (N/A)
    # Map: -1 (N/A) -> Gray, 0 (Wilcoxon) -> Orange, 1 (t-Test) -> Blue
    test_cmap = ListedColormap(['#9e9e9e', '#f57c00', '#1976d2'])  # Gray, Orange, Blue
    test_bounds = [-1.5, -0.5, 0.5, 1.5]
    test_norm = BoundaryNorm(test_bounds, test_cmap.N)
    
    im2 = ax2.imshow(test_matrix, cmap=test_cmap, norm=test_norm, aspect='auto')
    
    # Add test type annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if test_matrix[i, j] == 1:
                ax2.text(j, i, 't', ha='center', va='center', 
                        color='white', fontsize=10, fontweight='bold')
            elif test_matrix[i, j] == 0:
                ax2.text(j, i, 'W', ha='center', va='center', 
                        color='white', fontsize=10, fontweight='bold')
            elif test_matrix[i, j] == -1:
                ax2.text(j, i, 'N/A', ha='center', va='center', 
                        color='black', fontsize=8, fontweight='bold')
    
    ax2.set_xticks(range(n_cols))
    ax2.set_xticklabels(metrics)
    ax2.set_yticks(range(n_rows))
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_xlabel('Metrics', fontweight='bold')
    ax2.set_ylabel('Task - Occlusion', fontweight='bold')
    ax2.set_title('Test Selection\n(t = t-Test, W = Wilcoxon)', 
                 fontweight='bold', pad=15)
    
    # Add colorbar for test type
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[-1, 0, 1])
    cbar2.set_ticklabels(['N/A', 'Wilcoxon', 't-Test'])
    cbar2.set_label('Test Type', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'statistical_normality_assessment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to generate all visualizations."""
    try:
        print("=" * 60)
        print("Creating Statistical Analysis Visualizations")
        print("=" * 60)
        print(f"Working directory: {output_dir}")
        sys.stdout.flush()
        
        # Load data
        print("\nLoading statistical results...")
        sys.stdout.flush()
        df = load_statistical_results()
        print(f"✓ Loaded {len(df)} statistical test results")
        sys.stdout.flush()
        
        # Create visualizations
        print("\nCreating effect size plot...")
        sys.stdout.flush()
        create_effect_size_plot(df)
        
        print("Creating results summary plot...")
        sys.stdout.flush()
        create_results_summary_plot(df)
        
        print("Creating normality assessment plot...")
        sys.stdout.flush()
        create_normality_assessment_plot(df)
        
        print("Creating normality distributions plot...")
        sys.stdout.flush()
        create_normality_distributions_plot()
        
        print("\n" + "=" * 60)
        print("✓ All statistical visualizations created successfully!")
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

