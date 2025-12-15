"""
Create radar/spider charts comparing Transformer vs IV3-GRU performance
for classification and recognition tasks using Precision, Recall, and F1-Score.

Usage:
    python metrics/extract/overall/create_radar_chart.py

Output:
    - classification_radar_chart.png: Radar chart comparing Transformer and IV3-GRU
      for classification (category-based) metrics
    - recognition_radar_chart.png: Radar chart comparing Transformer and IV3-GRU
      for recognition (gloss-based) metrics
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import colorsys

# Add metrics directory to path to import color_palette
METRICS_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(METRICS_DIR))
from color_palette import COLOR_TRANSFORMER, COLOR_IV3_GRU, darken_color


def brighten_and_saturate(color_hex, brightness_factor=0.3, saturation_factor=0.5):
    """
    Brighten and increase saturation of a hex color.
    
    Args:
        color_hex: Hex color string
        brightness_factor: Factor to increase brightness (0-1)
        saturation_factor: Factor to increase saturation (0-1)
    
    Returns:
        Brightened and saturated hex color string
    """
    # Convert hex to RGB
    color_hex = color_hex.lstrip('#')
    r, g, b = [int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    
    # Convert to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Increase saturation
    s = min(1.0, s + (1.0 - s) * saturation_factor)
    
    # Increase brightness
    v = min(1.0, v + (1.0 - v) * brightness_factor)
    
    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

# Paths
SCRIPT_DIR = Path(__file__).parent
CLASSIFICATION_CSV = SCRIPT_DIR / "overall_classification_metrics.csv"
RECOGNITION_CSV = SCRIPT_DIR / "overall_recognition_metrics.csv"
OUTPUT_CLASSIFICATION = SCRIPT_DIR / "classification_radar_chart.png"
OUTPUT_RECOGNITION = SCRIPT_DIR / "recognition_radar_chart.png"


def create_radar_chart(csv_path, output_path, title):
    """
    Create a radar chart from CSV metrics file.
    
    Args:
        csv_path: Path to CSV file with metrics
        output_path: Path to save the chart
        title: Chart title
    """
    # Read data from CSV
    metrics_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row['Metric']
            if metric in ['Precision', 'Recall', 'F1-Score']:
                metrics_data[metric] = {
                    'Transformer': float(row['Transformer']),
                    'IV3-GRU': float(row['IV3-GRU'])
                }
    
    # Prepare data for radar chart
    categories = list(metrics_data.keys())
    transformer_values = [metrics_data[cat]['Transformer'] for cat in categories]
    iv3gru_values = [metrics_data[cat]['IV3-GRU'] for cat in categories]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add first value to end to close the plot
    transformer_values += transformer_values[:1]
    iv3gru_values += iv3gru_values[:1]
    
    # Create figure with polar subplot - adjusted size for better proportions
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(projection='polar'))
    
    # Create brighter and more saturated colors
    transformer_bright = brighten_and_saturate(COLOR_TRANSFORMER, brightness_factor=0.2, saturation_factor=0.6)
    iv3gru_bright = brighten_and_saturate(COLOR_IV3_GRU, brightness_factor=0.2, saturation_factor=0.6)
    
    # Use darker versions of bright colors for lines for better visibility
    transformer_line_color = darken_color(transformer_bright, 0.3)
    iv3gru_line_color = darken_color(iv3gru_bright, 0.3)
    
    # Plot data with improved styling
    ax.plot(angles, transformer_values, 'o-', linewidth=2.5, label='Transformer', 
            color=transformer_line_color, markersize=10, markeredgewidth=1.5, 
            markeredgecolor='white', zorder=3)
    ax.fill(angles, transformer_values, alpha=0.25, color=transformer_bright)
    
    ax.plot(angles, iv3gru_values, 's-', linewidth=2.5, label='IV3-GRU', 
            color=iv3gru_line_color, markersize=9, markeredgewidth=1.5, 
            markeredgecolor='white', zorder=3)
    ax.fill(angles, iv3gru_values, alpha=0.25, color=iv3gru_bright)
    
    # Set category labels with better positioning
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='semibold', 
                       color='#2C3E50')
    
    # Set y-axis limits with padding
    ax.set_ylim(0, 1.05)
    
    # Set y-axis ticks with better formatting
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                       fontsize=11, color='#5D6D7E')
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8, color='#BDC3C7')
    ax.set_facecolor('#FAFAFA')
    
    # Customize radial gridlines
    ax.spines['polar'].set_color('#95A5A6')
    ax.spines['polar'].set_linewidth(1.2)
    
    # Add title with better positioning
    plt.title(title, size=18, fontweight='bold', pad=30, color='#2C3E50')
    
    # Add legend with improved styling - positioned in lower right, moved left
    legend = plt.legend(loc='lower right', bbox_to_anchor=(1.05, -0.05), 
                       fontsize=13, frameon=True, shadow=True, 
                       fancybox=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#BDC3C7')
    
    # Adjust layout with left padding and prevent label cutoff
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.1, right=0.95)
    
    # Save figure with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved radar chart to: {output_path}")


def main():
    """Main function to create radar charts for both tasks."""
    # Create classification radar chart
    create_radar_chart(
        CLASSIFICATION_CSV,
        OUTPUT_CLASSIFICATION,
        "Classification Performance Comparison"
    )
    
    # Create recognition radar chart
    create_radar_chart(
        RECOGNITION_CSV,
        OUTPUT_RECOGNITION,
        "Recognition Performance Comparison"
    )
    
    print("Radar charts created successfully!")


if __name__ == "__main__":
    main()