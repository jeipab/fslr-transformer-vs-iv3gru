"""
Shared color palette for visualization scripts.

This module provides consistent colors for Transformer and IV3-GRU models
across all visualization scripts in the metrics directory.
"""

import colorsys

# Base colors for models
COLOR_TRANSFORMER = '#fff4d3'  # Light yellow/cream
COLOR_IV3_GRU = '#daedff'      # Light blue

def darken_color(color_hex, factor=0.3):
    """Darken a hex color by a factor (0-1)."""
    # Convert hex to RGB
    color_hex = color_hex.lstrip('#')
    r, g, b = [int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    # Convert to HSV, reduce value (brightness), convert back
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(0, v * (1 - factor))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def lighten_color(color_hex, factor=0.2):
    """Lighten a hex color by a factor (0-1)."""
    # Convert hex to RGB
    color_hex = color_hex.lstrip('#')
    r, g, b = [int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    # Convert to HSV, increase value (brightness), convert back
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = min(1, v + (1 - v) * factor)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

# Create distinct colors for occluded (darker) and non-occluded (lighter)
COLOR_TRANSFORMER_OCC = darken_color(COLOR_TRANSFORMER, 0.25)  # Darker for occluded
COLOR_TRANSFORMER_NON = lighten_color(COLOR_TRANSFORMER, 0.15)  # Lighter for non-occluded
COLOR_IV3_GRU_OCC = darken_color(COLOR_IV3_GRU, 0.25)  # Darker for occluded
COLOR_IV3_GRU_NON = lighten_color(COLOR_IV3_GRU, 0.15)  # Lighter for non-occluded

# For scatter plots comparing models: use model colors for occlusion types
COLOR_OCC = COLOR_TRANSFORMER_OCC  # Use Transformer darker yellow for occluded
COLOR_NON = COLOR_IV3_GRU_NON      # Use IV3-GRU lighter blue for non-occluded

