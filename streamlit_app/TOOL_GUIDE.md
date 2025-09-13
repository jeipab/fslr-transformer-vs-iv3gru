# Tool Guide

Interactive Streamlit application for FSLR visualization and analysis.

## Features

### Keypoint Visualization

- **Animated display**: Frame-by-frame keypoint viewing with slider
- **Skeleton overlay**: Connections between keypoints
- **Body part grouping**: Color-coded pose, hands, and face landmarks
- **Visibility indicators**: Shows detected vs. occluded keypoints

### Video Processing

- **Real-time extraction**: Process videos to extract keypoints
- **Video overlay**: Generate videos with keypoints drawn
- **Format support**: MP4, AVI, MOV, MKV, WMV, FLV, WebM

### Data Analysis

- **Feature charts**: Interactive keypoint trajectory visualization
- **Statistics**: Mean, std, min, max, range summaries
- **Temporal plots**: Heatmaps and line charts over time

## Usage

```bash
# From project root
streamlit run run_app.py

# From streamlit_app directory
cd streamlit_app
streamlit run main.py
```

## File Support

- **NPZ files**: Preprocessed keypoint data
- **Video files**: Multiple formats (MP4, AVI, MOV, etc.)

## Architecture

- `main.py`: Application entry point
- `components.py`: UI components and layout
- `data_processing.py`: Video and NPZ processing
- `visualization.py`: Keypoint visualization and charts
- `utils.py`: Utility functions
