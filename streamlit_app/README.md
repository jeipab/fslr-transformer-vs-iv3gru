# FSLR Streamlit App

This directory contains the modularized Streamlit application for the Filipino Sign Language Recognition (FSLR) demo.

## Structure

```
streamlit_app/
├── __init__.py          # Package initialization
├── main.py              # Main application entry point
├── components.py        # UI components and layout
├── data_processing.py   # Video and NPZ file processing
├── utils.py            # Utility functions
├── visualization.py    # Keypoint visualization and charts
└── README.md           # This file
```

## Features

### Enhanced Keypoint Visualization

- **Animated keypoint display**: Interactive slider to view keypoints frame by frame
- **Skeleton overlay**: Shows connections between keypoints for better understanding
- **Body part grouping**: Different colors for pose, hands, and face landmarks
- **Visibility indicators**: Shows which keypoints are detected vs. occluded

### Video Processing

- **Real-time keypoint extraction**: Process video files to extract keypoints
- **Video overlay**: Generate videos with keypoints drawn on top
- **Multiple format support**: MP4, AVI, MOV, MKV, WMV, FLV, WebM

### Data Analysis

- **Feature analysis**: Interactive charts showing keypoint trajectories
- **Statistical summaries**: Mean, std, min, max, range for each feature
- **Temporal visualization**: Heatmaps and line charts over time

## Running the App

### Option 1: From project root

```bash
streamlit run run_app.py
```

### Option 2: From streamlit_app directory

```bash
cd streamlit_app
streamlit run main.py
```

## Key Improvements

1. **Modular Architecture**: Code is now split into logical modules for better maintainability
2. **Enhanced Visualization**: New animated keypoint display with skeleton overlay
3. **Better Video Support**: Improved video processing with keypoint overlay generation
4. **Cleaner UI**: Better organized components and improved user experience

## Dependencies

- streamlit
- numpy
- pandas
- plotly
- opencv-python
- pathlib

## File Types Supported

- **NPZ files**: Preprocessed keypoint data
- **Video files**: MP4, AVI, MOV, MKV, WMV, FLV, WebM
