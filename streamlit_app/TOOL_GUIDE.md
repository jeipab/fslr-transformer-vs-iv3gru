# Tool Guide

Interactive Streamlit application for FSLR (First Sign Language Recognition) model comparison and analysis.

> **⚠️ Development Status**: This application is currently in active development. Features may change and some functionality may be experimental.

## Features

### Model Comparison

- **Dual Model Support**: Compare Transformer and IV3-GRU model architectures
- **Compatibility Detection**: Automatic detection of model compatibility based on data type
- **Model Type Indicators**: Clear labeling of intended model type (T/I/B)

### Video Preprocessing Pipeline

- **Flexible Extraction**: Choose between keypoints (156-D), IV3 features (2048-D), or both
- **Occlusion Detection**: Automatic detection of hand-face occlusion with metadata
- **Validation**: Prevents processing when no extraction options are selected
- **Real-time Processing**: Process videos directly in the browser interface

### Keypoint Visualization

- **Animated Display**: Frame-by-frame keypoint viewing with interactive slider
- **Skeleton Overlay**: Connections between keypoints with color-coded body parts
- **Body Part Grouping**: Pose (red), left hand (blue), right hand (green), face (orange)
- **Visibility Indicators**: Shows detected vs. occluded keypoints
- **Video Generation**: Export animated keypoint sequences as MP4 videos

### Data Analysis

- **Feature Charts**: Interactive keypoint trajectory visualization over time
- **Statistics Dashboard**: Mean, std, min, max, range summaries
- **Temporal Plots**: Heatmaps and line charts for different body parts
- **Metadata Display**: Processing parameters, model type, and occlusion status

### File Management

- **Multi-format Support**: NPZ files, video files (MP4, AVI, MOV, MKV, WMV, FLV, WebM)
- **Batch Processing**: Process multiple videos simultaneously
- **File Organization**: Clear workflow from upload → preprocessing → analysis

## Usage

```bash
# From project root
streamlit run run_app.py
```

## Workflow

1. **Upload Stage**: Upload video files or preprocessed NPZ files
2. **Preprocessing Stage**: Configure and run preprocessing with model-specific options
3. **Analysis Stage**: Visualize keypoints, analyze features, and compare model compatibility

## Model Types

- **T (Transformer)**: 156-D keypoint features for Transformer architecture
- **I (IV3-GRU)**: 2048-D InceptionV3 features for IV3-GRU architecture
- **B (Both)**: Contains both feature types for dual-model comparison

## File Support

- **NPZ files**: Preprocessed data with keypoints, features, and metadata
- **Video files**: Multiple formats with automatic keypoint extraction
- **Metadata**: Includes model type, occlusion flags, and processing parameters

## Architecture

- `main.py`: Application entry point and main workflow
- `preprocessing_manager.py`: Video preprocessing interface and controls
- `prediction_manager.py`: Model analysis and visualization interface
- `data_processing.py`: Video and NPZ processing backend
- `visualization.py`: Keypoint visualization and chart generation
- `utils.py`: Utility functions and compatibility checking
- `upload_manager.py`: File upload and management
- `components.py`: Reusable UI components

## Development Notes

- **Occlusion Detection**: Uses geometric analysis of hand-face interactions
- **Model Compatibility**: Based on metadata `model_type` field for accurate detection
- **Validation**: Prevents invalid preprocessing configurations
- **Backward Compatibility**: Supports legacy NPZ files without metadata
