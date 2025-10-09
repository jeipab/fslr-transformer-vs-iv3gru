# FSLR Demo Tool Guide

Interactive Streamlit application for Filipino Sign Language Recognition model comparison and analysis.

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

- **Multi-format Support**: NPZ files, video files (MP4, MOV)
- **Batch Processing**: Process multiple videos simultaneously
- **File Organization**: Clear workflow from upload → preprocessing → analysis

### Model Validation

- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score metrics
- **Confusion Matrix**: Visual analysis of model performance
- **Occlusion Analysis**: Separate performance metrics for occluded vs non-occluded samples
- **Per-class Performance**: Detailed breakdown by gloss and category classes
- **Results Export**: Download validation results as CSV/JSON

## Usage

```bash
# From project root
streamlit run run_app.py
```

## Workflow

1. **Upload Stage**: Upload video files or preprocessed NPZ files
2. **Preprocessing Stage**: Configure and run preprocessing with model-specific options
3. **Analysis Stage**: Visualize keypoints, analyze features, and compare model compatibility
4. **Validation Stage**: Evaluate model performance on validation datasets

## Model Types

- **T (Transformer)**: Flexible input - can use either 156-D keypoint features or 2048-D InceptionV3 features (auto-detected from model checkpoint)
- **I (IV3-GRU)**: 2048-D InceptionV3 features for IV3-GRU architecture
- **B (Both)**: Contains both feature types for dual-model comparison

## File Support

- **NPZ files**: Preprocessed data with keypoints, features, and metadata
- **Video files**: Multiple formats with automatic keypoint extraction
- **Metadata**: Includes model type, occlusion flags, and processing parameters

## Requirements

- Python 3.8+
- Streamlit
- OpenCV (for video processing)
- PyTorch (for model inference)
- Plotly (for visualizations)
- NumPy, Pandas (for data handling)

## Configuration

The application uses configuration files in `streamlit_app/core/config.py`:

- Model paths and settings
- Processing parameters (FPS, frame size, etc.)
- UI styling and layout options
- File size limits and supported formats

## Architecture

- `main.py`: Application entry point and main workflow
- `preprocessing_manager.py`: Video preprocessing interface and controls
- `prediction_manager.py`: Model analysis and visualization interface
- `validation_manager.py`: Model validation and evaluation interface
- `upload_manager.py`: File upload and management
- `data_processing.py`: Video and NPZ processing backend
- `visualization.py`: Keypoint visualization and chart generation
- `utils.py`: Utility functions and compatibility checking
- `components.py`: Reusable UI components
- `validation_components.py`: Validation-specific UI components

## Technical Notes

- **Occlusion Detection**: Uses geometric analysis of hand-face interactions
- **Model Compatibility**: Based on metadata `model_type` field for accurate detection
- **Validation**: Prevents invalid preprocessing configurations
- **Backward Compatibility**: Supports legacy NPZ files without metadata
- **GPU Acceleration**: Automatic detection and utilization of available GPU resources
- **Batch Processing**: Optimized multi-processing for video preprocessing
- **Session Management**: Persistent state across workflow stages
