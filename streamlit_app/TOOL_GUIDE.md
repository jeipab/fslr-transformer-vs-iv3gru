# PANSINAYAN Tool Guide

### _Where Every Sign Gets Attention_

Interactive application for Filipino Sign Language Recognition model comparison and analysis.

## Overview

**PANSINAYAN** is a comprehensive web-based tool for Filipino Sign Language Recognition, providing a complete interface for video preprocessing, model prediction, validation, and visualization. The name embodies the system's core innovation: using Multi-Head Attention mechanisms to give every sign the attention it deserves.

The application includes pre-trained models for 105 Filipino sign words across 10 semantic categories, leveraging attention-based architectures to achieve robust recognition even under challenging conditions like occlusion.

## Dataset Information

- **Glosses**: 105 sign words (IDs: 0-104)
- **Categories**: 10 semantic categories (IDs: 0-9)
  - GREETING, SURVIVAL, NUMBER, CALENDAR, DAYS, FAMILY, RELATIONSHIPS, COLOR, FOOD, DRINK
- **Pre-trained Models**: Transformer and IV3-GRU models trained on FSL-105 dataset
- **Demo Files**: Available in `data/demo/` for testing

## Quick Start

### Launch PANSINAYAN

```powershell
# From project root
streamlit run run_app.py

# Alternative: Run from streamlit_app directory
cd streamlit_app
streamlit run core\main.py

# Specify port
streamlit run run_app.py --server.port 8501
```

### Check Network Info

```powershell
# Get local IP and access URLs
python show_network_info.py
```

PANSINAYAN will open in your default browser at `http://localhost:8501`.

### Mobile Camera Uploads

PANSINAYAN supports direct camera capture on mobile devices:

- **iOS Safari**: Tap "Browse files" to access camera
- **Android Chrome**: Select camera from file picker
- **Enhanced Sync**: Automatic event handling for consistent uploads
- **Base64 Option**: Available for load-balanced deployments

## Features

### 1. Model Prediction

**Supported Inputs:**

- Preprocessed NPZ files (keypoints or features)
- Raw video files (MP4, MOV, AVI)
- Demo files from `data/demo/`

**Dual Model Support:**

- **Transformer**: Uses keypoints [T, 178]
- **IV3-GRU**: Uses InceptionV3 features [T, 2048]
- Real-time prediction with confidence scores
- Top-5 gloss predictions and top-3 category predictions

**Model Paths:**

- Transformer: `trained_models/transformer/FSL105_classification/SignTransformer_best.pt`
- IV3-GRU: `trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt`

### 2. Video Preprocessing Pipeline

**Extraction Options:**

- **Keypoints (178-D)**: MediaPipe pose, hands, and face landmarks
- **IV3 Features (2048-D)**: InceptionV3 CNN features
- **Both**: Extract both feature types simultaneously

**Processing Features:**

- Automatic occlusion detection (hand-face interactions)
- Configurable FPS and frame size
- Batch processing for multiple videos
- Real-time progress tracking
- NPZ output with metadata

**Occlusion Detection:**

- Frame-level: Keypoint visibility threshold (default: 60%)
- Clip-level: Occluded if ≥40% frames occluded OR consecutive run ≥15 frames
- Stored in NPZ metadata: `occluded_flag` (0 or 1)

### 3. Keypoint Visualization

**Animated Display:**

- Frame-by-frame viewing with interactive slider
- Play/pause animation controls
- Adjustable animation speed (FPS)

**Skeleton Overlay:**

- Body part color coding:
  - Pose landmarks: Red
  - Left hand: Blue
  - Right hand: Green
  - Face landmarks: Orange
- Keypoint connections showing body structure
- Visibility indicators (detected vs. occluded)

**Video Export:**

- Generate MP4 animations of keypoint sequences
- Customizable frame rate and resolution
- Download directly from browser

### 4. Data Analysis

**Feature Visualization:**

- Interactive trajectory plots over time
- Body part-specific analysis (pose, hands, face)
- Temporal heatmaps showing movement patterns
- Line charts for individual keypoint coordinates

**Statistics Dashboard:**

- Mean, standard deviation, min, max, range
- Per-body-part summary statistics
- Sequence length and frame count
- Timestamp analysis

**Metadata Display:**

- Processing parameters (FPS, image size)
- Model type compatibility (T/I/B)
- Occlusion status and detection details
- Source video information

### 5. Model Validation

**Comprehensive Evaluation:**

- Overall metrics: Accuracy, Precision, Recall, F1-score
- Gloss prediction performance (105 classes)
- Category prediction performance (10 classes)
- Confusion matrices for error analysis

**Occlusion Analysis:**

- Separate metrics for occluded samples
- Separate metrics for non-occluded samples
- Performance comparison between clean and occluded data
- Helps assess model robustness

**Per-class Performance:**

- Detailed breakdown by gloss (105 classes)
- Detailed breakdown by category (10 classes)
- Identifies difficult signs and categories
- Supports targeted model improvement

**Results Export:**

- Download validation results as JSON
- Download confusion matrices as CSV
- Save detailed per-class metrics
- Generate comprehensive reports

### 6. File Management

**Multi-format Support:**

- NPZ files (preprocessed data)
- Video files (MP4, MOV, AVI)
- Demo files for quick testing

**Upload Options:**

- Drag and drop interface
- File browser selection
- Batch upload for multiple files
- Mobile-friendly upload (up to 500MB)

**File Organization:**

- Clear workflow: Upload → Preprocess → Analyze → Validate
- Session-based file management
- Automatic cleanup of temporary files

## Workflow

### Step 1: Upload Stage

**Upload Files:**

1. Navigate to "Upload" section
2. Choose file type: Video or NPZ
3. Upload files via drag-and-drop or file browser
4. Review uploaded files list

**Try Demo Files:**

- Use pre-processed demo files from `data/demo/`
- Available demos: "nice to meet you", "nine", "grandfather", "green", "fish", "crab"

### Step 2: Preprocessing Stage (For Videos)

**Configure Processing:**

1. Select extraction type:
   - Keypoints only (Transformer)
   - IV3 Features only (IV3-GRU)
   - Both (Compare models)
2. Set FPS (15-30 recommended)
3. Set frame size (256 default)
4. Enable occlusion detection (automatic)

**Process Videos:**

1. Click "Process Videos"
2. Monitor progress bar
3. Review processing results
4. Download NPZ files

### Step 3: Prediction Stage

**Single Prediction:**

1. Select NPZ file or upload video
2. Choose model: Transformer or IV3-GRU
3. View prediction results:
   - Predicted gloss with confidence
   - Predicted category with confidence
   - Top-5 alternatives
4. View keypoint visualization (if keypoints available)

**Batch Prediction:**

1. Upload multiple files
2. Select model
3. Process batch
4. Review results for all files
5. Export results as CSV

### Step 4: Analysis Stage

**Keypoint Visualization:**

1. Select NPZ file with keypoints
2. View animated skeleton
3. Use slider to navigate frames
4. Generate video animation

**Feature Analysis:**

1. Select body part (pose, hands, face)
2. View trajectory plots
3. Analyze temporal patterns
4. Review statistics

### Step 5: Validation Stage

**Validate Model:**

1. Select validation dataset directory
2. Provide labels CSV file
3. Choose model and checkpoint
4. Run validation
5. View comprehensive metrics
6. Export results

## Model Types

The application automatically detects and displays model compatibility:

- **T (Transformer)**: Uses 178-D keypoints from MediaPipe

  - Input: Pose (25 points), Hands (21 points each), Face (22 points)
  - Processes sequential keypoint data
  - Provides attention weights for interpretability

- **I (IV3-GRU)**: Uses 2048-D InceptionV3 features

  - Input: Pre-computed CNN features
  - Leverages transfer learning from ImageNet
  - Processes visual appearance features

- **B (Both)**: Contains both feature types
  - Enables model comparison
  - Flexible for different use cases

## Configuration

### Application Settings

Located in `streamlit_app/core/config.py`:

**Model Settings:**

- Model paths and checkpoint locations
- Default model selection
- Number of classes (105 glosses, 10 categories)

**Processing Settings:**

- Default FPS: 30
- Default frame size: 256
- Batch size for GPU processing
- Number of workers for parallel processing

**UI Settings:**

- Page layout (wide mode)
- Color scheme and styling
- Animation speed
- Chart configurations

### Upload Limits

Configured in `.streamlit/config.toml`:

- `maxUploadSize = 500` MB
- `maxMessageSize = 500` MB
- `enableCORS = true` (mobile compatibility)
- `enableWebsocketCompression = true` (better performance)

## Architecture

### Core Components

**Entry Points:**

- `run_app.py`: Main application launcher
- `streamlit_app/core/main.py`: Application core and workflow

**Managers:**

- `upload_manager.py`: File upload and management interface
- `preprocessing_manager.py`: Video preprocessing controls
- `prediction_manager.py`: Model prediction and analysis
- `validation_manager.py`: Model validation and evaluation

**Backend:**

- `data_processing.py`: Video and NPZ processing logic
- `visualization.py`: Keypoint visualization and charts
- `utils.py`: Utility functions and compatibility checking
- `config.py`: Application configuration and settings

**UI Components:**

- `components.py`: Reusable UI elements
- `validation_components.py`: Validation-specific components
- `demo_video.py`: Demo file handling

### Data Flow

```
Upload → Preprocessing → NPZ Files → Prediction/Analysis → Results
                                   ↓
                              Validation → Metrics/Reports
```

## Requirements

### Python Dependencies

```powershell
pip install -r requirements.txt
```

**Core Libraries:**

- `streamlit` - Web application framework
- `torch` - Deep learning framework
- `opencv-python` - Video processing
- `mediapipe` - Keypoint extraction
- `numpy`, `pandas` - Data handling
- `plotly` - Interactive visualizations
- `scikit-learn` - Metrics and evaluation

**Optional:**

- `pyarrow` - Parquet file support

## Technical Notes

### Occlusion Detection

- **Method**: Geometric analysis of hand-face keypoint interactions
- **Thresholds**: Configurable visibility and temporal thresholds
- **Output**: Binary flag in NPZ metadata
- **Usage**: Filtering training data, analyzing model robustness

### Model Compatibility

- **Detection**: Based on NPZ metadata `model_type` field
- **Auto-selection**: Automatically chooses compatible model
- **Validation**: Prevents using wrong model for data type
- **Backward Compatibility**: Supports legacy files without metadata

### Performance Optimization

- **GPU Acceleration**: Automatic CUDA detection and usage
- **Batch Processing**: Multi-video parallel preprocessing
- **Session State**: Persistent data across interactions
- **Caching**: Streamlit caching for expensive operations

### Mobile Support

- **Responsive UI**: Adapts to mobile screen sizes
- **Large Uploads**: Supports up to 500MB files
- **WebSocket**: Compressed communication for mobile networks
- **CORS**: Enabled for mobile browser compatibility

## Troubleshooting

### Common Issues

**Model Loading Errors:**

- Ensure model checkpoints exist in `trained_models/` directory
- Verify model type matches input data type
- Check PyTorch version compatibility

**Video Processing Failures:**

- Verify video codec is supported (H.264 recommended)
- Check video file is not corrupted
- Ensure sufficient disk space for temporary files

**Upload Failures (Mobile):**

- Check file size (max 500MB)
- Ensure stable internet connection
- Try compressing video before upload
- Use gallery selection instead of camera

**Performance Issues:**

- Close other browser tabs
- Reduce FPS in preprocessing (use 15-20)
- Process fewer videos simultaneously
- Clear browser cache

### Debug Mode

Enable verbose logging:

```powershell
streamlit run run_app.py --logger.level=debug
```

## Demo Files

Pre-processed demo files in `data/demo/`:

- `clip_0138_nice to meet you.npz` - Greeting sign
- `clip_0585_nine.npz` - Number sign
- `clip_1146_grandfather.npz` - Family sign
- `clip_1493_green.npz` - Color sign
- `clip_1765_fish.npz` - Food sign
- `clip_1912_crab.npz` - Food sign

Use these to quickly test the application without preprocessing.

## Additional Resources

- [Data Guide](../data/DATA_GUIDE.md) - Data formats and structures
- [Model Guide](../models/MODEL_GUIDE.md) - Architecture details
- [Prediction Guide](../evaluation/prediction/PREDICTION_GUIDE.md) - Using models
- [Validation Guide](../evaluation/validation/VALIDATION_GUIDE.md) - Model evaluation
- [Preprocessing Guide](../preprocessing/docs/PREPROCESS_GUIDE.MD) - Video preprocessing
