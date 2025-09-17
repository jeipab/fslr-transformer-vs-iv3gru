# Occlusion Detection Guide

## Overview

The preprocessing pipeline includes hand-head occlusion detection for sign language video analysis. The system uses computer vision techniques based on MediaPipe keypoints and advanced detection algorithms.

## Quick Start

### Basic Usage

```bash
# Enable occlusion detection (default when keypoints are written)
python preprocessing/preprocess.py video.mp4 output/ --write-keypoints --occ-enable

# Multi-processing version
python preprocessing/multi_preprocess.py video.mp4 output/ --write-keypoints --occ-enable
```

### Detailed Results

```bash
# Get detailed occlusion analysis
python preprocessing/preprocess.py video.mp4 output/ --write-keypoints --occ-enable --occ-detailed

# Multi-processing with detailed results
python preprocessing/multi_preprocess.py video.mp4 output/ --write-keypoints --occ-enable --occ-detailed
```

## How It Works

The occlusion detection system implements sophisticated computer vision-based detection with:

- **5-Region Head Partitioning**: Divides the head into linguistically significant regions (forehead, cheeks, nose, mouth, neck)
- **Multi-Method Detection**: Uses ellipse intersection, proximity analysis, trajectory tracking, and orientation detection
- **Temporal Filtering**: Applies sliding window with majority voting for consistency
- **Conservative Thresholds**: Designed to minimize false positives while maintaining accuracy

### Head Regions

The system divides the head into 5 regions based on the Suvi dictionary:

- **Forehead**: Top of the head area
- **Cheeks**: Including eyes and ears
- **Nose**: Central face area
- **Mouth**: Including chin area
- **Neck**: Below chin area

### Detection Methods

1. **Direct Fingertip Intersection**: Detects when fingertips enter face regions
2. **Palm Center Proximity**: Analyzes palm center distance to face regions
3. **Trajectory Analysis**: Tracks hand movement patterns toward face
4. **Multi-Point Orientation**: Considers overall hand orientation relative to face

## Configuration

### Default Parameters

```python
# Conservative detection parameters
config = {
    'min_face_points': 5,           # Require more face points for accuracy
    'min_hand_points': 4,            # Require more hand points for reliability
    'min_fingertips_inside': 3,      # Require multiple fingertips for reliable detection
    'proximity_multiplier': 1.2,     # Conservative multiplier
    'occlusion_threshold': 0.30      # Higher threshold for reliability
}
```

### Output Formats

**Compatible Format** (default):

```python
# Returns binary flag
occlusion_flag = compute_occlusion_detection(video_path, output_format='compatible')
# Returns: 0 (not occluded) or 1 (occluded)
```

**Detailed Format**:

```python
# Returns comprehensive analysis
results = compute_occlusion_detection(video_path, output_format='detailed')
# Returns: {
#     'binary_flag': 0 or 1,
#     'occlusion_rate': 0.0-1.0,
#     'total_frames': int,
#     'occluded_frames': int,
#     'detailed_results': [...]
# }
```

## API Reference

### Core Functions

```python
from preprocessing.core.occlusion_detection import (
    compute_occlusion_detection,
    compute_occlusion_flag_from_keypoints,
    HandHeadOcclusionDetector,
    get_occlusion_config
)

# Main detection function
occlusion_flag = compute_occlusion_detection(
    video_path="path/to/video.mp4",
    output_format="compatible"  # or "detailed"
)

# Keypoint-based detection
occlusion_flag = compute_occlusion_detection(
    X=keypoints_array,  # [T, 156] normalized coordinates
    mask_bool_array=visibility_mask,  # [T, 78] visibility mask
    output_format="compatible"
)

# Legacy compatibility function
occlusion_flag = compute_occlusion_flag_from_keypoints(
    X=keypoints_array,
    mask_bool_array=visibility_mask
)
```

### Configuration

```python
# Get default configuration
config = get_occlusion_config()

# Validate configuration
is_valid = validate_occlusion_config(config)
```

## Integration Examples

### Preprocessing Pipeline

```python
from preprocessing.core.preprocess import process_video

# Process video with occlusion detection
process_video(
    video_path="input.mp4",
    out_dir="output/",
    write_keypoints=True,
    compute_occlusion=True,
    occ_detailed=False  # Set to True for detailed results
)
```

### Custom Detection

```python
from preprocessing.core.occlusion_detection import HandHeadOcclusionDetector
import cv2

# Initialize detector
detector = HandHeadOcclusionDetector(use_global_tracking=True)

# Process individual frames
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.process_frame(frame, frame_idx=0)
    print(f"Occlusion detected: {result['occlusion_detected']}")
    print(f"Occluded regions: {result['occluded_regions']}")
```

## Dependencies

The occlusion detection system requires:

```bash
pip install scipy scikit-learn opencv-python mediapipe
```

## Performance Notes

- **Speed**: Optimized for batch processing with multiprocessing support
- **Accuracy**: Conservative thresholds minimize false positives
- **Memory**: Efficient processing with configurable batch sizes
- **Scalability**: Supports parallel processing across multiple videos

## Troubleshooting

### Common Issues

1. **ImportError**: Install required dependencies

   ```bash
   pip install scipy scikit-learn
   ```

2. **Low Detection Rate**: Check video quality and lighting conditions

3. **False Positives**: System uses conservative thresholds by design

### Debug Mode

Enable detailed output for debugging:

```bash
python preprocessing/preprocess.py video.mp4 output/ --write-keypoints --occ-enable --occ-detailed
```

This will provide frame-by-frame analysis in the metadata.

## Research Background

This implementation is based on research in hand-head occlusion detection for sign language video analysis, incorporating:

- Computer vision techniques for robust detection
- Temporal consistency filtering
- Multi-region analysis for linguistic relevance
- Conservative thresholds for production use

The system is designed to be both accurate and reliable for sign language recognition applications.
