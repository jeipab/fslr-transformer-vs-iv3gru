# Occlusion Detection Guide

## Overview

Detects when hands obscure facial features during signing. Uses MediaPipe keypoints to identify hand-face occlusions in Filipino Sign Language videos.

**What It Does:**

- Analyzes hand positions (palm + fingertips) relative to 5 face regions
- Combines 4 detection methods with weighted scoring (0.5, 0.3, 0.15, 0.05)
- Requires detection in 3 out of 5 consecutive frames (filters noise)
- Returns binary flag: 0 (clean) or 1 (occluded)

## Quick Start

### Basic Usage

```powershell
# Enable occlusion detection (default when keypoints are written)
python -m preprocessing.core.preprocess data\raw\video.mp4 data\processed\output --write-keypoints --occ-enable

# Multi-processing version for batch processing
python -m preprocessing.core.preprocess data\raw\videos data\processed\output --write-keypoints --occ-enable --workers 8
```

### Detailed Results

```powershell
# Get detailed occlusion analysis
python -m preprocessing.core.preprocess data\raw\video.mp4 data\processed\output --write-keypoints --occ-enable --occ-detailed

# Multi-processing with detailed results
python -m preprocessing.core.preprocess data\raw\videos data\processed\output --write-keypoints --occ-enable --occ-detailed --workers 8
```

## How It Works

### Pipeline

```
Video Frame → MediaPipe Keypoints → Multi-Method Detection → Temporal Filtering → Binary Flag
     [T]            [89 points]           [4 methods]            [5-frame window]      [0 or 1]
```

### 1. Head Region Partitioning

Divides the face into 5 regions for detailed analysis:

| Region       | Description         | Radius (adaptive) | Purpose                                        |
| ------------ | ------------------- | ----------------- | ---------------------------------------------- |
| **Forehead** | Upper head area     | 0.12 × face_scale | Capture signs articulated at forehead          |
| **Cheeks**   | Eye and cheek areas | 0.15 × face_scale | Critical for visibility and facial expressions |
| **Nose**     | Central face region | 0.10 × face_scale | Key reference point for detection              |
| **Mouth**    | Lower face and chin | 0.12 × face_scale | Essential for mouth shape recognition          |
| **Neck**     | Below-chin area     | 0.15 × face_scale | Relevant for hand positioning                  |

Regions scale with face size - larger faces get larger detection zones.

### 2. Multi-Method Detection

**Implementation:** `_detect_occlusions_multi_method()` in `preprocessing/core/occlusion_detection.py`

```python
def _detect_occlusions_multi_method(
    palm_center: Optional[Tuple[float, float]],
    tips: List[Tuple[float, float]],
    face_regions: Dict[str, Dict],
    trajectory: List[Tuple[int, Tuple[float, float]]],
    current_frame: int,
    min_fingertips_inside: int = 1,
    proximity_multiplier: float = 1.5
) -> Dict[str, float]:
    """Detect occlusions using 4 methods, returns confidence per region."""
```

Four methods run in parallel:

| Method                     | Weight | Description                                        | Distance Metric                         |
| -------------------------- | ------ | -------------------------------------------------- | --------------------------------------- |
| **Fingertip Intersection** | 0.5    | Direct detection of fingertips inside face regions | Euclidean distance ≤ region radius      |
| **Palm Proximity**         | 0.3    | Palm center proximity analysis                     | Euclidean distance ≤ 1.5× region radius |
| **Trajectory Analysis**    | 0.15   | Hand movement toward face (5-frame history)        | Distance decreasing over time           |
| **Hand Orientation**       | 0.05   | Overall hand orientation toward face               | Minimum distance from all hand points   |

**Combined Score**: `min(0.5×method1 + 0.3×method2 + 0.15×method3 + 0.05×method4, 1.0)`

### 3. Temporal Filtering

**Implementation:** `_apply_temporal_filtering()` in `preprocessing/core/occlusion_detection.py`

```python
def _apply_temporal_filtering(
    results: List[Dict],
    config: Dict = None
) -> List[Dict]:
    """Apply consecutive frame filtering.

    Requires detection in 3 out of 5 consecutive frames.
    Allows up to 2 missed frames within window.
    """
```

**Parameters:**

- Window: 5 consecutive frames
- Required detections: ≥3 out of 5 (60%)
- Max gaps: 2 frames
- Confidence threshold: 0.2

**At 30 FPS:**

- 5 frames = 167ms window
- 3 detections = 100ms minimum duration

**Logic:**

```python
# For each frame, check 5-frame window
window = frames[i-2 : i+3]
valid = sum(frame.confidence ≥ 0.2 for frame in window)
occluded = (valid ≥ 3)
```

**Why?** Filters MediaPipe tracking noise (motion blur, lighting changes) while catching real occlusions.

### 4. Adaptive Region Sizing

**Implementation:** `_calculate_face_scale()` in `preprocessing/core/occlusion_detection.py`

Regions scale with face size:

```python
face_scale = (face_width + face_height) / 2 / 0.35
nose_radius = 0.10 * face_scale  # Scales automatically
```

Works at any distance or with different face sizes (child vs adult).

### Method Details

**Implementation:** Helper functions in `preprocessing/core/occlusion_detection.py`

```python
_hand_centers_and_tips()  # Extracts palm + 5 fingertips from MediaPipe
_validate_face_landmarks()  # Filters invalid face keypoints
```

#### Method 1: Fingertip Intersection (Weight=0.5)

Checks if any fingertip is inside a face region.

- Extract 5 fingertips from MediaPipe hands
- Compute distance to region center
- If distance ≤ region radius → add 0.5 to score
- **Strongest evidence** of occlusion

#### Method 2: Palm Proximity (Weight=0.3)

Checks if palm is near the face.

- Compute palm center from MCP joints (or use wrist)
- If distance ≤ 1.5× region radius → add score
- Formula: `(1 - distance/proximity_radius) × 0.3`
- **Accounts for hand size** - palm near = likely occlusion

#### Method 3: Trajectory (Weight=0.15)

Tracks if hand is moving toward face.

- Keep 5-frame history of hand positions
- Check if distance is decreasing over time
- If approaching + close → add score
- **Catches approaching hands** before full occlusion

#### Method 4: Orientation (Weight=0.05)

Checks overall hand direction.

- Get all 6 hand points (palm + tips)
- Find minimum distance to face region
- If close enough → add small score
- **Supplementary signal** for hand-face alignment

## Why These Numbers?

### 5 Frames @ 30 FPS = 167ms

**Reason:** Long enough to catch real occlusions, short enough to filter noise.

**Details:**

- FSL signs last 300-800ms → 167ms is shorter, catches even fast signs
- Intentional contact takes >150ms → 167ms separates deliberate from accidental touch
- MediaPipe has frame-level failures → multi-frame check filters them out

**Research:** Temporal filtering is standard for occlusion detection [[MIT]](https://visionbook.mit.edu/temporal_filters_v2.html), motor control studies show intentional movements need ~150-200ms [[Ref]](https://towardsdatascience.com/how-you-can-detect-partially-occluded-objects-using-temporal-context-3db1194e7171).

### 2-Frame Skip Tolerance

**Reason:** MediaPipe misses keypoints sometimes - we allow 2 gaps to handle this.

**Why MediaPipe fails:**

- Motion blur during fast movements
- Lighting changes (shadows, glare)
- Extreme hand angles
- Momentary self-occlusion

**Rule:** Need 3 out of 5 frames detecting occlusion (60% minimum). Allows 2 frames to fail while still catching real occlusions.

**Research:** Visual tracking systems use "gap tolerance" to handle detection failures [[PMC]](https://pmc.ncbi.nlm.nih.gov/articles/PMC7039229).

### Method Weights: 0.5, 0.3, 0.15, 0.05

**Reason:** Fingertips are strongest evidence (0.5), palm is next (0.3), trajectory/orientation help but aren't decisive (0.15, 0.05).

**Why these weights:**
| Method | Weight | Why |
|--------|--------|-----|
| Fingertips inside | 0.5 | Direct contact = strongest evidence |
| Palm nearby | 0.3 | Hand close = likely occlusion |
| Moving toward face | 0.15 | Predictive signal |
| Hand pointing at face | 0.05 | Weak signal, just adds context |

**Research:** Combining detection methods improves accuracy [[MDPI]](https://www.mdpi.com/2079-9292/10/1/43).

### Confidence: 0.4 per-frame, 0.2 temporal

**Reason:** Be strict for individual frames (0.4), lenient across time (0.2).

**Why two thresholds:**

- **0.4 per-frame**: Filters weak individual detections
- **0.2 temporal**: Allows variation across frames

Example: Frame scores 0.35 (fails alone) but surrounded by strong frames (0.6, 0.5, 0.35, 0.4, 0.6) → counts toward 3/5 requirement.

**Research:** Standard practice [[MDPI]](https://www.mdpi.com/1999-4893/12/5/92).

### Distance Metric: Euclidean in Normalized [0,1] Space

**Reason:** Use simple distance formula on MediaPipe's normalized coordinates.

**Why:** MediaPipe gives coordinates in [0,1] range → signer position/camera distance don't matter. Just compute: `√((x1-x2)² + (y1-y2)²)`

**Research:** Standard for pose estimation [[MDPI]](https://www.mdpi.com/1999-4893/12/5/92).

### Why Not Different Numbers?

| Our Choice | Alternative | Why Not?                                |
| ---------- | ----------- | --------------------------------------- |
| 5 frames   | 3 frames    | Too short - catches noise               |
| 5 frames   | 10 frames   | Too long - misses brief contact         |
| 2 skips    | 0 skips     | Too strict - too many false negatives   |
| 2 skips    | 4 skips     | Too loose - allows fragmented detection |
| 0.2/0.4    | 0.6/0.8     | Too strict - misses real occlusions     |
| 0.2/0.4    | 0.1/0.2     | Too loose - too many false positives    |

### References

All parameter choices are based on:

1. **Temporal Filtering** - [[MIT Vision Book]](https://visionbook.mit.edu/temporal_filters_v2.html)
2. **Occlusion Detection** - [[Towards Data Science]](https://towardsdatascience.com/how-you-can-detect-partially-occluded-objects-using-temporal-context-3db1194e7171)
3. **Visual Tracking** - [[PMC Study]](https://pmc.ncbi.nlm.nih.gov/articles/PMC7039229)
4. **Multi-Method Fusion** - [[MDPI Electronics]](https://www.mdpi.com/2079-9292/10/1/43)
5. **Euclidean Distance** - [[MDPI Algorithms]](https://www.mdpi.com/1999-4893/12/5/92)
6. **Occlusion Handling** - [[UnitX Labs]](https://resources.unitxlabs.com/occlusion-machine-vision-systems)

## Configuration

### Default Parameters

The system uses the following configuration (from `DEFAULT_OCCLUSION_CONFIG`):

```python
config = {
    # Core detection sensitivity parameters
    'min_face_points': 3,                    # Minimum face keypoints required
    'min_hand_points': 3,                    # Minimum hand keypoints required
    'min_fingertips_inside': 1,              # Minimum fingertips in face region
    'proximity_multiplier': 1.5,             # Proximity radius multiplier
    'occlusion_threshold': 0.15,             # Overall occlusion threshold
    'confidence_threshold': 0.4,             # Per-frame confidence minimum
    'temporal_confidence': 0.5,              # Temporal consistency threshold (unused in filtering)

    # Consecutive frame analysis parameters
    'consecutive_window_size': 5,            # Require 5 consecutive frames
    'max_consecutive_skips': 2,              # Allow up to 2 missed frames
    'min_consecutive_confidence': 0.2,       # Minimum confidence for temporal analysis

    # Advanced tracking parameters
    'use_global_tracking': True,             # Enable sophisticated tracking
    'gridlet_size': 4,                       # Size of point groups for tracking
    'tracking_window_size': 5,               # Temporal window for consistency
    'motion_threshold': 10,                  # Motion detection threshold
    'temporal_filtering': True,              # Enable temporal filtering
    'output_detailed_results': False         # Control output verbosity
}
```

### Parameter Descriptions

| Parameter                    | Value | Purpose                                     | Impact                                        |
| ---------------------------- | ----- | ------------------------------------------- | --------------------------------------------- |
| `min_face_points`            | 3     | Minimum face keypoints for valid frame      | Lower = more coverage, higher false positives |
| `min_hand_points`            | 3     | Minimum hand keypoints for analysis         | Lower = more sensitive, less reliable         |
| `min_fingertips_inside`      | 1     | Fingertips required for direct intersection | Lower = catch subtle occlusions               |
| `proximity_multiplier`       | 1.5   | Palm proximity radius multiplier            | Higher = more sensitive proximity detection   |
| `confidence_threshold`       | 0.4   | Per-frame detection confidence              | Higher = fewer false positives                |
| `min_consecutive_confidence` | 0.2   | Temporal filtering confidence               | Lower = more lenient across frames            |
| `consecutive_window_size`    | 5     | Frames in temporal window                   | Larger = more robust, less responsive         |
| `max_consecutive_skips`      | 2     | Allowed gaps in detection                   | Higher = more noise tolerance                 |

### Occlusion Criteria

#### Per-Frame Occlusion Detection

A frame is marked as occluded if:

```python
confidence_score ≥ 0.4  # Per-frame threshold
```

Where `confidence_score` is computed from:

```python
confidence = min(
    0.5 × fingertip_intersection +
    0.3 × palm_proximity +
    0.15 × trajectory_score +
    0.05 × orientation_score,
    1.0
)
```

#### Temporal Filtering (Consecutive Frame Analysis)

A frame's final occlusion status requires:

```python
# Check 5-frame window
valid_detections = count(confidence ≥ 0.2 in window)

# Require at least 3 out of 5 frames
frame_occluded = (valid_detections ≥ 3)
```

#### Clip-Level Occlusion Flag

The video receives binary flag `1` (occluded) if **any** temporal pattern is detected:

```python
# After temporal filtering
total_occlusions = sum(frame.occluded for frame in frames)

binary_flag = 1 if total_occlusions > 0 else 0
```

**Note**: The system prioritizes recall over precision—any detected occlusion pattern flags the entire clip for manual review.

### Customizing Parameters

You can override default parameters when calling detection functions:

```python
from preprocessing.core.occlusion_detection import compute_occlusion_detection

# Custom configuration
custom_results = compute_occlusion_detection(
    X=keypoints,
    mask=visibility_mask,
    output_format='detailed',
    consecutive_window_size=7,      # Stricter: 7 frames instead of 5
    max_consecutive_skips=1,        # Less tolerant: 1 skip instead of 2
    min_consecutive_confidence=0.3  # Higher confidence: 0.3 instead of 0.2
)
```

**When to Customize**:

- **More strict** (reduce false positives): Increase window size, reduce skips, increase confidence
- **More lenient** (reduce false negatives): Decrease window size, increase skips, decrease confidence
- **Different FPS**: Adjust window size proportionally (e.g., 60 FPS → 10 frames for same 167ms)

## API Reference

### Core Functions

```python
from preprocessing.core.occlusion_detection import (
    compute_occlusion_detection,
    compute_occlusion_detection_from_keypoints,
    HandHeadOcclusionDetector,
    get_occlusion_config,
    validate_occlusion_config,
    DEFAULT_OCCLUSION_CONFIG
)
```

### Main Detection Function

**Function**: `compute_occlusion_detection()`

```python
# From video file (requires OpenCV + MediaPipe)
result = compute_occlusion_detection(
    video_path="path/to/video.mp4",
    output_format="compatible",  # or "detailed"
    **kwargs  # Override config parameters
)

# From preprocessed keypoints (recommended)
result = compute_occlusion_detection(
    X=keypoints_array,          # [T, 178] normalized coordinates
    mask_bool_array=mask,       # [T, 89] visibility mask
    output_format="detailed",
    consecutive_window_size=5,  # Optional overrides
    max_consecutive_skips=2,
    min_consecutive_confidence=0.2
)
```

**Parameters**:

- `video_path` (str, optional): Path to video file
- `X` (np.ndarray, optional): Keypoint coordinates [T, 178]
- `mask_bool_array` (np.ndarray, optional): Visibility mask [T, 89]
- `output_format` (str): `'compatible'` (binary) or `'detailed'` (full results)
- `**kwargs`: Configuration overrides (see Configuration section)

**Returns**:

**Compatible format** (binary):

```python
0  # No occlusion detected
1  # Occlusion detected
```

**Detailed format** (dictionary):

```python
{
    'binary_flag': 0 or 1,
    'occlusion_rate': 0.35,           # 35% of frames
    'total_frames': 120,
    'occluded_frames': 42,
    'detailed_results': [
        {
            'frame_idx': 0,
            'occlusion_detected': False,
            'occluded_regions': [],
            'confidence': 0.0
        },
        # ... more frames
    ]
}
```

### Keypoint-Based Detection (Primary Method)

**Function**: `compute_occlusion_detection_from_keypoints()`

```python
result = compute_occlusion_detection_from_keypoints(
    X=keypoints,          # [T, 178]
    mask=visibility_mask,  # [T, 89]
    output_format='compatible',
    **kwargs
)
```

This is the **recommended method** as it works directly with preprocessed keypoints, bypassing video decoding.

### HandHeadOcclusionDetector Class

For raw video processing with frame-by-frame control:

```python
from preprocessing.core.occlusion_detection import HandHeadOcclusionDetector
import cv2

# Initialize detector
detector = HandHeadOcclusionDetector(use_global_tracking=True)

# Process video frame-by-frame
cap = cv2.VideoCapture("video.mp4")
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.process_frame(frame, frame_idx=len(results))
    results.append(result)

    print(f"Frame {result['frame_idx']}: "
          f"Occluded={result['occlusion_detected']}, "
          f"Regions={result['occluded_regions']}")

cap.release()
```

### Configuration Management

```python
# Get default configuration
config = get_occlusion_config()
print(config['consecutive_window_size'])  # 5

# Validate configuration
is_valid = validate_occlusion_config(config)

# Access default config constant
from preprocessing.core.occlusion_detection import DEFAULT_OCCLUSION_CONFIG
print(DEFAULT_OCCLUSION_CONFIG)
```

## Integration Examples

### Preprocessing Pipeline Integration

The occlusion detector integrates seamlessly with the preprocessing pipeline:

```python
from preprocessing.core.preprocess import process_video

# Process video with occlusion detection
process_video(
    video_path="input.mp4",
    out_dir="output/",
    write_keypoints=True,           # Required for occlusion detection
    compute_occlusion=True,         # Enable occlusion detection
    occ_detailed=False              # Set to True for detailed results
)

# The NPZ file will contain:
# - 'X': keypoints [T, 178]
# - 'mask': visibility [T, 89]
# - 'meta': {'occluded_flag': 0 or 1, ...}
```

### Batch Processing with Occlusion

```python
from preprocessing.core.preprocess import process_videos_multiprocess

# Process multiple videos
results = process_videos_multiprocess(
    video_files=['video1.mp4', 'video2.mp4', 'video3.mp4'],
    out_dir='output/',
    write_keypoints=True,
    compute_occlusion=True,
    workers=4  # Parallel processing
)

# Check occlusion flags
for filename, npz_data in results.items():
    meta = json.loads(str(npz_data['meta']))
    print(f"{filename}: Occluded={meta['occluded_flag']}")
```

## Batch Processing Example

```powershell
# Process entire dataset with occlusion detection
python -m preprocessing.core.preprocess ^
  data\raw\fsl-105 ^
  data\processed\fsl-105_output ^
  --write-keypoints ^
  --write-iv3-features ^
  --occ-enable ^
  --workers 8 ^
  --batch-size 32
```

## NPZ Metadata

Occlusion information is stored in the NPZ file's metadata:

```python
import numpy as np
import json

# Load NPZ file
data = np.load('clip_0001.npz', allow_pickle=True)

# Access metadata
meta = json.loads(str(data['meta']))
print(f"Occluded: {meta.get('occluded_flag', 0)}")
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

   ```powershell
   pip install scipy scikit-learn
   ```

2. **Low Detection Rate**: Check video quality and lighting conditions

3. **False Positives**: System uses conservative thresholds by design

### Debug Mode

Enable detailed output for debugging:

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\video.mp4 ^
  data\processed\output ^
  --write-keypoints ^
  --occ-enable ^
  --occ-detailed
```

This will provide frame-by-frame analysis in the metadata.

## Research Context

**Built for:** Filipino Sign Language recognition research (Transformer vs InceptionV3-GRU comparison).

**What it enables:**

- Flag videos with hand-face occlusions
- Compare model performance on clean vs occluded data
- Manual review of flagged videos

**Design:** Prioritizes catching all occlusions (high recall) over avoiding false positives. Better to flag for review than miss real occlusions.

**Tested on:** FSL-105 dataset (2,130 videos, 4 signers).

**Detailed justification:** See [Why These Numbers?](#why-these-numbers) section for parameter defense.
