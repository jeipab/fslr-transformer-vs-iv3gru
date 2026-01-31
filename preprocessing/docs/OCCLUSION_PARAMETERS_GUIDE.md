# Occlusion Detection Parameters Guide

This guide explains all parameters that control occlusion detection strictness and how to adjust them for stricter (less lenient) detection.

## Key Parameters for Stricter Detection

### 1. **Confidence Threshold** (Most Important)

**Current value:** `0.4`
**Location:** `DEFAULT_OCCLUSION_CONFIG` in `preprocessing/core/occlusion_detection.py`

Controls the minimum confidence score required for a frame to be considered occluded.

**To make stricter:**

- **Increase** to `0.5` - `0.7` (higher = more strict)
- This means detections need higher confidence scores

**Code location:**

```python
# In DEFAULT_OCCLUSION_CONFIG
'confidence_threshold': 0.4,  # Change this value
```

### 2. **Minimum Fingertips Inside Face Region**

**Current value:** `1`
**Location:** `DEFAULT_OCCLUSION_CONFIG` in `preprocessing/core/occlusion_detection.py`

Minimum number of fingertips that must be inside a face region to trigger occlusion.

**To make stricter:**

- **Increase** to `2` or `3` (requires more fingertips in face region)
- Currently only 1 fingertip needed, increasing requires more clear hand contact

**Code location:**

```python
# In DEFAULT_OCCLUSION_CONFIG
'min_fingertips_inside': 1,  # Change this value
```

### 3. **Proximity Multiplier**

**Current value:** `1.5`
**Location:** `DEFAULT_OCCLUSION_CONFIG` in `preprocessing/core/occlusion_detection.py`

Multiplier for proximity detection radius. Larger values detect hands further from face.

**To make stricter:**

- **Decrease** to `1.2` or `1.0` (smaller radius = hands must be closer)
- Reduces the "detection zone" around face regions

**Code location:**

```python
# In DEFAULT_OCCLUSION_CONFIG
'proximity_multiplier': 1.5,  # Decrease this value
```

### 4. **Face Region Radii**

**Current values:** Varies by region
**Location:** `_create_enhanced_face_regions()` in `preprocessing/core/occlusion_detection.py`

Size of detection regions for each face area.

**To make stricter:**

- **Decrease** all region radii multipliers:
  - Forehead: `0.12` → `0.10` or `0.08`
  - Cheeks: `0.15` → `0.12` or `0.10`
  - Nose: `0.10` → `0.08` or `0.06`
  - Mouth: `0.12` → `0.10` or `0.08`
  - Neck: `0.15` → `0.12` or `0.10`

**Code location:**

```python
# In _create_enhanced_face_regions()
'radius': 0.12 * face_scale,  # Forehead, mouth - decrease multipliers
'radius': 0.15 * face_scale,  # Cheeks, neck - decrease multipliers
'radius': 0.10 * face_scale,  # Nose - decrease multiplier
```

### 5. **Trajectory Analysis Multiplier**

**Current value:** `1.5`
**Location:** `_detect_occlusions_multi_method()` in `preprocessing/core/occlusion_detection.py`

Multiplier for trajectory-based detection (hand approaching face).

**To make stricter:**

- **Decrease** to `1.2` or `1.0` (hands must be closer when approaching)
- Reduces sensitivity for "approaching" detections

**Code location:**

```python
# In _detect_occlusions_multi_method()
if decreasing_count >= 2 and distances[-1] <= region_radius * 1.5:  # Decrease multiplier
```

### 6. **Orientation Detection Multiplier**

**Current value:** `1.8`
**Location:** `_detect_occlusions_multi_method()` in `preprocessing/core/occlusion_detection.py`

Multiplier for orientation-based detection (hand pointing toward face).

**To make stricter:**

- **Decrease** to `1.5` or `1.3` (hand must be more directly oriented)
- Currently very lenient, decreasing makes it stricter

**Code location:**

```python
# In _detect_occlusions_multi_method()
if min_distance <= region_radius * 1.8:  # Decrease multiplier
orientation_score = max(0, 1 - min_distance / (region_radius * 1.8))
```

### 7. **Consecutive Frame Parameters**

**Current values:**

- `consecutive_window_size`: `5`
- `max_consecutive_skips`: `2`
- `min_consecutive_confidence`: `0.2`

**Location:** `DEFAULT_OCCLUSION_CONFIG` in `preprocessing/core/occlusion_detection.py`

Controls temporal filtering - requires multiple consecutive frames to confirm occlusion.

**To make stricter:**

- **Increase** `consecutive_window_size` to `7` or `10` (more frames required)
- **Decrease** `max_consecutive_skips` to `1` or `0` (fewer missed frames allowed)
- **Increase** `min_consecutive_confidence` to `0.3` or `0.4` (higher confidence in window)

**Code location:**

```python
# In DEFAULT_OCCLUSION_CONFIG
'consecutive_window_size': 5,
'max_consecutive_skips': 2,
'min_consecutive_confidence': 0.2
```

### 8. **Method Weights** (Advanced)

**Current values:**

- Direct intersection: `0.5`
- Proximity: `0.3`
- Trajectory: `0.15`
- Orientation: `0.05`

**Location:** `_detect_occlusions_multi_method()` in `preprocessing/core/occlusion_detection.py`

Controls how much each detection method contributes to final confidence.

**To make stricter:**

- **Increase** direct intersection weight (more reliance on clear contact)
- **Decrease** proximity/trajectory/orientation weights (less reliance on indirect signals)

**Code location:**

```python
# In _detect_occlusions_multi_method()
confidence += 0.5   # Direct intersection - increase this
confidence += proximity_score * 0.3   # Proximity - decrease this
confidence += approach_score * 0.15   # Trajectory - decrease this
confidence += orientation_score * 0.05   # Orientation - decrease or remove
```

## How to Apply Changes

### Option 1: Modify Default Values in Code

Edit `DEFAULT_OCCLUSION_CONFIG` in `preprocessing/core/occlusion_detection.py` or modify the hardcoded values in the detection functions.

### Option 2: Pass Parameters via kwargs

Update your rerun scripts to pass parameters:

```python
# In rerun_occlusion_i.py or rerun_occlusion_c.py
occluded = compute_occlusion_detection_from_keypoints(
    X=X,
    mask=mask,
    output_format='compatible',
    min_fingertips_inside=2,      # Stricter: require 2 fingertips
    proximity_multiplier=1.2,      # Stricter: smaller detection radius
    confidence_threshold=0.5,      # Stricter: higher confidence required
)
```

However, note that some parameters are hardcoded in the function and would need code modifications.

## Recommended Stricter Settings

For **moderately stricter** detection:

- `confidence_threshold`: `0.4` → `0.5`
- `min_fingertips_inside`: `1` → `2`
- `proximity_multiplier`: `1.5` → `1.2`

For **very strict** detection:

- `confidence_threshold`: `0.4` → `0.6`
- `min_fingertips_inside`: `1` → `2` or `3`
- `proximity_multiplier`: `1.5` → `1.0`
- Face region radii: reduce all by 20-30%
- `trajectory_multiplier`: `1.5` → `1.2`
- `orientation_multiplier`: `1.8` → `1.3`

## Testing Strategy

1. Start with **confidence_threshold** (easiest to adjust)
2. Then adjust **min_fingertips_inside**
3. Finally adjust **proximity_multiplier** and region radii
4. Use your rerun scripts to compare old vs new counts
