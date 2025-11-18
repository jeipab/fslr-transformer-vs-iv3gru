# Occlusion Detection Parameters Guide

This guide explains all parameters that control occlusion detection strictness and how to adjust them for stricter (less lenient) detection.

## Key Parameters for Stricter Detection

### 1. **Confidence Threshold** (Most Important)

**Current value:** `0.4` (line 790, 1377)
**Location:** `compute_occlusion_detection_from_keypoints()`

Controls the minimum confidence score required for a frame to be considered occluded.

**To make stricter:**

- **Increase** to `0.5` - `0.7` (higher = more strict)
- This means detections need higher confidence scores

**Code location:**

```python
if confidence > 0.4:  # Line 790 - change this value
```

### 2. **Minimum Fingertips Inside Face Region**

**Current value:** `1` (line 705, 1374)
**Location:** `compute_occlusion_detection_from_keypoints()`

Minimum number of fingertips that must be inside a face region to trigger occlusion.

**To make stricter:**

- **Increase** to `2` or `3` (requires more fingertips in face region)
- Currently only 1 fingertip needed, increasing requires more clear hand contact

**Code location:**

```python
min_fingertips_inside = 1  # Line 705 - change this value
```

### 3. **Proximity Multiplier**

**Current value:** `1.5` (line 706, 1375)
**Location:** `_detect_occlusions_multi_method()`

Multiplier for proximity detection radius. Larger values detect hands further from face.

**To make stricter:**

- **Decrease** to `1.2` or `1.0` (smaller radius = hands must be closer)
- Reduces the "detection zone" around face regions

**Code location:**

```python
proximity_multiplier = 1.5  # Line 706 - decrease this value
proximity_radius = region_radius * proximity_multiplier  # Line 1000
```

### 4. **Face Region Radii**

**Current values:** Varies by region (lines 881, 892, 899, 910, 920)
**Location:** `_create_enhanced_face_regions()`

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
'radius': 0.12 * face_scale,  # Lines 881, 910 - decrease multipliers
'radius': 0.15 * face_scale,  # Lines 892, 920 - decrease multipliers
'radius': 0.10 * face_scale,  # Line 899 - decrease multiplier
```

### 5. **Trajectory Analysis Multiplier**

**Current value:** `1.5` (line 1018)
**Location:** `_detect_occlusions_multi_method()`

Multiplier for trajectory-based detection (hand approaching face).

**To make stricter:**

- **Decrease** to `1.2` or `1.0` (hands must be closer when approaching)
- Reduces sensitivity for "approaching" detections

**Code location:**

```python
if decreasing_count >= 2 and distances[-1] <= region_radius * 1.5:  # Line 1018 - decrease multiplier
```

### 6. **Orientation Detection Multiplier**

**Current value:** `1.8` (lines 1030, 1031)
**Location:** `_detect_occlusions_multi_method()`

Multiplier for orientation-based detection (hand pointing toward face).

**To make stricter:**

- **Decrease** to `1.5` or `1.3` (hand must be more directly oriented)
- Currently very lenient, decreasing makes it stricter

**Code location:**

```python
if min_distance <= region_radius * 1.8:  # Line 1030 - decrease multiplier
orientation_score = max(0, 1 - min_distance / (region_radius * 1.8))  # Line 1031
```

### 7. **Consecutive Frame Parameters**

**Current values:**

- `consecutive_window_size`: `5` (line 1381)
- `max_consecutive_skips`: `2` (line 1382)
- `min_consecutive_confidence`: `0.2` (line 1383)

Controls temporal filtering - requires multiple consecutive frames to confirm occlusion.

**To make stricter:**

- **Increase** `consecutive_window_size` to `7` or `10` (more frames required)
- **Decrease** `max_consecutive_skips` to `1` or `0` (fewer missed frames allowed)
- **Increase** `min_consecutive_confidence` to `0.3` or `0.4` (higher confidence in window)

**Code location:**

```python
'consecutive_window_size': 5,       # Line 1381
'max_consecutive_skips': 2,         # Line 1382
'min_consecutive_confidence': 0.2   # Line 1383
```

### 8. **Method Weights** (Advanced)

**Current values:**

- Direct intersection: `0.5` (line 994)
- Proximity: `0.3` (line 1003)
- Trajectory: `0.15` (line 1020)
- Orientation: `0.05` (line 1032)

Controls how much each detection method contributes to final confidence.

**To make stricter:**

- **Increase** direct intersection weight (more reliance on clear contact)
- **Decrease** proximity/trajectory/orientation weights (less reliance on indirect signals)

**Code location:**

```python
confidence += 0.5   # Line 994 - increase this
confidence += proximity_score * 0.3   # Line 1003 - decrease this
confidence += approach_score * 0.15   # Line 1020 - decrease this
confidence += orientation_score * 0.05   # Line 1032 - decrease or remove
```

## How to Apply Changes

### Option 1: Modify Default Values in Code

Edit `preprocessing/core/occlusion_detection.py` directly at the locations listed above.

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
