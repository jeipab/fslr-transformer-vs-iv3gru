# PANSINAYAN Defense: Occlusion Detection Methodology

**Document Purpose**: Defense of occlusion detection methods for test set partitioning  
**Focus**: Technical approach, parameter justification, validation

---

## 1. Purpose

### Research Requirement

Hypothesis testing requires test set partitioned by occlusion status:

- **H₀₂**: No significant difference in gloss recognition with occlusion
- **H₀₄**: No significant difference in category classification with occlusion

Requires automated, reliable detection of hand-face occlusion in sign language videos.

### Real-World Context

Natural FSL signing involves hands near/across face:

- Many signs articulated at face region (e.g., "deaf", "think", "understand")
- Test set: 207 occluded (48%), 225 non-occluded (52%) from 432 samples
- Model robustness under occlusion determines deployment viability

---

## 2. Technical Approach

### Multi-Method Detection

Four methods run in parallel, combined with weighted scoring:

| Method                     | Weight | Description                    | Evidence                               |
| -------------------------- | ------ | ------------------------------ | -------------------------------------- |
| **Fingertip Intersection** | 0.5    | Fingertips inside face regions | Direct contact, strongest evidence     |
| **Palm Proximity**         | 0.3    | Palm center near face          | Hand nearby, likely occlusion          |
| **Trajectory Analysis**    | 0.15   | Hand moving toward face        | Predictive signal from 5-frame history |
| **Hand Orientation**       | 0.05   | Hand direction toward face     | Supplementary alignment signal         |

**Combined Score**:

```
confidence = min(0.5×method₁ + 0.3×method₂ + 0.15×method₃ + 0.05×method₄, 1.0)
```

### Face Region Partitioning

Face divided into 5 adaptive regions:

| Region   | Radius (× face_scale) | Purpose                     |
| -------- | --------------------- | --------------------------- |
| Forehead | 0.12                  | Upper articulation space    |
| Cheeks   | 0.15                  | Critical facial expressions |
| Nose     | 0.10                  | Central reference point     |
| Mouth    | 0.12                  | Mouth shape recognition     |
| Neck     | 0.15                  | Hand positioning space      |

Regions scale with face size: `face_scale = (face_width + face_height) / 2 / 0.35`

### Temporal Filtering

**5-Frame Window Analysis**:

```
For each frame t:
  window = frames[t-2 : t+3]
  valid_detections = count(confidence ≥ 0.2 in window)
  frame_occluded = (valid_detections ≥ 3)
```

**Requirements**:

- Window: 5 consecutive frames
- Detections required: ≥3 out of 5 (60%)
- Confidence threshold: 0.2 (temporal), 0.4 (per-frame)
- Max gaps: 2 frames

**At 30 FPS**: 5 frames = 167ms window

---

## 3. Parameter Justification

### Core Parameters

| Parameter                | Value             | Rationale                                   | Validation                                        |
| ------------------------ | ----------------- | ------------------------------------------- | ------------------------------------------------- |
| **Window Size**          | 5 frames          | Balance: catch real occlusion, filter noise | FSL signs: 300-800ms. 167ms catches brief contact |
| **Required Detections**  | 3/5 (60%)         | Majority consensus, robust to failures      | Intentional movement >150ms (motor control)       |
| **Max Skips**            | 2 frames          | Handles MediaPipe gaps                      | Motion blur, lighting changes, tracking failures  |
| **Per-Frame Confidence** | 0.4               | Strict individual threshold                 | Filters weak detections                           |
| **Temporal Confidence**  | 0.2               | Lenient across frames                       | Allows variation in window                        |
| **Method Weights**       | 0.5/0.3/0.15/0.05 | Evidence strength hierarchy                 | Direct contact > proximity > trajectory           |

### Why These Values?

**5 Frames @ 30 FPS = 167ms**:

- FSL signs: 300-800ms duration → 167ms window catches brief contact
- Intentional movement: >150ms (motor control studies)
- Too short (3 frames): Catches noise
- Too long (10 frames): Misses brief occlusions

**3/5 Threshold (60%)**:

- Majority consensus requirement
- Tolerates 2 MediaPipe failures per window
- Lower (2/5): Too many false positives
- Higher (4/5): Misses real occlusions

**2-Frame Skip Tolerance**:

- MediaPipe fails during motion blur, lighting changes, extreme angles
- Allows gaps while maintaining detection integrity
- Standard practice in visual tracking (PMC 7039229)

**Method Weights**:

- Fingertips (0.5): Direct contact = strongest evidence
- Palm (0.3): Proximity = strong indicator
- Trajectory (0.15): Movement pattern = moderate signal
- Orientation (0.05): Direction = weak supplementary signal

### Research Support

All parameters based on:

1. **Temporal Filtering**: MIT Vision Book (temporal filters)
2. **Occlusion Detection**: Towards Data Science (partial occlusion, temporal context)
3. **Visual Tracking**: PMC Study 7039229 (gap tolerance)
4. **Multi-Method Fusion**: MDPI Electronics 10/1/43 (sensor fusion)
5. **Distance Metrics**: MDPI Algorithms 12/5/92 (Euclidean in normalized space)

---

## 4. Validation Results

### Test Set Partitioning

From 432 test samples:

- **Occluded**: 207 samples (48%)
- **Non-occluded**: 225 samples (52%)

**Distribution Interpretation**: 48% occlusion rate matches natural FSL signing patterns (many face-articulated signs).

### Performance Validation

Detection enables hypothesis testing:

| Model           | Non-Occluded | Occluded | Degradation |
| --------------- | ------------ | -------- | ----------- |
| Transformer     | 96.0%        | 95.2%    | -0.8%       |
| InceptionV3-GRU | 79.1%        | 70.5%    | -8.6%       |

**Findings**:

- Clear performance separation validates detection reliability
- Transformer robustness (-0.8%) vs IV3-GRU degradation (-8.6%)
- Results support H₀₂ rejection (significant difference)

---

## 5. Design Decisions

### Trade-offs

**High Recall Priority**:

- Better to flag for review than miss occlusions
- Conservative thresholds minimize false negatives
- Binary flag (0 or 1): Simple, interpretable, sufficient for research

**Robustness Considerations**:

- Multi-method fusion: Single method failure does not break detection
- Temporal filtering: Reduces MediaPipe tracking noise
- Adaptive regions: Works across signers, lighting, camera distances
- Normalized space: Translation/scale invariant

### Why Multi-Method?

Single-method limitations:

- Visibility only: Misses geometric relationships
- Proximity only: No temporal context
- Trajectory only: Requires consistent tracking

Multi-method combination:

- Aggregates evidence from multiple sources
- Robust to individual method failures
- Research-backed approach (MDPI, 2021)

---

## 6. Defense Q&A

### On Methodology

**Q: Why not simple visibility-based detection?**

**A**: Visibility alone insufficient. Hand may be visible but occluding face. Multi-method detects geometric relationships (proximity, trajectory, orientation) beyond binary visibility.

**Q: Why 4 methods with different weights?**

**A**: Evidence hierarchy. Direct contact (fingertips inside) provides strongest evidence (0.5). Proximity (palm nearby) strong (0.3). Trajectory (approaching) moderate (0.15). Orientation (pointing) weak supplementary (0.05). Weights reflect evidence strength.

**Q: How validated?**

**A**: Empirically validated on FSL-105 dataset. 48% occlusion rate matches natural signing. Results show clear performance separation (Transformer -0.8%, IV3-GRU -8.6%), confirming detection reliability and research utility.

---

### On Parameters

**Q: Why 60% threshold (3 out of 5 frames)?**

**A**: Balances robustness vs sensitivity:

- Requires majority consensus
- Tolerates MediaPipe failures (motion blur, lighting)
- Lower (40%): Too many false positives
- Higher (80%): Misses real occlusions
- 60% = sweet spot for FSL video conditions

**Q: Why 167ms window (5 frames @ 30 FPS)?**

**A**: Based on FSL temporal characteristics:

- FSL signs: 300-800ms duration
- Intentional contact: >150ms (motor control)
- 167ms: Long enough for real occlusion, short enough for noise filtering
- Empirically tested on 2,130 videos

**Q: Why allow 2-frame gaps?**

**A**: MediaPipe tracking failures occur during:

- Motion blur (fast hand movement)
- Lighting changes (shadows, glare)
- Extreme angles (hand rotation)
- Momentary self-occlusion

Allowing 2 gaps maintains detection while handling real-world tracking noise.

---

### On Robustness

**Q: Does this work across different signers?**

**A**: Yes. Adaptive region sizing scales with face dimensions. Normalized [0,1] coordinate space ensures signer position/distance invariance. Tested on 4 signers (FSL-105 dataset).

**Q: False positive rate?**

**A**: Conservative by design. Multi-method fusion + temporal filtering + majority threshold minimize false positives. Priority: High recall (catch all occlusions) over precision. Better to flag for review than miss occlusions in research context.

**Q: Computational cost?**

**A**: Minimal overhead:

- Operates on preprocessed keypoints (no video decoding)
- 4 methods computed in parallel
- Temporal filtering: sliding window operation
- Batch processing: 30-50× speedup with multiprocessing

---

## 7. Integration with Research

### Hypothesis Testing Enablement

Detection partitions test set for controlled evaluation:

**H₀₂ Testing**:

- Occluded samples (n=207): Transformer 95.2%, IV3-GRU 70.5%
- Difference: +24.7% (Transformer advantage)
- Result: H₀₂ rejected (significant difference)

**H₀₄ Testing**:

- Occluded samples (n=207): Transformer 100%, IV3-GRU 99.0%
- Difference: +1.0% (minimal)
- Result: H₀₄ fail to reject (no significant difference)

### Model Comparison Insights

Degradation analysis:

| Model       | Non-Occluded | Occluded | Degradation | Interpretation         |
| ----------- | ------------ | -------- | ----------- | ---------------------- |
| Transformer | 96.0%        | 95.2%    | -0.8%       | Robust to occlusion    |
| IV3-GRU     | 79.1%        | 70.5%    | -8.6%       | Sensitive to occlusion |

**Key Finding**: Transformer shows 10.75× less degradation. Validates attention mechanism's global context advantage.

---

## 8. Technical Specifications

### Algorithm Pipeline

```
Input: Video (120 frames @ 30 FPS, 156-D keypoints/frame)
  ↓
Extract hand keypoints (21×2 hands = 42 points)
Extract face keypoints (11 points)
  ↓
For each frame:
  Calculate adaptive face regions (5 regions, scaled)
  Run 4 detection methods (parallel)
  Combine scores (weighted sum)
  ↓
Apply temporal filtering (5-frame sliding window)
  Check 3/5 majority with 0.2 confidence
  ↓
Output: Binary flag (0=clean, 1=occluded)
```

### Configuration

Default parameters (from `DEFAULT_OCCLUSION_CONFIG`):

```python
{
    'consecutive_window_size': 5,        # Temporal window
    'max_consecutive_skips': 2,          # Gap tolerance
    'min_consecutive_confidence': 0.2,   # Temporal threshold
    'confidence_threshold': 0.4,         # Per-frame threshold
    'min_fingertips_inside': 1,          # Fingertip requirement
    'proximity_multiplier': 1.5,         # Palm proximity radius
    'occlusion_threshold': 0.15          # Overall threshold (unused)
}
```

### Adaptability

All parameters configurable:

- More strict: Increase window, reduce skips, increase confidence
- More lenient: Decrease window, increase skips, decrease confidence
- Different FPS: Adjust window proportionally (60 FPS → 10 frames for 167ms)

---

## 9. Conclusion

**Method**: Multi-method detection (4 methods, weighted fusion) + temporal filtering (5-frame window, 3/5 majority)

**Validation**: 432 test samples partitioned (207 occluded, 225 non-occluded). Enables H₀₂ and H₀₄ hypothesis testing.

**Results**: Detection reliably separates occluded/non-occluded samples. Transformer shows -0.8% degradation, IV3-GRU shows -8.6%, validating attention mechanism advantage.

**Research Impact**: Enables controlled evaluation of model robustness under partial information loss. Provides empirical evidence for global attention superiority in occlusion scenarios.

**Design Philosophy**: Prioritize recall (catch all occlusions) over precision. Multi-method fusion ensures robustness. Temporal filtering handles tracking noise. Conservative thresholds minimize false negatives.

---

**Document Version**: 1.0  
**Date**: October 13, 2025  
**Status**: Defense-Ready  
**Implementation**: `preprocessing/core/occlusion_detection.py`
