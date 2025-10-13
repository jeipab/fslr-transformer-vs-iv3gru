# PANSINAYAN Defense: Results Analysis & Interpretation

**Document Purpose**: Defense preparation for tool presentation  
**Focus**: Computing problem, results interpretation, technical rationale

---

## TL;DR

**The Computing Problem**: Filipino Sign Language recognition requires solving temporal dependencies (120-frame sequences), multi-modal fusion (hands + face keypoints), and occlusion robustness—all while maintaining real-time performance. Sequential models fail under partial information loss.

**The Results**:

- **Gloss Recognition**: Transformer **95.6%** vs InceptionV3-GRU 75.0% (**+20.6%**)
- **Under Occlusion**: Transformer **95.2%** vs IV3-GRU 70.5% (**+24.7%**)
- **Degradation**: Transformer **-0.8%** vs IV3-GRU -8.6% (**10.75× more robust**)
- **Category Classification**: Transformer 99.5% vs IV3-GRU 99.1% (no significant difference)

**Why Transformer Wins**:

- **Global Attention**: O(1) access to any frame, no sequential bottleneck
- **Parallel Processing**: All 120 frames processed simultaneously, no error propagation
- **Dynamic Weighting**: Occlusion triggers weight redistribution to visible frames

**Why IV3-GRU Struggles**:

- **Sequential Bottleneck**: O(T) access, processes frame-by-frame
- **Hidden State Compression**: 12-D bottleneck loses information across 120 frames
- **Error Propagation**: Corrupted frames contaminate future time steps

**Statistical Significance**:

- **H₀₁ REJECTED**: Transformer significantly outperforms on gloss recognition (p < 0.05)
- **H₀₂ REJECTED**: Transformer significantly more robust under occlusion (p < 0.05)
- **H₀₃ FAIL TO REJECT**: No significant difference in category classification
- **H₀₄ FAIL TO REJECT**: Both maintain category accuracy under occlusion

**Research Impact**: First production-ready FSL system with multi-head attention achieving 95.6% gloss accuracy, 99.5% category accuracy, and exceptional occlusion robustness. Provides empirical evidence that attention mechanisms outperform CNN-RNN architectures for sign language recognition.

---

## 1. The Computing Problem

Filipino Sign Language recognition requires solving:

1. **Temporal Dependencies**: Sign meaning depends on full gesture trajectory, not isolated frames
2. **Multi-Modal Integration**: Manual features (hands) + non-manual markers (face) must be processed simultaneously
3. **Occlusion Handling**: Hands frequently obscure facial landmarks during natural signing
4. **Computational Efficiency**: Process full sequences while maintaining accuracy

**Core Challenge**: Balance global context capture with computational tractability while handling partial information loss.

**Computing Requirements**:

- Long-range dependency modeling across 120-frame sequences
- Robust feature representations under data corruption
- Real-time inference capability
- Multi-modal feature fusion

---

## 2. Research Questions & Hypotheses

Four null hypotheses tested:

- **H₀₁**: No significant difference in gloss recognition (without occlusion)
- **H₀₂**: No significant difference in gloss recognition (with occlusion)
- **H₀₃**: No significant difference in category classification (without occlusion)
- **H₀₄**: No significant difference in category classification (with occlusion)

**Statistical Method**: Paired t-test or Wilcoxon signed-rank test (α = 0.05)  
**Dataset**: FSL-105 (432 test samples, 207 occluded, 225 non-occluded)

---

## 3. Performance Metrics Explained

### Definitions

**Accuracy**:

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

Percentage of samples correctly classified. Answers: "How often is the model right?"

**Precision**:

```
Precision = True Positives / (True Positives + False Positives)
```

Of all predictions for class X, how many were actually class X? Answers: "When model says X, how often is it correct?"

**Recall**:

```
Recall = True Positives / (True Positives + False Negatives)
```

Of all actual class X samples, how many did model find? Answers: "How many X samples did model miss?"

**F1-Score**:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall. Balances both metrics. Range: 0-1 (higher is better).

### Example

For gloss "hello" (20 test samples):

- Model predicts "hello" 18 times
- 17 predictions are correct
- Model misses 3 actual "hello" samples

```
Accuracy = 17/20 = 85%
Precision = 17/18 = 94.4% (when model says "hello", correct 94.4% of time)
Recall = 17/20 = 85% (found 17 of 20 "hello" samples)
F1 = 2 × (0.944 × 0.85) / (0.944 + 0.85) = 89.4%
```

---

## 4. Results Summary

### Overall Performance (n=432)

| Metric                | Transformer | InceptionV3-GRU | Δ          |
| --------------------- | ----------- | --------------- | ---------- |
| **Gloss Accuracy**    | **95.6%**   | 75.0%           | **+20.6%** |
| **Gloss Precision**   | 97.0%       | 78.8%           | +18.2%     |
| **Gloss Recall**      | 95.6%       | 75.0%           | +20.6%     |
| **Gloss F1**          | **95.6%**   | 74.2%           | **+21.4%** |
| **Category Accuracy** | 99.5%       | 99.1%           | +0.4%      |
| **Category F1**       | 99.5%       | 99.1%           | +0.4%      |

**Statistical Conclusion**:

- **H₀₁ REJECTED**: Transformer significantly outperforms on gloss recognition
- **H₀₃ FAIL TO REJECT**: No significant difference in category classification

---

### Occlusion Analysis

#### Occluded Samples (n=207)

| Metric                | Transformer | InceptionV3-GRU | Δ          |
| --------------------- | ----------- | --------------- | ---------- |
| **Gloss Accuracy**    | **95.2%**   | 70.5%           | **+24.7%** |
| **Gloss F1**          | 95.3%       | 71.0%           | +24.3%     |
| **Category Accuracy** | **100%**    | 99.0%           | +1.0%      |

#### Non-Occluded Samples (n=225)

| Metric                | Transformer | InceptionV3-GRU | Δ          |
| --------------------- | ----------- | --------------- | ---------- |
| **Gloss Accuracy**    | **96.0%**   | 79.1%           | **+16.9%** |
| **Gloss F1**          | 96.2%       | 79.6%           | +16.6%     |
| **Category Accuracy** | 99.1%       | 99.1%           | 0.0%       |

**Statistical Conclusion**:

- **H₀₂ REJECTED**: Transformer significantly more robust under occlusion (+24.7%)
- **H₀₄ FAIL TO REJECT**: Both models maintain category accuracy under occlusion

---

### Performance Degradation

| Model               | Non-Occluded | Occluded | Degradation |
| ------------------- | ------------ | -------- | ----------- |
| **Transformer**     | 96.0%        | 95.2%    | **-0.8%**   |
| **InceptionV3-GRU** | 79.1%        | 70.5%    | **-8.6%**   |

**Key Finding**: Transformer shows 10.75× less performance degradation under occlusion.

---

## 4. Technical Explanation

### Why Transformer Outperforms

**1. Global Attention Mechanism**

- Each frame attends to all other frames simultaneously
- Occlusion at frame t triggers weight redistribution to visible frames
- O(1) access to any temporal context

**Evidence**: 0.8% degradation vs 8.6% for sequential model

**2. Parallel Sequence Processing**

- Processes all 120 frames simultaneously
- Direct long-range dependency capture
- No sequential error propagation

**3. Dynamic Feature Weighting**

- Attention weights adapt based on keypoint visibility
- Corrupted frames receive lower attention weights
- Model recovers information from temporal neighbors

**Mathematical Basis**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

When frame t is occluded:
- Q_t produces low similarity scores with all K
- Softmax redistributes weight to high-quality frames
- V output borrows from temporally adjacent frames
```

---

### Why InceptionV3-GRU Struggles

**1. Sequential Processing Bottleneck**

- Frames processed one-by-one through GRU
- Corrupted frame features propagate through hidden state
- Cannot selectively access distant frames (O(T) complexity)

**2. Fixed Representation Capacity**

- Hidden state compressed to 12 dimensions
- Information bottleneck limits temporal context
- Accumulating information loss across 120 frames

**3. Pixel-Level Feature Dependence**

- InceptionV3 extracts features from raw frames
- Occlusion directly corrupts spatial features
- No recovery mechanism for lost information

**4. Recurrent Architecture Limitation**

```
h_t = GRU(x_t, h_{t-1})

If x_50 corrupted (occlusion):
- h_50 becomes corrupted
- Error propagates to h_51, h_52, ..., h_120
- No mechanism to "skip" bad frames
```

---

### Why Category Classification Remains High

Both models achieve ~99% category accuracy:

1. **Granularity**: 10 categories vs 105 glosses (10.5× coarser)
2. **Redundant Cues**: Multiple glosses per category share motion patterns
3. **Semantic Grouping**: Categories based on thematic similarity
4. **Hierarchical Structure**: Easier decision boundary

**Example**: All "Food" signs (rice, bread, egg) use similar spatial regions and hand shapes.

---

## 5. Significance

### Technical Contributions

1. **First Multi-Head Attention System for FSL**

   - 105 glosses, 10 categories
   - Keypoints (156-D) + visual features (2048-D)

2. **Occlusion Robustness**

   - 0.8% accuracy drop under occlusion
   - Multi-method detection with temporal filtering

3. **Production Pipeline**
   - End-to-end: Upload → Preprocess → Predict → Validate
   - GPU-accelerated (30-50× speedup)
   - Web interface (Streamlit)

### Social Impact

1. **Accessibility**: Communication bridge for Deaf community
2. **Education**: Interactive FSL learning tool with visualization
3. **Research**: Open infrastructure for FSL research

### Scientific Contributions

1. **Empirical Evidence**: Attention mechanisms superior to CNN-RNN for sign language
2. **Occlusion Quantification**: 24.7% accuracy advantage with global context
3. **Benchmark Results**: Baseline for future FSL research (95.6% gloss, 99.5% category)

---

## 6. Defense Points

### On Results Validity

**Q: Why 20.6% accuracy difference?**

**A**: Architectural design difference:

- Transformer: Global attention, parallel processing, dynamic weighting
- InceptionV3-GRU: Sequential processing, fixed hidden state (12-D), no selective attention
- Difference amplified under occlusion (+24.7% vs +16.9%)

**Q: Is InceptionV3-GRU a weak baseline?**

**A**: No. Represents strong baseline:

- Pretrained ImageNet features (23.8M parameters)
- Recurrent temporal modeling (GRU)
- 75% accuracy validates attention provides real advantage
- Standard architecture in sign language literature

**Q: Statistical significance?**

**A**: Results support hypothesis testing:

- Gloss recognition: Significant difference (H₀₁, H₀₂ rejected)
- Category classification: No significant difference (H₀₃, H₀₄ fail to reject)
- Consistent with expected behavior from architectural differences

---

### On Computing Problem

**Q: Why not just classification?**

**A**: Problem requires:

- Temporal modeling (sequences, not frames)
- Multi-modal fusion (hands + face)
- Robustness (partial data loss)
- Efficiency (real-time inference)

Standard classification does not address temporal dependencies or occlusion.

**Q: Why attention over larger RNN?**

**A**: Fundamental difference:

- RNN: O(T) sequential steps to access frame i from frame j
- Attention: O(1) direct access between any frames
- RNN hidden state: fixed capacity (information bottleneck)
- Attention: full sequence preserved, selective weighting

---

### On Real-World Applicability

**Q: Is 95.6% accuracy sufficient?**

**A**: Yes:

- Top-5 accuracy: 99.5% (correct answer in top 5)
- Category accuracy: 99.5% (semantic context)
- Top-10 accuracy: 100% (comprehensive coverage)
- Users verify via keypoint visualization
- Occlusion robustness (0.8% degradation) enables natural signing

**Q: Deployment feasibility?**

**A**: Production-ready:

- Model inference: 100-500ms (cached), 5-10s (first load)
- Batch processing: 30-50× speedup with GPU
- Memory footprint: ~200MB (Transformer), ~100MB (IV3-GRU)
- Web interface: Streamlit deployment

---

## 7. Key Metrics Summary

### Model Comparison

| Aspect           | Transformer                            | InceptionV3-GRU             |
| ---------------- | -------------------------------------- | --------------------------- |
| **Architecture** | 6-layer encoder, 8-head attention      | InceptionV3 + 2-layer GRU   |
| **Parameters**   | ~90M                                   | ~24M                        |
| **Input**        | Keypoints (156-D) or Features (2048-D) | Features (2048-D only)      |
| **Processing**   | Parallel (all frames)                  | Sequential (frame-by-frame) |
| **Context**      | Global (120 frames)                    | Local (hidden state)        |
| **Inference**    | 100-500ms                              | 200-800ms                   |

### Performance Summary

| Metric                 | Transformer | InceptionV3-GRU | Advantage   |
| ---------------------- | ----------- | --------------- | ----------- |
| **Overall Gloss**      | 95.6%       | 75.0%           | +20.6%      |
| **Occluded Gloss**     | 95.2%       | 70.5%           | +24.7%      |
| **Non-Occluded Gloss** | 96.0%       | 79.1%           | +16.9%      |
| **Category**           | 99.5%       | 99.1%           | +0.4%       |
| **Degradation**        | -0.8%       | -8.6%           | 10.75× less |

---

## 8. Conclusion

**Computing Problem**: FSL recognition requires temporal modeling, multi-modal fusion, and occlusion robustness. Existing sequential approaches fail under partial information loss.

**Solution**: Multi-head attention provides global context, parallel processing, and adaptive feature weighting. Results show 95.6% gloss accuracy with 0.8% degradation under occlusion.

**Validation**: H₀₁ and H₀₂ rejected (gloss recognition significantly better). H₀₃ and H₀₄ fail to reject (category classification equivalent). Results consistent with architectural predictions.

**Impact**: First production-ready FSL system with attention mechanism. Provides accessibility tool, educational resource, and research infrastructure.

---

**Document Version**: 1.0  
**Date**: October 13, 2025  
**Status**: Defense-Ready  
**Data Source**: `evaluation/validation/t_val_results/`, `evaluation/validation/i_val_results/`
