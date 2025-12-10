# Holm–Bonferroni Correction for Multiple Comparisons

This document provides the Holm–Bonferroni correction calculations applied to control the family-wise error rate (FWER) across multiple statistical tests performed in this study.

## Rationale

Since we perform multiple statistical tests (6 tests per task: 3 metrics × 2 occlusion conditions), we apply the Holm–Bonferroni correction to ensure that the probability of making at least one Type I error across all tests remains at or below α = 0.05.

## Method

The Holm–Bonferroni correction is a step-down procedure that adjusts p-values while maintaining monotonicity. For $m$ tests, the procedure:

1. Sorts p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. For each p-value at rank $i$ (1-indexed), computes the adjusted p-value:
   $$p_{\text{adj},(i)} = p_{(i)} \times (m - i + 1)$$
3. Ensures monotonicity: $p_{\text{adj},(i)} \geq p_{\text{adj},(i-1)}$
4. Caps adjusted p-values at 1.0

---

## Recognition Task (Hypothesis 1)

For the recognition task, we have $m = 6$ tests (all metrics under both occlusion conditions).

### Original p-values

| Test | Metric    | Condition   | Original p-value       |
| ---- | --------- | ----------- | ---------------------- |
| 1    | Precision | Occluded    | $1.41 \times 10^{-5}$  |
| 2    | Recall    | Occluded    | $2.90 \times 10^{-15}$ |
| 3    | F1-score  | Occluded    | $3.09 \times 10^{-11}$ |
| 4    | Precision | Nonoccluded | $4.12 \times 10^{-5}$  |
| 5    | Recall    | Nonoccluded | $1.49 \times 10^{-16}$ |
| 6    | F1-score  | Nonoccluded | $3.44 \times 10^{-13}$ |

### Step 1: Sort p-values in ascending order

| Rank $i$ | Test                    | Original $p_{(i)}$     |
| -------- | ----------------------- | ---------------------- |
| 1        | Recall (Nonoccluded)    | $1.49 \times 10^{-16}$ |
| 2        | Recall (Occluded)       | $2.90 \times 10^{-15}$ |
| 3        | F1-score (Nonoccluded)  | $3.44 \times 10^{-13}$ |
| 4        | F1-score (Occluded)     | $3.09 \times 10^{-11}$ |
| 5        | Precision (Occluded)    | $1.41 \times 10^{-5}$  |
| 6        | Precision (Nonoccluded) | $4.12 \times 10^{-5}$  |

### Step 2: Compute adjusted p-values

For each rank $i$, compute: $p_{\text{adj},(i)} = p_{(i)} \times (6 - i + 1)$

| Rank $i$ | Test                    | Original $p_{(i)}$     | Adjustment Factor $(m-i+1)$ | $p_{\text{adj},(i)}$   |
| -------- | ----------------------- | ---------------------- | --------------------------- | ---------------------- |
| 1        | Recall (Nonoccluded)    | $1.49 \times 10^{-16}$ | 6                           | $8.94 \times 10^{-16}$ |
| 2        | Recall (Occluded)       | $2.90 \times 10^{-15}$ | 5                           | $1.45 \times 10^{-14}$ |
| 3        | F1-score (Nonoccluded)  | $3.44 \times 10^{-13}$ | 4                           | $1.38 \times 10^{-12}$ |
| 4        | F1-score (Occluded)     | $3.09 \times 10^{-11}$ | 3                           | $9.26 \times 10^{-11}$ |
| 5        | Precision (Occluded)    | $1.41 \times 10^{-5}$  | 2                           | $2.82 \times 10^{-5}$  |
| 6        | Precision (Nonoccluded) | $4.12 \times 10^{-5}$  | 1                           | $4.12 \times 10^{-5}$  |

### Step 3: Ensure monotonicity

Since all adjusted p-values are already in ascending order, no further adjustment is needed.

### Final adjusted p-values

| Metric    | Condition   | Original p-value       | Adjusted p-value       |
| --------- | ----------- | ---------------------- | ---------------------- |
| Precision | Occluded    | $1.41 \times 10^{-5}$  | $2.82 \times 10^{-5}$  |
| Recall    | Occluded    | $2.90 \times 10^{-15}$ | $1.45 \times 10^{-14}$ |
| F1-score  | Occluded    | $3.09 \times 10^{-11}$ | $9.26 \times 10^{-11}$ |
| Precision | Nonoccluded | $4.12 \times 10^{-5}$  | $4.12 \times 10^{-5}$  |
| Recall    | Nonoccluded | $1.49 \times 10^{-16}$ | $8.94 \times 10^{-16}$ |
| F1-score  | Nonoccluded | $3.44 \times 10^{-13}$ | $1.38 \times 10^{-12}$ |

### Decision

All adjusted p-values remain < 0.05, confirming all decisions to **Reject Null Hypothesis 1** for all six tests.

---

## Classification Task (Hypothesis 2)

For the classification task, we have $m = 6$ tests (all metrics under both occlusion conditions).

### Original p-values

| Test | Metric    | Condition   | Original p-value      |
| ---- | --------- | ----------- | --------------------- |
| 1    | Precision | Occluded    | $1.30 \times 10^{-3}$ |
| 2    | Recall    | Occluded    | $5.00 \times 10^{-6}$ |
| 3    | F1-score  | Occluded    | $2.52 \times 10^{-6}$ |
| 4    | Precision | Nonoccluded | $2.50 \times 10^{-3}$ |
| 5    | Recall    | Nonoccluded | $2.00 \times 10^{-3}$ |
| 6    | F1-score  | Nonoccluded | $5.54 \times 10^{-5}$ |

### Step 1: Sort p-values in ascending order

| Rank $i$ | Test                    | Original $p_{(i)}$    |
| -------- | ----------------------- | --------------------- |
| 1        | F1-score (Occluded)     | $2.52 \times 10^{-6}$ |
| 2        | Recall (Occluded)       | $5.00 \times 10^{-6}$ |
| 3        | F1-score (Nonoccluded)  | $5.54 \times 10^{-5}$ |
| 4        | Precision (Occluded)    | $1.30 \times 10^{-3}$ |
| 5        | Recall (Nonoccluded)    | $2.00 \times 10^{-3}$ |
| 6        | Precision (Nonoccluded) | $2.50 \times 10^{-3}$ |

### Step 2: Compute adjusted p-values

For each rank $i$, compute: $p_{\text{adj},(i)} = p_{(i)} \times (6 - i + 1)$

| Rank $i$ | Test                    | Original $p_{(i)}$    | Adjustment Factor $(m-i+1)$ | $p_{\text{adj},(i)}$  |
| -------- | ----------------------- | --------------------- | --------------------------- | --------------------- |
| 1        | F1-score (Occluded)     | $2.52 \times 10^{-6}$ | 6                           | $1.51 \times 10^{-5}$ |
| 2        | Recall (Occluded)       | $5.00 \times 10^{-6}$ | 5                           | $2.50 \times 10^{-5}$ |
| 3        | F1-score (Nonoccluded)  | $5.54 \times 10^{-5}$ | 4                           | $2.22 \times 10^{-4}$ |
| 4        | Precision (Occluded)    | $1.30 \times 10^{-3}$ | 3                           | $3.90 \times 10^{-3}$ |
| 5        | Recall (Nonoccluded)    | $2.00 \times 10^{-3}$ | 2                           | $4.00 \times 10^{-3}$ |
| 6        | Precision (Nonoccluded) | $2.50 \times 10^{-3}$ | 1                           | $2.50 \times 10^{-3}$ |

### Step 3: Ensure monotonicity

Check if any adjusted p-value is less than the previous one:

- Rank 1: $1.51 \times 10^{-5}$
- Rank 2: $2.50 \times 10^{-5} \geq 1.51 \times 10^{-5}$ ✓
- Rank 3: $2.22 \times 10^{-4} \geq 2.50 \times 10^{-5}$ ✓
- Rank 4: $3.90 \times 10^{-3} \geq 2.22 \times 10^{-4}$ ✓
- Rank 5: $4.00 \times 10^{-3} \geq 3.90 \times 10^{-3}$ ✓
- Rank 6: $2.50 \times 10^{-3} < 4.00 \times 10^{-3}$ ✗

Adjust rank 6: $p_{\text{adj},(6)} = 4.00 \times 10^{-3}$ (set to previous value to maintain monotonicity)

### Final adjusted p-values

| Metric    | Condition   | Original p-value      | Adjusted p-value      |
| --------- | ----------- | --------------------- | --------------------- |
| Precision | Occluded    | $1.30 \times 10^{-3}$ | $3.90 \times 10^{-3}$ |
| Recall    | Occluded    | $5.00 \times 10^{-6}$ | $2.50 \times 10^{-5}$ |
| F1-score  | Occluded    | $2.52 \times 10^{-6}$ | $1.51 \times 10^{-5}$ |
| Precision | Nonoccluded | $2.50 \times 10^{-3}$ | $4.00 \times 10^{-3}$ |
| Recall    | Nonoccluded | $2.00 \times 10^{-3}$ | $4.00 \times 10^{-3}$ |
| F1-score  | Nonoccluded | $5.54 \times 10^{-5}$ | $2.22 \times 10^{-4}$ |

### Decision

All adjusted p-values remain < 0.05, confirming all decisions to **Reject Null Hypothesis 2** for all six tests.

---

## Summary

The Holm–Bonferroni correction was applied separately to each task (recognition and classification), correcting for 6 tests per task. All adjusted p-values remain below the significance threshold (α = 0.05), confirming that all statistical decisions remain unchanged after correction for multiple comparisons.
