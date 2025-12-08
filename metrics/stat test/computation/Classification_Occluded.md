# Classification Task: Occluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on occluded classification data.

**Hypothesis**: Null Hypothesis 2 — There is no significant difference in performance between Transformer and IV3-GRU models for the classification task under occluded conditions.

**Alpha level**: α = 0.05

---

## Occluded Precision

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed.

**All data pairs** (N = 10):

| Category Label | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| -------------- | --------------------- | ----------------- | ---------------- |
| GREETING       | 0.632                 | 0.487             | 0.145            |
| SURVIVAL       | 0.600                 | 0.604             | -0.004           |
| NUMBER         | 0.629                 | 0.346             | 0.283            |
| CALENDAR       | 0.017                 | 0.000             | 0.017            |
| DAYS           | 0.691                 | 0.456             | 0.235            |
| FAMILY         | 0.461                 | 0.492             | -0.031           |
| RELATIONSHIPS  | 0.467                 | 0.390             | 0.077            |
| COLOR          | 0.568                 | 0.620             | -0.052           |
| FOOD           | 0.655                 | 0.605             | 0.050            |
| DRINK          | 0.508                 | 0.553             | -0.045           |

**Total valid pairs**: N = 10

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.632 + 0.600 + 0.629 + 0.017 + 0.691 + 0.461 + 0.467 + 0.568 + 0.655 + 0.508) = 0.522800$$

$$\bar{Y} = \frac{1}{10}(0.487 + 0.604 + 0.346 + 0.000 + 0.456 + 0.492 + 0.390 + 0.620 + 0.605 + 0.553) = 0.455300$$

$$\bar{d} = 0.522800 - 0.455300 = 0.067500$$

**Arithmetic**

$$\bar{d} = 0.067500$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.522800$
- Mean IV3-GRU Precision: $\bar{Y} = 0.455300$
- Mean difference: $\bar{d} = 0.067500$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{10-1}\sum_{i=1}^{10}(d_i - 0.067500)^2}$$

**Arithmetic**

$$s_d = 0.117946$$

**Final Result**

Standard deviation of differences: $s_d = 0.117946$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Formula**

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients derived from the expected values of order statistics of a standard normal distribution.

**Test Result**

- Shapiro–Wilk statistic: $W = 0.950$ (computed by scipy)
- p-value: $p_{SW} = 0.1580$

**Decision**

Since $p_{SW} = 0.1580 > 0.05$, the differences are **normal**.

**Final Result**

Normal (Shapiro–Wilk p = .1580)

### Statistical Test Selection

Since the data is normal ($p_{SW} > 0.05$), $N = 10 \geq 2$, and variance exists, we use the **Paired Samples t-Test (two-tailed)**.

### Paired Samples t-Test Computation

**t-Statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.067500 \times \sqrt{10}}{0.117946}$$

**Arithmetic**

$$t = \frac{0.067500 \times 3.162}{0.117946} = \frac{0.213435}{0.117946} = 1.810$$

**Final Result**

$$t = 1.81$$

**Degrees of Freedom**

**Formula**

$$df = N - 1$$

**Substitution**

$$df = 10 - 1 = 9$$

**Final Result**

$$df = 9$$

**P-Value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

where $F_t$ is the cumulative distribution function of the t-distribution.

**Substitution**

$$p = 2(1 - F_t(1.81; 9))$$

**Arithmetic**

$$p = 0.1038$$

**Final Result**

$$p = 0.1038$$ (ns)

**Cohen's d**

**Formula**

$$d_{\text{cohen}} = \frac{\bar{d}}{s_d} = \frac{\text{mean}(d_i)}{\text{std}(d_i, \text{ddof}=1)}$$

**Substitution**

$$d_{\text{cohen}} = \frac{0.067500}{0.117946}$$

**Arithmetic**

$$d_{\text{cohen}} = 0.572$$

**Final Result**

$d = 0.57$ (medium effect size, since $0.5 \leq |d| < 0.8$)

### Hypothesis Decision

**Comparison**

$$p = 0.1038 \geq 0.05 = \alpha$$

**Decision**

Since $p \geq \alpha$, we **Fail to Reject Null Hypothesis 2**.

**Direction**

Since $\bar{d} = 0.067500 > 0$, **Transformer > IV3-GRU** (but not statistically significant).

**Final Result**

Fail to Reject Null Hypothesis 2. There is no statistically significant difference between Transformer and IV3-GRU on occluded precision (p = .1038 (ns), d = 0.57, medium effect).

---

## Occluded Recall

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed.

**All data pairs** (N = 10):

| Category Label | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| -------------- | ------------------ | -------------- | ---------------- |
| GREETING       | 0.875              | 0.573          | 0.302            |
| SURVIVAL       | 0.884              | 0.590          | 0.294            |
| NUMBER         | 0.886              | 0.298          | 0.588            |
| CALENDAR       | 0.727              | 0.000          | 0.727            |
| DAYS           | 0.889              | 0.538          | 0.351            |
| FAMILY         | 0.840              | 0.626          | 0.214            |
| RELATIONSHIPS  | 0.873              | 0.537          | 0.336            |
| COLOR          | 0.885              | 0.653          | 0.232            |
| FOOD           | 0.902              | 0.765          | 0.137            |
| DRINK          | 0.894              | 0.671          | 0.223            |

**Total valid pairs**: N = 10

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.875 + 0.884 + 0.886 + 0.727 + 0.889 + 0.840 + 0.873 + 0.885 + 0.902 + 0.894) = 0.865500$$

$$\bar{Y} = \frac{1}{10}(0.573 + 0.590 + 0.298 + 0.000 + 0.538 + 0.626 + 0.537 + 0.653 + 0.765 + 0.671) = 0.525100$$

$$\bar{d} = 0.865500 - 0.525100 = 0.340400$$

**Arithmetic**

$$\bar{d} = 0.340400$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.865500$
- Mean IV3-GRU Recall: $\bar{Y} = 0.525100$
- Mean difference: $\bar{d} = 0.340400$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{10-1}\sum_{i=1}^{10}(d_i - 0.340400)^2}$$

**Arithmetic**

$$s_d = 0.181783$$

**Final Result**

Standard deviation of differences: $s_d = 0.181783$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.920$ (computed by scipy)
- p-value: $p_{SW} = 0.0543$

**Decision**

Since $p_{SW} = 0.0543 > 0.05$, the differences are **normal**.

**Final Result**

Normal (Shapiro–Wilk p = .0543)

### Statistical Test Selection

Since the data is normal ($p_{SW} > 0.05$), $N = 10 \geq 2$, and variance exists, we use the **Paired Samples t-Test (two-tailed)**.

### Paired Samples t-Test Computation

**t-Statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.340400 \times \sqrt{10}}{0.181783}$$

**Arithmetic**

$$t = \frac{0.340400 \times 3.162}{0.181783} = \frac{1.076345}{0.181783} = 5.924$$

**Final Result**

$$t = 5.92$$

**Degrees of Freedom**

**Formula**

$$df = N - 1$$

**Substitution**

$$df = 10 - 1 = 9$$

**Final Result**

$$df = 9$$

**P-Value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

**Substitution**

$$p = 2(1 - F_t(5.92; 9))$$

**Arithmetic**

$$p = 0.0002$$

**Final Result**

$$p = 0.0002$$

**Cohen's d**

**Formula**

$$d_{\text{cohen}} = \frac{\bar{d}}{s_d}$$

**Substitution**

$$d_{\text{cohen}} = \frac{0.340400}{0.181783}$$

**Arithmetic**

$$d_{\text{cohen}} = 1.873$$

**Final Result**

$d = 1.87$ (large effect size, since $|d| \geq 0.8$)

### Hypothesis Decision

**Comparison**

$$p = 0.0002 < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 2**.

**Direction**

Since $\bar{d} = 0.340400 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 2. Transformer significantly outperforms IV3-GRU on occluded recall (p = .0002, d = 1.87, large effect).

---

## Occluded F1-score

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed.

**All data pairs** (N = 10):

| Category Label | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| -------------- | -------------------- | ---------------- | ---------------- |
| GREETING       | 0.734                | 0.527            | 0.207            |
| SURVIVAL       | 0.715                | 0.597            | 0.118            |
| NUMBER         | 0.735                | 0.320            | 0.415            |
| CALENDAR       | 0.033                | 0.000            | 0.033            |
| DAYS           | 0.777                | 0.493            | 0.284            |
| FAMILY         | 0.596                | 0.551            | 0.045            |
| RELATIONSHIPS  | 0.609                | 0.452            | 0.157            |
| COLOR          | 0.692                | 0.636            | 0.056            |
| FOOD           | 0.759                | 0.675            | 0.084            |
| DRINK          | 0.648                | 0.606            | 0.042            |

**Total valid pairs**: N = 10

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.734 + 0.715 + 0.735 + 0.033 + 0.777 + 0.596 + 0.609 + 0.692 + 0.759 + 0.648) = 0.629800$$

$$\bar{Y} = \frac{1}{10}(0.527 + 0.597 + 0.320 + 0.000 + 0.493 + 0.551 + 0.452 + 0.636 + 0.675 + 0.606) = 0.485700$$

$$\bar{d} = 0.629800 - 0.485700 = 0.144100$$

**Arithmetic**

$$\bar{d} = 0.144100$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.629800$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.485700$
- Mean difference: $\bar{d} = 0.144100$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{10-1}\sum_{i=1}^{10}(d_i - 0.144100)^2}$$

**Arithmetic**

$$s_d = 0.125488$$

**Final Result**

Standard deviation of differences: $s_d = 0.125488$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.930$ (computed by scipy)
- p-value: $p_{SW} = 0.0595$

**Decision**

Since $p_{SW} = 0.0595 > 0.05$, the differences are **normal**.

**Final Result**

Normal (Shapiro–Wilk p = .0595)

### Statistical Test Selection

Since the data is normal ($p_{SW} > 0.05$), $N = 10 \geq 2$, and variance exists, we use the **Paired Samples t-Test (two-tailed)**.

### Paired Samples t-Test Computation

**t-Statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.144100 \times \sqrt{10}}{0.125488}$$

**Arithmetic**

$$t = \frac{0.144100 \times 3.162}{0.125488} = \frac{0.455242}{0.125488} = 3.627$$

**Final Result**

$$t = 3.63$$

**Degrees of Freedom**

**Formula**

$$df = N - 1$$

**Substitution**

$$df = 10 - 1 = 9$$

**Final Result**

$$df = 9$$

**P-Value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

**Substitution**

$$p = 2(1 - F_t(3.63; 9))$$

**Arithmetic**

$$p = 0.0055$$

**Final Result**

$$p = 0.0055$$

**Cohen's d**

**Formula**

$$d_{\text{cohen}} = \frac{\bar{d}}{s_d}$$

**Substitution**

$$d_{\text{cohen}} = \frac{0.144100}{0.125488}$$

**Arithmetic**

$$d_{\text{cohen}} = 1.148$$

**Final Result**

$d = 1.15$ (large effect size, since $|d| \geq 0.8$)

### Hypothesis Decision

**Comparison**

$$p = 0.0055 < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 2**.

**Direction**

Since $\bar{d} = 0.144100 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 2. Transformer significantly outperforms IV3-GRU on occluded F1-score (p = .0055, d = 1.15, large effect).

---

## Summary

Two of three metrics (Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under occluded conditions:

- **Occluded Precision**: Fail to Reject Null Hypothesis 2 (p = .1038 (ns), d = 0.57, medium effect)
- **Occluded Recall**: Reject Null Hypothesis 2 (p = .0002, d = 1.87, large effect)
- **Occluded F1-score**: Reject Null Hypothesis 2 (p = .0055, d = 1.15, large effect)
