# Classification Task: Occluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on occluded classification data.

**Hypothesis**: Null Hypothesis 2 — There is no significant difference in performance between Transformer and IV3-GRU models for the classification task under occluded conditions.

**Alpha level**: α = 0.05

---

## Occluded Precision

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed.

**Total valid pairs**: N = 10

**All data pairs**:

| Category Label | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| -------------- | --------------------- | ----------------- | ---------------- |
| GREETING       | 0.584000              | 0.316456          | 0.267544         |
| SURVIVAL       | 0.355245              | 0.358696          | -0.003451        |
| NUMBER         | 0.273874              | 0.086093          | 0.187781         |
| CALENDAR       | 0.652704              | 0.322222          | 0.330482         |
| DAYS           | 0.579457              | 0.264151          | 0.315306         |
| FAMILY         | 0.526726              | 0.346880          | 0.179846         |
| RELATIONSHIPS  | 0.496158              | 0.383912          | 0.112246         |
| COLOR          | 0.444602              | 0.320455          | 0.124147         |
| FOOD           | 0.572864              | 0.288546          | 0.284318         |
| DRINK          | 0.516055              | 0.526690          | -0.010635        |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.584000 + 0.355245 + 0.273874 + 0.652704 + 0.579457 + 0.526726 + 0.496158 + 0.444602 + 0.572864 + 0.516055)$$

$$\bar{X} = \frac{1}{10}(5.001685) = 0.500169$$

$$\bar{Y} = \frac{1}{10}(0.316456 + 0.358696 + 0.086093 + 0.322222 + 0.264151 + 0.346880 + 0.383912 + 0.320455 + 0.288546 + 0.526690)$$

$$\bar{Y} = \frac{1}{10}(3.214100) = 0.321410$$

$$\bar{d} = 0.500169 - 0.321410 = 0.178759$$

**Arithmetic**

$$\bar{d} = 0.178759 \approx 0.178758$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.500168$
- Mean IV3-GRU Precision: $\bar{Y} = 0.321410$
- Mean difference: $\bar{d} = 0.178758$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

First, compute squared deviations from mean:

$$s_d = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(d_i - 0.178758)^2} = 0.123491$$

Variance exists (standard deviation is non-zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients from expected values of order statistics.

**Test Result**

Performed on N = 10 differences:

- Shapiro–Wilk statistic: $W = 0.920154$
- P-value: $p = 0.3582$

**Decision**

Since $p = 0.3582 > 0.05$, the data is **normal**.

**Normality Assessment**: Normal (Shapiro–Wilk p = .3582)

### Statistical Test Selection

Since the data is normal, the **Paired Samples t-Test** is used.

### Paired t-Test Computation

**Step 1: t-statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.178758 \times \sqrt{10}}{0.123491} = \frac{0.178758 \times 3.162278}{0.123491} = \frac{0.565685}{0.123491} = 4.584$$

**Arithmetic**

$$t = 4.584 \approx 4.58$$

**Step 2: Degrees of freedom**

$$df = N - 1 = 10 - 1 = 9$$

**Step 3: P-value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

where $F_t$ is the cumulative distribution function of the t-distribution.

**Computation**

For $t = 4.58$ with $df = 9$:

$$p = 2(1 - F_t(4.58; 9)) = 0.0013$$

**Final Result**

- Test statistic: $t(df = 9) = 4.58$
- P-value: $p = .0013$

### Effect Size (Cohen's d)

**Formula**

$$d_{cohen} = \frac{\bar{d}}{s_d} = \frac{0.178758}{0.123491} = 1.449$$

**Arithmetic**

$$d_{cohen} = 1.449 \approx 1.45$$

**Interpretation**

$d = 1.45$ indicates a **large** effect size (|d| ≥ 0.8).

### Hypothesis Decision

**Comparison**

$p = .0013 < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 2**

The Transformer model shows significantly higher precision than IV3-GRU under occluded conditions ($p = .0013$, $d = 1.45$ large effect).

---

## Occluded Recall

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed.

**Total valid pairs**: N = 10

**All data pairs**:

| Category Label | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| -------------- | ------------------ | -------------- | ---------------- |
| GREETING       | 0.839080           | 0.632184       | 0.206896         |
| SURVIVAL       | 0.861017           | 0.447458       | 0.413559         |
| NUMBER         | 0.883721           | 0.226744       | 0.656977         |
| CALENDAR       | 0.862173           | 0.350101       | 0.512072         |
| DAYS           | 0.874269           | 0.368421       | 0.505848         |
| FAMILY         | 0.847670           | 0.428315       | 0.419355         |
| RELATIONSHIPS  | 0.859316           | 0.399240       | 0.460076         |
| COLOR          | 0.836898           | 0.377005       | 0.459893         |
| FOOD           | 0.857143           | 0.656642       | 0.200501         |
| DRINK          | 0.867052           | 0.285164       | 0.581888         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.839080 + 0.861017 + 0.883721 + 0.862173 + 0.874269 + 0.847670 + 0.859316 + 0.836898 + 0.857143 + 0.867052)$$

$$\bar{X} = \frac{1}{10}(8.585431) = 0.858543$$

$$\bar{Y} = \frac{1}{10}(0.632184 + 0.447458 + 0.226744 + 0.350101 + 0.368421 + 0.428315 + 0.399240 + 0.377005 + 0.656642 + 0.285164)$$

$$\bar{Y} = \frac{1}{10}(4.171270) = 0.417127$$

$$\bar{d} = 0.858543 - 0.417127 = 0.441416$$

**Arithmetic**

$$\bar{d} = 0.441416 \approx 0.441706$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.858834$
- Mean IV3-GRU Recall: $\bar{Y} = 0.417127$
- Mean difference: $\bar{d} = 0.441706$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

$$s_d = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(d_i - 0.441706)^2} = 0.145424$$

Variance exists (standard deviation is non-zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Test Result**

Performed on N = 10 differences:

- Shapiro–Wilk statistic: $W = 0.913566$
- P-value: $p = 0.3064$

**Decision**

Since $p = 0.3064 > 0.05$, the data is **normal**.

**Normality Assessment**: Normal (Shapiro–Wilk p = .3064)

### Statistical Test Selection

Since the data is normal, the **Paired Samples t-Test** is used.

### Paired t-Test Computation

**Step 1: t-statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.441706 \times \sqrt{10}}{0.145424} = \frac{0.441706 \times 3.162278}{0.145424} = \frac{1.396998}{0.145424} = 9.604$$

**Arithmetic**

$$t = 9.604 \approx 9.60$$

**Step 2: Degrees of freedom**

$$df = N - 1 = 10 - 1 = 9$$

**Step 3: P-value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

**Computation**

For $t = 9.60$ with $df = 9$:

$$p = 2(1 - F_t(9.60; 9)) = 5.00 \times 10^{-6}$$

**Final Result**

- Test statistic: $t(df = 9) = 9.60$
- P-value: $p = 5.00 \times 10^{-6}$

### Effect Size (Cohen's d)

**Formula**

$$d_{cohen} = \frac{\bar{d}}{s_d} = \frac{0.441706}{0.145424} = 3.038$$

**Arithmetic**

$$d_{cohen} = 3.038 \approx 3.04$$

**Interpretation**

$d = 3.04$ indicates a **large** effect size (|d| ≥ 0.8).

### Hypothesis Decision

**Comparison**

$p = 5.00 \times 10^{-6} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 2**

The Transformer model shows significantly higher recall than IV3-GRU under occluded conditions ($p = 5.00 \times 10^{-6}$, $d = 3.04$ large effect).

---

## Occluded F1-score

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed.

**Total valid pairs**: N = 10

**All data pairs**:

| Category Label | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| -------------- | -------------------- | ---------------- | ---------------- |
| GREETING       | 0.688679             | 0.421779         | 0.266900         |
| SURVIVAL       | 0.502970             | 0.398190         | 0.104780         |
| NUMBER         | 0.418157             | 0.124800         | 0.293357         |
| CALENDAR       | 0.742956             | 0.335583         | 0.407373         |
| DAYS           | 0.696970             | 0.307692         | 0.389278         |
| FAMILY         | 0.649725             | 0.383320         | 0.266405         |
| RELATIONSHIPS  | 0.629088             | 0.391426         | 0.237662         |
| COLOR          | 0.580705             | 0.346437         | 0.234268         |
| FOOD           | 0.686747             | 0.400918         | 0.285829         |
| DRINK          | 0.647017             | 0.370000         | 0.277017         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.688679 + 0.502970 + 0.418157 + 0.742956 + 0.696970 + 0.649725 + 0.629088 + 0.580705 + 0.686747 + 0.647017)$$

$$\bar{X} = \frac{1}{10}(6.243014) = 0.624301$$

$$\bar{Y} = \frac{1}{10}(0.421779 + 0.398190 + 0.124800 + 0.335583 + 0.307692 + 0.383320 + 0.391426 + 0.346437 + 0.400918 + 0.370000)$$

$$\bar{Y} = \frac{1}{10}(3.480140) = 0.348014$$

$$\bar{d} = 0.624301 - 0.348014 = 0.276287$$

**Arithmetic**

$$\bar{d} = 0.276287$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.624301$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.348014$
- Mean difference: $\bar{d} = 0.276287$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

$$s_d = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(d_i - 0.276287)^2} = 0.083793$$

Variance exists (standard deviation is non-zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Test Result**

Performed on N = 10 differences:

- Shapiro–Wilk statistic: $W = 0.906393$
- P-value: $p = 0.2571$

**Decision**

Since $p = 0.2571 > 0.05$, the data is **normal**.

**Normality Assessment**: Normal (Shapiro–Wilk p = .2571)

### Statistical Test Selection

Since the data is normal, the **Paired Samples t-Test** is used.

### Paired t-Test Computation

**Step 1: t-statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.276287 \times \sqrt{10}}{0.083793} = \frac{0.276287 \times 3.162278}{0.083793} = \frac{0.873998}{0.083793} = 10.432$$

**Arithmetic**

$$t = 10.432 \approx 10.43$$

**Step 2: Degrees of freedom**

$$df = N - 1 = 10 - 1 = 9$$

**Step 3: P-value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

**Computation**

For $t = 10.43$ with $df = 9$:

$$p = 2(1 - F_t(10.43; 9)) = 2.52 \times 10^{-6}$$

**Final Result**

- Test statistic: $t(df = 9) = 10.43$
- P-value: $p = 2.52 \times 10^{-6}$

### Effect Size (Cohen's d)

**Formula**

$$d_{cohen} = \frac{\bar{d}}{s_d} = \frac{0.276287}{0.083793} = 3.298$$

**Arithmetic**

$$d_{cohen} = 3.298 \approx 3.30$$

**Interpretation**

$d = 3.30$ indicates a **large** effect size (|d| ≥ 0.8).

### Hypothesis Decision

**Comparison**

$p = 2.52 \times 10^{-6} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 2**

The Transformer model shows significantly higher F1-score than IV3-GRU under occluded conditions ($p = 2.52 \times 10^{-6}$, $d = 3.30$ large effect).

---

## Summary

All three metrics (Precision, Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under occluded conditions:

- **Occluded Precision**: Transformer > IV3-GRU ($p = .0013$, $d = 1.45$ large effect)
- **Occluded Recall**: Transformer > IV3-GRU ($p = 5.00 \times 10^{-6}$, $d = 3.04$ large effect)
- **Occluded F1-score**: Transformer > IV3-GRU ($p = 2.52 \times 10^{-6}$, $d = 3.30$ large effect)

**Overall Decision**: **Reject Null Hypothesis 2** for all three metrics. The Transformer model demonstrates superior performance compared to IV3-GRU for the classification task under occluded conditions.
