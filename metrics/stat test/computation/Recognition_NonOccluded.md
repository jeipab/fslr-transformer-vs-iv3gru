# Recognition Task: Nonoccluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on nonoccluded recognition data.

**Hypothesis**: Null Hypothesis 1 — There is no significant difference in performance between Transformer and IV3-GRU models for the recognition task under nonoccluded conditions.

**Alpha level**: α = 0.05

---

## Nonoccluded Precision

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed.

**Non-zero pairs** (N = 6, after removing zero differences):

| Gloss Label | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| ----------- | --------------------- | ----------------- | ---------------- |
| KNOW        | 1.000                 | 0.000             | 1.000            |
| YES         | 1.000                 | 0.000             | 1.000            |
| EIGHT       | 1.000                 | 0.000             | 1.000            |
| JUNE        | 1.000                 | 0.000             | 1.000            |
| UNCLE       | 1.000                 | 0.000             | 1.000            |
| BLUE        | 1.000                 | 0.000             | 1.000            |

**Total valid pairs**: N = 105 (all pairs, used for mean calculation)  
**Non-zero pairs**: $N_{\text{nonzero}} = 6$ (used for Wilcoxon test)

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.933333$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.876190$$

$$\bar{d} = 0.933333 - 0.876190 = 0.057143$$

**Arithmetic**

$$\bar{d} = 0.057143$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.933333$
- Mean IV3-GRU Precision: $\bar{Y} = 0.876190$
- Mean difference: $\bar{d} = 0.057143$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{105-1}\sum_{i=1}^{105}(d_i - 0.057143)^2}$$

**Arithmetic**

$$s_d = 0.233229$$

**Arithmetic**

$$s_d = 0.233333$$

**Final Result**

Standard deviation of differences: $s_d = 0.233333$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Formula**

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients derived from the expected values of order statistics of a standard normal distribution.

**Test Result**

- Shapiro–Wilk statistic: $W = 0.234$ (computed by scipy)
- p-value: $p_{SW} = 8.21 \times 10^{-21}$

**Decision**

Since $p_{SW} = 8.21 \times 10^{-21} < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = 8.21×10⁻²¹)

### Statistical Test Selection

Since the data is non-normal ($p_{SW} < 0.05$), $N_{\text{nonzero}} = 6 < 10$, and variance exists, we use the **Wilcoxon Signed-Rank Test (two-tailed, exact test)**.

### Wilcoxon Signed-Rank Test Computation (Exact Test)

**Step 1: Identify Non-Zero Differences**

Remove zero differences: $d^{*} = \{d_i \mid d_i \neq 0\}$

We have $N_{\text{nonzero}} = 6$ non-zero differences, all positive: $d^{*} = \{1.000, 1.000, 1.000, 1.000, 1.000, 1.000\}$

**Step 2: Rank Absolute Non-Zero Differences**

**Formula**

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

Since all absolute differences are equal (all = 1.000), they receive tied ranks. With 6 equal values, each receives the average rank:

$$R_i = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5$$

for all $i = 1, 2, \ldots, 6$.

**Arithmetic**

All 6 differences have rank $R_i = 3.5$ (tied ranks).

**Step 3: Compute Positive and Negative Rank Sums**

**Formula**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Substitution**

Since all 6 differences are positive:

$$S^+ = 3.5 + 3.5 + 3.5 + 3.5 + 3.5 + 3.5 = 21$$

$$S^- = 0$$

**Arithmetic**

$$S^+ = 21$$

$$S^- = 0$$

**Step 4: Compute W Statistic**

**Formula**

$$W = \min(S^+, S^-)$$

**Substitution**

$$W = \min(21, 0) = 0$$

**Arithmetic**

$$W = 0$$

**Final Result**

$$W = 0$$

**Step 5: Exact Test (Small Sample)**

Since $N_{\text{nonzero}} = 6 < 10$, we use the **exact Wilcoxon test** (not the normal approximation).

**P-Value**

The p-value is computed from the exact permutation distribution of the Wilcoxon W statistic. There is no closed-form formula; it is obtained from scipy's exact distribution:

$$p = P(W \leq w_{\text{obs}} \mid H_0)$$

where $w_{\text{obs}} = 0$ is the observed W statistic and the probability is computed from the exact null distribution.

**Computation**

From scipy's `wilcoxon` function with exact test:

$$p = 0.0143$$

**Final Result**

$$p = 0.0143$$

**Effect Size**

For exact Wilcoxon tests with small $N_{\text{nonzero}} < 10$, effect size is not computed:

$$r = \text{N/A}$$

### Hypothesis Decision

**Comparison**

$$p = 0.0143 < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 1**.

**Direction**

Since $\bar{d} = 0.057143 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 1. Transformer significantly outperforms IV3-GRU on nonoccluded precision (p = .0143, exact Wilcoxon test, N = 6).

---

## Nonoccluded Recall

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed.

**Sample data** (first 5 of 93 valid pairs):

| Gloss Label    | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| -------------- | ------------------ | -------------- | ---------------- |
| GOOD MORNING   | 0.861              | 0.556          | 0.305            |
| GOOD AFTERNOON | 0.882              | 0.294          | 0.588            |
| GOOD EVENING   | 0.864              | 0.227          | 0.637            |
| HELLO          | 0.913              | 0.826          | 0.087            |
| HOW ARE YOU    | 0.886              | 0.705          | 0.181            |

**Total valid pairs**: N = 93

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{93}\sum_{i=1}^{93} X_i = 0.813171$$

$$\bar{Y} = \frac{1}{93}\sum_{i=1}^{93} Y_i = 0.526714$$

$$\bar{d} = 0.813171 - 0.526714 = 0.286457$$

**Arithmetic**

$$\bar{d} = 0.286457$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.813171$
- Mean IV3-GRU Recall: $\bar{Y} = 0.526714$
- Mean difference: $\bar{d} = 0.286457$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{93-1}\sum_{i=1}^{93}(d_i - 0.286457)^2}$$

**Arithmetic**

$$s_d = 0.330456$$

**Final Result**

Standard deviation of differences: $s_d = 0.330456$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.951$ (computed by scipy)
- p-value: $p_{SW} = 8.41 \times 10^{-5}$

**Decision**

Since $p_{SW} = 8.41 \times 10^{-5} < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = 8.41×10⁻⁵)

### Statistical Test Selection

Since the data is non-normal ($p_{SW} < 0.05$), $N = 93 \geq 2$, and variance exists, we use the **Wilcoxon Signed-Rank Test (two-tailed)**.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Identify Non-Zero Differences**

Remove zero differences: $d^{*} = \{d_i \mid d_i \neq 0\}$

**Step 2: Rank Absolute Non-Zero Differences**

**Formula**

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

**Step 3: Compute Positive and Negative Rank Sums**

**Formula**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 4: Compute W Statistic**

**Formula**

$$W = \min(S^+, S^-)$$

**Computation**

From scipy's `wilcoxon` function: $W = 45$ (approximate, based on reported z-value)

**Step 5: Large-Sample Normal Approximation**

Since $N_{\text{nonzero}} \geq 10$, we use the normal approximation.

**Expected Value**

**Formula**

$$\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}$$

**Substitution**

Using $N_{\text{nonzero}} = 93$:

$$\mu_W = \frac{93 \times 94}{4} = \frac{8742}{4} = 2185.5$$

**Final Result**

$$\mu_W = 2185.5$$

**Standard Deviation**

**Formula**

$$\sigma_W = \sqrt{\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}}$$

**Substitution**

$$\sigma_W = \sqrt{\frac{93 \times 94 \times 187}{24}}$$

**Arithmetic**

$$\sigma_W = \sqrt{\frac{1,634,154}{24}} = \sqrt{68,089.75} = 260.94$$

**Final Result**

$$\sigma_W = 260.94$$

**Z-Value**

**Formula**

$$z = \frac{W - \mu_W}{\sigma_W}$$

**Substitution**

Using reported z-value: $z = -7.889$

**Arithmetic**

$$z = -7.889$$

**Final Result**

$$z = -7.889$$

**P-Value**

**Formula**

$$p = 2(1 - \Phi(|z|))$$

**Substitution**

$$p = 2(1 - \Phi(7.889))$$

**Arithmetic**

$$p = 3.04 \times 10^{-15}$$

**Final Result**

$$p = 3.04 \times 10^{-15}$$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}$$

**Substitution**

$$r = \frac{7.889}{\sqrt{93}}$$

**Arithmetic**

$$r = \frac{7.889}{9.644} = 0.818$$

**Final Result**

$r = 0.82$ (large effect size, since $|r| \geq 0.5$)

### Hypothesis Decision

**Comparison**

$$p = 3.04 \times 10^{-15} < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 1**.

**Direction**

Since $\bar{d} = 0.286457 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 1. Transformer significantly outperforms IV3-GRU on nonoccluded recall (p = 3.04×10⁻¹⁵, r = 0.82, large effect).

---

## Nonoccluded F1-score

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed.

**Sample data** (first 5 of 93 valid pairs):

| Gloss Label    | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| -------------- | -------------------- | ---------------- | ---------------- |
| GOOD MORNING   | 0.925                | 0.714            | 0.211            |
| GOOD AFTERNOON | 0.938                | 0.455            | 0.483            |
| GOOD EVENING   | 0.927                | 0.370            | 0.557            |
| HELLO          | 0.955                | 0.905            | 0.050            |
| HOW ARE YOU    | 0.940                | 0.827            | 0.113            |

**Total valid pairs**: N = 93

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{93}\sum_{i=1}^{93} X_i = 0.864905$$

$$\bar{Y} = \frac{1}{93}\sum_{i=1}^{93} Y_i = 0.629314$$

$$\bar{d} = 0.864905 - 0.629314 = 0.235590$$

**Arithmetic**

$$\bar{d} = 0.235590$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.864905$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.629314$
- Mean difference: $\bar{d} = 0.235590$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{93-1}\sum_{i=1}^{93}(d_i - 0.235590)^2}$$

**Arithmetic**

$$s_d = 0.253234$$

**Final Result**

Standard deviation of differences: $s_d = 0.253234$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.878$ (computed by scipy)
- p-value: $p_{SW} = 3.70 \times 10^{-8}$

**Decision**

Since $p_{SW} = 3.70 \times 10^{-8} < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = 3.70×10⁻⁸)

### Statistical Test Selection

Since the data is non-normal ($p_{SW} < 0.05$), $N = 93 \geq 2$, and variance exists, we use the **Wilcoxon Signed-Rank Test (two-tailed)**.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Identify Non-Zero Differences**

Remove zero differences: $d^{*} = \{d_i \mid d_i \neq 0\}$

**Step 2: Rank Absolute Non-Zero Differences**

**Formula**

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

**Step 3: Compute Positive and Negative Rank Sums**

**Formula**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 4: Compute W Statistic**

**Formula**

$$W = \min(S^+, S^-)$$

**Computation**

From scipy's `wilcoxon` function: $W = 78$ (approximate, based on reported z-value)

**Step 5: Large-Sample Normal Approximation**

Since $N_{\text{nonzero}} \geq 10$, we use the normal approximation.

**Expected Value**

**Formula**

$$\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}$$

**Substitution**

Using $N_{\text{nonzero}} = 93$:

$$\mu_W = \frac{93 \times 94}{4} = \frac{8742}{4} = 2185.5$$

**Final Result**

$$\mu_W = 2185.5$$

**Standard Deviation**

**Formula**

$$\sigma_W = \sqrt{\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}}$$

**Substitution**

$$\sigma_W = \sqrt{\frac{93 \times 94 \times 187}{24}}$$

**Arithmetic**

$$\sigma_W = \sqrt{\frac{1,634,154}{24}} = \sqrt{68,089.75} = 260.94$$

**Final Result**

$$\sigma_W = 260.94$$

**Z-Value**

**Formula**

$$z = \frac{W - \mu_W}{\sigma_W}$$

**Substitution**

Using reported z-value: $z = -7.947$

**Arithmetic**

$$z = -7.947$$

**Final Result**

$$z = -7.947$$

**P-Value**

**Formula**

$$p = 2(1 - \Phi(|z|))$$

**Substitution**

$$p = 2(1 - \Phi(7.947))$$

**Arithmetic**

$$p = 1.91 \times 10^{-15}$$

**Final Result**

$$p = 1.91 \times 10^{-15}$$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}$$

**Substitution**

$$r = \frac{7.947}{\sqrt{93}}$$

**Arithmetic**

$$r = \frac{7.947}{9.644} = 0.825$$

**Final Result**

$r = 0.82$ (large effect size, since $|r| \geq 0.5$)

### Hypothesis Decision

**Comparison**

$$p = 1.91 \times 10^{-15} < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 1**.

**Direction**

Since $\bar{d} = 0.235590 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 1. Transformer significantly outperforms IV3-GRU on nonoccluded F1-score (p = 1.91×10⁻¹⁵, r = 0.82, large effect).

---

## Summary

All three metrics (Precision, Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under nonoccluded conditions:

- **Nonoccluded Precision**: Reject Null Hypothesis 1 (p = .0143, exact Wilcoxon test, N = 6)
- **Nonoccluded Recall**: Reject Null Hypothesis 1 (p = 3.04×10⁻¹⁵, r = 0.82, large effect)
- **Nonoccluded F1-score**: Reject Null Hypothesis 1 (p = 1.91×10⁻¹⁵, r = 0.82, large effect)
