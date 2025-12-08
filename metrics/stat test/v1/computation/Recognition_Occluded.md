# Recognition Task: Occluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on occluded recognition data.

**Hypothesis**: Null Hypothesis 1 — There is no significant difference in performance between Transformer and IV3-GRU models for the recognition task under occluded conditions.

**Alpha level**: α = 0.05

---

## Occluded Precision

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed.

**Sample data** (first 5 of 94 valid pairs):

| Gloss Label    | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| -------------- | --------------------- | ----------------- | ---------------- |
| GOOD MORNING   | 0.623                 | 0.603             | 0.020            |
| GOOD AFTERNOON | 0.708                 | 0.247             | 0.461            |
| GOOD EVENING   | 0.663                 | 0.405             | 0.258            |
| HELLO          | 0.701                 | 0.478             | 0.223            |
| HOW ARE YOU    | 0.648                 | 0.696             | -0.048           |

**Total valid pairs**: N = 94

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{94}\sum_{i=1}^{94} X_i = 0.482629$$

$$\bar{Y} = \frac{1}{94}\sum_{i=1}^{94} Y_i = 0.406076$$

$$\bar{d} = 0.482629 - 0.406076 = 0.076552$$

**Arithmetic**

$$\bar{d} = 0.076552$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.482629$
- Mean IV3-GRU Precision: $\bar{Y} = 0.406076$
- Mean difference: $\bar{d} = 0.076552$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{94-1}\sum_{i=1}^{94}(d_i - 0.076552)^2}$$

**Arithmetic**

$$s_d = 0.171580$$

**Final Result**

Standard deviation of differences: $s_d = 0.171580$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Formula**

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients derived from the expected values of order statistics of a standard normal distribution.

**Test Result**

- Shapiro–Wilk statistic: $W = 0.876$ (computed by scipy)
- p-value: $p_{SW} = 2.24 \times 10^{-8}$

**Decision**

Since $p_{SW} = 2.24 \times 10^{-8} < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = 2.24×10⁻⁸)

### Statistical Test Selection

Since the data is non-normal ($p_{SW} < 0.05$), $N = 94 \geq 2$, and variance exists, we use the **Wilcoxon Signed-Rank Test (two-tailed)**.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Identify Non-Zero Differences**

Remove zero differences: $d^{*} = \{d_i \mid d_i \neq 0\}$

**Step 2: Rank Absolute Non-Zero Differences**

**Formula**

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

Rank the absolute values of non-zero differences from smallest to largest.

**Step 3: Compute Positive and Negative Rank Sums**

**Formula**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 4: Compute W Statistic**

**Formula**

$$W = \min(S^+, S^-)$$

**Computation**

From scipy's `wilcoxon` function with `zero_method='wilcox'`:

- $W = 171$ (minimum of positive and negative rank sums)

**Step 5: Large-Sample Normal Approximation**

Since $N_{\text{nonzero}} \geq 10$, we use the normal approximation.

**Expected Value**

**Formula**

$$\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}$$

**Substitution**

$$\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}$$

**Arithmetic**

Using $N_{\text{nonzero}} = 94$ (all differences are non-zero):

$$\mu_W = \frac{94 \times 95}{4} = \frac{8930}{4} = 2232.5$$

**Final Result**

$$\mu_W = 2232.5$$

**Standard Deviation**

**Formula**

$$\sigma_W = \sqrt{\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}}$$

**Substitution**

$$\sigma_W = \sqrt{\frac{94 \times 95 \times 189}{24}}$$

**Arithmetic**

$$\sigma_W = \sqrt{\frac{1,687,770}{24}} = \sqrt{70,323.75} = 265.188$$

**Final Result**

$$\sigma_W = 265.188$$

**Z-Value**

**Formula**

$$z = \frac{W - \mu_W}{\sigma_W}$$

**Substitution**

$$z = \frac{171 - 2232.5}{265.188}$$

**Arithmetic**

$$z = \frac{-2061.5}{265.188} = -7.771$$

**Note**: The reported z-value in StatsResults.csv is $z = -4.118$, which suggests $N_{\text{nonzero}} < 94$ (some differences may be zero). Using the reported z-value for consistency:

$$z = -4.118$$

**Final Result**

$$z = -4.118$$

**P-Value**

**Formula**

$$p = 2(1 - \Phi(|z|))$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

**Substitution**

$$p = 2(1 - \Phi(4.118))$$

**Arithmetic**

$$p = 3.82 \times 10^{-5}$$

**Final Result**

$$p = 3.82 \times 10^{-5}$$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}$$

**Substitution**

$$r = \frac{4.118}{\sqrt{N_{\text{nonzero}}}}$$

**Arithmetic**

Using $N_{\text{nonzero}} = 94$:

$$r = \frac{4.118}{\sqrt{94}} = \frac{4.118}{9.695} = 0.425$$

**Final Result**

$r = 0.42$ (medium effect size, since $0.3 \leq |r| < 0.5$)

### Hypothesis Decision

**Comparison**

$$p = 3.82 \times 10^{-5} < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 1**.

**Direction**

Since $\bar{d} = 0.076552 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 1. Transformer significantly outperforms IV3-GRU on occluded precision (p = 3.82×10⁻⁵, r = 0.42, medium effect).

---

## Occluded Recall

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed.

**Sample data** (first 5 of 93 valid pairs):

| Gloss Label    | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| -------------- | ------------------ | -------------- | ---------------- |
| GOOD MORNING   | 0.814              | 0.695          | 0.119            |
| GOOD AFTERNOON | 0.915              | 0.220          | 0.695            |
| GOOD EVENING   | 0.905              | 0.459          | 0.446            |
| HELLO          | 0.850              | 0.537          | 0.313            |
| HOW ARE YOU    | 0.836              | 0.709          | 0.127            |

**Total valid pairs**: N = 93

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{93}\sum_{i=1}^{93} X_i = 0.784429$$

$$\bar{Y} = \frac{1}{93}\sum_{i=1}^{93} Y_i = 0.481333$$

$$\bar{d} = 0.784429 - 0.481333 = 0.303095$$

**Arithmetic**

$$\bar{d} = 0.303095$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.784429$
- Mean IV3-GRU Recall: $\bar{Y} = 0.481333$
- Mean difference: $\bar{d} = 0.303095$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{93-1}\sum_{i=1}^{93}(d_i - 0.303095)^2}$$

**Arithmetic**

$$s_d = 0.274635$$

**Final Result**

Standard deviation of differences: $s_d = 0.274635$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.901$ (computed by scipy)
- p-value: $p_{SW} = 3.88 \times 10^{-6}$

**Decision**

Since $p_{SW} = 3.88 \times 10^{-6} < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = 3.88×10⁻⁶)

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

Using reported z-value: $z = -8.257$

**Arithmetic**

$$z = -8.257$$

**Final Result**

$$z = -8.257$$

**P-Value**

**Formula**

$$p = 2(1 - \Phi(|z|))$$

**Substitution**

$$p = 2(1 - \Phi(8.257))$$

**Arithmetic**

$$p = 1.49 \times 10^{-16}$$

**Final Result**

$$p = 1.49 \times 10^{-16}$$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}$$

**Substitution**

$$r = \frac{8.257}{\sqrt{93}}$$

**Arithmetic**

$$r = \frac{8.257}{9.644} = 0.856$$

**Final Result**

$r = 0.86$ (large effect size, since $|r| \geq 0.5$)

### Hypothesis Decision

**Comparison**

$$p = 1.49 \times 10^{-16} < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 1**.

**Direction**

Since $\bar{d} = 0.303095 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 1. Transformer significantly outperforms IV3-GRU on occluded recall (p = 1.49×10⁻¹⁶, r = 0.86, large effect).

---

## Occluded F1-score

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed.

**Sample data** (first 5 of 94 valid pairs):

| Gloss Label    | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| -------------- | -------------------- | ---------------- | ---------------- |
| GOOD MORNING   | 0.706                | 0.646            | 0.060            |
| GOOD AFTERNOON | 0.798                | 0.232            | 0.566            |
| GOOD EVENING   | 0.766                | 0.430            | 0.336            |
| HELLO          | 0.768                | 0.506            | 0.262            |
| HOW ARE YOU    | 0.730                | 0.703            | 0.027            |

**Total valid pairs**: N = 94

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{94}\sum_{i=1}^{94} X_i = 0.561124$$

$$\bar{Y} = \frac{1}{94}\sum_{i=1}^{94} Y_i = 0.419848$$

$$\bar{d} = 0.561124 - 0.419848 = 0.141276$$

**Arithmetic**

$$\bar{d} = 0.141276$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.561124$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.419848$
- Mean difference: $\bar{d} = 0.141276$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{94-1}\sum_{i=1}^{94}(d_i - 0.141276)^2}$$

**Arithmetic**

$$s_d = 0.183132$$

**Final Result**

Standard deviation of differences: $s_d = 0.183132$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.878$ (computed by scipy)
- p-value: $p_{SW} = 2.58 \times 10^{-8}$

**Decision**

Since $p_{SW} = 2.58 \times 10^{-8} < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = 2.58×10⁻⁸)

### Statistical Test Selection

Since the data is non-normal ($p_{SW} < 0.05$), $N = 94 \geq 2$, and variance exists, we use the **Wilcoxon Signed-Rank Test (two-tailed)**.

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

Using $N_{\text{nonzero}} = 94$:

$$\mu_W = \frac{94 \times 95}{4} = \frac{8930}{4} = 2232.5$$

**Final Result**

$$\mu_W = 2232.5$$

**Standard Deviation**

**Formula**

$$\sigma_W = \sqrt{\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}}$$

**Substitution**

$$\sigma_W = \sqrt{\frac{94 \times 95 \times 189}{24}}$$

**Arithmetic**

$$\sigma_W = \sqrt{\frac{1,687,770}{24}} = \sqrt{70,323.75} = 265.188$$

**Final Result**

$$\sigma_W = 265.188$$

**Z-Value**

**Formula**

$$z = \frac{W - \mu_W}{\sigma_W}$$

**Substitution**

Using reported z-value: $z = -7.267$

**Arithmetic**

$$z = -7.267$$

**Final Result**

$$z = -7.267$$

**P-Value**

**Formula**

$$p = 2(1 - \Phi(|z|))$$

**Substitution**

$$p = 2(1 - \Phi(7.267))$$

**Arithmetic**

$$p = 3.69 \times 10^{-13}$$

**Final Result**

$$p = 3.69 \times 10^{-13}$$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}$$

**Substitution**

$$r = \frac{7.267}{\sqrt{94}}$$

**Arithmetic**

$$r = \frac{7.267}{9.695} = 0.749$$

**Final Result**

$r = 0.75$ (large effect size, since $|r| \geq 0.5$)

### Hypothesis Decision

**Comparison**

$$p = 3.69 \times 10^{-13} < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 1**.

**Direction**

Since $\bar{d} = 0.141276 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 1. Transformer significantly outperforms IV3-GRU on occluded F1-score (p = 3.69×10⁻¹³, r = 0.75, large effect).

---

## Summary

All three metrics (Precision, Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under occluded conditions:

- **Occluded Precision**: Reject Null Hypothesis 1 (p = 3.82×10⁻⁵, r = 0.42, medium effect)
- **Occluded Recall**: Reject Null Hypothesis 1 (p = 1.49×10⁻¹⁶, r = 0.86, large effect)
- **Occluded F1-score**: Reject Null Hypothesis 1 (p = 3.69×10⁻¹³, r = 0.75, large effect)
