# Classification Task: Nonoccluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on nonoccluded classification data.

**Hypothesis**: Null Hypothesis 2 — There is no significant difference in performance between Transformer and IV3-GRU models for the classification task under nonoccluded conditions.

**Alpha level**: α = 0.05

---

## Nonoccluded Precision

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed.

**All data pairs** (N = 10):

| Category Label | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| -------------- | --------------------- | ----------------- | ---------------- |
| GREETING       | 1.000                 | 1.000             | 0.000            |
| SURVIVAL       | 1.000                 | 1.000             | 0.000            |
| NUMBER         | 1.000                 | 1.000             | 0.000            |
| CALENDAR       | 1.000                 | 1.000             | 0.000            |
| DAYS           | 1.000                 | 1.000             | 0.000            |
| FAMILY         | 1.000                 | 1.000             | 0.000            |
| RELATIONSHIPS  | 1.000                 | 1.000             | 0.000            |
| COLOR          | 1.000                 | 1.000             | 0.000            |
| FOOD           | 1.000                 | 1.000             | 0.000            |
| DRINK          | 1.000                 | 1.000             | 0.000            |

**Total valid pairs**: N = 10

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000) = 1.000$$

$$\bar{Y} = \frac{1}{10}(1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000 + 1.000) = 1.000$$

$$\bar{d} = 1.000 - 1.000 = 0.000$$

**Arithmetic**

$$\bar{d} = 0.000$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 1.000$
- Mean IV3-GRU Precision: $\bar{Y} = 1.000$
- Mean difference: $\bar{d} = 0.000$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

Since all differences $d_i = 0.000$ and $\bar{d} = 0.000$:

$$s_d = \sqrt{\frac{1}{10-1}\sum_{i=1}^{10}(0.000 - 0.000)^2} = \sqrt{\frac{1}{9} \times 0} = 0$$

**Arithmetic**

$$s_d = 0$$

**Final Result**

Standard deviation of differences: $s_d = 0.000$

**No variance exists** ($s_d = 0$ and all differences are zero), so statistical testing is **not applicable**.

### Normality Testing

**Shapiro–Wilk Test**

Normality testing cannot be performed because there is no variance in the differences (all $d_i = 0$).

**Final Result**

Normality cannot be assessed (no variance)

### Statistical Test Selection

Since there is no variance (all differences are zero), statistical tests are **not applicable**.

**Final Result**

Test not applicable — no variance

### Hypothesis Decision

**Comparison**

Since all differences are zero ($\bar{d} = 0.000$), both models have identical performance.

**Decision**

We **Fail to Reject Null Hypothesis 2** (no statistical test can be performed due to lack of variance).

**Direction**

**Equal performance** (both models achieve perfect precision of 1.000)

**Final Result**

Fail to Reject Null Hypothesis 2. Both Transformer and IV3-GRU achieve identical precision (1.000) under nonoccluded conditions. No statistical test is applicable due to lack of variance.

---

## Nonoccluded Recall

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed.

**All data pairs** (N = 10):

| Category Label | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| -------------- | ------------------ | -------------- | ---------------- |
| GREETING       | 0.880              | 0.653          | 0.227            |
| SURVIVAL       | 0.864              | 0.769          | 0.095            |
| NUMBER         | 0.913              | 0.297          | 0.616            |
| CALENDAR       | 0.874              | 0.370          | 0.504            |
| DAYS           | 0.898              | 0.635          | 0.263            |
| FAMILY         | 0.866              | 0.575          | 0.291            |
| RELATIONSHIPS  | 0.882              | 0.644          | 0.238            |
| COLOR          | 0.866              | 0.727          | 0.139            |
| FOOD           | 0.902              | 0.749          | 0.153            |
| DRINK          | 0.884              | 0.680          | 0.204            |

**Total valid pairs**: N = 10

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.880 + 0.864 + 0.913 + 0.874 + 0.898 + 0.866 + 0.882 + 0.866 + 0.902 + 0.884) = 0.882900$$

$$\bar{Y} = \frac{1}{10}(0.653 + 0.769 + 0.297 + 0.370 + 0.635 + 0.575 + 0.644 + 0.727 + 0.749 + 0.680) = 0.609900$$

$$\bar{d} = 0.882900 - 0.609900 = 0.273000$$

**Arithmetic**

$$\bar{d} = 0.273000$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.882900$
- Mean IV3-GRU Recall: $\bar{Y} = 0.609900$
- Mean difference: $\bar{d} = 0.273000$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{10-1}\sum_{i=1}^{10}(d_i - 0.273000)^2}$$

**Arithmetic**

$$s_d = 0.164511$$

**Final Result**

Standard deviation of differences: $s_d = 0.164511$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.925$ (computed by scipy)
- p-value: $p_{SW} = 0.0659$

**Decision**

Since $p_{SW} = 0.0659 > 0.05$, the differences are **normal**.

**Final Result**

Normal (Shapiro–Wilk p = .0659)

### Statistical Test Selection

Since the data is normal ($p_{SW} > 0.05$), $N = 10 \geq 2$, and variance exists, we use the **Paired Samples t-Test (two-tailed)**.

### Paired Samples t-Test Computation

**t-Statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.273000 \times \sqrt{10}}{0.164511}$$

**Arithmetic**

$$t = \frac{0.273000 \times 3.162}{0.164511} = \frac{0.863226}{0.164511} = 5.250$$

**Final Result**

$$t = 5.25$$

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

$$p = 2(1 - F_t(5.25; 9))$$

**Arithmetic**

$$p = 0.0005$$

**Final Result**

$$p = 0.0005$$

**Cohen's d**

**Formula**

$$d_{\text{cohen}} = \frac{\bar{d}}{s_d}$$

**Substitution**

$$d_{\text{cohen}} = \frac{0.273000}{0.164511}$$

**Arithmetic**

$$d_{\text{cohen}} = 1.659$$

**Final Result**

$d = 1.66$ (large effect size, since $|d| \geq 0.8$)

### Hypothesis Decision

**Comparison**

$$p = 0.0005 < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 2**.

**Direction**

Since $\bar{d} = 0.273000 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 2. Transformer significantly outperforms IV3-GRU on nonoccluded recall (p = .0005, d = 1.66, large effect).

---

## Nonoccluded F1-score

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed.

**All data pairs** (N = 10):

| Category Label | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| -------------- | -------------------- | ---------------- | ---------------- |
| GREETING       | 0.936                | 0.790            | 0.146            |
| SURVIVAL       | 0.927                | 0.870            | 0.057            |
| NUMBER         | 0.954                | 0.457            | 0.497            |
| CALENDAR       | 0.933                | 0.540            | 0.393            |
| DAYS           | 0.946                | 0.776            | 0.170            |
| FAMILY         | 0.928                | 0.730            | 0.198            |
| RELATIONSHIPS  | 0.937                | 0.784            | 0.153            |
| COLOR          | 0.928                | 0.842            | 0.086            |
| FOOD           | 0.949                | 0.857            | 0.092            |
| DRINK          | 0.939                | 0.810            | 0.129            |

**Total valid pairs**: N = 10

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.936 + 0.927 + 0.954 + 0.933 + 0.946 + 0.928 + 0.937 + 0.928 + 0.949 + 0.939) = 0.937700$$

$$\bar{Y} = \frac{1}{10}(0.790 + 0.870 + 0.457 + 0.540 + 0.776 + 0.730 + 0.784 + 0.842 + 0.857 + 0.810) = 0.745600$$

$$\bar{d} = 0.937700 - 0.745600 = 0.192100$$

**Arithmetic**

$$\bar{d} = 0.192100$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.937700$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.745600$
- Mean difference: $\bar{d} = 0.192100$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Substitution**

$$s_d = \sqrt{\frac{1}{10-1}\sum_{i=1}^{10}(d_i - 0.192100)^2}$$

**Arithmetic**

$$s_d = 0.141819$$

**Final Result**

Standard deviation of differences: $s_d = 0.141819$

Variance exists ($s_d > 0$), so statistical testing is applicable.

### Normality Testing

**Shapiro–Wilk Test**

**Test Result**

- Shapiro–Wilk statistic: $W = 0.880$ (computed by scipy)
- p-value: $p_{SW} = 0.0162$

**Decision**

Since $p_{SW} = 0.0162 < 0.05$, the differences are **non-normal**.

**Final Result**

Non-normal (Shapiro–Wilk p = .0162)

### Statistical Test Selection

Since the data is non-normal ($p_{SW} < 0.05$), $N = 10 \geq 2$, and variance exists, we use the **Wilcoxon Signed-Rank Test (two-tailed)**.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Identify Non-Zero Differences**

Remove zero differences: $d^{*} = \{d_i \mid d_i \neq 0\}$

All 10 differences are non-zero: $d^{*} = \{0.146, 0.057, 0.497, 0.393, 0.170, 0.198, 0.153, 0.086, 0.092, 0.129\}$

**Step 2: Rank Absolute Non-Zero Differences**

**Formula**

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

Rank the absolute values of non-zero differences from smallest to largest.

**Step 3: Compute Positive and Negative Rank Sums**

**Formula**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

Since all 10 differences are positive, $S^- = 0$ and $S^+$ equals the sum of all ranks.

**Step 4: Compute W Statistic**

**Formula**

$$W = \min(S^+, S^-)$$

**Substitution**

$$W = \min(S^+, 0) = 0$$

**Arithmetic**

$$W = 0$$

**Final Result**

$$W = 0$$

**Step 5: Large-Sample Normal Approximation**

Since $N_{\text{nonzero}} = 10 \geq 10$, we use the normal approximation.

**Expected Value**

**Formula**

$$\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}$$

**Substitution**

$$\mu_W = \frac{10 \times 11}{4} = \frac{110}{4} = 27.5$$

**Final Result**

$$\mu_W = 27.5$$

**Standard Deviation**

**Formula**

$$\sigma_W = \sqrt{\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}}$$

**Substitution**

$$\sigma_W = \sqrt{\frac{10 \times 11 \times 21}{24}}$$

**Arithmetic**

$$\sigma_W = \sqrt{\frac{2310}{24}} = \sqrt{96.25} = 9.811$$

**Final Result**

$$\sigma_W = 9.811$$

**Z-Value**

**Formula**

$$z = \frac{W - \mu_W}{\sigma_W}$$

**Substitution**

$$z = \frac{0 - 27.5}{9.811}$$

**Arithmetic**

$$z = \frac{-27.5}{9.811} = -2.803$$

**Final Result**

$$z = -2.803$$

**P-Value**

**Formula**

$$p = 2(1 - \Phi(|z|))$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

**Substitution**

$$p = 2(1 - \Phi(2.803))$$

**Arithmetic**

$$p = 0.0020$$

**Final Result**

$$p = 0.0020$$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}$$

**Substitution**

$$r = \frac{2.803}{\sqrt{10}}$$

**Arithmetic**

$$r = \frac{2.803}{3.162} = 0.886$$

**Final Result**

$r = 0.89$ (large effect size, since $|r| \geq 0.5$)

### Hypothesis Decision

**Comparison**

$$p = 0.0020 < 0.05 = \alpha$$

**Decision**

Since $p < \alpha$, we **Reject Null Hypothesis 2**.

**Direction**

Since $\bar{d} = 0.192100 > 0$, **Transformer > IV3-GRU**.

**Final Result**

Reject Null Hypothesis 2. Transformer significantly outperforms IV3-GRU on nonoccluded F1-score (p = .0020, r = 0.89, large effect).

---

## Summary

Two of three metrics (Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under nonoccluded conditions:

- **Nonoccluded Precision**: Fail to Reject Null Hypothesis 2 (no variance — both models achieve perfect precision of 1.000)
- **Nonoccluded Recall**: Reject Null Hypothesis 2 (p = .0005, d = 1.66, large effect)
- **Nonoccluded F1-score**: Reject Null Hypothesis 2 (p = .0020, r = 0.89, large effect)
