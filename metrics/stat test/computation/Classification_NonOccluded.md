# Classification Task: Nonoccluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on nonoccluded classification data.

**Hypothesis**: Null Hypothesis 2 — There is no significant difference in performance between Transformer and IV3-GRU models for the classification task under nonoccluded conditions.

**Alpha level**: α = 0.05

---

## Nonoccluded Precision

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed.

**Total valid pairs**: N = 10

**All data pairs**:

| Category Label | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| -------------- | --------------------- | ----------------- | ---------------- |
| GREETING       | 0.643855              | 0.319489          | 0.324366         |
| SURVIVAL       | 0.570085              | 0.562284          | 0.007801         |
| NUMBER         | 0.617757              | 0.312903          | 0.304854         |
| CALENDAR       | 0.015152              | 0.000000          | 0.015152         |
| DAYS           | 0.696429              | 0.320965          | 0.375464         |
| FAMILY         | 0.438642              | 0.303571          | 0.135071         |
| RELATIONSHIPS  | 0.442118              | 0.243182          | 0.198936         |
| COLOR          | 0.570346              | 0.402151          | 0.168195         |
| FOOD           | 0.639045              | 0.311044          | 0.328001         |
| DRINK          | 0.481436              | 0.468619          | 0.012817         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.643855 + 0.570085 + 0.617757 + 0.015152 + 0.696429 + 0.438642 + 0.442118 + 0.570346 + 0.639045 + 0.481436)$$

$$\bar{X} = \frac{1}{10}(5.114860) = 0.511486$$

$$\bar{Y} = \frac{1}{10}(0.319489 + 0.562284 + 0.312903 + 0.000000 + 0.320965 + 0.303571 + 0.243182 + 0.402151 + 0.311044 + 0.468619)$$

$$\bar{Y} = \frac{1}{10}(3.244210) = 0.324421$$

$$\bar{d} = 0.511486 - 0.324421 = 0.187065$$

**Arithmetic**

$$\bar{d} = 0.187065 \approx 0.187066$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.511486$
- Mean IV3-GRU Precision: $\bar{Y} = 0.324421$
- Mean difference: $\bar{d} = 0.187066$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

First, compute squared deviations from mean:

$$s_d = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(d_i - 0.187066)^2} = 0.142726$$

Variance exists (standard deviation is non-zero). All differences are non-zero, so there is variation in the paired differences.

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients from expected values of order statistics.

**Substitution**

Order the N = 10 differences from smallest to largest to obtain $d_{(1)}, d_{(2)}, \ldots, d_{(10)}$. The coefficients $a_i$ are derived from expected values of order statistics of a standard normal distribution (computed via scipy's algorithm).

The denominator $\sum_{i=1}^{10}(d_i - \bar{d})^2$ was computed in the Variance Assessment section.

**Computation**

Using the ordered differences and coefficients $a_i$:

$$W = \frac{\left( \sum_{i=1}^{10} a_i d_{(i)} \right)^2}{\sum_{i=1}^{10}(d_i - \bar{d})^2} = 0.885768$$

**P-value Computation**

The p-value is computed from the distribution of the Shapiro–Wilk statistic under the null hypothesis of normality:

$$p = P(W \leq 0.885768 \mid H_0: \text{data is normal}, N = 10)$$

For N = 10, the p-value is computed using the distribution of W (via scipy's algorithm):

$$p = 0.1519$$

**Test Result**

Performed on N = 10 differences:

- Shapiro–Wilk statistic: $W = 0.885768$
- P-value: $p = 0.1519$

**Decision**

Since $p = 0.1519 > 0.05$, the data is **normal**.

**Normality Assessment**: Normal (Shapiro–Wilk p = .1519)

### Statistical Test Selection

Since the data is normal, the **Paired Samples t-Test** is used.

### Paired t-Test Computation

**Step 1: t-statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.187066 \times \sqrt{10}}{0.142726} = \frac{0.187066 \times 3.162278}{0.142726} = \frac{0.591528}{0.142726} = 4.144$$

**Arithmetic**

$$t = 4.144 \approx 4.14$$

**Step 2: Degrees of freedom**

$$df = N - 1 = 10 - 1 = 9$$

**Step 3: P-value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

where $F_t$ is the cumulative distribution function of the t-distribution.

**Computation**

For $t = 4.14$ with $df = 9$:

$$p = 2(1 - F_t(4.14; 9)) = 0.0025$$

**Final Result**

- Test statistic: $t(df = 9) = 4.14$
- P-value: $p = .0025$

### Effect Size (Cohen's d)

**Formula**

$$d_{cohen} = \frac{\bar{d}}{s_d} = \frac{0.187066}{0.142726} = 1.311$$

**Arithmetic**

$$d_{cohen} = 1.311 \approx 1.31$$

**Interpretation**

$d = 1.31$ indicates a **large** effect size (|d| ≥ 0.8).

### Hypothesis Decision

**Comparison**

$p = .0025 < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 2**

The Transformer model shows significantly higher precision than IV3-GRU under nonoccluded conditions ($p = .0025$, $d = 1.31$ large effect).

---

## Nonoccluded Recall

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed.

**Total valid pairs**: N = 10

**All data pairs**:

| Category Label | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| -------------- | ------------------ | -------------- | ---------------- |
| GREETING       | 0.848987           | 0.552486       | 0.296501         |
| SURVIVAL       | 0.868195           | 0.465616       | 0.402579         |
| NUMBER         | 0.868594           | 0.254928       | 0.613666         |
| CALENDAR       | 0.636364           | 0.000000       | 0.636364         |
| DAYS           | 0.852101           | 0.290756       | 0.561345         |
| FAMILY         | 0.815534           | 0.536408       | 0.279126         |
| RELATIONSHIPS  | 0.860911           | 0.256595       | 0.604316         |
| COLOR          | 0.866776           | 0.307566       | 0.559210         |
| FOOD           | 0.892157           | 0.601961       | 0.290196         |
| DRINK          | 0.858720           | 0.247241       | 0.611479         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.848987 + 0.868195 + 0.868594 + 0.636364 + 0.852101 + 0.815534 + 0.860911 + 0.866776 + 0.892157 + 0.858720)$$

$$\bar{X} = \frac{1}{10}(8.368339) = 0.836834$$

$$\bar{Y} = \frac{1}{10}(0.552486 + 0.465616 + 0.254928 + 0.000000 + 0.290756 + 0.536408 + 0.256595 + 0.307566 + 0.601961 + 0.247241)$$

$$\bar{Y} = \frac{1}{10}(3.513557) = 0.351356$$

$$\bar{d} = 0.836834 - 0.351356 = 0.485478$$

**Arithmetic**

$$\bar{d} = 0.485478$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.836834$
- Mean IV3-GRU Recall: $\bar{Y} = 0.351356$
- Mean difference: $\bar{d} = 0.485478$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

$$s_d = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(d_i - 0.485478)^2} = 0.150440$$

Variance exists (standard deviation is non-zero). All differences are positive (Transformer > IV3-GRU for all pairs), but there is variation in the magnitude of differences.

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Substitution**

Order the N = 10 differences from smallest to largest to obtain $d_{(1)}, d_{(2)}, \ldots, d_{(10)}$. The coefficients $a_i$ are derived from expected values of order statistics of a standard normal distribution (computed via scipy's algorithm).

The denominator $\sum_{i=1}^{10}(d_i - \bar{d})^2$ was computed in the Variance Assessment section.

**Computation**

Using the ordered differences and coefficients $a_i$:

$$W = \frac{\left( \sum_{i=1}^{10} a_i d_{(i)} \right)^2}{\sum_{i=1}^{10}(d_i - \bar{d})^2} = 0.800251$$

**P-value Computation**

The p-value is computed from the distribution of the Shapiro–Wilk statistic under the null hypothesis of normality:

$$p = P(W \leq 0.800251 \mid H_0: \text{data is normal}, N = 10)$$

For N = 10, the p-value is computed using the distribution of W (via scipy's algorithm):

$$p = 0.0146$$

**Test Result**

Performed on N = 10 differences:

- Shapiro–Wilk statistic: $W = 0.800251$
- P-value: $p = 0.0146$

**Decision**

Since $p = 0.0146 < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = .0146)

### Statistical Test Selection

Since the data is non-normal, the **Wilcoxon Signed-Rank Test** is used.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Rank absolute non-zero differences**

For N = 10 pairs, all differences are non-zero and positive. Rank the absolute values:

$$R_i = \operatorname{rank}(|d_i|)$$

Since all differences are positive, we rank them from smallest to largest:

| Rank | Difference (d_i) | Absolute Value |
| ---- | ---------------- | -------------- |
| 1    | 0.296501         | 0.296501       |
| 2    | 0.279126         | 0.279126       |
| 3    | 0.290196         | 0.290196       |
| 4    | 0.402579         | 0.402579       |
| 5    | 0.559210         | 0.559210       |
| 6    | 0.561345         | 0.561345       |
| 7    | 0.604316         | 0.604316       |
| 8    | 0.611479         | 0.611479       |
| 9    | 0.613666         | 0.613666       |
| 10   | 0.636364         | 0.636364       |

**Step 2: Compute positive and negative rank sums**

Since all differences are positive ($d_i > 0$ for all $i$):

$$S^+ = \sum_{d_i > 0} R_i = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55$$

$$S^- = \sum_{d_i < 0} R_i = 0$$

**Step 3: W statistic**

$$W = \min(S^+, S^-) = \min(55, 0) = 0$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 10 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{10 \times 11}{4} = \frac{110}{4} = 27.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{10 \times 11 \times 21}{24}} = \sqrt{\frac{2310}{24}} = \sqrt{96.25} = 9.811$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{0 - 27.50}{9.811} = \frac{-27.50}{9.811} = -2.803$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(2.803)) = 0.0020$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

**Final Result**

- Test statistic: $z = -2.803$
- P-value: $p = .0020$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{2.803}{\sqrt{10}} = \frac{2.803}{3.162} = 0.886$$

**Arithmetic**

$$r = 0.886 \approx 0.89$$

**Interpretation**

$r = 0.89$ indicates a **large** effect size (|r| ≥ 0.5).

### Hypothesis Decision

**Comparison**

$p = .0020 < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 2**

The Transformer model shows significantly higher recall than IV3-GRU under nonoccluded conditions ($p = .0020$, $r = 0.89$ large effect).

---

## Nonoccluded F1-score

### Data Extraction

Data extracted from `Classification-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed.

**Total valid pairs**: N = 10

**All data pairs**:

| Category Label | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| -------------- | -------------------- | ---------------- | ---------------- |
| GREETING       | 0.732327             | 0.404858         | 0.327469         |
| SURVIVAL       | 0.688245             | 0.509404         | 0.178841         |
| NUMBER         | 0.722010             | 0.280956         | 0.441054         |
| CALENDAR       | 0.029598             | 0.000000         | 0.029598         |
| DAYS           | 0.766440             | 0.305115         | 0.461325         |
| FAMILY         | 0.570458             | 0.387719         | 0.182739         |
| RELATIONSHIPS  | 0.584215             | 0.249708         | 0.334507         |
| COLOR          | 0.687990             | 0.348555         | 0.339435         |
| FOOD           | 0.744681             | 0.410154         | 0.334527         |
| DRINK          | 0.616971             | 0.323699         | 0.293272         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

$$\bar{X} = \frac{1}{10}(0.732327 + 0.688245 + 0.722010 + 0.029598 + 0.766440 + 0.570458 + 0.584215 + 0.687990 + 0.744681 + 0.616971)$$

$$\bar{X} = \frac{1}{10}(6.142944) = 0.614294$$

$$\bar{Y} = \frac{1}{10}(0.404858 + 0.509404 + 0.280956 + 0.000000 + 0.305115 + 0.387719 + 0.249708 + 0.348555 + 0.410154 + 0.323699)$$

$$\bar{Y} = \frac{1}{10}(3.220164) = 0.322017$$

$$\bar{d} = 0.614294 - 0.322017 = 0.292277$$

**Arithmetic**

$$\bar{d} = 0.292277$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.614294$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.322017$
- Mean difference: $\bar{d} = 0.292277$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

$$s_d = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(d_i - 0.292277)^2} = 0.129790$$

Variance exists (standard deviation is non-zero). All differences are positive, but there is variation in the magnitude of differences.

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Substitution**

Order the N = 10 differences from smallest to largest to obtain $d_{(1)}, d_{(2)}, \ldots, d_{(10)}$. The coefficients $a_i$ are derived from expected values of order statistics of a standard normal distribution (computed via scipy's algorithm).

The denominator $\sum_{i=1}^{10}(d_i - \bar{d})^2$ was computed in the Variance Assessment section.

**Computation**

Using the ordered differences and coefficients $a_i$:

$$W = \frac{\left( \sum_{i=1}^{10} a_i d_{(i)} \right)^2}{\sum_{i=1}^{10}(d_i - \bar{d})^2} = 0.917105$$

**P-value Computation**

The p-value is computed from the distribution of the Shapiro–Wilk statistic under the null hypothesis of normality:

$$p = P(W \leq 0.917105 \mid H_0: \text{data is normal}, N = 10)$$

For N = 10, the p-value is computed using the distribution of W (via scipy's algorithm):

$$p = 0.3334$$

**Test Result**

Performed on N = 10 differences:

- Shapiro–Wilk statistic: $W = 0.917105$
- P-value: $p = 0.3334$

**Decision**

Since $p = 0.3334 > 0.05$, the data is **normal**.

**Normality Assessment**: Normal (Shapiro–Wilk p = .3334)

### Statistical Test Selection

Since the data is normal, the **Paired Samples t-Test** is used.

### Paired t-Test Computation

**Step 1: t-statistic**

**Formula**

$$t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}$$

**Substitution**

$$t = \frac{0.292277 \times \sqrt{10}}{0.129790} = \frac{0.292277 \times 3.162278}{0.129790} = \frac{0.924499}{0.129790} = 7.124$$

**Arithmetic**

$$t = 7.124 \approx 7.12$$

**Step 2: Degrees of freedom**

$$df = N - 1 = 10 - 1 = 9$$

**Step 3: P-value**

**Formula**

$$p = 2(1 - F_t(|t|; df))$$

**Computation**

For $t = 7.12$ with $df = 9$:

$$p = 2(1 - F_t(7.12; 9)) = 5.54 \times 10^{-5}$$

**Final Result**

- Test statistic: $t(df = 9) = 7.12$
- P-value: $p = 5.54 \times 10^{-5}$

### Effect Size (Cohen's d)

**Formula**

$$d_{cohen} = \frac{\bar{d}}{s_d} = \frac{0.292277}{0.129790} = 2.253$$

**Arithmetic**

$$d_{cohen} = 2.253 \approx 2.25$$

**Interpretation**

$d = 2.25$ indicates a **large** effect size (|d| ≥ 0.8).

### Hypothesis Decision

**Comparison**

$p = 5.54 \times 10^{-5} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 2**

The Transformer model shows significantly higher F1-score than IV3-GRU under nonoccluded conditions ($p = 5.54 \times 10^{-5}$, $d = 2.25$ large effect).

---

## Summary

All three metrics (Precision, Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under nonoccluded conditions:

- **Nonoccluded Precision**: Transformer > IV3-GRU ($p = .0025$, $d = 1.31$ large effect)
- **Nonoccluded Recall**: Transformer > IV3-GRU ($p = .0020$, $r = 0.89$ large effect)
- **Nonoccluded F1-score**: Transformer > IV3-GRU ($p = 5.54 \times 10^{-5}$, $d = 2.25$ large effect)

**Overall Decision**: **Reject Null Hypothesis 2** for all three metrics. The Transformer model demonstrates superior performance compared to IV3-GRU for the classification task under nonoccluded conditions.

**Note on Variance**: All three metrics show variance in the paired differences. The Nonoccluded Precision metric has variance (standard deviation = 0.142726), contrary to a potential "no variance" case. All differences are non-zero, indicating meaningful variation between the two models' performance.
