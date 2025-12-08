# Recognition Task: Occluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on occluded recognition data.

**Hypothesis**: Null Hypothesis 1 — There is no significant difference in performance between Transformer and IV3-GRU models for the recognition task under occluded conditions.

**Alpha level**: α = 0.05

---

## Occluded Precision

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed. Pairs where both models scored 0.0 (both-zero pairs) were excluded from the Wilcoxon test but included in mean calculations.

**Total valid pairs**: N = 105 (all pairs for mean calculation)
**Pairs after removing both-zero**: N = 98 (for Wilcoxon test)
**Non-zero differences**: N_nonzero = 98

**Sample data** (first 10 of 105 valid pairs):

| Gloss Label      | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| ---------------- | --------------------- | ----------------- | ---------------- |
| GOOD MORNING     | 0.516667              | 0.425532          | 0.091135         |
| GOOD AFTERNOON   | 0.326087              | 0.083333          | 0.242754         |
| GOOD EVENING     | 0.358491              | 0.090909          | 0.267582         |
| HELLO            | 0.420000              | 0.287879          | 0.132121         |
| HOW ARE YOU      | 0.609375              | 0.645833          | -0.036458        |
| IM FINE          | 0.000000              | 0.000000          | 0.000000         |
| NICE TO MEET YOU | 0.712963              | 0.757895          | -0.044932        |
| THANK YOU        | 0.422222              | 0.484848          | -0.062626        |
| YOURE WELCOME    | 0.806452              | 0.671642          | 0.134810         |
| SEE YOU TOMORROW | 0.713115              | 0.577236          | 0.135879         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

Using all N = 105 valid pairs:

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.470068$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.378707$$

$$\bar{d} = 0.470068 - 0.378707 = 0.091361 \approx 0.091360$$

**Arithmetic**

$$\bar{d} = 0.091360$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.470068$
- Mean IV3-GRU Precision: $\bar{Y} = 0.378707$
- Mean difference: $\bar{d} = 0.091360$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

Using N = 98 (after removing both-zero pairs) for statistical testing:

The standard deviation of differences is computed from the 98 non-zero pairs. Variance exists (not all differences are zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients from expected values of order statistics.

**Test Result**

Performed on N = 98 differences (after removing both-zero pairs):

- Shapiro–Wilk statistic: $W = 0.941027$
- P-value: $p = 2.92 \times 10^{-5}$

**Decision**

Since $p = 2.92 \times 10^{-5} < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = 2.92×10⁻⁵)

### Statistical Test Selection

Since the data is non-normal, the **Wilcoxon Signed-Rank Test** is used.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Rank absolute non-zero differences**

For N_nonzero = 98 non-zero differences, rank the absolute values:

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

where $d^{*}_i$ are the non-zero differences.

**Step 2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3: W statistic**

$$W = \min(S^+, S^-) = 1200.0$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 98 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{98 \times 99}{4} = \frac{9702}{4} = 2425.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{98 \times 99 \times 197}{24}} = \sqrt{\frac{1905894}{24}} = \sqrt{79412.25} = 281.82$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{1200.0 - 2425.50}{281.82} = \frac{-1225.50}{281.82} = -4.343$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(4.343)) = 1.41 \times 10^{-5}$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

**Final Result**

- Test statistic: $z = -4.343$
- P-value: $p = 1.41 \times 10^{-5}$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{4.343}{\sqrt{98}} = \frac{4.343}{9.899} = 0.439$$

**Interpretation**

$r = 0.44$ indicates a **medium** effect size (0.3 ≤ |r| < 0.5).

### Hypothesis Decision

**Comparison**

$p = 1.41 \times 10^{-5} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 1**

The Transformer model shows significantly higher precision than IV3-GRU under occluded conditions ($p = 1.41 \times 10^{-5}$, $r = 0.44$ medium effect).

---

## Occluded Recall

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed. Pairs where both models scored 0.0 (both-zero pairs) were excluded from the Wilcoxon test but included in mean calculations.

**Total valid pairs**: N = 105 (all pairs for mean calculation)
**Pairs after removing both-zero**: N = 98 (for Wilcoxon test)
**Non-zero differences**: N_nonzero = 93

**Sample data** (first 10 of 105 valid pairs):

| Gloss Label      | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| ---------------- | ------------------ | -------------- | ---------------- |
| GOOD MORNING     | 0.861111           | 0.555556       | 0.305555         |
| GOOD AFTERNOON   | 0.882353           | 0.294118       | 0.588235         |
| GOOD EVENING     | 0.863636           | 0.227273       | 0.636363         |
| HELLO            | 0.913043           | 0.826087       | 0.086956         |
| HOW ARE YOU      | 0.886364           | 0.704545       | 0.181819         |
| IM FINE          | 0.000000           | 0.000000       | 0.000000         |
| NICE TO MEET YOU | 0.885057           | 0.827586       | 0.057471         |
| THANK YOU        | 0.826087           | 0.695652       | 0.130435         |
| YOURE WELCOME    | 0.862069           | 0.517241       | 0.344828         |
| SEE YOU TOMORROW | 0.90625            | 0.739583       | 0.166667         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

Using all N = 105 valid pairs:

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.813161$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.526710$$

$$\bar{d} = 0.813161 - 0.526710 = 0.286451$$

**Arithmetic**

$$\bar{d} = 0.286451$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.813161$
- Mean IV3-GRU Recall: $\bar{Y} = 0.526710$
- Mean difference: $\bar{d} = 0.286451$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

Using N = 98 (after removing both-zero pairs) for statistical testing:

The standard deviation of differences is computed from the 98 non-zero pairs. Variance exists (not all differences are zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Test Result**

Performed on N = 98 differences (after removing both-zero pairs):

- Shapiro–Wilk statistic: $W = 0.948846$
- P-value: $p = 8.37 \times 10^{-5}$

**Decision**

Since $p = 8.37 \times 10^{-5} < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = 8.37×10⁻⁵)

### Statistical Test Selection

Since the data is non-normal, the **Wilcoxon Signed-Rank Test** is used.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Rank absolute non-zero differences**

For N_nonzero = 93 non-zero differences, rank the absolute values:

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

**Step 2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3: W statistic**

$$W = \min(S^+, S^-) = 125.0$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 93 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{93 \times 94}{4} = \frac{8742}{4} = 2185.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{93 \times 94 \times 187}{24}} = \sqrt{\frac{1634154}{24}} = \sqrt{68173.08} = 261.10$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{125.0 - 2185.50}{261.10} = \frac{-2060.50}{261.10} = -7.895$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(7.895)) = 2.90 \times 10^{-15}$$

**Final Result**

- Test statistic: $z = -7.895$
- P-value: $p = 2.90 \times 10^{-15}$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{7.895}{\sqrt{93}} = \frac{7.895}{9.644} = 0.819$$

**Interpretation**

$r = 0.82$ indicates a **large** effect size (|r| ≥ 0.5).

### Hypothesis Decision

**Comparison**

$p = 2.90 \times 10^{-15} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 1**

The Transformer model shows significantly higher recall than IV3-GRU under occluded conditions ($p = 2.90 \times 10^{-15}$, $r = 0.82$ large effect).

---

## Occluded F1-score

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "occluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed. Pairs where both models scored 0.0 (both-zero pairs) were excluded from the Wilcoxon test but included in mean calculations.

**Total valid pairs**: N = 105 (all pairs for mean calculation)
**Pairs after removing both-zero**: N = 98 (for Wilcoxon test)
**Non-zero differences**: N_nonzero = 98

**Sample data** (first 10 of 105 valid pairs):

| Gloss Label      | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| ---------------- | -------------------- | ---------------- | ---------------- |
| GOOD MORNING     | 0.645833             | 0.481928         | 0.163905         |
| GOOD AFTERNOON   | 0.476190             | 0.129870         | 0.346320         |
| GOOD EVENING     | 0.506667             | 0.129870         | 0.376797         |
| HELLO            | 0.575342             | 0.426966         | 0.148376         |
| HOW ARE YOU      | 0.722222             | 0.673913         | 0.048309         |
| IM FINE          | 0.000000             | 0.000000         | 0.000000         |
| NICE TO MEET YOU | 0.789744             | 0.791209         | -0.001465        |
| THANK YOU        | 0.558824             | 0.571429         | -0.012605        |
| YOURE WELCOME    | 0.833333             | 0.584416         | 0.248917         |
| SEE YOU TOMORROW | 0.798165             | 0.648402         | 0.149763         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

Using all N = 105 valid pairs:

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.563301$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.411522$$

$$\bar{d} = 0.563301 - 0.411522 = 0.151779$$

**Arithmetic**

$$\bar{d} = 0.151778$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.563301$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.411522$
- Mean difference: $\bar{d} = 0.151778$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

Using N = 98 (after removing both-zero pairs) for statistical testing:

The standard deviation of differences is computed from the 98 non-zero pairs. Variance exists (not all differences are zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Test Result**

Performed on N = 98 differences (after removing both-zero pairs):

- Shapiro–Wilk statistic: $W = 0.930729$
- P-value: $p = 4.21 \times 10^{-6}$

**Decision**

Since $p = 4.21 \times 10^{-6} < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = 4.21×10⁻⁶)

### Statistical Test Selection

Since the data is non-normal, the **Wilcoxon Signed-Rank Test** is used.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Rank absolute non-zero differences**

For N_nonzero = 98 non-zero differences, rank the absolute values:

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

**Step 2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3: W statistic**

$$W = \min(S^+, S^-) = 551.0$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 98 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{98 \times 99}{4} = \frac{9702}{4} = 2425.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{98 \times 99 \times 197}{24}} = \sqrt{\frac{1905894}{24}} = \sqrt{79412.25} = 281.82$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{551.0 - 2425.50}{281.82} = \frac{-1874.50}{281.82} = -6.642$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(6.642)) = 3.09 \times 10^{-11}$$

**Final Result**

- Test statistic: $z = -6.642$
- P-value: $p = 3.09 \times 10^{-11}$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{6.642}{\sqrt{98}} = \frac{6.642}{9.899} = 0.671$$

**Interpretation**

$r = 0.67$ indicates a **large** effect size (|r| ≥ 0.5).

### Hypothesis Decision

**Comparison**

$p = 3.09 \times 10^{-11} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 1**

The Transformer model shows significantly higher F1-score than IV3-GRU under occluded conditions ($p = 3.09 \times 10^{-11}$, $r = 0.67$ large effect).

---

## Summary

All three metrics (Precision, Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under occluded conditions:

- **Occluded Precision**: Transformer > IV3-GRU ($p = 1.41 \times 10^{-5}$, $r = 0.44$ medium effect)
- **Occluded Recall**: Transformer > IV3-GRU ($p = 2.90 \times 10^{-15}$, $r = 0.82$ large effect)
- **Occluded F1-score**: Transformer > IV3-GRU ($p = 3.09 \times 10^{-11}$, $r = 0.67$ large effect)

**Overall Decision**: **Reject Null Hypothesis 1** for all three metrics. The Transformer model demonstrates superior performance compared to IV3-GRU for the recognition task under occluded conditions.
