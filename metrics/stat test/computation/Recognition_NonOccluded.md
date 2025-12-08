# Recognition Task: Nonoccluded Condition — Statistical Analysis

This document provides detailed step-by-step computations for all three metrics (Precision, Recall, F1-score) comparing Transformer and IV3-GRU models on nonoccluded recognition data.

**Hypothesis**: Null Hypothesis 1 — There is no significant difference in performance between Transformer and IV3-GRU models for the recognition task under nonoccluded conditions.

**Alpha level**: α = 0.05

---

## Nonoccluded Precision

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Precision were removed. Pairs where both models scored 0.0 (both-zero pairs) were excluded from the Wilcoxon test but included in mean calculations.

**Total valid pairs**: N = 105 (all pairs for mean calculation)
**Pairs after removing both-zero**: N = 94 (for Wilcoxon test)
**Non-zero differences**: N_nonzero = 94

**Sample data** (first 10 of 105 valid pairs):

| Gloss Label      | Transformer Precision | IV3-GRU Precision | Difference (d_i) |
| ---------------- | --------------------- | ----------------- | ---------------- |
| GOOD MORNING     | 0.623377              | 0.602941          | 0.020436         |
| GOOD AFTERNOON   | 0.707547              | 0.246575          | 0.460972         |
| GOOD EVENING     | 0.663366              | 0.404762          | 0.258604         |
| HELLO            | 0.701031              | 0.477778          | 0.223253         |
| HOW ARE YOU      | 0.647887              | 0.696429          | -0.048542        |
| IM FINE          | 0.816327              | 0.804598          | 0.011729         |
| NICE TO MEET YOU | 0.184211              | 0.178571          | 0.005640         |
| THANK YOU        | 0.729167              | 0.757143          | -0.027976        |
| YOURE WELCOME    | 0.437500              | 0.266667          | 0.170833         |
| SEE YOU TOMORROW | 0.000000              | 0.000000          | 0.000000         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

Using all N = 105 valid pairs:

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.482643$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.406100$$

$$\bar{d} = 0.482643 - 0.406100 = 0.076543$$

**Arithmetic**

$$\bar{d} = 0.076543$$

**Final Result**

- Mean Transformer Precision: $\bar{X} = 0.482643$
- Mean IV3-GRU Precision: $\bar{Y} = 0.406100$
- Mean difference: $\bar{d} = 0.076543$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

Using N = 94 (after removing both-zero pairs) for statistical testing:

The standard deviation of differences is computed from the 94 non-zero pairs. Variance exists (not all differences are zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients from expected values of order statistics.

**Test Result**

Performed on N = 94 differences (after removing both-zero pairs):

- Shapiro–Wilk statistic: $W = 0.890257$
- P-value: $p = 2.26 \times 10^{-8}$

**Decision**

Since $p = 2.26 \times 10^{-8} < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = 2.26×10⁻⁸)

### Statistical Test Selection

Since the data is non-normal, the **Wilcoxon Signed-Rank Test** is used.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Rank absolute non-zero differences**

For N_nonzero = 94 non-zero differences, rank the absolute values:

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

where $d^{*}_i$ are the non-zero differences.

**Step 2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3: W statistic**

$$W = \min(S^+, S^-) = 1145.0$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 94 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{94 \times 95}{4} = \frac{8930}{4} = 2232.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{94 \times 95 \times 189}{24}} = \sqrt{\frac{1687770}{24}} = \sqrt{70323.75} = 265.19$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{1145.0 - 2232.50}{265.19} = \frac{-1087.50}{265.19} = -4.101$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(4.101)) = 4.12 \times 10^{-5}$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

**Final Result**

- Test statistic: $z = -4.101$
- P-value: $p = 4.12 \times 10^{-5}$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{4.101}{\sqrt{94}} = \frac{4.101}{9.695} = 0.423$$

**Interpretation**

$r = 0.42$ indicates a **medium** effect size (0.3 ≤ |r| < 0.5).

### Hypothesis Decision

**Comparison**

$p = 4.12 \times 10^{-5} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 1**

The Transformer model shows significantly higher precision than IV3-GRU under nonoccluded conditions ($p = 4.12 \times 10^{-5}$, $r = 0.42$ medium effect).

---

## Nonoccluded Recall

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU Recall were removed. Pairs where both models scored 0.0 (both-zero pairs) were excluded from the Wilcoxon test but included in mean calculations.

**Total valid pairs**: N = 105 (all pairs for mean calculation)
**Pairs after removing both-zero**: N = 94 (for Wilcoxon test)
**Non-zero differences**: N_nonzero = 93

**Sample data** (first 10 of 105 valid pairs):

| Gloss Label      | Transformer Recall | IV3-GRU Recall | Difference (d_i) |
| ---------------- | ------------------ | -------------- | ---------------- |
| GOOD MORNING     | 0.813559           | 0.694915       | 0.118644         |
| GOOD AFTERNOON   | 0.914634           | 0.219512       | 0.695122         |
| GOOD EVENING     | 0.905405           | 0.459459       | 0.445946         |
| HELLO            | 0.850000           | 0.537500       | 0.312500         |
| HOW ARE YOU      | 0.836364           | 0.709091       | 0.127273         |
| IM FINE          | 0.833333           | 0.729167       | 0.104166         |
| NICE TO MEET YOU | 0.875000           | 0.625000       | 0.250000         |
| THANK YOU        | 0.945946           | 0.716216       | 0.229730         |
| YOURE WELCOME    | 0.933333           | 0.533333       | 0.400000         |
| SEE YOU TOMORROW | 0.000000           | 0.000000       | 0.000000         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

Using all N = 105 valid pairs:

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.784460$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.481308$$

$$\bar{d} = 0.784460 - 0.481308 = 0.303152$$

**Arithmetic**

$$\bar{d} = 0.303152$$

**Final Result**

- Mean Transformer Recall: $\bar{X} = 0.784460$
- Mean IV3-GRU Recall: $\bar{Y} = 0.481308$
- Mean difference: $\bar{d} = 0.303152$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

Using N = 94 (after removing both-zero pairs) for statistical testing:

The standard deviation of differences is computed from the 94 non-zero pairs. Variance exists (not all differences are zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Test Result**

Performed on N = 94 differences (after removing both-zero pairs):

- Shapiro–Wilk statistic: $W = 0.923691$
- P-value: $p = 3.90 \times 10^{-6}$

**Decision**

Since $p = 3.90 \times 10^{-6} < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = 3.90×10⁻⁶)

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

$$W = \min(S^+, S^-) = 30.5$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 93 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{93 \times 94}{4} = \frac{8742}{4} = 2185.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{93 \times 94 \times 187}{24}} = \sqrt{\frac{1634154}{24}} = \sqrt{68173.08} = 261.10$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{30.5 - 2185.50}{261.10} = \frac{-2155.00}{261.10} = -8.257$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(8.257)) = 1.49 \times 10^{-16}$$

**Final Result**

- Test statistic: $z = -8.257$
- P-value: $p = 1.49 \times 10^{-16}$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{8.257}{\sqrt{93}} = \frac{8.257}{9.644} = 0.857$$

**Interpretation**

$r = 0.86$ indicates a **large** effect size (|r| ≥ 0.5).

### Hypothesis Decision

**Comparison**

$p = 1.49 \times 10^{-16} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 1**

The Transformer model shows significantly higher recall than IV3-GRU under nonoccluded conditions ($p = 1.49 \times 10^{-16}$, $r = 0.86$ large effect).

---

## Nonoccluded F1-score

### Data Extraction

Data extracted from `Recognition-Breakdown.csv` filtered for `Occlusion == "nonoccluded"`. Pairs with NaN values in either Transformer or IV3-GRU F1-score were removed. Pairs where both models scored 0.0 (both-zero pairs) were excluded from the Wilcoxon test but included in mean calculations.

**Total valid pairs**: N = 105 (all pairs for mean calculation)
**Pairs after removing both-zero**: N = 94 (for Wilcoxon test)
**Non-zero differences**: N_nonzero = 94

**Sample data** (first 10 of 105 valid pairs):

| Gloss Label      | Transformer F1-score | IV3-GRU F1-score | Difference (d_i) |
| ---------------- | -------------------- | ---------------- | ---------------- |
| GOOD MORNING     | 0.705882             | 0.645669         | 0.060213         |
| GOOD AFTERNOON   | 0.797872             | 0.232258         | 0.565614         |
| GOOD EVENING     | 0.765714             | 0.430380         | 0.335334         |
| HELLO            | 0.768362             | 0.505882         | 0.262480         |
| HOW ARE YOU      | 0.730159             | 0.702703         | 0.027456         |
| IM FINE          | 0.824742             | 0.765027         | 0.059715         |
| NICE TO MEET YOU | 0.304348             | 0.277778         | 0.026570         |
| THANK YOU        | 0.823529             | 0.736111         | 0.087418         |
| YOURE WELCOME    | 0.595745             | 0.355556         | 0.240189         |
| SEE YOU TOMORROW | 0.000000             | 0.000000         | 0.000000         |

### Descriptive Statistics

**Formula**

$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i$$

$$\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$$

$$\bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar{X} - \bar{Y}$$

**Substitution**

Using all N = 105 valid pairs:

$$\bar{X} = \frac{1}{105}\sum_{i=1}^{105} X_i = 0.561103$$

$$\bar{Y} = \frac{1}{105}\sum_{i=1}^{105} Y_i = 0.419827$$

$$\bar{d} = 0.561103 - 0.419827 = 0.141276$$

**Arithmetic**

$$\bar{d} = 0.141276$$

**Final Result**

- Mean Transformer F1-score: $\bar{X} = 0.561103$
- Mean IV3-GRU F1-score: $\bar{Y} = 0.419827$
- Mean difference: $\bar{d} = 0.141276$

### Variance Assessment

**Formula**

$$s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

**Computation**

Using N = 94 (after removing both-zero pairs) for statistical testing:

The standard deviation of differences is computed from the 94 non-zero pairs. Variance exists (not all differences are zero).

### Normality Testing

**Formula**

Shapiro–Wilk test statistic:

$$W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

**Test Result**

Performed on N = 94 differences (after removing both-zero pairs):

- Shapiro–Wilk statistic: $W = 0.887962$
- P-value: $p = 2.57 \times 10^{-8}$

**Decision**

Since $p = 2.57 \times 10^{-8} < 0.05$, the data is **non-normal**.

**Normality Assessment**: Non-normal (Shapiro–Wilk p = 2.57×10⁻⁸)

### Statistical Test Selection

Since the data is non-normal, the **Wilcoxon Signed-Rank Test** is used.

### Wilcoxon Signed-Rank Test Computation

**Step 1: Rank absolute non-zero differences**

For N_nonzero = 94 non-zero differences, rank the absolute values:

$$R_i = \operatorname{rank}(|d^{*}_i|)$$

**Step 2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i$$

$$S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3: W statistic**

$$W = \min(S^+, S^-) = 303.0$$

**Step 4: Expected value and standard deviation**

For large sample (N_nonzero = 94 ≥ 10), use normal approximation:

$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4} = \frac{94 \times 95}{4} = \frac{8930}{4} = 2232.50$$

$$\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}} = \sqrt{\frac{94 \times 95 \times 189}{24}} = \sqrt{\frac{1687770}{24}} = \sqrt{70323.75} = 265.19$$

**Step 5: Z-value**

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{303.0 - 2232.50}{265.19} = \frac{-1929.50}{265.19} = -7.276$$

**Step 6: P-value**

$$p = 2(1 - \Phi(|z|)) = 2(1 - \Phi(7.276)) = 3.44 \times 10^{-13}$$

**Final Result**

- Test statistic: $z = -7.276$
- P-value: $p = 3.44 \times 10^{-13}$

### Effect Size

**Formula**

$$r = \frac{|z|}{\sqrt{N_{nonzero}}} = \frac{7.276}{\sqrt{94}} = \frac{7.276}{9.695} = 0.750$$

**Interpretation**

$r = 0.75$ indicates a **large** effect size (|r| ≥ 0.5).

### Hypothesis Decision

**Comparison**

$p = 3.44 \times 10^{-13} < \alpha = 0.05$

**Decision**: **Reject Null Hypothesis 1**

The Transformer model shows significantly higher F1-score than IV3-GRU under nonoccluded conditions ($p = 3.44 \times 10^{-13}$, $r = 0.75$ large effect).

---

## Summary

All three metrics (Precision, Recall, F1-score) show statistically significant differences favoring the Transformer model over IV3-GRU under nonoccluded conditions:

- **Nonoccluded Precision**: Transformer > IV3-GRU ($p = 4.12 \times 10^{-5}$, $r = 0.42$ medium effect)
- **Nonoccluded Recall**: Transformer > IV3-GRU ($p = 1.49 \times 10^{-16}$, $r = 0.86$ large effect)
- **Nonoccluded F1-score**: Transformer > IV3-GRU ($p = 3.44 \times 10^{-13}$, $r = 0.75$ large effect)

**Overall Decision**: **Reject Null Hypothesis 1** for all three metrics. The Transformer model demonstrates superior performance compared to IV3-GRU for the recognition task under nonoccluded conditions.
