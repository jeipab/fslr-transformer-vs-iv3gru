# Recognition Task: Nonoccluded Condition — Statistical Analysis (Compact)

**Hypothesis**: Null Hypothesis 1 — There is no significant difference in performance between Transformer and IV3-GRU models for the recognition task under nonoccluded conditions.

**Alpha level**: α = 0.05

---

## Nonoccluded Precision

**Data**: N = 105 valid pairs (all pairs for mean calculation), N = 94 pairs after removing both-zero (for Wilcoxon test), N_nonzero = 94

### Step 1: Descriptive Statistics

**Formula**:
$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i, \quad \bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i, \quad \bar{d} = \bar{X} - \bar{Y}$$

**Substitution**: Using all N = 105 valid pairs

**Result**: $\bar{X} = 0.482643$, $\bar{Y} = 0.406100$, $\bar{d} = 0.076543$

### Step 2: Normality Testing

**Formula**: Shapiro–Wilk test statistic $W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$

**Result**: $W = 0.890257$, $p = 2.26 \times 10^{-8} < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute non-zero differences**

For N_nonzero = 94 non-zero differences, rank the absolute values: $R_i = \operatorname{rank}(|d^{*}_i|)$

**Step 3.2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i, \quad S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 1145.0$

**Step 3.4: Expected value and standard deviation**

**Formula**:
$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4}, \quad \sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}}$$

**Substitution**:
$$\mu_W = \frac{94 \times 95}{4} = 2232.50, \quad \sigma_W = \sqrt{\frac{94 \times 95 \times 189}{24}} = 265.19$$

**Step 3.5: Z-value**

**Formula**: $z = \frac{W - \mu_W}{\sigma_W}$

**Substitution**: $z = \frac{1145.0 - 2232.50}{265.19} = -4.101$

**Step 3.6: P-value**

**Formula**: $p = 2(1 - \Phi(|z|))$

**Result**: $z = -4.101$, $p = 4.12 \times 10^{-5}$

### Step 4: Effect Size

**Formula**: $r = \frac{|z|}{\sqrt{N_{nonzero}}}$

**Result**: $r = 0.42$ (medium effect, 0.3 ≤ |r| < 0.5)

### Step 5: Hypothesis Decision

$p = 4.12 \times 10^{-5} < \alpha = 0.05$ → **Reject Null Hypothesis 1**

Transformer > IV3-GRU ($p = 4.12 \times 10^{-5}$, $r = 0.42$ medium effect)

---

## Nonoccluded Recall

**Data**: N = 105 valid pairs (all pairs for mean calculation), N = 94 pairs after removing both-zero (for Wilcoxon test), N_nonzero = 93

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: Using all N = 105 valid pairs

**Result**: $\bar{X} = 0.784460$, $\bar{Y} = 0.481308$, $\bar{d} = 0.303152$

### Step 2: Normality Testing

**Result**: $W = 0.923691$, $p = 3.90 \times 10^{-6} < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute non-zero differences**

For N_nonzero = 93 non-zero differences, rank the absolute values: $R_i = \operatorname{rank}(|d^{*}_i|)$

**Step 3.2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i, \quad S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 30.5$

**Step 3.4: Expected value and standard deviation**

**Formula**:
$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4}, \quad \sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}}$$

**Substitution**:
$$\mu_W = \frac{93 \times 94}{4} = 2185.50, \quad \sigma_W = \sqrt{\frac{93 \times 94 \times 187}{24}} = 261.10$$

**Step 3.5: Z-value**

**Substitution**: $z = \frac{30.5 - 2185.50}{261.10} = -8.257$

**Step 3.6: P-value**

**Result**: $z = -8.257$, $p = 1.49 \times 10^{-16}$

### Step 4: Effect Size

**Result**: $r = 0.86$ (large effect, |r| ≥ 0.5)

### Step 5: Hypothesis Decision

$p = 1.49 \times 10^{-16} < \alpha = 0.05$ → **Reject Null Hypothesis 1**

Transformer > IV3-GRU ($p = 1.49 \times 10^{-16}$, $r = 0.86$ large effect)

---

## Nonoccluded F1-score

**Data**: N = 105 valid pairs (all pairs for mean calculation), N = 94 pairs after removing both-zero (for Wilcoxon test), N_nonzero = 94

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: Using all N = 105 valid pairs

**Result**: $\bar{X} = 0.561103$, $\bar{Y} = 0.419827$, $\bar{d} = 0.141276$

### Step 2: Normality Testing

**Result**: $W = 0.887962$, $p = 2.57 \times 10^{-8} < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute non-zero differences**

For N_nonzero = 94 non-zero differences, rank the absolute values: $R_i = \operatorname{rank}(|d^{*}_i|)$

**Step 3.2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i, \quad S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 303.0$

**Step 3.4: Expected value and standard deviation**

**Substitution**:
$$\mu_W = \frac{94 \times 95}{4} = 2232.50, \quad \sigma_W = \sqrt{\frac{94 \times 95 \times 189}{24}} = 265.19$$

**Step 3.5: Z-value**

**Substitution**: $z = \frac{303.0 - 2232.50}{265.19} = -7.276$

**Step 3.6: P-value**

**Result**: $z = -7.276$, $p = 3.44 \times 10^{-13}$

### Step 4: Effect Size

**Result**: $r = 0.75$ (large effect, |r| ≥ 0.5)

### Step 5: Hypothesis Decision

$p = 3.44 \times 10^{-13} < \alpha = 0.05$ → **Reject Null Hypothesis 1**

Transformer > IV3-GRU ($p = 3.44 \times 10^{-13}$, $r = 0.75$ large effect)

---

## Summary

All three metrics show statistically significant differences favoring Transformer over IV3-GRU under nonoccluded conditions:

- **Precision**: Transformer > IV3-GRU ($p = 4.12 \times 10^{-5}$, $r = 0.42$ medium effect)
- **Recall**: Transformer > IV3-GRU ($p = 1.49 \times 10^{-16}$, $r = 0.86$ large effect)
- **F1-score**: Transformer > IV3-GRU ($p = 3.44 \times 10^{-13}$, $r = 0.75$ large effect)

**Overall Decision**: **Reject Null Hypothesis 1** for all three metrics.
