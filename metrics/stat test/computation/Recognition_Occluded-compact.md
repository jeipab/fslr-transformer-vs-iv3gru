# Recognition Task: Occluded Condition — Statistical Analysis (Compact)

**Hypothesis**: Null Hypothesis 1 — There is no significant difference in performance between Transformer and IV3-GRU models for the recognition task under occluded conditions.

**Alpha level**: α = 0.05

---

## Occluded Precision

**Data**: N = 105 valid pairs (all pairs for mean calculation), N = 98 pairs after removing both-zero (for Wilcoxon test), N_nonzero = 98

### Step 1: Descriptive Statistics

**Formula**:
$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i, \quad \bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i, \quad \bar{d} = \bar{X} - \bar{Y}$$

**Substitution**: Using all N = 105 valid pairs

**Result**: $\bar{X} = 0.470068$, $\bar{Y} = 0.378707$, $\bar{d} = 0.091360$

### Step 2: Normality Testing

**Formula**: Shapiro–Wilk test statistic $W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$

**Result**: $W = 0.941027$, $p = 2.92 \times 10^{-5} < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute non-zero differences**

For N_nonzero = 98 non-zero differences, rank the absolute values: $R_i = \operatorname{rank}(|d^{*}_i|)$

**Step 3.2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i, \quad S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 1200.0$

**Step 3.4: Expected value and standard deviation**

**Formula**:
$$\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4}, \quad \sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}}$$

**Substitution**:
$$\mu_W = \frac{98 \times 99}{4} = 2425.50, \quad \sigma_W = \sqrt{\frac{98 \times 99 \times 197}{24}} = 281.82$$

**Step 3.5: Z-value**

**Formula**: $z = \frac{W - \mu_W}{\sigma_W}$

**Substitution**: $z = \frac{1200.0 - 2425.50}{281.82} = -4.343$

**Step 3.6: P-value**

**Formula**: $p = 2(1 - \Phi(|z|))$

**Result**: $z = -4.343$, $p = 1.41 \times 10^{-5}$

### Step 4: Effect Size

**Formula**: $r = \frac{|z|}{\sqrt{N_{nonzero}}}$

**Result**: $r = 0.44$ (medium effect, 0.3 ≤ |r| < 0.5)

### Step 5: Hypothesis Decision

$p = 1.41 \times 10^{-5} < \alpha = 0.05$ → **Reject Null Hypothesis 1**

Transformer > IV3-GRU ($p = 1.41 \times 10^{-5}$, $r = 0.44$ medium effect)

---

## Occluded Recall

**Data**: N = 105 valid pairs (all pairs for mean calculation), N = 98 pairs after removing both-zero (for Wilcoxon test), N_nonzero = 93

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: Using all N = 105 valid pairs

**Result**: $\bar{X} = 0.813161$, $\bar{Y} = 0.526710$, $\bar{d} = 0.286451$

### Step 2: Normality Testing

**Result**: $W = 0.948846$, $p = 8.37 \times 10^{-5} < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute non-zero differences**

For N_nonzero = 93 non-zero differences, rank the absolute values: $R_i = \operatorname{rank}(|d^{*}_i|)$

**Step 3.2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i, \quad S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 125.0$

**Step 3.4: Expected value and standard deviation**

**Substitution**:
$$\mu_W = \frac{93 \times 94}{4} = 2185.50, \quad \sigma_W = \sqrt{\frac{93 \times 94 \times 187}{24}} = 261.10$$

**Step 3.5: Z-value**

**Substitution**: $z = \frac{125.0 - 2185.50}{261.10} = -7.895$

**Step 3.6: P-value**

**Result**: $z = -7.895$, $p = 2.90 \times 10^{-15}$

### Step 4: Effect Size

**Result**: $r = 0.82$ (large effect, |r| ≥ 0.5)

### Step 5: Hypothesis Decision

$p = 2.90 \times 10^{-15} < \alpha = 0.05$ → **Reject Null Hypothesis 1**

Transformer > IV3-GRU ($p = 2.90 \times 10^{-15}$, $r = 0.82$ large effect)

---

## Occluded F1-score

**Data**: N = 105 valid pairs (all pairs for mean calculation), N = 98 pairs after removing both-zero (for Wilcoxon test), N_nonzero = 98

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: Using all N = 105 valid pairs

**Result**: $\bar{X} = 0.563301$, $\bar{Y} = 0.411522$, $\bar{d} = 0.151778$

### Step 2: Normality Testing

**Result**: $W = 0.930729$, $p = 4.21 \times 10^{-6} < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute non-zero differences**

For N_nonzero = 98 non-zero differences, rank the absolute values: $R_i = \operatorname{rank}(|d^{*}_i|)$

**Step 3.2: Compute positive and negative rank sums**

$$S^+ = \sum_{d^{*}_i > 0} R_i, \quad S^- = \sum_{d^{*}_i < 0} R_i$$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 551.0$

**Step 3.4: Expected value and standard deviation**

**Substitution**:
$$\mu_W = \frac{98 \times 99}{4} = 2425.50, \quad \sigma_W = \sqrt{\frac{98 \times 99 \times 197}{24}} = 281.82$$

**Step 3.5: Z-value**

**Substitution**: $z = \frac{551.0 - 2425.50}{281.82} = -6.642$

**Step 3.6: P-value**

**Result**: $z = -6.642$, $p = 3.09 \times 10^{-11}$

### Step 4: Effect Size

**Result**: $r = 0.67$ (large effect, |r| ≥ 0.5)

### Step 5: Hypothesis Decision

$p = 3.09 \times 10^{-11} < \alpha = 0.05$ → **Reject Null Hypothesis 1**

Transformer > IV3-GRU ($p = 3.09 \times 10^{-11}$, $r = 0.67$ large effect)

---

## Summary

All three metrics show statistically significant differences favoring Transformer over IV3-GRU under occluded conditions:

- **Precision**: Transformer > IV3-GRU ($p = 1.41 \times 10^{-5}$, $r = 0.44$ medium effect)
- **Recall**: Transformer > IV3-GRU ($p = 2.90 \times 10^{-15}$, $r = 0.82$ large effect)
- **F1-score**: Transformer > IV3-GRU ($p = 3.09 \times 10^{-11}$, $r = 0.67$ large effect)

**Overall Decision**: **Reject Null Hypothesis 1** for all three metrics.
