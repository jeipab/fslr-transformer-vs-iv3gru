# Classification Task: Nonoccluded Condition — Statistical Analysis (Compact)

**Hypothesis**: Null Hypothesis 2 — There is no significant difference in performance between Transformer and IV3-GRU models for the classification task under nonoccluded conditions.

**Alpha level**: α = 0.05

---

## Nonoccluded Precision

**Data**: N = 10 valid pairs

### Step 1: Descriptive Statistics

**Formula**:
$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i, \quad \bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i, \quad \bar{d} = \bar{X} - \bar{Y}$$

**Substitution**:
$$\bar{X} = \frac{1}{10}(5.114860) = 0.511486, \quad \bar{Y} = \frac{1}{10}(3.244210) = 0.324421$$

**Result**: $\bar{d} = 0.187066$

### Step 2: Normality Testing

**Formula**: Shapiro–Wilk test statistic $W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$

**Result**: $W = 0.885768$, $p = 0.1519 > 0.05$ → **Normal**

### Step 3: Paired t-Test Computation

**Step 3.1: t-statistic**

**Formula**: $t = \frac{\bar{d}\sqrt{N}}{s_d}$

**Substitution**: $t = \frac{0.187066 \times 3.162278}{0.142726} = 4.144$

**Step 3.2: Degrees of freedom**

$df = N - 1 = 9$

**Step 3.3: P-value**

**Formula**: $p = 2(1 - F_t(|t|; df))$

**Result**: $t(9) = 4.14$, $p = .0025$

### Step 4: Effect Size (Cohen's d)

**Formula**: $d_{cohen} = \frac{\bar{d}}{s_d}$

**Result**: $d = 1.31$ (large effect, |d| ≥ 0.8)

### Step 5: Hypothesis Decision

$p = .0025 < \alpha = 0.05$ → **Reject Null Hypothesis 2**

Transformer > IV3-GRU ($p = .0025$, $d = 1.31$ large effect)

---

## Nonoccluded Recall

**Data**: N = 10 valid pairs

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: $\bar{X} = 0.836834$, $\bar{Y} = 0.351356$

**Result**: $\bar{d} = 0.485478$

### Step 2: Normality Testing

**Result**: $W = 0.800251$, $p = 0.0146 < 0.05$ → **Non-normal**

### Step 3: Wilcoxon Signed-Rank Test Computation

**Step 3.1: Rank absolute differences**

All 10 differences are positive. Rank from smallest to largest.

**Step 3.2: Compute rank sums**

$S^+ = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55$, $S^- = 0$

**Step 3.3: W statistic**

$W = \min(S^+, S^-) = 0$

**Step 3.4: Expected value and standard deviation**

**Formula**: $\mu_W = \frac{N_{nonzero}(N_{nonzero}+1)}{4}$, $\sigma_W = \sqrt{\frac{N_{nonzero}(N_{nonzero}+1)(2N_{nonzero}+1)}{24}}$

**Substitution**: $\mu_W = \frac{10 \times 11}{4} = 27.50$, $\sigma_W = \sqrt{\frac{10 \times 11 \times 21}{24}} = 9.811$

**Step 3.5: Z-value**

**Formula**: $z = \frac{W - \mu_W}{\sigma_W}$

**Substitution**: $z = \frac{0 - 27.50}{9.811} = -2.803$

**Step 3.6: P-value**

**Formula**: $p = 2(1 - \Phi(|z|))$

**Result**: $z = -2.803$, $p = .0020$

### Step 4: Effect Size

**Formula**: $r = \frac{|z|}{\sqrt{N_{nonzero}}}$

**Result**: $r = 0.89$ (large effect, |r| ≥ 0.5)

### Step 5: Hypothesis Decision

$p = .0020 < \alpha = 0.05$ → **Reject Null Hypothesis 2**

Transformer > IV3-GRU ($p = .0020$, $r = 0.89$ large effect)

---

## Nonoccluded F1-score

**Data**: N = 10 valid pairs

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: $\bar{X} = 0.614294$, $\bar{Y} = 0.322017$

**Result**: $\bar{d} = 0.292277$

### Step 2: Normality Testing

**Result**: $W = 0.917105$, $p = 0.3334 > 0.05$ → **Normal**

### Step 3: Paired t-Test Computation

**Step 3.1: t-statistic**

**Formula**: $t = \frac{\bar{d}\sqrt{N}}{s_d}$

**Substitution**: $t = \frac{0.292277 \times 3.162278}{0.129790} = 7.124$

**Step 3.2: Degrees of freedom**

$df = 9$

**Step 3.3: P-value**

**Result**: $t(9) = 7.12$, $p = 5.54 \times 10^{-5}$

### Step 4: Effect Size (Cohen's d)

**Result**: $d = 2.25$ (large effect, |d| ≥ 0.8)

### Step 5: Hypothesis Decision

$p = 5.54 \times 10^{-5} < \alpha = 0.05$ → **Reject Null Hypothesis 2**

Transformer > IV3-GRU ($p = 5.54 \times 10^{-5}$, $d = 2.25$ large effect)

---

## Summary

All three metrics show statistically significant differences favoring Transformer over IV3-GRU under nonoccluded conditions:

- **Precision**: Transformer > IV3-GRU ($p = .0025$, $d = 1.31$ large effect)
- **Recall**: Transformer > IV3-GRU ($p = .0020$, $r = 0.89$ large effect)
- **F1-score**: Transformer > IV3-GRU ($p = 5.54 \times 10^{-5}$, $d = 2.25$ large effect)

**Overall Decision**: **Reject Null Hypothesis 2** for all three metrics.
