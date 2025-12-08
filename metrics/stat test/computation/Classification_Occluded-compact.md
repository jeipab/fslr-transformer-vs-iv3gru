# Classification Task: Occluded Condition — Statistical Analysis (Compact)

**Hypothesis**: Null Hypothesis 2 — There is no significant difference in performance between Transformer and IV3-GRU models for the classification task under occluded conditions.

**Alpha level**: α = 0.05

---

## Occluded Precision

**Data**: N = 10 valid pairs

### Step 1: Descriptive Statistics

**Formula**:
$$\bar{X} = \frac{1}{N}\sum_{i=1}^{N} X_i, \quad \bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i, \quad \bar{d} = \bar{X} - \bar{Y}$$

**Substitution**:
$$\bar{X} = \frac{1}{10}(5.001685) = 0.500169, \quad \bar{Y} = \frac{1}{10}(3.214100) = 0.321410$$

**Result**: $\bar{d} = 0.178758$

### Step 2: Normality Testing

**Formula**: Shapiro–Wilk test statistic $W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}$

**Result**: $W = 0.920154$, $p = 0.3582 > 0.05$ → **Normal**

### Step 3: Paired t-Test Computation

**Step 3.1: t-statistic**

**Formula**: $t = \frac{\bar{d}\sqrt{N}}{s_d}$

**Substitution**: $t = \frac{0.178758 \times 3.162278}{0.123491} = 4.584$

**Step 3.2: Degrees of freedom**

$df = N - 1 = 9$

**Step 3.3: P-value**

**Formula**: $p = 2(1 - F_t(|t|; df))$

**Result**: $t(9) = 4.58$, $p = .0013$

### Step 4: Effect Size (Cohen's d)

**Formula**: $d_{cohen} = \frac{\bar{d}}{s_d}$

**Result**: $d = 1.45$ (large effect, |d| ≥ 0.8)

### Step 5: Hypothesis Decision

$p = .0013 < \alpha = 0.05$ → **Reject Null Hypothesis 2**

Transformer > IV3-GRU ($p = .0013$, $d = 1.45$ large effect)

---

## Occluded Recall

**Data**: N = 10 valid pairs

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: $\bar{X} = 0.858543$, $\bar{Y} = 0.417127$

**Result**: $\bar{d} = 0.441706$

### Step 2: Normality Testing

**Result**: $W = 0.913566$, $p = 0.3064 > 0.05$ → **Normal**

### Step 3: Paired t-Test Computation

**Step 3.1: t-statistic**

**Formula**: $t = \frac{\bar{d}\sqrt{N}}{s_d}$

**Substitution**: $t = \frac{0.441706 \times 3.162278}{0.145424} = 9.604$

**Step 3.2: Degrees of freedom**

$df = 9$

**Step 3.3: P-value**

**Result**: $t(9) = 9.60$, $p = 5.00 \times 10^{-6}$

### Step 4: Effect Size (Cohen's d)

**Result**: $d = 3.04$ (large effect, |d| ≥ 0.8)

### Step 5: Hypothesis Decision

$p = 5.00 \times 10^{-6} < \alpha = 0.05$ → **Reject Null Hypothesis 2**

Transformer > IV3-GRU ($p = 5.00 \times 10^{-6}$, $d = 3.04$ large effect)

---

## Occluded F1-score

**Data**: N = 10 valid pairs

### Step 1: Descriptive Statistics

**Formula**: $\bar{d} = \bar{X} - \bar{Y}$

**Substitution**: $\bar{X} = 0.624301$, $\bar{Y} = 0.348014$

**Result**: $\bar{d} = 0.276287$

### Step 2: Normality Testing

**Result**: $W = 0.906393$, $p = 0.2571 > 0.05$ → **Normal**

### Step 3: Paired t-Test Computation

**Step 3.1: t-statistic**

**Formula**: $t = \frac{\bar{d}\sqrt{N}}{s_d}$

**Substitution**: $t = \frac{0.276287 \times 3.162278}{0.083793} = 10.432$

**Step 3.2: Degrees of freedom**

$df = 9$

**Step 3.3: P-value**

**Result**: $t(9) = 10.43$, $p = 2.52 \times 10^{-6}$

### Step 4: Effect Size (Cohen's d)

**Result**: $d = 3.30$ (large effect, |d| ≥ 0.8)

### Step 5: Hypothesis Decision

$p = 2.52 \times 10^{-6} < \alpha = 0.05$ → **Reject Null Hypothesis 2**

Transformer > IV3-GRU ($p = 2.52 \times 10^{-6}$, $d = 3.30$ large effect)

---

## Summary

All three metrics show statistically significant differences favoring Transformer over IV3-GRU under occluded conditions:

- **Precision**: Transformer > IV3-GRU ($p = .0013$, $d = 1.45$ large effect)
- **Recall**: Transformer > IV3-GRU ($p = 5.00 \times 10^{-6}$, $d = 3.04$ large effect)
- **F1-score**: Transformer > IV3-GRU ($p = 2.52 \times 10^{-6}$, $d = 3.30$ large effect)

**Overall Decision**: **Reject Null Hypothesis 2** for all three metrics.
