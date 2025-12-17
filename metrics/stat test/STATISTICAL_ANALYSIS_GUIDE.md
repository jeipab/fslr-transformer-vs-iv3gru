# Statistical Analysis Methodology and Formulas Reference

## Overview

This document describes the statistical analysis pipeline for comparing Transformer and IV3-GRU models on Filipino Sign Language recognition and classification tasks. The analysis employs paired-sample hypothesis testing with appropriate parametric or non-parametric tests based on data distribution assumptions.

**Purpose (Layman's Terms)**: When comparing two machine learning models, we need to determine if any performance differences are real or just due to random chance. This analysis uses statistical tests to answer: "Is one model genuinely better than the other, or could the difference have happened by luck?"

## Input Data

The analysis processes breakdown CSV files containing per-category or per-gloss performance metrics:

- **Classification**: `classification/Classification-Breakdown.csv`
- **Recognition**: `recognition/Recognition-Breakdown.csv`

Each breakdown file contains precision, recall, and F1-score values for both models, separated by occlusion condition (occluded vs. non-occluded).

**Data Structure**: Each row represents a category/gloss, with columns for Transformer and IV3-GRU scores for each metric (Precision, Recall, F1-score) under occluded and non-occluded conditions.

## Statistical Testing Pipeline

### 1. Data Preparation

**Data Used**: Raw performance scores from breakdown CSV files (Transformer scores and IV3-GRU scores for each category/gloss).

**Process**:

- Extract paired scores: Transformer vs. IV3-GRU
- Compute paired differences: `diff = Transformer - IV3-GRU`
- Remove pairs with missing values (NaN)
- Check for variance in differences

**Purpose (Layman's Terms)**: Before testing, we clean the data by removing incomplete entries and calculate how much each model differs for the same category. This gives us a list of differences to analyze.

### 2. Normality Assessment

**Data Used**: Paired differences (the list of differences between Transformer and IV3-GRU scores).

**Shapiro-Wilk Test** is applied to the paired differences:

- **Null Hypothesis**: Differences are normally distributed
- **Alpha level**: α = 0.05
- **Decision rule**:
  - If p > 0.05 → assume normality → proceed to parametric test
  - If p ≤ 0.05 → assume non-normality → proceed to non-parametric test

**Note**: Normality test requires minimum N ≥ 3 and non-zero variance.

**Purpose (Layman's Terms)**: We check if the differences follow a bell curve pattern. If they do, we can use more powerful statistical tests. If not, we use tests that don't require this assumption.

### 3. Test Selection

#### Parametric Test: Paired Samples t-Test

**Data Used**: Paired differences that passed normality test (normally distributed differences).

**Conditions**:

- Normality assumption met (Shapiro-Wilk p > 0.05)
- N ≥ 2
- Non-zero variance in differences

**Test Details**:

- **Type**: Two-tailed paired samples t-test
- **Null Hypothesis**: No difference between models (μ_diff = 0)
- **Test Statistic**: t-value with df = N - 1
- **Effect Size**: Cohen's d (paired)
  - Formula: `d = mean(diff) / std(diff, ddof=1)`
  - Interpretation: |d| < 0.5 (small), 0.5 ≤ |d| < 0.8 (medium), |d| ≥ 0.8 (large)

**Purpose (Layman's Terms)**: When data follows a bell curve, the t-test tells us if the average difference is statistically meaningful. It's like asking: "Is the average difference big enough that it's unlikely to be zero?"

#### Non-Parametric Test: Wilcoxon Signed-Rank Test

**Data Used**: Paired differences that failed normality test (non-normally distributed differences), with zero differences removed.

**Conditions**:

- Normality assumption violated (Shapiro-Wilk p ≤ 0.05)
- N_nonzero ≥ 2 (pairs with non-zero differences)
- Non-zero variance in differences

**Test Details**:

- **Type**: Two-tailed Wilcoxon signed-rank test
- **Null Hypothesis**: No difference between models (median difference = 0)
- **Test Statistic**:
  - For N ≥ 10: z-value (normal approximation)
  - For N < 10: W statistic (exact test)
- **Effect Size**:
  - For N ≥ 10: r (from z-value), `r = |z| / √N`
  - For N < 10: Not computed (exact test)
  - Interpretation: |r| < 0.3 (small), 0.3 ≤ |r| < 0.5 (medium), |r| ≥ 0.5 (large)

**Purpose (Layman's Terms)**: When data doesn't follow a bell curve, this test ranks the differences and checks if one model consistently performs better. It's more robust to unusual data patterns.

#### Special Cases

- **No variance**: All differences are zero or constant → Test not applicable
- **Insufficient data**: N < 2 or N_nonzero < 2 → Test not applicable

### 4. Hypothesis Testing

**Data Used**: Test statistics and p-values from selected statistical tests.

**Alpha level**: α = 0.05 (two-tailed)

**Decision rule**:

- If p < 0.05 → Reject null hypothesis (significant difference)
- If p ≥ 0.05 → Fail to reject null hypothesis (no significant difference)

**Hypothesis numbering**:

- **Hypothesis 1**: Recognition task (Transformer vs. IV3-GRU)
- **Hypothesis 2**: Classification task (Transformer vs. IV3-GRU)

**Purpose (Layman's Terms)**: The p-value tells us the probability that we'd see this difference by chance alone. If it's less than 5%, we conclude the difference is real and not just luck.

### 5. Effect Size Interpretation

**Data Used**: Test statistics and sample sizes from completed tests.

Effect sizes are reported with magnitude labels following established conventions:

**Cohen's d** (for t-tests):

- Small: |d| < 0.5
- Medium: 0.5 ≤ |d| < 0.8
- Large: |d| ≥ 0.8

**r (from z)** (for Wilcoxon, N ≥ 10):

- Small: |r| < 0.3
- Medium: 0.3 ≤ |r| < 0.5
- Large: |r| ≥ 0.5

**Purpose (Layman's Terms)**: Effect size tells us not just whether there's a difference, but how big that difference is. A statistically significant difference might still be tiny in practice—effect size helps us understand the practical importance.

## Formulas Reference

### I. Data Cleaning and Pairing

#### 1. Remove NaN pairs

**Data Used**: Raw Transformer and IV3-GRU scores from breakdown files.

The script removes any row where _either value_ is NaN:

$$
X_i = \text{Transformer}[i],\quad
Y_i = \text{GRU}[i]
$$

$$
\text{valid\_mask}_i = \neg(\text{isnan}(X_i) \lor \text{isnan}(Y_i))
$$

$$
X_i \leftarrow X_i[\text{valid\_mask}],\quad
Y_i \leftarrow Y_i[\text{valid\_mask}]
$$

**Purpose (Layman's Terms)**: We can't compare models when data is missing, so we remove incomplete pairs.

#### 2. Compute paired differences

**Data Used**: Cleaned Transformer and IV3-GRU scores.

$$
d_i = X_i - Y_i
$$

**Purpose (Layman's Terms)**: We calculate how much better (or worse) the Transformer model is compared to IV3-GRU for each category.

#### 3. Remove zero-differences for Wilcoxon

**Data Used**: Paired differences (for Wilcoxon test only).

(Scipy's `zero_method='wilcox'`)

$$
d^{*} = {d_i \mid d_i \neq 0}
$$

$$
N = |d|,\qquad N_{\text{nonzero}} = |d^{*}|
$$

**Purpose (Layman's Terms)**: For the Wilcoxon test, ties (where both models score the same) don't provide information, so we focus only on categories where models actually differ.

### II. Descriptive Statistics

#### 4. Mean of each model

**Data Used**: Cleaned Transformer and IV3-GRU scores.

$$
\texttt{transformer\_mean} = \frac{1}{N}\sum X_i
$$

$$
\texttt{gru\_mean} = \frac{1}{N}\sum Y_i
$$

**Purpose (Layman's Terms)**: We calculate the average performance of each model across all categories to get an overall picture.

#### 5. Mean difference

**Data Used**: Paired differences.

$$
\texttt{difference} = \bar{d} = \frac{1}{N}\sum d_i
$$

**Purpose (Layman's Terms)**: This tells us the average difference between models—positive means Transformer is better on average, negative means IV3-GRU is better.

#### 6. Standard deviation of differences

**Data Used**: Paired differences.

(Used for t-test and Cohen's d calculation; computed with `ddof=1` for sample standard deviation)

$$
s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}
$$

This matches:

```python
s_d = diff.std(ddof=1)
```

**Purpose (Layman's Terms)**: This measures how consistent the differences are. A small standard deviation means the difference is similar across categories; a large one means it varies a lot.

#### 6b. Variance check

**Data Used**: Paired differences.

(Performed before statistical testing)

$$
\text{has\_variance} = \neg(\text{all}(d_i = 0) \lor (N > 1 \land s_d = 0))
$$

If no variance exists, statistical tests are not applicable.

**Purpose (Layman's Terms)**: If all differences are the same (or zero), there's nothing to test—we need variation to determine if differences are meaningful.

### III. Normality Testing — Shapiro–Wilk

**Data Used**: Paired differences.

(applied to `diff`)

#### 7. Shapiro–Wilk statistic

(Computed by Scipy's `shapiro` function)

$$
W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}
$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients derived from the expected values of order statistics of a standard normal distribution. The test is only performed when $N \geq 3$ and variance exists.

**Purpose (Layman's Terms)**: This formula checks if the differences form a bell curve pattern by comparing them to what we'd expect from a normal distribution.

#### 8. Normality decision rule

**Data Used**: Shapiro-Wilk p-value, sample size, and variance status.

$$
p_{\text{SW}} > 0.05 \text{ and } N \geq 2 \text{ and variance exists} \Rightarrow \text{Use paired t-test}
$$

$$
p_{\text{SW}} \le 0.05 \text{ and } N_{\text{nonzero}} \geq 2 \text{ and variance exists} \Rightarrow \text{Use Wilcoxon}
$$

**Note**: If $N < 3$ or no variance exists, normality test is skipped and test selection depends on other conditions.

**Purpose (Layman's Terms)**: Based on the normality test result, we choose the appropriate statistical test—t-test for bell-curve data, Wilcoxon for other patterns.

### IV. Paired t-test

**Data Used**: Paired differences that are normally distributed.

(Only used if data is normal, $N \geq 2$, and variance exists)

**Conditions**: Normal distribution (Shapiro–Wilk $p > 0.05$), $N \geq 2$, and non-zero variance in differences.

#### 9. t statistic

(Computed by Scipy's `ttest_rel` function)

$$
t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}
$$

**Purpose (Layman's Terms)**: The t-statistic measures how many standard errors the average difference is away from zero. Larger absolute values suggest a more significant difference.

#### 10. Degrees of freedom

**Data Used**: Sample size.

$$
df = N - 1
$$

**Purpose (Layman's Terms)**: This adjusts for the sample size—with more data, we have more confidence in our results.

#### 11. p-value

**Data Used**: t-statistic and degrees of freedom.

$$
p = 2\left(1 - F_t(|t|; df)\right)
$$

**Purpose (Layman's Terms)**: The p-value is the probability of seeing this t-statistic (or more extreme) if there's actually no difference between models.

#### 12. Cohen's d

**Data Used**: Mean and standard deviation of paired differences.

(Paired samples Cohen's d, computed exactly as in code)

$$
d_{\text{cohen}} = \frac{\bar{d}}{s_d} = \frac{\text{mean}(d_i)}{\text{std}(d_i, \text{ddof}=1)}
$$

This matches:

```python
cohens_d = diff.mean() / diff.std(ddof=1)
```

**Note**: Returns `NaN` if standard deviation is zero (no variance).

**Purpose (Layman's Terms)**: Cohen's d standardizes the difference by the variability, giving us a measure of effect size that's comparable across different metrics.

### V. Wilcoxon Signed-Rank Test

**Data Used**: Non-zero paired differences (zero differences removed).

(Used when data is non-normal, $N_{\text{nonzero}} \geq 2$, and variance exists)

**Conditions**: Non-normal distribution (Shapiro–Wilk $p \leq 0.05$) or normality test not applicable, $N_{\text{nonzero}} \geq 2$, and non-zero variance in differences.

#### 13. Rank absolute non-zero differences

$$
R_i = \operatorname{rank}(|d^{*}_i|)
$$

**Purpose (Layman's Terms)**: We rank the differences by their size (ignoring direction), so the biggest difference gets the highest rank.

#### 14. Positive and negative rank sums

$$
S^+ = \sum_{d^{*}_i > 0} R_i
$$

$$
S^- = \sum_{d^{*}_i < 0} R_i
$$

**Purpose (Layman's Terms)**: We sum up the ranks where Transformer is better (positive) and where IV3-GRU is better (negative) to see which model has more and larger advantages.

#### 15. W statistic used by Scipy

(Scipy's `wilcoxon` function with `zero_method='wilcox'` and `alternative='two-sided'`)

Scipy returns the _smaller_ of the positive/negative rank sums:

$$
W = \min(S^+, S^-)
$$

This matches:

```python
w_stat, p_val = wilcoxon(transformer, gru, zero_method='wilcox', alternative='two-sided')
```

**Purpose (Layman's Terms)**: The W statistic is the smaller rank sum. If one model is consistently better, this value will be small, indicating a significant difference.

### VI. Wilcoxon Z-Value

**Data Used**: W statistic and number of non-zero pairs.

(Computed explicitly by the script using `calculate_wilcoxon_z`)
**IMPORTANT:** The script uses **non-zero pairs** ($N_{\text{nonzero}}$), NOT total N.

#### 16. Expected value of W

$$
\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}
$$

**Purpose (Layman's Terms)**: This is what we'd expect the W statistic to be if there's no real difference between models.

#### 17. Standard deviation of W

$$
\sigma_W =
\sqrt{
\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}
}
$$

**Purpose (Layman's Terms)**: This measures how much the W statistic typically varies when there's no real difference.

#### 18. Z-value computed by the script

$$
z = \frac{W - \mu_W}{\sigma_W}
$$

This matches:

```python
z = calculate_wilcoxon_z(w_stat, n_nonzero)
```

**Purpose (Layman's Terms)**: The z-value standardizes the W statistic, telling us how many standard deviations away from expected it is. This helps us calculate the p-value.

### VII. P-value for Wilcoxon (large N)

**Data Used**: Z-value from Wilcoxon test.

(When `n_nonzero >= 10`)

#### 19. Two-tailed p-value

(Computed by Scipy for large $N_{\text{nonzero}} \geq 10$)

$$
p = 2\left(1 - \Phi(|z|)\right)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

**Purpose (Layman's Terms)**: For larger samples, we use the normal distribution to calculate the probability of seeing this z-value if there's no real difference.

### VIII. Effect Size for Wilcoxon (r)

**Data Used**: Z-value and number of non-zero pairs.

(Only when `n_nonzero >= 10`)

#### 20. Effect size formula

$$
r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}
$$

This matches:

```python
effect_size = abs(z) / sqrt(n_nonzero)
```

**Purpose (Layman's Terms)**: This converts the z-value into an effect size measure that tells us the practical magnitude of the difference, accounting for sample size.

### IX. Exact Wilcoxon Test (small N)

**Data Used**: W statistic and exact distribution.

(When `n_nonzero < 10`)

#### 21. Exact statistic

$$
W = \min(S^+, S^-)
$$

#### 22. p-value

(Computed by Scipy's exact distribution for small $N_{\text{nonzero}} < 10$)

No closed-form formula; obtained from Scipy's exact permutation distribution of the Wilcoxon W statistic:

$$
p = P(W \leq w_{\text{obs}} \mid H_0)
$$

where $w_{\text{obs}}$ is the observed W statistic and the probability is computed from the exact null distribution.

**Purpose (Layman's Terms)**: For small samples, we can't use approximations, so we calculate the exact probability by checking all possible outcomes.

#### 23. Effect size

Not computed:

$$
r = \text{N/A}
$$

**Purpose (Layman's Terms)**: Effect size isn't reliable with very small samples, so we don't calculate it in these cases.

### X. Direction and Decision Rules

**Data Used**: Mean difference and p-value from statistical test.

#### 24. Direction

$$
\text{If } \bar{d} > 0 \Rightarrow \text{Transformer > IV3-GRU}
$$

$$
\text{If } \bar{d} < 0 \Rightarrow \text{IV3-GRU > Transformer}
$$

**Purpose (Layman's Terms)**: This simply tells us which model performed better on average.

#### 25. Hypothesis decision

(Alpha level: $\alpha = 0.05$)

$$
p < 0.05 \Rightarrow \text{Reject Null Hypothesis}
$$

$$
p \ge 0.05 \Rightarrow \text{Fail to Reject Null Hypothesis}
$$

**Note**: In the output, hypotheses are labeled as "Null Hypothesis 1" (for recognition) or "Null Hypothesis 2" (for classification). If no variance exists, the decision is always "Fail to Reject Null Hypothesis".

**Purpose (Layman's Terms)**: Based on the p-value, we make a final decision: if p < 0.05, we conclude there's a real difference; otherwise, we can't be confident the difference isn't just due to chance.

## Output Format

**Data Used**: All computed statistics, test results, and decisions.

Results are saved to CSV files with the following columns:

| Column             | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `metric`           | Metric name (e.g., "Occluded Precision")                    |
| `n`                | Sample size (total for t-test, non-zero pairs for Wilcoxon) |
| `transformer_mean` | Mean Transformer score                                      |
| `gru_mean`         | Mean IV3-GRU score                                          |
| `difference`       | Mean difference (Transformer - IV3-GRU)                     |
| `normality`        | Normality assessment with exact p-value                     |
| `test_used`        | Statistical test applied                                    |
| `test_statistic`   | Formatted test statistic (t, z, or W)                       |
| `p_value`          | Formatted p-value with significance indicator               |
| `effect_size`      | Effect size with magnitude interpretation                   |
| `direction`        | Direction of difference (Transformer > IV3-GRU, etc.)       |
| `decision`         | Statistical decision (Reject/Fail to Reject H0/H1/H2)       |

## P-Value Formatting

**Data Used**: Raw p-values from statistical tests.

P-values are formatted according to academic standards:

- **Very small** (p < 0.0001): Scientific notation (e.g., "2.34×10⁻⁵")
- **Larger values**: Decimal notation without leading zero (e.g., ".0234")
- **Non-significant** (p ≥ 0.05): Includes "(ns)" indicator

## Academic Standards

The analysis follows established statistical practices:

- **Assumption checking**: Normality tested before test selection
- **Appropriate test selection**: Parametric vs. non-parametric based on data distribution
- **Effect size reporting**: Always reported when applicable
- **Exact p-values**: Reported rather than thresholds (p < 0.05)
- **Two-tailed tests**: Appropriate for exploratory comparisons
- **Sample size considerations**: Different handling for small N (exact tests)

## Usage

```bash
python statistical_analysis.py
```

The script automatically:

1. Loads breakdown CSV files from subdirectories
2. Processes all metrics (Precision, Recall, F1-score) for both occlusion conditions
3. Performs appropriate statistical tests
4. Saves formatted results to:
   - `classification/Classification-StatsResults.csv`
   - `recognition/Recognition-StatsResults.csv`
