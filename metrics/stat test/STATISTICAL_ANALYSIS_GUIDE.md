# Statistical Analysis Methodology

## Overview

This document describes the statistical analysis pipeline for comparing Transformer and IV3-GRU models on Filipino Sign Language recognition and classification tasks. The analysis employs paired-sample hypothesis testing with appropriate parametric or non-parametric tests based on data distribution assumptions.

## Input Data

The analysis processes breakdown CSV files containing per-category or per-gloss performance metrics:

- **Classification**: `classification/Classification-Breakdown.csv`
- **Recognition**: `recognition/Recognition-Breakdown.csv`

Each breakdown file contains precision, recall, and F1-score values for both models, separated by occlusion condition (occluded vs. non-occluded).

## Statistical Testing Pipeline

### 1. Data Preparation

For each metric (Precision, Recall, F1-score) and occlusion condition:

- Extract paired scores: Transformer vs. IV3-GRU
- Compute paired differences: `diff = Transformer - IV3-GRU`
- Remove pairs with missing values (NaN)
- Check for variance in differences

### 2. Normality Assessment

**Shapiro-Wilk Test** is applied to the paired differences:

- **Null Hypothesis**: Differences are normally distributed
- **Alpha level**: α = 0.05
- **Decision rule**:
  - If p > 0.05 → assume normality → proceed to parametric test
  - If p ≤ 0.05 → assume non-normality → proceed to non-parametric test

**Note**: Normality test requires minimum N ≥ 3 and non-zero variance.

### 3. Test Selection

#### Parametric Test: Paired Samples t-Test

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

#### Non-Parametric Test: Wilcoxon Signed-Rank Test

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

#### Special Cases

- **No variance**: All differences are zero or constant → Test not applicable
- **Insufficient data**: N < 2 or N_nonzero < 2 → Test not applicable

### 4. Hypothesis Testing

**Alpha level**: α = 0.05 (two-tailed)

**Decision rule**:

- If p < 0.05 → Reject null hypothesis (significant difference)
- If p ≥ 0.05 → Fail to reject null hypothesis (no significant difference)

**Hypothesis numbering**:

- **Hypothesis 1**: Recognition task (Transformer vs. IV3-GRU)
- **Hypothesis 2**: Classification task (Transformer vs. IV3-GRU)

### 5. Effect Size Interpretation

Effect sizes are reported with magnitude labels following established conventions:

**Cohen's d** (for t-tests):

- Small: |d| < 0.5
- Medium: 0.5 ≤ |d| < 0.8
- Large: |d| ≥ 0.8

**r (from z)** (for Wilcoxon, N ≥ 10):

- Small: |r| < 0.3
- Medium: 0.3 ≤ |r| < 0.5
- Large: |r| ≥ 0.5

## Output Format

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
