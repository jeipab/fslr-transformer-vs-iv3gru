"""
Statistical Analysis Pipeline for Transformer vs IV3-GRU Comparison

This script performs comprehensive statistical testing comparing Transformer and IV3-GRU
models for Filipino Sign Language recognition and classification tasks.

USAGE:
    python "metrics\stat test\statistical_analysis.py"

REQUIREMENTS:
    - Python 3.7+
    - pandas
    - numpy
    - scipy

INPUT FILES (expected in subdirectories):
    - classification/Classification-Breakdown.csv
    - recognition/Recognition-Breakdown.csv

OUTPUT FILES (generated in subdirectories):
    - classification/Classification-StatsResults.csv
    - recognition/Recognition-StatsResults.csv

STATISTICAL TESTS PERFORMED:
    1. Shapiro-Wilk normality test on paired differences
    2. If normal (p > 0.05) -> Paired t-test with Cohen's d effect size
    3. If non-normal (p <= 0.05) -> Wilcoxon Signed-Rank test
       - Effect size r (from z-value) for N >= 10
       - Exact test for N < 10 (no effect size)

OUTPUT COLUMNS:
    - metric: Metric name (e.g., "Occluded Precision")
    - transformer_mean: Mean Transformer score
    - gru_mean: Mean IV3-GRU score
    - difference: Mean difference (Transformer - IV3-GRU)
    - normality: Normality assessment text
    - test_used: Statistical test applied
    - test_statistic: Test statistic value (formatted)
    - p_value: P-value from statistical test (formatted)
    - effect_size: Effect size with interpretation
    - direction: Direction of difference
    - decision: Statistical decision

NOTES:
    - Alpha level: 0.05
    - All metrics (Precision, Recall, F1-score) are tested for both
      occluded and non-occluded conditions
    - Results are automatically saved to CSV files for thesis documentation
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, wilcoxon, ttest_rel
from pathlib import Path
import warnings
import math

warnings.filterwarnings('ignore')


def effect_size_r_from_z(z, n):
    """Calculate effect size r from z-value for Wilcoxon test."""
    return abs(z) / np.sqrt(n)


def effect_size_cohens_d(diff):
    """Calculate Cohen's d (paired) for parametric tests."""
    if diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


def calculate_wilcoxon_z(w_stat, n):
    """
    Calculate z-value from Wilcoxon W statistic.
    
    Parameters:
    -----------
    w_stat : float
        Wilcoxon W statistic
    n : int
        Sample size (non-zero pairs)
        
    Returns:
    --------
    float : z-value
    """
    mean_w = n * (n + 1) / 4
    sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_stat - mean_w) / sd_w
    return z


def sci_notation_str(p, sig=2):
    """Format number in scientific notation."""
    if p == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(p))))
    mant = p / (10**exp)
    return f"{round(mant, sig)}×10^{exp}"


def format_pvalue_exact(p):
    """
    Format p-value as exact value (academic standard).
    Uses scientific notation for very small values.
    """
    if p is None or p == "N/A" or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    try:
        p = float(p)
    except:
        return str(p)

    # For very small p-values, use scientific notation
    if p < 1e-4:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10**exp)
        return f"{mant:.2f}×10⁻{abs(exp)}"
    else:
        # For larger p-values, use decimal notation (remove leading zero)
        return f"{p:.4f}".replace("0.", ".")


def format_pvalue(p):
    """
    Format p-value for p_value column (includes significance indicator).
    Always shows exact value.
    """
    if p is None or p == "N/A" or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    try:
        p = float(p)
    except:
        return str(p)

    # Format exact p-value
    if p < 1e-4:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10**exp)
        p_str = f"{mant:.2f}×10⁻{abs(exp)}"
    else:
        p_str = f"{p:.4f}".replace("0.", ".")
    
    # Add significance indicator
    if p >= 0.05:
        return f"p = {p_str} (ns)"
    else:
        return f"p = {p_str}"


def effect_label_r(r):
    """Get effect size label for r."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "N/A"
    r = abs(float(r))
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "medium"
    return "large"


def effect_label_d(d):
    """Get effect size label for d."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "N/A"
    d = abs(float(d))
    if d < 0.50:
        return "small"
    if d < 0.80:
        return "medium"
    return "large"


def format_normality_text(normality_p, has_variance):
    """
    Format normality assessment as text.
    
    Parameters:
    -----------
    normality_p : float
        Shapiro-Wilk p-value
    has_variance : bool
        Whether there is variance in the differences
        
    Returns:
    --------
    str : Formatted normality text
    """
    if not has_variance:
        return "Normality cannot be assessed (no variance)"
    
    if np.isnan(normality_p):
        return "N/A"
    
    if normality_p < 0.00001:
        return f"Non-normal (Shapiro–Wilk p < .00001)"
    elif normality_p < 0.05:
        # Format with scientific notation
        if normality_p < 0.0001:
            exp = int(np.floor(np.log10(normality_p)))
            coeff = normality_p / (10 ** exp)
            return f"Non-normal (Shapiro–Wilk p = {coeff:.2f}×10⁻{abs(exp)})"
        else:
            # Remove leading zero
            p_str = f"{normality_p:.4f}".lstrip('0')
            if p_str.startswith('.'):
                return f"Non-normal (Shapiro–Wilk p = {p_str})"
            else:
                return f"Non-normal (Shapiro–Wilk p = {normality_p:.4f})"
    else:
        # Remove leading zero
        p_str = f"{normality_p:.4f}".lstrip('0')
        if p_str.startswith('.'):
            return f"Normal (Shapiro–Wilk p = {p_str})"
        else:
            return f"Normal (Shapiro–Wilk p = {normality_p:.4f})"


def format_p_value_text(p_val):
    """
    Format p-value for reporting with proper notation.
    
    Parameters:
    -----------
    p_val : float
        P-value
        
    Returns:
    --------
    str : Formatted p-value string
    """
    if np.isnan(p_val):
        return "N/A"
    
    if p_val == 0 or p_val < 0.00001:
        return "p < .00001"
    elif p_val < 0.0001:
        return "p < .0001"
    elif p_val < 0.001:
        return "p < .001"
    elif p_val < 0.05:
        # Remove leading zero
        p_str = f"{p_val:.4f}".lstrip('0')
        if p_str.startswith('.'):
            return f"p = {p_str}"
        else:
            return f"p = {p_val:.4f}"
    else:
        # Non-significant, add (ns)
        p_str = f"{p_val:.4f}".lstrip('0')
        if p_str.startswith('.'):
            return f"p = {p_str} (ns)"
        else:
            return f"p = {p_val:.4f} (ns)"


def format_effect_size_text(effect_size, effect_type):
    """
    Format effect size with magnitude interpretation (APA standards).
    
    Parameters:
    -----------
    effect_size : float
        Effect size value
    effect_type : str
        Type of effect size ('Cohen's d' or 'r (from z)')
        
    Returns:
    --------
    str : Formatted effect size string with interpretation
    """
    if np.isnan(effect_size) or effect_type == "N/A":
        return "N/A"
    
    if effect_type == "Cohen's d":
        abs_d = abs(effect_size)
        if abs_d >= 0.8:
            magnitude = "large"
        elif abs_d >= 0.5:
            magnitude = "medium"
        else:
            magnitude = "small"
        return f"d = {effect_size:.2f} ({magnitude})"
    
    elif effect_type == "r (from z)":
        abs_r = abs(effect_size)
        if abs_r >= 0.5:
            magnitude = "large"
        elif abs_r >= 0.3:
            magnitude = "medium"
        else:
            magnitude = "small"
        return f"r = {effect_size:.2f} ({magnitude})"
    
    return "N/A"


def format_test_statistic_text(test_statistic, test_used, n, df=None, w_stat=None):
    """
    Format test statistic based on test type.
    
    Parameters:
    -----------
    test_statistic : float
        Test statistic value
    test_used : str
        Name of test used
    n : int
        Sample size (non-zero pairs for Wilcoxon)
    df : int, optional
        Degrees of freedom for t-test
    w_stat : float, optional
        Wilcoxon W statistic (for small N)
        
    Returns:
    --------
    str : Formatted test statistic
    """
    if np.isnan(test_statistic):
        return "N/A"
    
    if "Wilcoxon" in test_used:
        if n >= 10:
            # Use z-value
            return f"z = {test_statistic:.3f}"
        else:
            # Use W statistic for small N
            if w_stat is not None:
                return f"W = {w_stat:.0f} (exact test)"
            else:
                return f"W = {test_statistic:.0f} (exact test)"
    
    elif "t-Test" in test_used or "t-test" in test_used:
        if df is not None:
            return f"t(df = {df}) = {test_statistic:.2f}"
        else:
            return f"t = {test_statistic:.2f}"
    
    return str(test_statistic)


def format_direction_text(mean_diff):
    """
    Format direction of difference.
    
    Parameters:
    -----------
    mean_diff : float
        Mean difference (Transformer - IV3-GRU)
        
    Returns:
    --------
    str : Direction text
    """
    if np.isnan(mean_diff):
        return "N/A"
    
    if mean_diff > 0:
        return "Transformer > IV3-GRU"
    elif mean_diff < 0:
        return "IV3-GRU > Transformer"
    else:
        return "Equal performance"


def format_decision_text(p_val, has_variance, hypothesis_num=1):
    """
    Format statistical decision.
    
    Parameters:
    -----------
    p_val : float
        P-value
    has_variance : bool
        Whether test was applicable
    hypothesis_num : int
        Hypothesis number (1 for recognition, 2 for classification)
        
    Returns:
    --------
    str : Decision text
    """
    if not has_variance:
        return f"Fail to Reject Null Hypothesis {hypothesis_num}"
    
    if np.isnan(p_val):
        return "N/A"
    
    if p_val < 0.05:
        return f"Reject Null Hypothesis {hypothesis_num}"
    else:
        return f"Fail to Reject Null Hypothesis {hypothesis_num}"


def format_statistic_row(metric, n, transformer_mean, gru_mean, diff,
                         normality_p, test_used, test_stat_raw, p_value_raw,
                         effect_val, effect_type, direction, decision, hypothesis_num=1):
    """
    Format a single test result row for CSV output.
    
    Parameters:
    -----------
    metric : str
        Metric name
    n : int
        Sample size (total for t-test, non-zero for Wilcoxon)
    transformer_mean : float
        Mean Transformer score
    gru_mean : float
        Mean IV3-GRU score
    diff : float
        Mean difference
    normality_p : float or None
        Shapiro-Wilk p-value
    test_used : str
        Test name
    test_stat_raw : float or dict
        Test statistic (raw value or dict with W/z/n for Wilcoxon)
    p_value_raw : float or None
        P-value
    effect_val : float or None
        Effect size value
    effect_type : str
        Effect type ('r' or 'd')
    direction : str
        Direction text
    decision : str
        Decision text (will be formatted with hypothesis_num)
    hypothesis_num : int
        Hypothesis number (1 for recognition, 2 for classification)
        
    Returns:
    --------
    dict : Formatted result dictionary
    """
    # Normality formatting (always show exact p-values)
    if normality_p is None or (isinstance(normality_p, float) and np.isnan(normality_p)):
        normality_text = "Normality cannot be assessed (no variance)"
    else:
        p = float(normality_p)
        p_exact = format_pvalue_exact(p)
        if p < 0.05:
            normality_text = f"Non-normal (Shapiro–Wilk p = {p_exact})"
        else:
            normality_text = f"Normal (Shapiro–Wilk p = {p_exact})"

    # Test label
    if "wilcoxon" in test_used.lower():
        test_label = "Wilcoxon Signed-Rank Test (two-tailed)"
    elif "t-test" in test_used.lower() or "t-test" in test_used:
        test_label = "Paired Samples t-Test (two-tailed)"
    elif "Test not applicable" in test_used or "N/A" in test_used:
        test_label = "Test not applicable — no variance"
    else:
        test_label = test_used

    # Test statistic formatting
    if "Test not applicable" in test_used or "N/A" in test_used:
        stat_text = "N/A"
    elif "wilcoxon" in test_used.lower():
        if isinstance(test_stat_raw, dict):
            n_nonzero = test_stat_raw.get("n", n)
            if n_nonzero < 10:
                stat_text = f"W = {test_stat_raw['W']:.0f} (exact test, N = {n_nonzero})"
            else:
                z = test_stat_raw.get("z", None)
                if z is not None:
                    stat_text = f"z = {z:.3f}"
                else:
                    stat_text = "N/A"
        else:
            # Fallback: use raw value if it's a number
            if not (isinstance(test_stat_raw, float) and np.isnan(test_stat_raw)):
                stat_text = f"z = {test_stat_raw:.3f}"
            else:
                stat_text = "N/A"
    else:
        # t-test
        tval = float(test_stat_raw) if not (isinstance(test_stat_raw, float) and np.isnan(test_stat_raw)) else None
        if tval is not None:
            stat_text = f"t(df = {n-1}) = {tval:.2f}"
        else:
            stat_text = "N/A"

    p_text = format_pvalue(p_value_raw)

    # Effect size formatting
    if effect_type == "r" and effect_val is not None and not (isinstance(effect_val, float) and np.isnan(effect_val)):
        eff_text = f"r = {float(effect_val):.2f} ({effect_label_r(effect_val)})"
    elif effect_type == "d" and effect_val is not None and not (isinstance(effect_val, float) and np.isnan(effect_val)):
        eff_text = f"d = {float(effect_val):.2f} ({effect_label_d(effect_val)})"
    else:
        eff_text = "N/A"

    # Format decision with hypothesis number
    # Handle no variance case
    if "Test not applicable" in test_label or test_label == "N/A":
        decision_text = f"Fail to Reject Null Hypothesis {hypothesis_num}"
    elif p_value_raw is not None and not (isinstance(p_value_raw, float) and np.isnan(p_value_raw)):
        # Determine decision based on p-value
        p_val = float(p_value_raw)
        if p_val < 0.05:
            decision_text = f"Reject Null Hypothesis {hypothesis_num}"
        else:
            decision_text = f"Fail to Reject Null Hypothesis {hypothesis_num}"
    elif "Reject" in decision:
        decision_text = f"Reject Null Hypothesis {hypothesis_num}"
    elif "Fail to Reject" in decision:
        decision_text = f"Fail to Reject Null Hypothesis {hypothesis_num}"
    else:
        decision_text = decision

    return {
        "metric": metric,
        "n": n,
        "transformer_mean": round(transformer_mean, 6) if not (isinstance(transformer_mean, float) and np.isnan(transformer_mean)) else transformer_mean,
        "gru_mean": round(gru_mean, 6) if not (isinstance(gru_mean, float) and np.isnan(gru_mean)) else gru_mean,
        "difference": round(diff, 6) if not (isinstance(diff, float) and np.isnan(diff)) else diff,
        "normality": normality_text,
        "test_used": test_label,
        "test_statistic": stat_text,
        "p_value": p_text,
        "effect_size": eff_text,
        "direction": direction,
        "decision": decision_text
    }


def analyze_metric(transformer, gru, metric_name, verbose=True):
    """
    Analyze differences between Transformer and IV3-GRU for a given metric.
    
    Parameters:
    -----------
    transformer : array-like
        Transformer model scores
    gru : array-like
        IV3-GRU model scores
    metric_name : str
        Name of the metric being analyzed
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    dict : Dictionary containing all statistical test results (raw, unformatted)
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"METRIC: {metric_name}")
        print("=" * 60)
    
    # Ensure arrays are numpy arrays and remove any NaN values
    transformer = np.array(transformer)
    gru = np.array(gru)
    
    # Remove pairs where either value is NaN
    valid_mask = ~(np.isnan(transformer) | np.isnan(gru))
    transformer = transformer[valid_mask]
    gru = gru[valid_mask]
    
    if len(transformer) == 0:
        if verbose:
            print("WARNING: No valid data pairs found!")
        return {
            'metric': metric_name,
            'n': 0,
            'n_nonzero': 0,
            'transformer_mean': np.nan,
            'gru_mean': np.nan,
            'difference': np.nan,
            'normality_p': np.nan,
            'test_used': 'N/A',
            'test_statistic': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'effect_size_type': 'N/A',
            'direction': 'N/A',
            'decision': 'N/A',
            'df': None,
            'w_stat': None
        }

    # Compute paired differences
    diff = transformer - gru
    diff_nonzero = diff[diff != 0]  # For Wilcoxon

    n = len(diff)
    n_nonzero = len(diff_nonzero)
    
    transformer_mean = transformer.mean()
    gru_mean = gru.mean()
    mean_diff = diff.mean()
    
    # Check if there's variance (all differences are zero or std is zero)
    has_variance = not (np.all(diff == 0) or (len(diff) > 1 and diff.std(ddof=1) == 0))
    
    if verbose:
        print(f"Number of samples: {n}")
        print(f"Non-zero pairs (Wilcoxon N): {n_nonzero}")
        print(f"Transformer mean: {transformer_mean:.4f}")
        print(f"IV3-GRU mean: {gru_mean:.4f}")
        print(f"Mean difference: {mean_diff:.4f}")

    # 1. Shapiro-Wilk Normality Test
    if n < 3 or not has_variance:
        if verbose:
            if n < 3:
                print("WARNING: Sample size too small for normality test (N < 3)")
            else:
                print("WARNING: No variance - cannot assess normality")
        normality_p = np.nan
        normal = False
    else:
        sh_stat, normality_p = shapiro(diff)
        normal = normality_p > 0.05
        if verbose:
            print(f"Shapiro-Wilk: W = {sh_stat:.4f}, p = {normality_p:.6f}")
    
    # Determine direction
    if not has_variance:
        direction = "Equal performance"
    elif mean_diff > 0:
        direction = "Transformer > IV3-GRU"
    elif mean_diff < 0:
        direction = "IV3-GRU > Transformer"
    else:
        direction = "Equal performance"
    
    # 2. Choose and perform appropriate test
    if normal and n >= 2 and has_variance:
        # Paired t-test
        if verbose:
            print("-> Normal distribution -> Paired t-test")
        
        t_stat, p_val = ttest_rel(transformer, gru)
        df = n - 1
        
        # Effect size: Cohen's d (paired)
        cohens_d = effect_size_cohens_d(diff)
        
        if verbose:
            print(f"t({df}) = {t_stat:.4f}, p = {p_val:.6f}")
            if not np.isnan(cohens_d):
                print(f"Cohen's d = {cohens_d:.4f}")
        
        test_used = "Paired t-test (two-tailed)"
        test_statistic = t_stat
        effect_size = cohens_d
        effect_size_type = "d"
        w_stat = None
        
    elif n_nonzero >= 2 and has_variance:
        # Wilcoxon Signed-Rank Test
        if verbose:
            print("-> Non-normal -> Wilcoxon Signed-Rank Test")
        
        # Use exact test for small N, approximate for larger N
        w_stat, p_val = wilcoxon(transformer, gru, zero_method='wilcox', alternative='two-sided')
        
        # Calculate z-value from W statistic
        z = calculate_wilcoxon_z(w_stat, n_nonzero)
        
        # Store test statistic as dict for Wilcoxon
        test_statistic = {"W": w_stat, "z": z, "n": n_nonzero}
        
        # Calculate effect size r from z-value
        if n_nonzero >= 10:
            r = effect_size_r_from_z(z, n_nonzero)
            
            if verbose:
                print(f"W = {w_stat}, z = {z:.4f}, p = {p_val:.6f}")
                print(f"Effect size (r) = {r:.4f}")
            
            effect_size = r
            effect_size_type = "r"
        else:
            # Small N: exact test, no effect size
            if verbose:
                print(f"W = {w_stat}, p = {p_val:.6f} (exact test, small N)")
                print("Effect size not computed due to small sample size (N < 10).")
            
            effect_size = np.nan
            effect_size_type = "N/A"
        
        test_used = "Wilcoxon Signed-Rank (two-tailed)"
        df = None
        
    else:
        # No variance or insufficient data
        if verbose:
            if not has_variance:
                print("WARNING: No variance - test not applicable")
            else:
                print("WARNING: Insufficient data for statistical testing")
        
        test_used = "Test not applicable — no variance" if not has_variance else "N/A"
        test_statistic = np.nan
        p_val = np.nan
        effect_size = np.nan
        effect_size_type = "N/A"
        df = None
        w_stat = None
    
    # Decision (alpha = 0.05)
    if not has_variance:
        decision = "Fail to Reject H0"
    elif not np.isnan(p_val):
        decision = "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
    else:
        decision = "N/A"
    
    if verbose:
        print(f"Decision (alpha = 0.05): {decision}")
        print("-" * 60)
    
    return {
        'metric': metric_name,
        'n': n,
        'n_nonzero': n_nonzero,
        'transformer_mean': transformer_mean,
        'gru_mean': gru_mean,
        'difference': mean_diff,
        'normality_p': normality_p,
        'test_used': test_used,
        'test_statistic': test_statistic,
        'p_value': p_val,
        'effect_size': effect_size,
        'effect_size_type': effect_size_type,
        'direction': direction,
        'decision': decision,
        'df': df,
        'w_stat': w_stat
    }


def load_breakdown_csv(csv_path):
    """
    Load breakdown CSV file and handle multi-line headers.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the breakdown CSV file
        
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe with proper column names
    """
    # Read the CSV, handling potential multi-line headers
    df = pd.read_csv(csv_path)
    
    # Clean column names (remove newlines and extra spaces)
    df.columns = [col.replace('\n', ' ').strip() for col in df.columns]
    
    # Standardize column names
    column_mapping = {
        'Category ID': 'ID',
        'Gloss ID': 'ID',
        'Category Label': 'Label',
        'Gloss Label': 'Label',
        'Transformer\nPrecision': 'Transformer Precision',
        'Transformer\nRecall': 'Transformer Recall',
        'Transformer\nF1-score': 'Transformer F1-score',
        'IV3-GRU\nPrecision': 'IV3-GRU Precision',
        'IV3-GRU\nRecall': 'IV3-GRU Recall',
        'IV3-GRU\nF1-score': 'IV3-GRU F1-score',
    }
    
    df = df.rename(columns=column_mapping)
    
    return df


def process_breakdown_data(df, occlusion_type, metric):
    """
    Extract data for a specific occlusion type and metric from breakdown dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Breakdown dataframe
    occlusion_type : str
        'occluded' or 'nonoccluded'
    metric : str
        'Precision', 'Recall', or 'F1-score'
        
    Returns:
    --------
    tuple : (transformer_scores, gru_scores)
    """
    # Filter by occlusion type
    mask = df['Occlusion'].str.lower() == occlusion_type.lower()
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return np.array([]), np.array([])
    
    # Extract scores
    transformer_col = f'Transformer {metric}'
    gru_col = f'IV3-GRU {metric}'
    
    transformer_scores = filtered_df[transformer_col].values
    gru_scores = filtered_df[gru_col].values
    
    return transformer_scores, gru_scores


def process_task(task_name, breakdown_path, verbose=True):
    """
    Process all metrics for a given task (classification or recognition).
    
    Parameters:
    -----------
    task_name : str
        'classification' or 'recognition'
    breakdown_path : str or Path
        Path to the breakdown CSV file
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    pd.DataFrame : Results dataframe with all statistical tests (formatted)
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {task_name.upper()}")
        print("=" * 80)
    
    # Load breakdown data
    df = load_breakdown_csv(breakdown_path)
    
    # Determine hypothesis number based on task
    hypothesis_num = 2 if task_name.lower() == 'classification' else 1
    
    results = []
    metrics = ['Precision', 'Recall', 'F1-score']
    occlusion_types = ['occluded', 'nonoccluded']
    
    for occlusion_type in occlusion_types:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"OCCLUSION TYPE: {occlusion_type.upper()}")
            print(f"{'=' * 80}")
        
        for metric in metrics:
            metric_name = f"{occlusion_type.capitalize()} {metric}"
            
            # Extract data
            transformer_scores, gru_scores = process_breakdown_data(
                df, occlusion_type, metric
            )
            
            if len(transformer_scores) == 0:
                if verbose:
                    print(f"\nWARNING: No data found for {metric_name}")
                continue
            
            # Analyze (returns raw results)
            raw_result = analyze_metric(
                transformer_scores, 
                gru_scores, 
                metric_name,
                verbose=verbose
            )
            
            # Determine N for formatting (non-zero for Wilcoxon, total for t-test)
            if "Wilcoxon" in raw_result['test_used']:
                n_for_format = raw_result.get('n_nonzero', raw_result['n'])
            else:
                n_for_format = raw_result['n']
            
            # Format results using format_statistic_row
            formatted_result = format_statistic_row(
                metric=raw_result['metric'],
                n=n_for_format,
                transformer_mean=raw_result['transformer_mean'],
                gru_mean=raw_result['gru_mean'],
                diff=raw_result['difference'],
                normality_p=raw_result['normality_p'],
                test_used=raw_result['test_used'],
                test_stat_raw=raw_result['test_statistic'],
                p_value_raw=raw_result['p_value'],
                effect_val=raw_result['effect_size'],
                effect_type=raw_result['effect_size_type'],
                direction=raw_result['direction'],
                decision=raw_result['decision'],
                hypothesis_num=hypothesis_num
            )
            results.append(formatted_result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    return results_df


def main():
    """Main function to run the complete statistical analysis pipeline."""
    print("=" * 80)
    print("STATISTICAL ANALYSIS PIPELINE: Transformer vs IV3-GRU")
    print("=" * 80)
    
    # Define paths
    base_path = Path(__file__).parent
    classification_breakdown = base_path / 'classification' / 'Classification-Breakdown.csv'
    recognition_breakdown = base_path / 'recognition' / 'Recognition-Breakdown.csv'
    
    classification_output = base_path / 'classification' / 'Classification-StatsResults.csv'
    recognition_output = base_path / 'recognition' / 'Recognition-StatsResults.csv'
    
    # Process classification
    if classification_breakdown.exists():
        classification_results = process_task(
            'classification',
            classification_breakdown,
            verbose=True
        )
        
        # Save results (already formatted)
        classification_results.to_csv(classification_output, index=False)
        print(f"\n[OK] Classification results saved to: {classification_output}")
    else:
        print(f"\n[ERROR] Classification breakdown file not found: {classification_breakdown}")
    
    # Process recognition
    if recognition_breakdown.exists():
        recognition_results = process_task(
            'recognition',
            recognition_breakdown,
            verbose=True
        )
        
        # Save results (already formatted)
        recognition_results.to_csv(recognition_output, index=False)
        print(f"\n[OK] Recognition results saved to: {recognition_output}")
    else:
        print(f"\n[ERROR] Recognition breakdown file not found: {recognition_breakdown}")
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
