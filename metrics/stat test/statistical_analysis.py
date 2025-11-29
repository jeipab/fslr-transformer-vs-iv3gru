"""
Statistical Analysis Pipeline for Transformer vs IV3-GRU Comparison

This script performs comprehensive statistical testing comparing Transformer and IV3-GRU
models for Filipino Sign Language recognition and classification tasks.

USAGE:
    python statistical_analysis.py

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
    - n: Sample size
    - transformer_mean: Mean Transformer score
    - gru_mean: Mean IV3-GRU score
    - difference: Mean difference (Transformer - IV3-GRU)
    - normality_p: Shapiro-Wilk p-value
    - test_used: Statistical test applied
    - test_statistic: Test statistic value
    - p_value: P-value from statistical test
    - effect_size: Effect size (Cohen's d or r)
    - effect_size_type: Type of effect size
    - direction: Direction of difference
    - decision: Statistical decision (Reject H0 / Fail to Reject H0)

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

warnings.filterwarnings('ignore')


def effect_size_r_from_z(z, n):
    """Calculate effect size r from z-value for Wilcoxon test (N > 20)."""
    return abs(z) / np.sqrt(n)


def effect_size_cohens_d(diff):
    """Calculate Cohen's d (paired) for parametric tests."""
    if diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


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
    dict : Dictionary containing all statistical test results
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
            'transformer_mean': np.nan,
            'gru_mean': np.nan,
            'difference': np.nan,
            'normality_p': np.nan,
            'test_used': 'N/A',
            'test_statistic': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'direction': 'N/A',
            'decision': 'N/A'
        }

    # Compute paired differences
    diff = transformer - gru
    diff_nonzero = diff[diff != 0]  # For Wilcoxon

    n = len(diff)
    n_nonzero = len(diff_nonzero)
    
    transformer_mean = transformer.mean()
    gru_mean = gru.mean()
    mean_diff = diff.mean()
    
    if verbose:
        print(f"Number of samples: {n}")
        print(f"Non-zero pairs (Wilcoxon N): {n_nonzero}")
        print(f"Transformer mean: {transformer_mean:.4f}")
        print(f"IV3-GRU mean: {gru_mean:.4f}")
        print(f"Mean difference: {mean_diff:.4f}")

    # 1. Shapiro-Wilk Normality Test
    if n < 3:
        if verbose:
            print("WARNING: Sample size too small for normality test (N < 3)")
        normality_p = np.nan
        normal = False
    else:
        sh_stat, normality_p = shapiro(diff)
        normal = normality_p > 0.05
        if verbose:
            print(f"Shapiro-Wilk: W = {sh_stat:.4f}, p = {normality_p:.6f}")
    
    # Determine direction
    if mean_diff > 0:
        direction = "Transformer > IV3-GRU"
    elif mean_diff < 0:
        direction = "IV3-GRU > Transformer"
    else:
        direction = "Equal"
    
    # 2. Choose and perform appropriate test
    if normal and n >= 2:
        # Check if all differences are zero
        if np.all(diff == 0):
            if verbose:
                print("-> Normal distribution -> Paired t-test")
                print("WARNING: All differences are zero. Cannot perform t-test.")
            
            test_used = "Paired t-test (N/A - no variance)"
            test_statistic = np.nan
            p_val = np.nan
            effect_size = np.nan
            effect_size_type = "N/A"
        else:
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
            
            test_used = "Paired t-test"
            test_statistic = t_stat
            effect_size = cohens_d
            effect_size_type = "Cohen's d"
        
    elif n_nonzero >= 2:
        # Wilcoxon Signed-Rank Test
        if verbose:
            print("-> Non-normal -> Wilcoxon Signed-Rank Test")
        
        # Use exact test for small N, approximate for larger N
        w_stat, p_val = wilcoxon(transformer, gru, zero_method='wilcox', alternative='two-sided')
        
        # Calculate effect size r from z-value (only valid for N > 20)
        if n_nonzero > 20:
            # Approximate z from W statistic
            mean_w = n_nonzero * (n_nonzero + 1) / 4
            sd_w = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)
            z = (w_stat - mean_w) / sd_w
            r = effect_size_r_from_z(z, n_nonzero)
            
            if verbose:
                print(f"W = {w_stat}, approx z = {z:.4f}, p = {p_val:.6f}")
                print(f"Effect size (r) = {r:.4f}")
            
            effect_size = r
            effect_size_type = "r (from z)"
        elif n_nonzero >= 10:
            # Can compute approximate z for N >= 10
            mean_w = n_nonzero * (n_nonzero + 1) / 4
            sd_w = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)
            z = (w_stat - mean_w) / sd_w
            r = effect_size_r_from_z(z, n_nonzero)
            
            if verbose:
                print(f"W = {w_stat}, approx z = {z:.4f}, p = {p_val:.6f}")
                print(f"Effect size (r) = {r:.4f}")
            
            effect_size = r
            effect_size_type = "r (from z)"
        else:
            # Small N: exact test, no effect size
            if verbose:
                print(f"W = {w_stat}, p = {p_val:.6f} (exact test, small N)")
                print("Effect size not computed due to small sample size (N < 10).")
            
            effect_size = np.nan
            effect_size_type = "N/A"
        
        test_used = "Wilcoxon Signed-Rank"
        test_statistic = w_stat
        
    else:
        # Insufficient data
        if verbose:
            print("WARNING: Insufficient data for statistical testing")
        
        test_used = "N/A"
        test_statistic = np.nan
        p_val = np.nan
        effect_size = np.nan
        effect_size_type = "N/A"
    
    # Decision (alpha = 0.05)
    if not np.isnan(p_val):
        decision = "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
    else:
        decision = "N/A"
    
    if verbose:
        print(f"Decision (alpha = 0.05): {decision}")
        print("-" * 60)
    
    return {
        'metric': metric_name,
        'n': n,
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
        'decision': decision
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


def format_results_df(df):
    """
    Format results dataframe for clean CSV output.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
        
    Returns:
    --------
    pd.DataFrame : Formatted dataframe
    """
    df = df.copy()
    
    # Round numeric columns to 6 decimal places
    numeric_cols = ['transformer_mean', 'gru_mean', 'difference', 'normality_p', 
                    'test_statistic', 'p_value', 'effect_size']
    
    for col in numeric_cols:
        if col in df.columns:
            # Round, but keep as float (not string) for proper CSV formatting
            df[col] = df[col].round(6)
    
    return df


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
    pd.DataFrame : Results dataframe with all statistical tests
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {task_name.upper()}")
        print("=" * 80)
    
    # Load breakdown data
    df = load_breakdown_csv(breakdown_path)
    
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
            
            # Analyze
            result = analyze_metric(
                transformer_scores, 
                gru_scores, 
                metric_name,
                verbose=verbose
            )
            
            results.append(result)
    
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
        
        # Format and save results
        classification_results_formatted = format_results_df(classification_results)
        classification_results_formatted.to_csv(classification_output, index=False)
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
        
        # Format and save results
        recognition_results_formatted = format_results_df(recognition_results)
        recognition_results_formatted.to_csv(recognition_output, index=False)
        print(f"\n[OK] Recognition results saved to: {recognition_output}")
    else:
        print(f"\n[ERROR] Recognition breakdown file not found: {recognition_breakdown}")
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
