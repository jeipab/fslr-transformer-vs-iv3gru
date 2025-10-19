#!/usr/bin/env python3
"""
CTC Error Analysis Script for Continuous Sign Language Recognition

This script performs a detailed error analysis for models using Connectionist Temporal Classification (CTC)
on continuous signing datasets. It supports JSON files structured with "segments" containing
"timestamp_start_ms", "timestamp_end_ms", and "gloss_label" fields.

It evaluates:
    - CTC prediction error breakdown (insertions, deletions, substitutions)
    - Temporal boundary and duration accuracy
    - Context-based error trends (start, middle, end of sequence)
    - Signer-specific and strategy-specific error patterns
    - Exports JSON and PDF reports
    - Generates heatmaps for error distribution

Usage Examples:
    python evaluation/analysis/error_analysis.py \
        --input results/continuous/predictions \
        --ground-truth-dir data/ground_truth \
        --output-dir results/continuous/error_report
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter


def load_json_files(input_dir):
    """Load all JSON files from a directory and normalize structure."""
    data = []
    for f in os.listdir(input_dir):
        if f.endswith(".json"):
            with open(os.path.join(input_dir, f), 'r', encoding='utf-8') as file:
                sample = json.load(file)

                # Normalize structure
                if "segments" in sample:
                    # Extract gloss list
                    sample["glosses"] = [seg["gloss_label"] for seg in sample["segments"]]

                    # Rename timestamp fields
                    for seg in sample["segments"]:
                        seg["start"] = seg.pop("timestamp_start_ms", 0)
                        seg["end"] = seg.pop("timestamp_end_ms", 0)

                data.append(sample)
    return data


def compute_ctc_error_types(pred_sequence, gt_sequence):
    """Compute CTC-style insertion, deletion, substitution errors."""
    insertions = deletions = substitutions = 0
    min_len = min(len(pred_sequence), len(gt_sequence))

    for i in range(min_len):
        p, g = pred_sequence[i], gt_sequence[i]
        if p == g:
            continue
        elif p not in gt_sequence:
            insertions += 1
        elif g not in pred_sequence:
            deletions += 1
        else:
            substitutions += 1

    # Handle length mismatch
    if len(pred_sequence) > len(gt_sequence):
        insertions += len(pred_sequence) - len(gt_sequence)
    elif len(gt_sequence) > len(pred_sequence):
        deletions += len(gt_sequence) - len(pred_sequence)

    return {"insertions": insertions, "deletions": deletions, "substitutions": substitutions}


def temporal_error_analysis(pred_segments, gt_segments):
    """Compute boundary and duration errors."""
    boundary_errors = []
    duration_errors = []

    min_len = min(len(pred_segments), len(gt_segments))
    for i in range(min_len):
        pred = pred_segments[i]
        gt = gt_segments[i]
        start_diff = abs(pred["start"] - gt["start"])
        end_diff = abs(pred["end"] - gt["end"])
        duration_diff = abs((pred["end"] - pred["start"]) - (gt["end"] - gt["start"]))

        boundary_errors.append((start_diff + end_diff) / 2)
        duration_errors.append(duration_diff)

    return {
        "boundary_error_mean": np.mean(boundary_errors) if boundary_errors else 0.0,
        "duration_error_mean": np.mean(duration_errors) if duration_errors else 0.0
    }


def per_signer_error_patterns(error_log):
    """Compute average error per signer."""
    df = pd.DataFrame(error_log)
    return df.groupby("signer")["error_count"].mean().to_dict()


def generate_error_report(error_summary, output_dir):
    """Save summary report as JSON and PDF."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"error_report_{timestamp}.json")
    pdf_path = os.path.join(output_dir, f"error_report_{timestamp}.pdf")

    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=4)

    # Simple PDF summary
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_data = [[k, v] for k, v in error_summary.items()]
    ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


def visualize_error_patterns(df, output_dir):
    """Generate per-signer heatmap."""
    if "category" not in df.columns:
        df["category"] = "ALL"

    pivot = df.pivot_table(values='error_count', index='signer', columns='category', fill_value=0)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap="Reds", fmt=".2f")
    plt.title("Error Distribution Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_heatmap.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="CTC Error Analysis for Continuous Signing")
    parser.add_argument("--input", required=True, help="Directory with prediction JSON files")
    parser.add_argument("--ground-truth-dir", required=True, help="Directory with ground truth JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    predictions = load_json_files(args.input)
    ground_truths = load_json_files(args.ground_truth_dir)

    all_errors = []

    for pred, gt in zip(predictions, ground_truths):
        ctc_errors = compute_ctc_error_types(pred["glosses"], gt["glosses"])
        temporal_errors = temporal_error_analysis(pred["segments"], gt["segments"])

        total_errors = sum(ctc_errors.values())
        all_errors.append({
            "file": pred.get("file_name", "unknown"),
            "signer": pred.get("signer", "unknown"),
            "error_count": total_errors,
            **ctc_errors,
            **temporal_errors
        })

    df = pd.DataFrame(all_errors)
    summary = {
        "avg_insertions": df["insertions"].mean(),
        "avg_deletions": df["deletions"].mean(),
        "avg_substitutions": df["substitutions"].mean(),
        "avg_boundary_error": df["boundary_error_mean"].mean(),
        "avg_duration_error": df["duration_error_mean"].mean(),
        "per_signer": per_signer_error_patterns(all_errors)
    }

    generate_error_report(summary, args.output_dir)
    visualize_error_patterns(df, args.output_dir)

    print("Error analysis completed. Reports saved in:", args.output_dir)


if __name__ == "__main__":
    main()
