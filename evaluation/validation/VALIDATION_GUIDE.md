# Continuous Validation Guide

This guide explains how to run the continuous validation pipeline and interpret the confusion-matrix metrics it produces. Content unrelated to the continuous workflow has been removed.

## Running Continuous Validation

```bash
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --data-dir data/processed/fsl_val \
    --labels-csv data/processed/fsl_val.csv
```

The command computes streaming predictions, aligns them with ground-truth activity windows, and saves confusion statistics to `confusion_matrices.json`.

## Confusion Matrix Enhancements

The continuous validator produces a richer confusion matrix than the legacy batch reports:

- Stores TP, FP, TN, FN counts per gloss and per category.
- Persists the raw counts alongside derived metrics so downstream tools can recompute thresholds.
- Tracks the decision windows used for each count, enabling timeline cross-checks in the dashboard.

# Enhanced Confusion Matrix

- **TP (True Positive)**: High-confidence gloss matches that overlap ground truth while the signer is active (hands visible)
- **FP (False Positive)**: High-confidence glosses that mismatch ground truth or occur during inactive periods (hands hidden)
- **TN (True Negative)**: Low-confidence outputs that end inside inactive periods with no overlapping ground truth (model abstained correctly)
- **FN (False Negative)**: Low-confidence outputs that fall inside active regions or overlap the ground truth (missed sign during activity)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 _ (Precision _ Recall) / (Precision + Recall)

Use these metrics to monitor how well continuous decoding balances correctness and restraint. Spot-check the raw timeline outputs alongside the confusion matrix when diagnosing regressions.
