"""
Utilities for continuous evaluation metrics.

Provides helpers for computing per-sequence TP/FP/FN breakdowns
and aggregate statistics.
"""

from .continuous import (  # noqa: F401
    ContinuousEvaluationConfig,
    SequencePrediction,
    SequenceGroundTruth,
    SequenceEvaluationResult,
    Timestamp,
    compute_overall_metrics_micro,
    compute_overall_metrics_macro,
    compute_stratified_metrics,
    split_predictions_by_confidence,
    match_predictions_to_ground_truth,
    check_active_overlap,
    augment_matches_with_order,
    compute_sequence_metrics,
)

__all__ = [
    "ContinuousEvaluationConfig",
    "SequencePrediction",
    "SequenceGroundTruth",
    "SequenceEvaluationResult",
    "Timestamp",
    "compute_overall_metrics_micro",
    "compute_overall_metrics_macro",
    "compute_stratified_metrics",
    "split_predictions_by_confidence",
    "match_predictions_to_ground_truth",
    "check_active_overlap",
    "augment_matches_with_order",
    "compute_sequence_metrics",
]

