"""
Continuous sequence evaluation helpers.

This module provides the building blocks for computing TP/FP/FN breakdowns
and precision/recall/F1-style metrics for continuous sign-language recognition
sequences.  The implementation follows the comprehensive overhaul plan and is
structured to keep the public API clean, with well-documented helpers that can
be composed by offline validation scripts and interactive tooling.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration & Data Structures
# --------------------------------------------------------------------------- #


@dataclass
class ContinuousEvaluationConfig:
    """Holds threshold configuration used during continuous evaluation."""

    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5
    inactive_threshold: float = 0.9
    active_overlap_threshold: float = 0.1
    min_gap_duration_ms: int = 200
    enable_lenient_matching: bool = True


@dataclass
class Timestamp:
    """Represents a temporal span for a gloss prediction or ground truth."""

    start_ms: float
    end_ms: float
    gloss: Optional[int] = None

    def duration(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class SequencePrediction:
    """Model prediction information for a single sequence."""

    gloss_ids: List[int] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    timestamps: List[Timestamp] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.gloss_ids) != len(self.timestamps):
            raise ValueError("Prediction gloss_ids and timestamps lengths must match.")
        if self.confidence_scores and len(self.confidence_scores) != len(self.gloss_ids):
            raise ValueError("Prediction confidence_scores length must match gloss_ids.")


@dataclass
class SequenceGroundTruth:
    """Ground-truth annotation information for a single sequence."""

    gloss_ids: List[int] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    timestamps: List[Timestamp] = field(default_factory=list)
    occlusion_flags: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if len(self.gloss_ids) != len(self.timestamps):
            raise ValueError("Ground-truth gloss_ids and timestamps lengths must match.")
        if self.occlusion_flags and len(self.occlusion_flags) != len(self.gloss_ids):
            raise ValueError("Ground-truth occlusion_flags length must match gloss_ids.")


@dataclass
class SequenceEvaluationResult:
    """Holds per-sequence evaluation aggregates with detailed breakdowns."""

    num_tp: int
    num_fp: int
    num_fn: int
    precision: float
    recall: float
    f1_score: float
    mean_iou: float
    tp_breakdown: Dict[str, int] = field(default_factory=dict)
    fp_breakdown: Dict[str, int] = field(default_factory=dict)
    fn_breakdown: Dict[str, int] = field(default_factory=dict)
    matched_pairs: List[Dict[str, float]] = field(default_factory=list)
    tp_indices: List[int] = field(default_factory=list)
    fp_indices: List[int] = field(default_factory=list)
    fn_indices: List[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core Helpers
# --------------------------------------------------------------------------- #


def split_predictions_by_confidence(
    prediction: SequencePrediction,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """
    Split prediction indices into high- and low-confidence groups.

    Args:
        prediction: SequencePrediction instance containing gloss IDs and confidences.
        threshold: Confidence threshold; predictions >= threshold are treated as sign
                   hypotheses, while those below threshold are handled separately.

    Returns:
        Tuple of (high_conf_indices, low_conf_indices).
    """
    high_conf = []
    low_conf = []

    for idx, score in enumerate(prediction.confidence_scores or []):
        if score >= threshold:
            high_conf.append(idx)
        else:
            low_conf.append(idx)

    return high_conf, low_conf


def calculate_temporal_iou(pred: Timestamp, gt: Timestamp) -> float:
    """Return the temporal IoU between two timestamp intervals."""
    overlap_start = max(pred.start_ms, gt.start_ms)
    overlap_end = min(pred.end_ms, gt.end_ms)
    overlap = max(0.0, overlap_end - overlap_start)

    union_start = min(pred.start_ms, gt.start_ms)
    union_end = max(pred.end_ms, gt.end_ms)
    union = max(0.0, union_end - union_start)

    return overlap / union if union > 0 else 0.0


def match_predictions_to_ground_truth(
    pred_glosses: Sequence[int],
    pred_timestamps: Sequence[Timestamp],
    gt_glosses: Sequence[int],
    gt_timestamps: Sequence[Timestamp],
    iou_threshold: float,
) -> Tuple[List[int], List[int], List[int], List[Dict[str, float]], float]:
    """
    Greedy IoU-based matching between predictions and ground truth.

    Returns:
        tp_indices: prediction indices that matched a ground-truth instance.
        fp_indices: prediction indices not matched.
        fn_indices: ground-truth indices without a match.
        matched_pairs: metadata describing each matched (pred_idx, gt_idx, iou).
        mean_iou: mean IoU across matched pairs.
    """
    num_pred = len(pred_glosses)
    num_gt = len(gt_glosses)

    if num_pred == 0:
        return [], [], list(range(num_gt)), [], 0.0
    if num_gt == 0:
        return [], list(range(num_pred)), [], [], 0.0

    iou_matrix = np.zeros((num_pred, num_gt), dtype=float)

    for i, pred_ts in enumerate(pred_timestamps):
        for j, gt_ts in enumerate(gt_timestamps):
            if pred_glosses[i] != gt_glosses[j]:
                continue
            iou_matrix[i, j] = calculate_temporal_iou(pred_ts, gt_ts)

    candidates: List[Tuple[int, int, float]] = [
        (i, j, iou_matrix[i, j])
        for i in range(num_pred)
        for j in range(num_gt)
        if iou_matrix[i, j] >= iou_threshold
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)

    pred_matched = [False] * num_pred
    gt_matched = [False] * num_gt
    matched_pairs: List[Dict[str, float]] = []

    for pred_idx, gt_idx, iou in candidates:
        if pred_matched[pred_idx] or gt_matched[gt_idx]:
            continue
        matched_pairs.append(
            {"pred_idx": pred_idx, "gt_idx": gt_idx, "iou": float(iou)}
        )
        pred_matched[pred_idx] = True
        gt_matched[gt_idx] = True

    tp_indices = [pair["pred_idx"] for pair in matched_pairs]
    fp_indices = [i for i, matched in enumerate(pred_matched) if not matched]
    fn_indices = [j for j, matched in enumerate(gt_matched) if not matched]
    mean_iou = (
        float(np.mean([pair["iou"] for pair in matched_pairs])) if matched_pairs else 0.0
    )

    return tp_indices, fp_indices, fn_indices, matched_pairs, mean_iou


def compute_active_ratio(
    pred_timestamp: Timestamp,
    mask: Optional[np.ndarray],
    timestamps_ms: Optional[np.ndarray],
    left_hand_slice: slice = slice(25, 46),
    right_hand_slice: slice = slice(46, 67),
) -> float:
    """
    Calculate the ratio of frames within a timestamp that contain active hands.
    """
    if mask is None or timestamps_ms is None or len(mask) == 0:
        return 1.0

    frame_indices = np.where(
        (timestamps_ms >= pred_timestamp.start_ms)
        & (timestamps_ms <= pred_timestamp.end_ms)
    )[0]
    if len(frame_indices) == 0:
        return 0.0

    active_count = 0
    for frame_idx in frame_indices:
        frame = mask[frame_idx]
        left_active = bool(np.any(frame[left_hand_slice]))
        right_active = bool(np.any(frame[right_hand_slice]))
        if left_active or right_active:
            active_count += 1

    return active_count / len(frame_indices)


def check_active_overlap(
    pred_timestamp: Timestamp,
    mask: Optional[np.ndarray],
    timestamps_ms: Optional[np.ndarray],
    min_overlap_ratio: float,
    left_hand_slice: slice = slice(25, 46),
    right_hand_slice: slice = slice(46, 67),
) -> bool:
    """
    Determine whether a prediction overlaps sufficiently with active frames.

    Args:
        pred_timestamp: Timestamp describing the prediction span.
        mask: [T, K] boolean array where True indicates visible keypoint.
        timestamps_ms: Array of frame timestamps aligned with mask.
        min_overlap_ratio: Minimal ratio of active frames within prediction span.
        left_hand_slice/right_hand_slice: Slices pointing to left/right hand
            keypoints inside mask; defaults align with Mediapipe layout.
    """
    if mask is None or timestamps_ms is None or len(mask) == 0:
        return True  # Graceful fallback when activity data is unavailable.

    overlap_ratio = compute_active_ratio(
        pred_timestamp,
        mask,
        timestamps_ms,
        left_hand_slice=left_hand_slice,
        right_hand_slice=right_hand_slice,
    )
    return overlap_ratio >= min_overlap_ratio


def augment_matches_with_order(
    pred_glosses: Sequence[int],
    gt_glosses: Sequence[int],
    unmatched_pred_indices: Sequence[int],
    unmatched_gt_indices: Sequence[int],
) -> List[Dict[str, float]]:
    """
    Lenient order-preserving matching when IoU matching fails.

    Returns list of dicts with pred_idx and gt_idx.
    """
    augmented: List[Dict[str, float]] = []
    p_ptr = 0
    g_ptr = 0

    while p_ptr < len(unmatched_pred_indices) and g_ptr < len(unmatched_gt_indices):
        pred_idx = unmatched_pred_indices[p_ptr]
        gt_idx = unmatched_gt_indices[g_ptr]
        if pred_glosses[p_ptr] == gt_glosses[g_ptr]:
            augmented.append(
                {"pred_idx": int(pred_idx), "gt_idx": int(gt_idx), "iou": 0.0}
            )
            p_ptr += 1
            g_ptr += 1
        else:
            p_ptr += 1

    return augmented


# --------------------------------------------------------------------------- #
# Evaluation Pipeline
# --------------------------------------------------------------------------- #


def _compute_counts_to_metrics(
    num_tp: int, num_fp: int, num_fn: int
) -> Tuple[float, float, float]:
    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1_score


def compute_sequence_metrics(
    prediction: SequencePrediction,
    ground_truth: SequenceGroundTruth,
    *,
    config: ContinuousEvaluationConfig,
    mask: Optional[np.ndarray] = None,
    timestamps_ms: Optional[np.ndarray] = None,
) -> SequenceEvaluationResult:
    """
    Compute per-sequence metrics according to the comprehensive plan.

    This function handles high-level orchestration: splitting predictions by
    confidence, IoU matching, active-region validation, optional lenient
    matching, and deferred classification of FP/FN subcases (populated in
    later phases).
    """

    if not prediction.gloss_ids:
        prediction.confidence_scores = prediction.confidence_scores or []
    if not ground_truth.gloss_ids:
        ground_truth.occlusion_flags = (
            ground_truth.occlusion_flags or [0] * len(ground_truth.gloss_ids)
        )

    high_conf_indices, low_conf_indices = split_predictions_by_confidence(
        prediction, config.confidence_threshold
    )
    high_conf_glosses = [prediction.gloss_ids[i] for i in high_conf_indices]
    high_conf_timestamps = [prediction.timestamps[i] for i in high_conf_indices]

    (
        tp_indices_relative,
        fp_indices_relative,
        fn_indices,
        matched_pairs_relative,
        mean_iou,
    ) = match_predictions_to_ground_truth(
        pred_glosses=high_conf_glosses,
        pred_timestamps=high_conf_timestamps,
        gt_glosses=ground_truth.gloss_ids,
        gt_timestamps=ground_truth.timestamps,
        iou_threshold=config.iou_threshold,
    )

    rel_match_lookup = {
        pair["pred_idx"]: pair["gt_idx"] for pair in matched_pairs_relative
    }

    matched_pairs = [
        {
            "pred_idx": high_conf_indices[pair["pred_idx"]],
            "gt_idx": pair["gt_idx"],
            "iou": pair["iou"],
        }
        for pair in matched_pairs_relative
    ]

    validated_tp_indices: List[int] = []
    demoted_fp_indices: List[int] = []

    fn_indices_absolute = list(fn_indices)

    for rel_idx in tp_indices_relative:
        pred_idx = high_conf_indices[rel_idx]
        gt_idx = rel_match_lookup.get(rel_idx)
        if check_active_overlap(
            prediction.timestamps[pred_idx],
            mask,
            timestamps_ms,
            config.active_overlap_threshold,
        ):
            validated_tp_indices.append(pred_idx)
        else:
            demoted_fp_indices.append(pred_idx)
            if gt_idx is not None and gt_idx not in fn_indices_absolute:
                fn_indices_absolute.append(gt_idx)
            matched_pairs = [
                pair
                for pair in matched_pairs
                if not (pair["pred_idx"] == pred_idx and pair["gt_idx"] == gt_idx)
            ]

    tp_indices_absolute = validated_tp_indices
    fp_indices_absolute = [high_conf_indices[i] for i in fp_indices_relative]

    for idx in demoted_fp_indices:
        if idx not in fp_indices_absolute:
            fp_indices_absolute.append(idx)

    if config.enable_lenient_matching:
        matched_gt_indices = {pair["gt_idx"] for pair in matched_pairs}
        unmatched_preds = [
            idx
            for idx in high_conf_indices
            if idx not in tp_indices_absolute and idx not in fp_indices_absolute
        ]
        unmatched_gts = [
            i
            for i in range(len(ground_truth.gloss_ids))
            if i not in matched_gt_indices
        ]

        augmented_pairs = augment_matches_with_order(
            pred_glosses=[prediction.gloss_ids[i] for i in unmatched_preds],
            gt_glosses=[ground_truth.gloss_ids[j] for j in unmatched_gts],
            unmatched_pred_indices=unmatched_preds,
            unmatched_gt_indices=unmatched_gts,
        )

        for pair in augmented_pairs:
            pred_idx = int(pair["pred_idx"])
            gt_idx = int(pair["gt_idx"])
            if check_active_overlap(
                prediction.timestamps[pred_idx],
                mask,
                timestamps_ms,
                config.active_overlap_threshold,
            ):
                tp_indices_absolute.append(pred_idx)
                matched_pairs.append(
                    {"pred_idx": pred_idx, "gt_idx": gt_idx, "iou": pair["iou"]}
                )
                if pred_idx in fp_indices_absolute:
                    fp_indices_absolute.remove(pred_idx)
                if gt_idx in fn_indices_absolute:
                    fn_indices_absolute.remove(gt_idx)

    tp_indices_absolute = sorted(set(tp_indices_absolute))
    fp_indices_absolute = sorted(set(fp_indices_absolute))
    fn_indices_absolute = sorted(set(fn_indices_absolute))

    tp_breakdown = {"TP": len(tp_indices_absolute)}
    fp_breakdown = {"FP": len(fp_indices_absolute)}
    fn_breakdown = {"FN": len(fn_indices_absolute)}

    num_tp = len(tp_indices_absolute)
    num_fp = len(fp_indices_absolute)
    num_fn = len(fn_indices_absolute)

    precision, recall, f1_score = _compute_counts_to_metrics(
        num_tp, num_fp, num_fn
    )

    return SequenceEvaluationResult(
        num_tp=num_tp,
        num_fp=num_fp,
        num_fn=num_fn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        mean_iou=mean_iou,
        tp_breakdown=tp_breakdown,
        fp_breakdown=fp_breakdown,
        fn_breakdown=fn_breakdown,
        matched_pairs=matched_pairs,
        tp_indices=tp_indices_absolute,
        fp_indices=fp_indices_absolute,
        fn_indices=fn_indices_absolute,
    )


# --------------------------------------------------------------------------- #
# Aggregation Helpers
# --------------------------------------------------------------------------- #


def _extract_metric(source: Any, key: str, default: float = 0.0) -> float:
    if isinstance(source, Mapping):
        return float(source.get(key, default))
    return float(getattr(source, key, default))


def _extract_int(source: Any, key: str, default: int = 0) -> int:
    if isinstance(source, Mapping):
        return int(source.get(key, default))
    return int(getattr(source, key, default))


def compute_overall_metrics_micro(
    per_sequence_results: Sequence[Any],
) -> Dict[str, Any]:
    if not per_sequence_results:
        return {
            "aggregation_method": "micro",
            "total_sequences": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_gt_instances": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    total_tp = sum(_extract_int(r, "num_tp", 0) for r in per_sequence_results)
    total_fp = sum(_extract_int(r, "num_fp", 0) for r in per_sequence_results)
    total_fn = sum(_extract_int(r, "num_fn", 0) for r in per_sequence_results)
    total_gt = sum(_extract_int(r, "num_gt", 0) for r in per_sequence_results)

    precision, recall, f1_score = _compute_counts_to_metrics(
        total_tp, total_fp, total_fn
    )

    return {
        "aggregation_method": "micro",
        "total_sequences": len(per_sequence_results),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "total_gt_instances": int(total_gt),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }


def compute_overall_metrics_macro(
    per_sequence_results: Sequence[Any],
) -> Dict[str, Any]:
    if not per_sequence_results:
        return {
            "aggregation_method": "macro",
            "total_sequences": 0,
            "mean_precision": 0.0,
            "std_precision": 0.0,
            "mean_recall": 0.0,
            "std_recall": 0.0,
            "mean_f1_score": 0.0,
            "std_f1_score": 0.0,
            "median_f1_score": 0.0,
        }

    precisions = np.array(
        [_extract_metric(r, "precision", 0.0) for r in per_sequence_results]
    )
    recalls = np.array(
        [_extract_metric(r, "recall", 0.0) for r in per_sequence_results]
    )
    f1_scores = np.array(
        [_extract_metric(r, "f1_score", 0.0) for r in per_sequence_results]
    )
    return {
        "aggregation_method": "macro",
        "total_sequences": len(per_sequence_results),
        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_f1_score": float(np.mean(f1_scores)),
        "std_f1_score": float(np.std(f1_scores)),
        "median_f1_score": float(np.median(f1_scores)),
    }


def _group_by_key(
    per_sequence_results: Sequence[Any], key: str
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for result in per_sequence_results:
        value = None
        if isinstance(result, Mapping):
            value = result.get(key)
        else:
            value = getattr(result, key, None)
        if value in (None, ""):
            continue
        grouped[str(value)].append(result)
    return {
        group_key: compute_overall_metrics_micro(group_results)
        for group_key, group_results in grouped.items()
        if group_results
    }


def compute_stratified_metrics(
    per_sequence_results: Sequence[Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        "per_signer": _group_by_key(per_sequence_results, "signer"),
        "per_strategy": _group_by_key(per_sequence_results, "strategy"),
    }

