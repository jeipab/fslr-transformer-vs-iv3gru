"""Shared utilities for computing TP/FP/FN case maps for CTC visualizations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # Optional imports used for label enrichment
    from data.labels.label_mapping import load_label_mappings  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_label_mappings = None  # type: ignore


DEFAULT_CASE_PALETTE: Dict[str, str] = {
    "TP": "#22c55e",
    "FP": "#ef4444",
    "FN": "#f97316",
}


def _augment_matches_with_order_categories(
    pred_categories: Sequence[int],
    gt_categories: Sequence[int],
    pred_indices: Sequence[int],
    gt_indices: Sequence[int],
) -> List[Dict[str, int]]:
    """
    Order-preserving augmentation of matches based on category equality.

    Mirrors backend logic used during prediction to keep frontend in sync.
    """
    if not pred_indices or not gt_indices:
        return []

    additional_pairs: List[Dict[str, int]] = []
    p_i = 0
    g_i = 0
    while p_i < len(pred_indices) and g_i < len(gt_indices):
        pi = pred_indices[p_i]
        gi = gt_indices[g_i]
        if pi >= len(pred_categories) or gi >= len(gt_categories):
            break
        if pred_categories[pi] == gt_categories[gi]:
            additional_pairs.append({"pred_idx": pi, "gt_idx": gi})
            p_i += 1
            g_i += 1
        else:
            p_i += 1

    return additional_pairs


def derive_category_case_maps(
    metrics: Dict[str, Any],
    predicted_categories: Optional[Sequence[int]],
    ground_truth_categories: Optional[Sequence[int]],
    matched_pairs: Optional[List[Dict[str, Any]]],
    pred_len: int,
    gt_len: int,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Construct category TP/FP/FN maps for predicted and ground truth indexes.

    Prefer backend-provided indices; fall back to recomputing locally when absent.
    """
    pred_map: Dict[int, str] = {}
    gt_map: Dict[int, str] = {}
    matched_pairs = matched_pairs or []

    # Prefer explicit indices provided by backend (newer exports)
    if metrics.get("category_tp_pred_indices") is not None:
        for idx in metrics.get("category_tp_pred_indices") or []:
            idx = int(idx)
            if 0 <= idx < pred_len:
                pred_map[idx] = "TP"
    if metrics.get("category_fp_pred_indices") is not None:
        for idx in metrics.get("category_fp_pred_indices") or []:
            idx = int(idx)
            if 0 <= idx < pred_len:
                pred_map[idx] = "FP"
    if metrics.get("category_tp_gt_indices") is not None:
        for idx in metrics.get("category_tp_gt_indices") or []:
            idx = int(idx)
            if 0 <= idx < gt_len:
                gt_map[idx] = "TP"
    if metrics.get("category_fn_gt_indices") is not None:
        for idx in metrics.get("category_fn_gt_indices") or []:
            idx = int(idx)
            if 0 <= idx < gt_len:
                gt_map[idx] = "FN"

    # If any maps were populated, return early
    if pred_map or gt_map:
        return pred_map, gt_map

    # Fallback: recompute using available sequence information
    if not predicted_categories or not ground_truth_categories:
        return pred_map, gt_map

    pred_len = min(pred_len, len(predicted_categories))
    gt_len = min(gt_len, len(ground_truth_categories))

    matched_pred = set()
    matched_gt = set()

    for pair in matched_pairs:
        pi = pair.get("pred_idx")
        gi = pair.get("gt_idx")
        if pi is None or gi is None:
            continue
        pi = int(pi)
        gi = int(gi)
        if 0 <= pi < pred_len and 0 <= gi < gt_len:
            if predicted_categories[pi] == ground_truth_categories[gi]:
                matched_pred.add(pi)
                matched_gt.add(gi)

    remaining_pred = [i for i in range(pred_len) if i not in matched_pred]
    remaining_gt = [j for j in range(gt_len) if j not in matched_gt]

    additional_pairs = _augment_matches_with_order_categories(
        predicted_categories,
        ground_truth_categories,
        remaining_pred,
        remaining_gt,
    )

    for pair in additional_pairs:
        pi = pair["pred_idx"]
        gi = pair["gt_idx"]
        if 0 <= pi < pred_len:
            matched_pred.add(pi)
        if 0 <= gi < gt_len:
            matched_gt.add(gi)

    for idx in matched_pred:
        pred_map[idx] = "TP"
    for idx in matched_gt:
        gt_map[idx] = "TP"

    for idx in range(pred_len):
        if idx not in matched_pred:
            pred_map[idx] = "FP"

    for idx in range(gt_len):
        if idx not in matched_gt:
            gt_map[idx] = "FN"

    return pred_map, gt_map


def build_case_maps(
    metrics: Dict[str, Any],
    predicted_sequence: Optional[Sequence[int]] = None,
    ground_truth_sequence: Optional[Sequence[int]] = None,
    confidence_scores: Optional[Sequence[Optional[float]]] = None,
    predicted_categories: Optional[Sequence[int]] = None,
    ground_truth_categories: Optional[Sequence[int]] = None,
    category_confidences: Optional[Sequence[Optional[float]]] = None,
    confidence_threshold: Optional[float] = None,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Build gloss and category case maps (TP/FP/FN) shared by inference and validation views.
    """
    predicted_sequence = list(predicted_sequence or [])
    ground_truth_sequence = list(ground_truth_sequence or [])
    len_pred = len(predicted_sequence)
    len_gt = len(ground_truth_sequence)

    prediction_case_map: Dict[int, str] = {}
    ground_truth_case_map: Dict[int, str] = {}
    matched_pairs = metrics.get("matched_pairs") or []

    # Mark matched pairs as true positives
    for pair in matched_pairs:
        pred_idx = pair.get("pred_idx")
        gt_idx = pair.get("gt_idx")
        if pred_idx is None or gt_idx is None:
            continue
        pred_idx = int(pred_idx)
        gt_idx = int(gt_idx)
        if 0 <= pred_idx < len_pred and 0 <= gt_idx < len_gt:
            prediction_case_map[pred_idx] = "TP"
            ground_truth_case_map[gt_idx] = "TP"

    # Include explicit TP/FP/FN indices if provided
    for idx in metrics.get("tp_indices", []) or []:
        idx = int(idx)
        if 0 <= idx < len_pred:
            prediction_case_map.setdefault(idx, "TP")

    for idx in metrics.get("unmatched_predictions", []) or []:
        idx = int(idx)
        if 0 <= idx < len_pred:
            prediction_case_map[idx] = "FP"

    for idx in metrics.get("fp_indices", []) or []:
        idx = int(idx)
        if 0 <= idx < len_pred:
            prediction_case_map.setdefault(idx, "FP")

    for idx in metrics.get("unmatched_ground_truth", []) or []:
        idx = int(idx)
        if 0 <= idx < len_gt:
            ground_truth_case_map[idx] = "FN"

    for idx in metrics.get("fn_indices", []) or []:
        idx = int(idx)
        if 0 <= idx < len_gt:
            ground_truth_case_map.setdefault(idx, "FN")

    # Apply confidence-based fallback
    threshold = confidence_threshold
    if threshold is None:
        try:
            threshold = float(metrics.get("confidence_threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5

    for idx, conf in enumerate(confidence_scores or []):
        if conf is None:
            continue
        if 0 <= idx < len_pred and conf < threshold:
            prediction_case_map.setdefault(idx, "FP")

    # Derive category-level case maps
    category_prediction_case_map, category_ground_truth_case_map = derive_category_case_maps(
        metrics=metrics,
        predicted_categories=predicted_categories,
        ground_truth_categories=ground_truth_categories,
        matched_pairs=matched_pairs,
        pred_len=len_pred,
        gt_len=len_gt,
    )

    if (
        predicted_categories
        and category_confidences is not None
        and category_prediction_case_map is not None
    ):
        for idx, cat_conf in enumerate(category_confidences):
            if cat_conf is None:
                continue
            if 0 <= idx < len_pred and cat_conf < threshold:
                category_prediction_case_map.setdefault(idx, "FP")

    return (
        prediction_case_map,
        ground_truth_case_map,
        category_prediction_case_map,
        category_ground_truth_case_map,
    )


def build_case_maps_for_inference(
    metrics: Dict[str, Any],
    predicted_sequence: Optional[Sequence[int]] = None,
    ground_truth_sequence: Optional[Sequence[int]] = None,
    confidence_scores: Optional[Sequence[Optional[float]]] = None,
    predicted_categories: Optional[Sequence[int]] = None,
    ground_truth_categories: Optional[Sequence[int]] = None,
    category_confidences: Optional[Sequence[Optional[float]]] = None,
    confidence_threshold: Optional[float] = None,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Build case maps for INFERENCE UI using explicit indices only (never mixed with matched_pairs).
    This matches the logic that makes category work correctly.
    
    Validation continues using build_case_maps() which works perfectly.
    """
    predicted_sequence = list(predicted_sequence or [])
    ground_truth_sequence = list(ground_truth_sequence or [])
    len_pred = len(predicted_sequence)
    len_gt = len(ground_truth_sequence)

    prediction_case_map: Dict[int, str] = {}
    ground_truth_case_map: Dict[int, str] = {}

    # Use explicit indices ONLY (like category does) - never touch matched_pairs
    # This is the authoritative source from backend validation

    # Mirror validation outputs strictly: use only explicit tp/fp/fn indices
    # Note: backend returns FPs/FNs under "unmatched_*" keys in results; include them.
    tp_pred_indices = [int(i) for i in (metrics.get("tp_indices") or [])]
    fp_pred_indices = [
        int(i)
        for i in (
            (metrics.get("fp_indices") or [])
            + (metrics.get("unmatched_predictions") or [])
        )
    ]
    fn_gt_indices = [
        int(i)
        for i in (
            (metrics.get("fn_indices") or [])
            + (metrics.get("unmatched_ground_truth") or [])
        )
    ]

    # Predicted: assign TP/FP from explicit lists only
    for idx in tp_pred_indices:
        if 0 <= idx < len_pred:
            prediction_case_map[idx] = "TP"
    for idx in fp_pred_indices:
        if 0 <= idx < len_pred:
            prediction_case_map[idx] = "FP"

    # Ground truth: assign FN from explicit list, all other GT entries are TP
    fn_gt_set = set(i for i in fn_gt_indices if 0 <= i < len_gt)
    for idx in fn_gt_set:
        ground_truth_case_map[idx] = "FN"
    for idx in range(len_gt):
        if idx not in ground_truth_case_map:
            ground_truth_case_map[idx] = "TP"

    # Do not override with confidence threshold here; validation has already applied it

    # Derive category case maps using existing logic
    category_prediction_case_map, category_ground_truth_case_map = derive_category_case_maps(
        metrics=metrics,
        predicted_categories=predicted_categories,
        ground_truth_categories=ground_truth_categories,
        matched_pairs=metrics.get("matched_pairs", []),
        pred_len=len_pred,
        gt_len=len_gt,
    )

    # Do not apply category confidence threshold; mirror validation-provided indices

    return (
        prediction_case_map,
        ground_truth_case_map,
        category_prediction_case_map,
        category_ground_truth_case_map,
    )


def enrich_ground_truth_timestamps(
    timestamps: Optional[List[Dict[str, Any]]],
    gloss_labels: Optional[Sequence[str]],
    gloss_sequence: Sequence[int],
    category_ids: Optional[Sequence[int]],
    category_labels: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    """
    Ensure ground-truth timestamps include gloss/category metadata for visualization.
    Mirrors the augmentation used in validation.
    """
    if not timestamps:
        return []

    gloss_labels = list(gloss_labels or [])
    category_ids = list(category_ids or [])
    category_labels = list(category_labels or [])

    if load_label_mappings:
        try:
            gloss_mapping, category_mapping = load_label_mappings()
        except Exception:  # pragma: no cover - fallback gracefully
            gloss_mapping = {}
            category_mapping = {}
    else:  # pragma: no cover - when mappings are unavailable
        gloss_mapping = {}
        category_mapping = {}

    enriched: List[Dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        ts_copy = dict(ts)

        gloss_id = ts_copy.get("gloss")
        if gloss_id is None and idx < len(gloss_sequence):
            gloss_id = gloss_sequence[idx]
            ts_copy["gloss"] = gloss_id

        gloss_label = ts_copy.get("gloss_label")
        if not gloss_label:
            if idx < len(gloss_labels) and gloss_labels[idx]:
                gloss_label = gloss_labels[idx]
            elif gloss_id is not None:
                gloss_label = gloss_mapping.get(int(gloss_id), str(gloss_id))
            else:
                gloss_label = ""
            ts_copy["gloss_label"] = gloss_label

        cat_id = ts_copy.get("category")
        if cat_id is None and idx < len(category_ids):
            cat_id = category_ids[idx]
            ts_copy["category"] = cat_id

        cat_label = ts_copy.get("category_label")
        if not cat_label:
            if idx < len(category_labels) and category_labels[idx]:
                cat_label = category_labels[idx]
            elif cat_id is not None:
                cat_label = category_mapping.get(int(cat_id), str(cat_id))
            else:
                cat_label = ""
            ts_copy["category_label"] = cat_label

        enriched.append(ts_copy)

    return enriched

