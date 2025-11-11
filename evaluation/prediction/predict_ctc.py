"""
CTC Prediction for Continuous Sign Language Recognition

Predicts gloss sequences from continuous sign language videos using CTC models.
Supports batch prediction, ground truth comparison, and comprehensive metrics.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models import SignTransformerCtc, MediaPipeGRUCtc, InceptionV3GRUCtc
from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder
from evaluation.metrics import (
    ContinuousEvaluationConfig,
    SequenceGroundTruth,
    SequencePrediction,
    Timestamp as MetricTimestamp,
    TPCase,
    FPCase,
    FNCase,
    TNCase,
    compute_sequence_metrics,
    compute_overall_metrics_micro,
    compute_overall_metrics_macro,
    compute_stratified_metrics,
)
from streamlit_app.core.config import CTC_CONFIG, CTC_CONFIG_SUBSET, MODEL_CONFIG
from data.labels.label_mapping import load_label_mappings


def load_ground_truth_json(json_path: Path) -> Dict:
    """Load ground truth from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def estimate_timestamps(predicted_sequence: List[int], total_frames: int, fps: int = 30) -> List[Dict]:
    """Estimate timestamps for predicted glosses using frame-based distribution."""
    if len(predicted_sequence) == 0:
        return []
    
    timestamps = []
    frames_per_gloss = total_frames / len(predicted_sequence)
    
    for idx, gloss in enumerate(predicted_sequence):
        start_frame = int(idx * frames_per_gloss)
        end_frame = int((idx + 1) * frames_per_gloss)
        
        start_ms = int((start_frame / fps) * 1000)
        end_ms = int((end_frame / fps) * 1000)
        
        timestamps.append({
            'index': idx,
            'gloss': gloss,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'duration_ms': end_ms - start_ms
        })
    
    return timestamps


def calculate_temporal_alignment_accuracy(
    pred_timestamps: List[Dict],
    gt_timestamps: List[Dict],
    tolerance_ms: int = 500
) -> float:
    """
    Calculate temporal alignment accuracy.
    
    Matches predicted glosses with ground truth glosses (same gloss ID) and checks
    if both start and end times are within tolerance.
    """
    if len(gt_timestamps) == 0:
        return 0.0
    
    aligned = 0
    for gt_ts in gt_timestamps:
        for pred_ts in pred_timestamps:
            if (gt_ts['gloss'] == pred_ts['gloss'] and
                abs(gt_ts['start_ms'] - pred_ts['start_ms']) <= tolerance_ms and
                abs(gt_ts['end_ms'] - pred_ts['end_ms']) <= tolerance_ms):
                aligned += 1
                break
    
    return aligned / len(gt_timestamps)


def calculate_temporal_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Calculate temporal IoU (Intersection over Union) between two time intervals.
    
    Args:
        pred_start: Prediction start time (ms)
        pred_end: Prediction end time (ms)
        gt_start: Ground truth start time (ms)
        gt_end: Ground truth end time (ms)
    
    Returns:
        IoU value between 0.0 and 1.0
    """
    # Calculate intersection
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    overlap_duration = max(0.0, overlap_end - overlap_start)
    
    # Calculate union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union_duration = union_end - union_start
    
    if union_duration == 0:
        return 0.0
    
    return overlap_duration / union_duration


def match_predictions_to_ground_truth(
    pred_glosses: List[int],
    pred_timestamps: List[Dict],
    gt_glosses: List[int],
    gt_timestamps: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[int], List[int], List[int], List[Dict], float]:
    """
    Match predictions to ground truth using temporal IoU.
    
    Args:
        pred_glosses: List of predicted gloss IDs
        pred_timestamps: List of timestamp dicts with 'start_ms' and 'end_ms'
        gt_glosses: List of ground truth gloss IDs
        gt_timestamps: List of timestamp dicts with 'start_ms' and 'end_ms'
        iou_threshold: Minimum IoU required for a match (default: 0.5)
    
    Returns:
        Tuple of:
        - tp_indices: Indices of true positive predictions (prediction index)
        - fp_indices: Indices of false positive predictions (prediction index)
        - fn_indices: Indices of false negative ground truths (ground truth index)
        - matched_pairs: List of match info dicts with keys: pred_idx, gt_idx, iou, gloss
        - mean_iou: Average IoU for all TP matches
    """
    if len(pred_timestamps) != len(pred_glosses):
        raise ValueError(f"Mismatch: {len(pred_timestamps)} timestamps for {len(pred_glosses)} predictions")
    if len(gt_timestamps) != len(gt_glosses):
        raise ValueError(f"Mismatch: {len(gt_timestamps)} timestamps for {len(gt_glosses)} ground truth")
    
    num_pred = len(pred_glosses)
    num_gt = len(gt_glosses)
    
    # Edge cases
    if num_pred == 0:
        return [], [], list(range(num_gt)), [], 0.0
    if num_gt == 0:
        return [], list(range(num_pred)), [], [], 0.0
    
    # Calculate IoU matrix (num_pred x num_gt)
    iou_matrix = np.zeros((num_pred, num_gt))
    for i, pred_ts in enumerate(pred_timestamps):
        for j, gt_ts in enumerate(gt_timestamps):
            iou = calculate_temporal_iou(
                pred_ts['start_ms'], pred_ts['end_ms'],
                gt_ts['start_ms'], gt_ts['end_ms']
            )
            iou_matrix[i, j] = iou
    
    # Match predictions to ground truth (one-to-one matching)
    # Strategy: Greedy matching - match highest IoU first, but only if:
    #   1. IoU >= threshold
    #   2. Gloss IDs match
    #   3. Not already matched
    
    matched_pairs = []
    pred_matched = [False] * num_pred
    gt_matched = [False] * num_gt
    
    # Create list of candidate matches (pred_idx, gt_idx, iou)
    candidates = []
    for i in range(num_pred):
        for j in range(num_gt):
            if pred_glosses[i] == gt_glosses[j] and iou_matrix[i, j] >= iou_threshold:
                candidates.append((i, j, iou_matrix[i, j]))
    
    # Sort by IoU (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching
    for pred_idx, gt_idx, iou in candidates:
        if not pred_matched[pred_idx] and not gt_matched[gt_idx]:
            matched_pairs.append({
                'pred_idx': pred_idx,
                'gt_idx': gt_idx,
                'iou': float(iou),
                'gloss': pred_glosses[pred_idx]
            })
            pred_matched[pred_idx] = True
            gt_matched[gt_idx] = True
    
    # Classify predictions and ground truth
    tp_indices = [pair['pred_idx'] for pair in matched_pairs]
    fp_indices = [i for i in range(num_pred) if not pred_matched[i]]
    fn_indices = [j for j in range(num_gt) if not gt_matched[j]]
    
    # Calculate mean IoU for TP matches
    mean_iou = float(np.mean([pair['iou'] for pair in matched_pairs])) if matched_pairs else 0.0
    
    return tp_indices, fp_indices, fn_indices, matched_pairs, mean_iou


def _augment_matches_with_order(
    pred_glosses: List[int],
    gt_glosses: List[int],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
) -> List[Dict]:
    """
    Lenient, order-preserving greedy matcher to augment strict temporal matches.
    Pairs remaining unmatched predictions and ground-truth by label equality
    while preserving sequence order. Does not use timestamps.
    """
    if not unmatched_pred or not unmatched_gt:
        return []

    add_pairs: List[Dict] = []
    p_i = 0
    g_i = 0
    while p_i < len(unmatched_pred) and g_i < len(unmatched_gt):
        pi = unmatched_pred[p_i]
        gi = unmatched_gt[g_i]
        if pred_glosses[pi] == gt_glosses[gi]:
            add_pairs.append({'pred_idx': pi, 'gt_idx': gi, 'iou': 0.0, 'gloss': pred_glosses[pi]})
            p_i += 1
            g_i += 1
        else:
            # Advance predictions pointer first to be more permissive on extra predictions
            p_i += 1
    return add_pairs


def _compute_category_metrics_balanced(
    pred_categories: List[int],
    gt_categories: List[int],
    matched_pairs: List[Dict],
    pred_len: int,
    gt_len: int,
    gloss_fp_indices: List[int],
    gloss_fn_indices: List[int],
) -> Tuple[int, int, int]:
    """
    Compute category TP/FP/FN by:
    1) Using gloss matched pairs where categories also match
    2) Then augmenting with order-preserving matches over the remaining
       unmatched predictions and GT purely by category equality
    """
    if not pred_categories or not gt_categories:
        return 0, 0, len(gt_categories)

    # Bound checks
    pred_len = min(pred_len, len(pred_categories))
    gt_len = min(gt_len, len(gt_categories))

    pred_matched = set()
    gt_matched = set()
    cat_tp = 0

    # Step 1: accept gloss matches where categories also match
    for pair in matched_pairs:
        pi = pair.get('pred_idx')
        gi = pair.get('gt_idx')
        if pi is None or gi is None:
            continue
        if 0 <= pi < pred_len and 0 <= gi < gt_len:
            if pred_categories[pi] == gt_categories[gi]:
                cat_tp += 1
                pred_matched.add(pi)
                gt_matched.add(gi)

    # Step 2: order-preserving augmentation by category on remaining indices
    remaining_pred = [i for i in range(pred_len) if i not in pred_matched]
    remaining_gt = [j for j in range(gt_len) if j not in gt_matched]

    # Build additional matches by treating categories as sequences
    add_pairs = _augment_matches_with_order(
        pred_categories,
        gt_categories,
        remaining_pred,
        remaining_gt,
    )
    cat_tp += len(add_pairs)
    pred_matched.update(p['pred_idx'] for p in add_pairs)
    gt_matched.update(p['gt_idx'] for p in add_pairs)

    # Remaining unmatched are FP/FN for category
    cat_fp = len([i for i in range(pred_len) if i not in pred_matched])
    cat_fn = len([j for j in range(gt_len) if j not in gt_matched])

    return cat_tp, cat_fp, cat_fn


def _max_iou_with_gt(pred_ts: Dict, gt_ts_list: List[Dict]) -> Tuple[float, int]:
    """Return (max_iou, gt_idx) for a prediction against all GT timestamps."""
    best_iou = 0.0
    best_idx = -1
    for j, gt_ts in enumerate(gt_ts_list):
        iou = calculate_temporal_iou(
            pred_ts['start_ms'], pred_ts['end_ms'], gt_ts['start_ms'], gt_ts['end_ms']
        )
        if iou > best_iou:
            best_iou = iou
            best_idx = j
    return best_iou, best_idx


def _compute_occlusion_split_metrics(
    pred_timestamps: List[Dict],
    gt_timestamps: List[Dict],
    gt_occluded: Optional[List[int]],
    matched_pairs: List[Dict],
    fp_indices: List[int],
    fn_indices: List[int]
) -> Dict:
    """
    Compute TP/FP/FN split by occlusion (without=0, with=1).
    FP assignment uses max IoU to nearest GT; FPs with no overlap are excluded.
    """
    if not gt_occluded or not gt_timestamps:
        return {
            'without_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
            'with_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
        }

    # TPs by matched pair GT occlusion
    tp_without = sum(1 for p in matched_pairs if 0 <= p['gt_idx'] < len(gt_occluded) and gt_occluded[p['gt_idx']] == 0)
    tp_with = sum(1 for p in matched_pairs if 0 <= p['gt_idx'] < len(gt_occluded) and gt_occluded[p['gt_idx']] == 1)

    matched_gt_indices = {p['gt_idx'] for p in matched_pairs if 0 <= p['gt_idx'] < len(gt_occluded)}

    # FNs are unmatched GTs
    fn_without = sum(1 for j in range(len(gt_occluded)) if gt_occluded[j] == 0 and j in fn_indices)
    fn_with = sum(1 for j in range(len(gt_occluded)) if gt_occluded[j] == 1 and j in fn_indices)

    # FPs assigned by nearest overlapping GT (if any)
    fp_without = 0
    fp_with = 0
    for i in fp_indices:
        if i < 0 or i >= len(pred_timestamps):
            continue
        max_iou, best_gt = _max_iou_with_gt(pred_timestamps[i], gt_timestamps)
        if max_iou > 0 and 0 <= best_gt < len(gt_occluded):
            if gt_occluded[best_gt] == 0:
                fp_without += 1
            else:
                fp_with += 1
        # else: ignore FP in gaps

    return {
        'without_occlusion': {'tp': tp_without, 'fp': fp_without, 'fn': fn_without},
        'with_occlusion': {'tp': tp_with, 'fp': fp_with, 'fn': fn_with},
    }


def _compute_category_occlusion_split_metrics(
    pred_categories: List[int],
    gt_categories: List[int],
    gt_occluded: Optional[List[int]],
    matched_pairs: List[Dict],
    fp_indices: List[int],
    fn_indices: List[int],
    pred_timestamps: List[Dict],
    gt_timestamps: List[Dict],
) -> Dict:
    """
    Category TP/FP/FN split by GT occlusion, using:
    1) Strict gloss-matched pairs where categories match (split by GT occlusion)
    2) Order-preserving augmentation on categories for remaining indices
    3) FPs assigned to split by nearest overlapping GT (if any). FNs by GT flags
    """
    if not pred_categories or not gt_categories or not gt_occluded:
        return {
            'without_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
            'with_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
        }

    n_pred = len(pred_categories)
    n_gt = len(gt_categories)
    pred_matched = set()
    gt_matched = set()

    tp_without = 0
    tp_with = 0

    # Step 1: gloss-matched pairs where category matches
    for pair in matched_pairs:
        pi = pair.get('pred_idx')
        gi = pair.get('gt_idx')
        if pi is None or gi is None:
            continue
        if 0 <= pi < n_pred and 0 <= gi < n_gt:
            if pred_categories[pi] == gt_categories[gi]:
                if 0 <= gi < len(gt_occluded) and gt_occluded[gi] == 1:
                    tp_with += 1
                else:
                    tp_without += 1
                pred_matched.add(pi)
                gt_matched.add(gi)

    # Step 2: order-based augmentation by category on remaining indices
    remaining_pred = [i for i in range(n_pred) if i not in pred_matched]
    remaining_gt = [j for j in range(n_gt) if j not in gt_matched]
    add_pairs = _augment_matches_with_order(
        pred_categories,
        gt_categories,
        remaining_pred,
        remaining_gt,
    )
    for p in add_pairs:
        gi = p['gt_idx']
        if 0 <= gi < len(gt_occluded) and gt_occluded[gi] == 1:
            tp_with += 1
        else:
            tp_without += 1
        pred_matched.add(p['pred_idx'])
        gt_matched.add(gi)

    # Step 3: FPs assigned by nearest overlapping GT; FNs by GT occlusion
    fp_without = 0
    fp_with = 0
    for i in range(n_pred):
        if i in pred_matched:
            continue
        if i in fp_indices and 0 <= i < len(pred_timestamps):
            max_iou, best_gt = _max_iou_with_gt(pred_timestamps[i], gt_timestamps)
            if max_iou > 0 and 0 <= best_gt < len(gt_occluded):
                if gt_occluded[best_gt] == 1:
                    fp_with += 1
                else:
                    fp_without += 1

    fn_without = 0
    fn_with = 0
    for j in range(n_gt):
        if j in gt_matched:
            continue
        if j in fn_indices and 0 <= j < len(gt_occluded):
            if gt_occluded[j] == 1:
                fn_with += 1
            else:
                fn_without += 1

    return {
        'without_occlusion': {'tp': tp_without, 'fp': fp_without, 'fn': fn_without},
        'with_occlusion': {'tp': tp_with, 'fp': fp_with, 'fn': fn_with},
    }


def calculate_detection_metrics(
    num_tp: int, num_fp: int, num_fn: int
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score from TP/FP/FN counts.
    
    Args:
        num_tp: Number of true positives
        num_fp: Number of false positives
        num_fn: Number of false negatives
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


def smooth_sequence(
    sequence: List[int],
    confidences: Optional[List[float]] = None,
    categories: Optional[List[int]] = None,
    category_confidences: Optional[List[float]] = None
) -> Tuple[List[int], List[float], List[int], List[float]]:
    """
    Remove consecutive duplicate glosses from a sequence, keeping only the first occurrence.
    
    This helps reduce repeated predictions that occur due to sliding window overlap.
    For example: [good, morning, good, morning] -> [good, morning]
    
    Args:
        sequence: List of gloss IDs
        confidences: Optional list of confidence scores (take max for consecutive duplicates)
        categories: Optional list of category IDs (keep first occurrence)
        category_confidences: Optional list of category confidence scores (take max for consecutive duplicates)
    
    Returns:
        Tuple of (smoothed_sequence, smoothed_confidences, smoothed_categories, smoothed_category_confidences)
    """
    if not sequence:
        return [], [], [], []
    
    smoothed_seq = []
    smoothed_conf = []
    smoothed_cats = []
    smoothed_cat_conf = []
    
    prev_gloss = None
    
    for i, gloss in enumerate(sequence):
        if gloss != prev_gloss:
            # New gloss, add it
            smoothed_seq.append(gloss)
            
            if confidences and i < len(confidences):
                smoothed_conf.append(confidences[i])
            else:
                smoothed_conf.append(1.0)
            
            if categories and i < len(categories):
                smoothed_cats.append(categories[i])
            else:
                smoothed_cats.append(0)
            
            if category_confidences and i < len(category_confidences):
                smoothed_cat_conf.append(category_confidences[i])
            else:
                smoothed_cat_conf.append(0.0)
            
            prev_gloss = gloss
        else:
            # Consecutive duplicate - merge confidences (take max)
            if confidences and i < len(confidences):
                if smoothed_conf:
                    smoothed_conf[-1] = max(smoothed_conf[-1], confidences[i])
            
            # Merge category confidences (take max)
            if category_confidences and i < len(category_confidences):
                if smoothed_cat_conf:
                    smoothed_cat_conf[-1] = max(smoothed_cat_conf[-1], category_confidences[i])
    
    return smoothed_seq, smoothed_conf, smoothed_cats, smoothed_cat_conf


class CTCPredictor:
    """CTC-based sign language recognition predictor."""
    
    def __init__(self, model_type: str, checkpoint_path: str, blank_id: Optional[int] = None, device: Optional[torch.device] = None):
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Auto-detect blank_id from model configuration or use provided value
        if blank_id is not None:
            self.blank_id = blank_id
        else:
            # Check if this is a subset model (e.g., GREETINGS-only)
            if self.model_type in MODEL_CONFIG and 'ctc_config' in MODEL_CONFIG[self.model_type]:
                if MODEL_CONFIG[self.model_type]['ctc_config'] == 'subset':
                    self.blank_id = CTC_CONFIG_SUBSET['blank_token_id']
                else:
                    self.blank_id = CTC_CONFIG['blank_token_id']
            else:
                # Fallback to default CTC config
                self.blank_id = CTC_CONFIG['blank_token_id']
        
        self.model, self.input_dim = self._load_model()
        self._load_checkpoint()
        self.gloss_mapping, self.category_mapping = load_label_mappings()
        self.metrics_config = ContinuousEvaluationConfig()
    
    def _load_model(self) -> Tuple[torch.nn.Module, int]:
        if self.model_type == 'transformer_ctc':
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                input_dim = state_dict['embedding.weight'].shape[1] if 'embedding.weight' in state_dict else 178
                
                # Extract CTC class count from CTC head weights
                if 'ctc_head.weight' in state_dict:
                    num_ctc_classes = state_dict['ctc_head.weight'].shape[0]
                else:
                    # Fallback to trained model defaults
                    num_ctc_classes = 11
                
                # Extract category class count from category head weights (if present)
                num_cat = None
                if 'category_head.weight' in state_dict:
                    num_cat = state_dict['category_head.weight'].shape[0]
                    
            except:
                # Fallback to trained model defaults if checkpoint loading fails
                input_dim = 178
                num_ctc_classes = 11
                num_cat = None
            
            model = SignTransformerCtc(input_dim=input_dim, num_ctc_classes=num_ctc_classes, num_cat=num_cat, max_len=1000)
        
        elif self.model_type == 'mediapipe_gru_ctc':
            input_dim = 178
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Extract CTC class count from checkpoint
                if 'ctc_head.weight' in state_dict:
                    num_ctc_classes = state_dict['ctc_head.weight'].shape[0]
                else:
                    # Use subset config for GREETINGS models
                    if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                        num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    else:
                        num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                
                # Check for category head in checkpoint
                num_cat = None
                if 'category_head.weight' in state_dict:
                    num_cat = state_dict['category_head.weight'].shape[0]
                
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    # For bidirectional GRU: [3*hidden*2, hidden] -> hidden = shape[1]
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[1]
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[1]
                else:
                    gru1_hidden = 256
                    gru2_hidden = 128
            except:
                # Use subset config for GREETINGS models
                if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                    num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    num_cat = 1  # GREETINGS subset has 1 category
                else:
                    num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                    num_cat = None
                gru1_hidden = 256
                gru2_hidden = 128
            
            model = MediaPipeGRUCtc(
                input_dim=input_dim, 
                num_ctc_classes=num_ctc_classes,
                hidden1=gru1_hidden, 
                hidden2=gru2_hidden,
                num_cat=num_cat  # Pass category classes to enable category head
            )
        
        elif self.model_type == 'iv3_gru_ctc':
            # InceptionV3GRUCtc uses 2048-D features
            input_dim = 2048
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Extract CTC class count from checkpoint
                if 'ctc_head.weight' in state_dict:
                    num_ctc_classes = state_dict['ctc_head.weight'].shape[0]
                else:
                    # Use subset config for GREETINGS models
                    if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                        num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    else:
                        num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                
                # Check for category head in checkpoint
                num_cat = None
                if 'category_head.weight' in state_dict:
                    num_cat = state_dict['category_head.weight'].shape[0]
                
                # Try to extract hidden dimensions from checkpoint
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    # For bidirectional GRU: weight_hh_l0 has shape [3*hidden_size*2, hidden_size]
                    # The hidden_size is actually shape[1], not shape[0] // 3 // 2
                    gru1_shape = state_dict['gru1.weight_hh_l0'].shape
                    gru2_shape = state_dict['gru2.weight_hh_l0'].shape
                    
                    # For bidirectional GRU: [3*hidden*2, hidden] -> hidden = shape[1]
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[1]
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[1]
                else:
                    gru1_hidden = 256
                    gru2_hidden = 128
            except:
                # Use subset config for GREETINGS models
                if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                    num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    num_cat = 1  # GREETINGS subset has 1 category
                else:
                    num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                    num_cat = None
                gru1_hidden = 256
                gru2_hidden = 128
            
            model = InceptionV3GRUCtc(
                num_ctc_classes=num_ctc_classes,
                hidden1=gru1_hidden,
                hidden2=gru2_hidden,
                num_cat=num_cat  # Pass category classes to enable category head
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device), input_dim
    
    def _load_checkpoint(self):
        """Load model checkpoint with robust error handling."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        
        # Get current model state dict
        model_state_dict = self.model.state_dict()
        
        # Filter state dict to only include keys that exist in current model
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                # Handle positional encoding size mismatch
                if key == 'pos_encoder.pe':
                    current_shape = model_state_dict[key].shape
                    checkpoint_shape = value.shape
                    
                    if current_shape != checkpoint_shape:
                        # Always expand to the current model's max_len (1000)
                        if current_shape[1] > checkpoint_shape[1]:
                            # Need to expand - pad with zeros (match device and dtype)
                            padding = value.new_zeros(current_shape[0], current_shape[1] - checkpoint_shape[1], current_shape[2])
                            value = torch.cat([value, padding], dim=1)
                        else:
                            # Need to truncate
                            value = value[:, :current_shape[1], :]
                
                filtered_state_dict[key] = value
            else:
                pass
        
        # Load the filtered state dict
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.eval()
    
    def predict_sequence(
        self,
        npz_path: Path,
        ground_truth: Optional[Dict] = None,
        decode_method: str = 'greedy',
        beam_width: int = 10,
        fps: int = 30,
        temporal_tolerance: int = 500,
        iou_threshold: float = 0.5
    ) -> Dict:
        """Predict single continuous sequence with full metrics."""
        data = np.load(npz_path)

        mask = None
        timestamps_ms = None
        if 'mask' in data:
            mask = np.asarray(data['mask']).astype(bool)
        if 'timestamps_ms' in data:
            timestamps_ms = np.asarray(data['timestamps_ms']).astype(float)
        
        if self.input_dim == 2048:
            if 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X2048' key")
            X = torch.from_numpy(data['X2048']).float().unsqueeze(0)
        elif self.input_dim == 178:
            if 'X' not in data:
                raise ValueError(f"NPZ file missing 'X' key")
            X = torch.from_numpy(data['X']).float().unsqueeze(0)
        elif self.input_dim == 156:
            if 'X' not in data:
                raise ValueError(f"NPZ file missing 'X' key")
            X = torch.from_numpy(data['X']).float().unsqueeze(0)
        elif self.input_dim == 2204:
            if 'X' not in data or 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X' or 'X2048' key")
            X_kp = torch.from_numpy(data['X']).float()
            X_feat = torch.from_numpy(data['X2048']).float()
            X = torch.cat([X_kp, X_feat], dim=1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input dimension {self.input_dim}")
        
        X = X.to(self.device)
        input_length = torch.tensor([X.shape[1]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # InceptionV3GRUCtc requires features_already parameter
            if self.model_type == 'iv3_gru_ctc':
                output = self.model(X, features_already=True)
            else:
                output = self.model(X)
            
            # Handle dual-task models (CTC + Category)
            cat_logits = None
            if isinstance(output, tuple):
                log_probs, cat_logits = output  # [B,T,num_ctc], [B,T,num_cat]
            else:
                log_probs = output
            
        # Decode gloss sequence
        if decode_method == 'greedy':
            predicted_sequence = greedy_ctc_decoder(log_probs, self.blank_id, input_length)[0]
            probs = torch.exp(log_probs[0])
            confidence_scores = [float(probs[:, g].max()) for g in predicted_sequence] if len(predicted_sequence) > 0 else []
            
        else:
            predicted_sequence, log_prob = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_length)[0]
            avg_conf = np.exp(log_prob / max(len(predicted_sequence), 1))
            confidence_scores = [float(avg_conf)] * len(predicted_sequence)
        
        # Decode category sequence (per-frame predictions -> per-sign categories)
        predicted_categories = []
        category_confidences = []
        if cat_logits is not None:
            # Get per-frame category predictions [T, num_cat]
            cat_probs = torch.softmax(cat_logits[0, :input_length[0]], dim=1)
            
            if len(predicted_sequence) > 0:
                # Assign categories to each predicted sign based on frame distribution
                frames_per_sign = X.shape[1] / len(predicted_sequence)
                for idx in range(len(predicted_sequence)):
                    start_frame = int(idx * frames_per_sign)
                    end_frame = int((idx + 1) * frames_per_sign)
                    # Majority vote for this sign's frames
                    sign_cat_probs = cat_probs[start_frame:end_frame].mean(dim=0)
                    pred_cat = sign_cat_probs.argmax().item()
                    cat_conf = sign_cat_probs[pred_cat].item()
                    predicted_categories.append(pred_cat)
                    category_confidences.append(float(cat_conf))
                
        else:
            # No category head available - provide fallback
            # Fallback: assign categories based on gloss predictions
            if len(predicted_sequence) > 0:
                for gloss_id in predicted_sequence:
                    # Simple fallback: assign category based on gloss ID
                    if 0 <= gloss_id <= 9:  # Greetings
                        predicted_categories.append(0)  # GREETING
                        category_confidences.append(0.8)  # High confidence for greetings
                    elif 10 <= gloss_id <= 19:  # Survival
                        predicted_categories.append(1)  # SURVIVAL
                        category_confidences.append(0.7)
                    elif 20 <= gloss_id <= 29:  # Numbers
                        predicted_categories.append(2)  # NUMBER
                        category_confidences.append(0.7)
                    else:
                        predicted_categories.append(0)  # Default to GREETING
                        category_confidences.append(0.5)  # Lower confidence for unknown
        
        # Smooth sequence by removing consecutive duplicates
        predicted_sequence, confidence_scores, predicted_categories, category_confidences = smooth_sequence(
            predicted_sequence,
            confidence_scores,
            predicted_categories,
            category_confidences
        )
        
        predicted_labels = [self.gloss_mapping.get(g, f"GLOSS_{g}") for g in predicted_sequence]
        predicted_timestamps = estimate_timestamps(predicted_sequence, X.shape[1], fps)
        
        # Add category information to predicted_timestamps
        if predicted_categories and len(predicted_categories) == len(predicted_timestamps):
            for i, ts in enumerate(predicted_timestamps):
                ts['category'] = predicted_categories[i] if i < len(predicted_categories) else None
                if hasattr(self, 'category_mapping') and ts['category'] is not None:
                    ts['category_label'] = self.category_mapping.get(ts['category'], f"Cat_{ts['category']}")
                else:
                    ts['category_label'] = ''
        
        result = {
            'file_name': npz_path.name,
            'predicted_sequence': predicted_sequence,
            'predicted_labels': predicted_labels,
            'predicted_categories': predicted_categories,
            'confidence_scores': confidence_scores,
            'category_confidences': category_confidences,
            'predicted_timestamps': predicted_timestamps,
            'num_predicted': len(predicted_sequence)
        }
        
        if ground_truth:
            result['signer'] = ground_truth['signer']
            result['strategy'] = ground_truth['strategy']
            result['ground_truth_sequence'] = ground_truth['ground_truth_sequence']
            result['ground_truth_labels'] = ground_truth['ground_truth_labels']
            result['ground_truth_timestamps'] = ground_truth.get('ground_truth_timestamps', [])
            
            # Ground truth categories if available
            if 'ground_truth_categories' in ground_truth:
                result['ground_truth_categories'] = ground_truth['ground_truth_categories']
            
            if len(confidence_scores) != len(predicted_sequence):
                if len(predicted_sequence) == 0:
                    confidence_scores = []
                else:
                    confidence_scores = [
                        confidence_scores[i] if i < len(confidence_scores) else 1.0
                        for i in range(len(predicted_sequence))
                    ]

            gt_timestamps = ground_truth.get('ground_truth_timestamps', [])

            try:
                if gt_timestamps:
                    pred_ts_objects = [
                        MetricTimestamp(
                            start_ms=float(ts.get('start_ms', 0.0)),
                            end_ms=float(ts.get('end_ms', 0.0)),
                            gloss=int(predicted_sequence[i]) if i < len(predicted_sequence) else None,
                        )
                        for i, ts in enumerate(predicted_timestamps)
                    ]
                    gt_ts_objects = [
                        MetricTimestamp(
                            start_ms=float(ts.get('start_ms', 0.0)),
                            end_ms=float(ts.get('end_ms', 0.0)),
                            gloss=int(ts.get('gloss', ground_truth['ground_truth_sequence'][idx])),
                        )
                        for idx, ts in enumerate(gt_timestamps)
                    ]

                    seq_prediction = SequencePrediction(
                        gloss_ids=[int(g) for g in predicted_sequence],
                        labels=predicted_labels,
                        timestamps=pred_ts_objects,
                        confidence_scores=[float(c) for c in confidence_scores],
                    )
                    seq_ground_truth = SequenceGroundTruth(
                        gloss_ids=[int(g) for g in ground_truth['ground_truth_sequence']],
                        labels=ground_truth.get('ground_truth_labels', []),
                        timestamps=gt_ts_objects,
                        occlusion_flags=ground_truth.get('ground_truth_occluded'),
                    )

                    metrics_result = compute_sequence_metrics(
                        prediction=seq_prediction,
                        ground_truth=seq_ground_truth,
                        config=ContinuousEvaluationConfig(
                            iou_threshold=iou_threshold,
                            confidence_threshold=self.metrics_config.confidence_threshold,
                            inactive_threshold=self.metrics_config.inactive_threshold,
                            active_overlap_threshold=self.metrics_config.active_overlap_threshold,
                            min_gap_duration_ms=self.metrics_config.min_gap_duration_ms,
                            enable_lenient_matching=self.metrics_config.enable_lenient_matching,
                        ),
                        mask=mask,
                        timestamps_ms=timestamps_ms,
                    )

                    result['num_tp'] = metrics_result.num_tp
                    result['num_fp'] = metrics_result.num_fp
                    result['num_fn'] = metrics_result.num_fn
                    result['num_tn'] = metrics_result.num_tn
                    result['num_gt'] = len(ground_truth['ground_truth_sequence'])
                    result['precision'] = float(metrics_result.precision)
                    result['recall'] = float(metrics_result.recall)
                    result['f1_score'] = float(metrics_result.f1_score)
                    result['iou_threshold'] = iou_threshold
                    result['matched_pairs'] = metrics_result.matched_pairs
                    result['unmatched_predictions'] = metrics_result.fp_indices
                    result['unmatched_ground_truth'] = metrics_result.fn_indices
                    result['tp_indices'] = metrics_result.tp_indices
                    result['tp_breakdown'] = {
                        case.value: count
                        for case, count in metrics_result.tp_breakdown.items()
                    }
                    result['fp_breakdown'] = {
                        case.value: count
                        for case, count in metrics_result.fp_breakdown.items()
                    }
                    result['fn_breakdown'] = {
                        case.value: count
                        for case, count in metrics_result.fn_breakdown.items()
                    }
                    result['tn_breakdown'] = {
                        case.value: count
                        for case, count in metrics_result.tn_breakdown.items()
                    }

                    if 'ground_truth_occluded' in ground_truth:
                        occ = _compute_occlusion_split_metrics(
                            pred_timestamps=predicted_timestamps,
                            gt_timestamps=gt_timestamps,
                            gt_occluded=ground_truth['ground_truth_occluded'],
                            matched_pairs=metrics_result.matched_pairs,
                            fp_indices=metrics_result.fp_indices,
                            fn_indices=metrics_result.fn_indices,
                        )
                        result['occlusion_metrics'] = occ
                        if predicted_categories and 'ground_truth_categories' in ground_truth:
                            occ_cat = _compute_category_occlusion_split_metrics(
                                pred_categories=predicted_categories,
                                gt_categories=ground_truth['ground_truth_categories'],
                                gt_occluded=ground_truth['ground_truth_occluded'],
                                matched_pairs=metrics_result.matched_pairs,
                                fp_indices=metrics_result.fp_indices,
                                fn_indices=metrics_result.fn_indices,
                                pred_timestamps=predicted_timestamps,
                                gt_timestamps=gt_timestamps,
                            )
                            result['occlusion_metrics_category'] = occ_cat

                    if predicted_categories and 'ground_truth_categories' in ground_truth:
                        gt_cats = ground_truth['ground_truth_categories']
                        if gt_cats:
                            cat_tp, cat_fp, cat_fn = _compute_category_metrics_balanced(
                                pred_categories=predicted_categories,
                                gt_categories=gt_cats,
                                matched_pairs=metrics_result.matched_pairs,
                                pred_len=len(predicted_sequence),
                                gt_len=len(ground_truth['ground_truth_sequence']),
                                gloss_fp_indices=metrics_result.fp_indices,
                                gloss_fn_indices=metrics_result.fn_indices,
                            )
                            cat_precision, cat_recall, cat_f1 = calculate_detection_metrics(cat_tp, cat_fp, cat_fn)
                            result['category_num_tp'] = cat_tp
                            result['category_num_fp'] = cat_fp
                            result['category_num_fn'] = cat_fn
                            result['category_precision'] = float(cat_precision)
                            result['category_recall'] = float(cat_recall)
                            result['category_f1_score'] = float(cat_f1)
                else:
                    raise ValueError("Ground truth timestamps are missing or empty. Detection metrics require timestamps.")
            except Exception as e:
                result['num_tp'] = 0
                result['num_fp'] = len(predicted_sequence)
                result['num_fn'] = len(ground_truth['ground_truth_sequence'])
                result['num_tn'] = 0
                result['num_gt'] = len(ground_truth['ground_truth_sequence'])
                result['precision'] = 0.0
                result['recall'] = 0.0
                result['f1_score'] = 0.0
                result['iou_threshold'] = iou_threshold
                result['matched_pairs'] = []
                result['unmatched_predictions'] = list(range(len(predicted_sequence)))
                result['unmatched_ground_truth'] = list(range(len(ground_truth['ground_truth_sequence'])))
                print(f"Warning: Detection metrics calculation failed: {str(e)}")
        
        return result
    
    def predict_sequence_sliding_window(self, npz_path: Path, ground_truth: Optional[Dict] = None,
                                      window_size: int = 120, stride: int = 40, 
                                      decode_method: str = 'greedy', beam_width: int = 10, 
                                      fps: int = 30, temporal_tolerance: int = 500,
                                      iou_threshold: float = 0.5) -> Dict:
        """
        Predict continuous sequence using sliding window approach.
        
        This method applies the CTC model to overlapping windows of the input sequence,
        then aggregates the predictions to get the final continuous sequence.
        
        Args:
            npz_path: Path to NPZ file
            ground_truth: Optional ground truth dictionary
            window_size: Size of each window in frames
            stride: Step size between windows
            decode_method: Decoding method ('greedy' or 'beam_search')
            beam_width: Beam width for beam search
            fps: Frames per second for timestamp estimation
            temporal_tolerance: Temporal tolerance for alignment (ms)
            
        Returns:
            Dictionary with sliding window prediction results
        """
        data = np.load(npz_path)
        
        # Load input data
        if self.input_dim == 2048:
            if 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X2048' key")
            X = torch.from_numpy(data['X2048']).float()
        elif self.input_dim == 178:
            if 'X' not in data:
                raise ValueError(f"NPZ file missing 'X' key")
            X = torch.from_numpy(data['X']).float()
        elif self.input_dim == 156:
            if 'X' not in data:
                raise ValueError(f"NPZ file missing 'X' key")
            X = torch.from_numpy(data['X']).float()
        elif self.input_dim == 2204:
            if 'X' not in data or 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X' or 'X2048' key")
            X_kp = torch.from_numpy(data['X']).float()
            X_feat = torch.from_numpy(data['X2048']).float()
            X = torch.cat([X_kp, X_feat], dim=1)
        else:
            raise ValueError(f"Unsupported input dimension {self.input_dim}")
        
        seq_len = X.shape[0]
        
        # Generate sliding windows
        windows = []
        window_predictions = []
        window_confidences = []
        window_categories = []
        window_category_confidences = []
        
        # Handle sequences shorter than window_size
        if seq_len < window_size:
            # Use the entire sequence as a single window
            window_data = X.unsqueeze(0).to(self.device)  # [1, seq_len, features]
            input_length = torch.tensor([seq_len], dtype=torch.long).to(self.device)
            
            windows.append((0, seq_len))
            
            # Get model prediction for this single window
            with torch.no_grad():
                if self.model_type == 'iv3_gru_ctc':
                    output = self.model(window_data, features_already=True)
                else:
                    output = self.model(window_data)
                
                # Handle dual-task models
                cat_logits = None
                if isinstance(output, tuple):
                    log_probs, cat_logits = output
                else:
                    log_probs = output
            
            # Decode window prediction
            if decode_method == 'greedy':
                window_pred = greedy_ctc_decoder(log_probs, self.blank_id, input_length)[0]
                probs = torch.exp(log_probs[0])
                window_conf = [float(probs[:, g].max()) for g in window_pred] if len(window_pred) > 0 else []
                
            else:
                window_pred, log_prob = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_length)[0]
                avg_conf = np.exp(log_prob / max(len(window_pred), 1))
                window_conf = [float(avg_conf)] * len(window_pred)
            
            # Process category predictions
            window_cat_preds = []
            window_cat_confs = []
            if cat_logits is not None:
                cat_probs = torch.softmax(cat_logits[0], dim=1)  # [T, num_categories]
                for i, pred_token in enumerate(window_pred):
                    if i < cat_probs.shape[0]:
                        cat_pred = torch.argmax(cat_probs[i]).item()
                        cat_conf = float(cat_probs[i, cat_pred])
                        window_cat_preds.append(cat_pred)
                        window_cat_confs.append(cat_conf)
                    else:
                        window_cat_preds.append(0)  # Default category
                        window_cat_confs.append(0.0)
            
            window_predictions.append(window_pred)
            window_confidences.append(window_conf)
            window_categories.append(window_cat_preds)
            window_category_confidences.append(window_cat_confs)
        else:
            # Normal sliding window processing for longer sequences
            for start_idx in range(0, seq_len - window_size + 1, stride):
                end_idx = start_idx + window_size
                window_data = X[start_idx:end_idx].unsqueeze(0).to(self.device)  # [1, window_size, features]
                input_length = torch.tensor([window_size], dtype=torch.long).to(self.device)
                
                windows.append((start_idx, end_idx))
                
                # Get model prediction for this window
                with torch.no_grad():
                    if self.model_type == 'iv3_gru_ctc':
                        output = self.model(window_data, features_already=True)
                    else:
                        output = self.model(window_data)
                    
                    # Handle dual-task models
                    cat_logits = None
                    if isinstance(output, tuple):
                        log_probs, cat_logits = output
                    else:
                        log_probs = output
                
                # Decode window prediction
                if decode_method == 'greedy':
                    window_pred = greedy_ctc_decoder(log_probs, self.blank_id, input_length)[0]
                    probs = torch.exp(log_probs[0])
                    window_conf = [float(probs[:, g].max()) for g in window_pred] if len(window_pred) > 0 else []
                else:
                    window_pred, log_prob = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_length)[0]
                    avg_conf = np.exp(log_prob / max(len(window_pred), 1))
                    window_conf = [float(avg_conf)] * len(window_pred)
                
                # Process category predictions
                window_cat_preds = []
                window_cat_confs = []
                if cat_logits is not None:
                    cat_probs = torch.softmax(cat_logits[0], dim=1)  # [T, num_categories]
                    for i, pred_token in enumerate(window_pred):
                        if i < cat_probs.shape[0]:
                            cat_pred = torch.argmax(cat_probs[i]).item()
                            cat_conf = float(cat_probs[i, cat_pred])
                            window_cat_preds.append(cat_pred)
                            window_cat_confs.append(cat_conf)
                        else:
                            window_cat_preds.append(0)  # Default category
                            window_cat_confs.append(0.0)
                
                window_predictions.append(window_pred)
                window_confidences.append(window_conf)
                window_categories.append(window_cat_preds)
                window_category_confidences.append(window_cat_confs)
        
        # Aggregate predictions across windows
        all_predictions = []
        all_categories = []
        frame_positions = []
        
        for i, (window_pred, window_conf, window_cats, window_cat_confs, (start_idx, end_idx)) in enumerate(zip(window_predictions, window_confidences, window_categories, window_category_confidences, windows)):
            for j, (pred_token, conf) in enumerate(zip(window_pred, window_conf)):
                # Estimate frame position within the window
                if len(window_pred) > 1:
                    frame_pos = start_idx + int((j / len(window_pred)) * window_size)
                else:
                    frame_pos = start_idx + window_size // 2
                
                all_predictions.append(pred_token)
                frame_positions.append(frame_pos)
                
                # Get corresponding category prediction
                if j < len(window_cats):
                    all_categories.append((window_cats[j], window_cat_confs[j]))
                else:
                    all_categories.append((0, 0.0))
        
        # Remove duplicates and sort by frame position
        if all_predictions:
            # Group predictions by frame position and take the most confident
            position_groups = {}
            for pred, cat_info, pos, conf in zip(all_predictions, all_categories, frame_positions, [max(c) if c else 0.0 for c in window_confidences]):
                if pos not in position_groups or conf > position_groups[pos][1]:
                    position_groups[pos] = (pred, conf, cat_info)
            
            # Sort by frame position and extract final sequence
            sorted_positions = sorted(position_groups.keys())
            final_sequence = [position_groups[pos][0] for pos in sorted_positions]
            final_confidences = [position_groups[pos][1] for pos in sorted_positions]
            final_categories = [position_groups[pos][2][0] for pos in sorted_positions]
            final_category_confidences = [position_groups[pos][2][1] for pos in sorted_positions]
        else:
            final_sequence = []
            final_confidences = []
            final_categories = []
            final_category_confidences = []
        
        # Smooth sequence by removing consecutive duplicates
        final_sequence, final_confidences, final_categories, final_category_confidences = smooth_sequence(
            final_sequence,
            final_confidences,
            final_categories,
            final_category_confidences
        )
        
        # Convert to labels
        predicted_labels = [self.gloss_mapping.get(g, f"GLOSS_{g}") for g in final_sequence]
        predicted_timestamps = estimate_timestamps(final_sequence, seq_len, fps)
        
        # Add category information to predicted_timestamps
        if final_categories and len(final_categories) == len(predicted_timestamps):
            for i, ts in enumerate(predicted_timestamps):
                ts['category'] = final_categories[i] if i < len(final_categories) else None
                if hasattr(self, 'category_mapping') and ts['category'] is not None:
                    ts['category_label'] = self.category_mapping.get(ts['category'], f"Cat_{ts['category']}")
                else:
                    ts['category_label'] = ''
        
        result = {
            'file_name': npz_path.name,
            'predicted_sequence': final_sequence,
            'predicted_labels': predicted_labels,
            'confidence_scores': final_confidences,
            'predicted_categories': final_categories,
            'category_confidences': final_category_confidences,
            'predicted_timestamps': predicted_timestamps,
            'num_predicted': len(final_sequence),
            'num_windows': len(windows),
            'window_size': window_size,
            'stride': stride,
            'method': 'sliding_window',
            'confidence_threshold': float(self.metrics_config.confidence_threshold),
        }
        
        # Add ground truth comparison if available
        if ground_truth:
            # Handle both metadata format (with segments) and ground truth format
            if 'segments' in ground_truth:
                # Original metadata format - segments is a list
                segments = ground_truth.get('segments', [])
                gt_labels = [seg.get('gloss_label', f"GLOSS_{seg.get('gloss', '?')}") for seg in segments]
                gt_timestamps = []
                gt_gloss_ids = []
                # Also extract per-segment categories if available
                gt_categories = []
                gt_category_labels = []
                gt_occluded = []
                for seg in segments:
                    gt_gloss_ids.append(seg.get('gloss', 0))
                    gt_timestamps.append({
                        'start_ms': seg.get('timestamp_start_ms', 0),
                        'end_ms': seg.get('timestamp_end_ms', 0),
                        'duration_ms': seg.get('timestamp_end_ms', 0) - seg.get('timestamp_start_ms', 0)
                    })
                    if 'category' in seg:
                        gt_categories.append(seg.get('category'))
                    if 'category_label' in seg:
                        gt_category_labels.append(seg.get('category_label'))
                    if 'occluded' in seg:
                        gt_occluded.append(int(seg.get('occluded', 0)))
                # Expose flattened GT categories/labels for UI and metrics
                if gt_categories:
                    ground_truth['ground_truth_categories'] = gt_categories
                if gt_category_labels:
                    ground_truth['ground_truth_category_labels'] = gt_category_labels
                if gt_occluded:
                    ground_truth['ground_truth_occluded'] = gt_occluded
            else:
                # Converted ground truth format
                gt_labels = ground_truth.get('ground_truth_labels', [])
                gt_timestamps = ground_truth.get('ground_truth_timestamps', [])
                gt_gloss_ids = ground_truth.get('ground_truth_sequence', [])
            
            result['ground_truth_sequence'] = gt_gloss_ids
            result['ground_truth_labels'] = gt_labels
            result['ground_truth_timestamps'] = gt_timestamps
            # Pass through per-segment categories if present
            if 'ground_truth_category_labels' in ground_truth:
                result['ground_truth_category_labels'] = ground_truth['ground_truth_category_labels']
            if 'ground_truth_categories' in ground_truth:
                result['ground_truth_categories'] = ground_truth['ground_truth_categories']
            if 'ground_truth_occluded' in ground_truth:
                result['ground_truth_occluded'] = ground_truth['ground_truth_occluded']
            result['signer'] = ground_truth.get('signer')
            result['strategy'] = ground_truth.get('strategy', ground_truth.get('strategy_name'))
            
            # Calculate detection metrics using unified evaluation pipeline
            try:
                pred_ts_objects = [
                    MetricTimestamp(
                        start_ms=float(ts.get('start_ms', 0.0)),
                        end_ms=float(ts.get('end_ms', 0.0)),
                        gloss=int(final_sequence[i]) if i < len(final_sequence) else None,
                    )
                    for i, ts in enumerate(predicted_timestamps)
                ]
                gt_ts_objects = [
                    MetricTimestamp(
                        start_ms=float(ts.get('start_ms', 0.0)),
                        end_ms=float(ts.get('end_ms', 0.0)),
                        gloss=int(gt_gloss_ids[idx]) if idx < len(gt_gloss_ids) else ts.get('gloss'),
                    )
                    for idx, ts in enumerate(gt_timestamps)
                ]

                seq_prediction = SequencePrediction(
                    gloss_ids=list(final_sequence),
                    labels=predicted_labels,
                    timestamps=pred_ts_objects,
                    confidence_scores=[float(c) for c in final_confidences],
                )
                seq_ground_truth = SequenceGroundTruth(
                    gloss_ids=list(gt_gloss_ids),
                    labels=gt_labels,
                    timestamps=gt_ts_objects,
                    occlusion_flags=ground_truth.get('ground_truth_occluded'),
                )

                metrics_result = compute_sequence_metrics(
                    prediction=seq_prediction,
                    ground_truth=seq_ground_truth,
                    config=ContinuousEvaluationConfig(
                        iou_threshold=iou_threshold,
                        confidence_threshold=self.metrics_config.confidence_threshold,
                        inactive_threshold=self.metrics_config.inactive_threshold,
                        active_overlap_threshold=self.metrics_config.active_overlap_threshold,
                        min_gap_duration_ms=self.metrics_config.min_gap_duration_ms,
                        enable_lenient_matching=self.metrics_config.enable_lenient_matching,
                    ),
                    mask=None,
                    timestamps_ms=None,
                )

                result['num_tp'] = metrics_result.num_tp
                result['num_fp'] = metrics_result.num_fp
                result['num_fn'] = metrics_result.num_fn
                result['num_tn'] = metrics_result.num_tn
                result['num_gt'] = len(gt_gloss_ids)
                result['precision'] = float(metrics_result.precision)
                result['recall'] = float(metrics_result.recall)
                result['f1_score'] = float(metrics_result.f1_score)
                result['confidence_threshold'] = float(self.metrics_config.confidence_threshold)
                result['tp_indices'] = metrics_result.tp_indices
                result['matched_pairs'] = metrics_result.matched_pairs
                result['unmatched_predictions'] = metrics_result.fp_indices
                result['unmatched_ground_truth'] = metrics_result.fn_indices
                result['tp_breakdown'] = {
                    case.value: count for case, count in metrics_result.tp_breakdown.items()
                }
                result['fp_breakdown'] = {
                    case.value: count for case, count in metrics_result.fp_breakdown.items()
                }
                result['fn_breakdown'] = {
                    case.value: count for case, count in metrics_result.fn_breakdown.items()
                }
                result['tn_breakdown'] = {
                    case.value: count for case, count in metrics_result.tn_breakdown.items()
                }

                # Occlusion split metrics if occlusion flags available
                if 'ground_truth_occluded' in ground_truth:
                    occ = _compute_occlusion_split_metrics(
                        pred_timestamps=predicted_timestamps,
                        gt_timestamps=gt_timestamps,
                        gt_occluded=ground_truth['ground_truth_occluded'],
                        matched_pairs=metrics_result.matched_pairs,
                        fp_indices=metrics_result.fp_indices,
                        fn_indices=metrics_result.fn_indices,
                    )
                    result['occlusion_metrics'] = occ
                    # Category occlusion split (if categories available)
                    if final_categories and 'ground_truth_categories' in ground_truth:
                        occ_cat = _compute_category_occlusion_split_metrics(
                            pred_categories=final_categories,
                            gt_categories=gt_categories,
                            gt_occluded=ground_truth['ground_truth_occluded'],
                            matched_pairs=metrics_result.matched_pairs,
                            fp_indices=metrics_result.fp_indices,
                            fn_indices=metrics_result.fn_indices,
                            pred_timestamps=predicted_timestamps,
                            gt_timestamps=gt_timestamps,
                        )
                        result['occlusion_metrics_category'] = occ_cat
                
                # Category-level detection metrics (balanced, independent of gloss identity)
                if final_categories and 'ground_truth_categories' in ground_truth:
                    gt_categories = ground_truth['ground_truth_categories']
                    if gt_categories:
                        cat_tp, cat_fp, cat_fn = _compute_category_metrics_balanced(
                            pred_categories=final_categories,
                            gt_categories=gt_categories,
                            matched_pairs=metrics_result.matched_pairs,
                            pred_len=len(final_sequence),
                            gt_len=len(gt_gloss_ids),
                            gloss_fp_indices=metrics_result.fp_indices,
                            gloss_fn_indices=metrics_result.fn_indices,
                        )
                        cat_precision, cat_recall, cat_f1 = calculate_detection_metrics(cat_tp, cat_fp, cat_fn)
                        result['category_num_tp'] = cat_tp
                        result['category_num_fp'] = cat_fp
                        result['category_num_fn'] = cat_fn
                        result['category_precision'] = float(cat_precision)
                        result['category_recall'] = float(cat_recall)
                        result['category_f1_score'] = float(cat_f1)
                
            except Exception as e:
                # Fallback: set default values if matching fails
                result['num_tp'] = 0
                result['num_fp'] = len(final_sequence)
                result['num_fn'] = len(gt_gloss_ids)
                result['num_tn'] = 0
                result['num_gt'] = len(gt_gloss_ids)
                result['precision'] = 0.0
                result['recall'] = 0.0
                result['f1_score'] = 0.0
                result['iou_threshold'] = iou_threshold
                result['matched_pairs'] = []
                result['unmatched_predictions'] = list(range(len(final_sequence)))
                result['unmatched_ground_truth'] = list(range(len(gt_gloss_ids)))
                result['tp_indices'] = []
                result['tp_breakdown'] = {case.value: 0 for case in TPCase}
                result['fp_breakdown'] = {case.value: 0 for case in FPCase}
                result['fn_breakdown'] = {case.value: 0 for case in FNCase}
                result['tn_breakdown'] = {case.value: 0 for case in TNCase}
                print(f"Warning: Detection metrics calculation failed for {result.get('file_name', 'unknown')}: {str(e)}")
        
        return result
    
    def predict_batch(
        self,
        input_dir: Path,
        ground_truth_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        decode_method: str = 'greedy',
        beam_width: int = 10,
        fps: int = 30,
        temporal_tolerance: int = 500
    ) -> Dict:
        """Predict batch of continuous sequences."""
        npz_files = sorted(input_dir.glob('*.npz'))
        
        if not npz_files:
            raise ValueError(f"No NPZ files found in {input_dir}")
        
        predictions = []
        
        print(f"\nPredicting {len(npz_files)} sequences...")
        for npz_path in tqdm(npz_files, desc="Processing"):
            ground_truth = None
            if ground_truth_dir:
                gt_path = ground_truth_dir / (npz_path.stem + '_gt.json')
                if gt_path.exists():
                    ground_truth = load_ground_truth_json(gt_path)
            
            try:
                result = self.predict_sequence(
                    npz_path,
                    ground_truth,
                    decode_method,
                    beam_width,
                    fps,
                    temporal_tolerance
                )
                predictions.append(result)
                
                if output_dir:
                    pred_path = output_dir / (npz_path.stem + '_pred.json')
                    with open(pred_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
            
            except Exception as e:
                print(f"\nError processing {npz_path.name}: {e}")
                continue
        
        summary = self._generate_summary(predictions)
        
        if output_dir:
            summary_path = output_dir / 'prediction_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            self._generate_confusion_matrices(predictions, output_dir)
        
        return {
            'predictions': predictions,
            'summary': summary
        }
    
    def _generate_summary(self, predictions: List[Dict]) -> Dict:
        """Generate summary statistics using instance-level detection metrics."""
        if not predictions:
            return {'total_sequences': 0}
        
        # Filter predictions with ground truth (detection metrics required)
        predictions_with_gt = [p for p in predictions if 'f1_score' in p]
        has_gt = len(predictions_with_gt) > 0
        has_categories = 'predicted_categories' in predictions[0] and len(predictions[0].get('predicted_categories', [])) > 0
        
        summary = {
            'total_sequences': len(predictions),
            'total_sequences_with_gt': len(predictions_with_gt),
            'model_type': self.model_type,
            'has_category_predictions': has_categories
        }
        
        if has_gt:
            micro_metrics = compute_overall_metrics_micro(predictions_with_gt)
            macro_metrics = compute_overall_metrics_macro(predictions_with_gt)
            stratified_metrics = compute_stratified_metrics(predictions_with_gt)

            combined_overall = dict(micro_metrics)
            combined_overall.update({
                'overall_precision': micro_metrics.get('precision', 0.0),
                'overall_recall': micro_metrics.get('recall', 0.0),
                'overall_f1_score': micro_metrics.get('f1_score', 0.0),
                'mean_precision': macro_metrics.get('mean_precision', 0.0),
                'mean_recall': macro_metrics.get('mean_recall', 0.0),
                'mean_f1_score': macro_metrics.get('mean_f1_score', 0.0),
                'median_f1_score': macro_metrics.get('median_f1_score', 0.0),
            })

            summary['overall_metrics'] = combined_overall
            summary['macro_metrics'] = macro_metrics
            summary['stratified_metrics'] = stratified_metrics
            summary['per_signer_metrics'] = {
                signer: dict(metrics, num_sequences=metrics.get('total_sequences', 0))
                for signer, metrics in stratified_metrics.get('per_signer', {}).items()
            }
            summary['per_strategy_metrics'] = {
                strategy: dict(metrics, num_sequences=metrics.get('total_sequences', 0))
                for strategy, metrics in stratified_metrics.get('per_strategy', {}).items()
            }
            
            # Category-level metrics (if available)
            if has_categories:
                cat_tp_total = sum(p.get('category_num_tp', 0) for p in predictions_with_gt if 'category_num_tp' in p)
                cat_fp_total = sum(p.get('category_num_fp', 0) for p in predictions_with_gt if 'category_num_fp' in p)
                cat_fn_total = sum(p.get('category_num_fn', 0) for p in predictions_with_gt if 'category_num_fn' in p)
                
                if cat_tp_total + cat_fp_total + cat_fn_total > 0:
                    cat_precision, cat_recall, cat_f1 = calculate_detection_metrics(
                        cat_tp_total, cat_fp_total, cat_fn_total
                    )
                    summary['overall_metrics']['category_total_tp'] = int(cat_tp_total)
                    summary['overall_metrics']['category_total_fp'] = int(cat_fp_total)
                    summary['overall_metrics']['category_total_fn'] = int(cat_fn_total)
                    summary['overall_metrics']['category_overall_precision'] = float(cat_precision)
                    summary['overall_metrics']['category_overall_recall'] = float(cat_recall)
                    summary['overall_metrics']['category_overall_f1_score'] = float(cat_f1)

                # Mean per-sequence category metrics (macro) if present on sequences
                cat_precisions = [p.get('category_precision') for p in predictions_with_gt if 'category_precision' in p]
                cat_recalls = [p.get('category_recall') for p in predictions_with_gt if 'category_recall' in p]
                cat_f1s = [p.get('category_f1_score') for p in predictions_with_gt if 'category_f1_score' in p]
                if cat_precisions:
                    summary['overall_metrics']['category_mean_precision'] = float(np.mean(cat_precisions))
                if cat_recalls:
                    summary['overall_metrics']['category_mean_recall'] = float(np.mean(cat_recalls))
                if cat_f1s:
                    summary['overall_metrics']['category_mean_f1_score'] = float(np.mean(cat_f1s))

            # Occlusion split metrics aggregation (if present on sequences)
            occ_with = {'tp': 0, 'fp': 0, 'fn': 0}
            occ_without = {'tp': 0, 'fp': 0, 'fn': 0}
            occ_cat_with = {'tp': 0, 'fp': 0, 'fn': 0}
            occ_cat_without = {'tp': 0, 'fp': 0, 'fn': 0}
            for p in predictions_with_gt:
                occ = p.get('occlusion_metrics')
                if not occ:
                    continue
                wout = occ.get('without_occlusion') or {}
                wth = occ.get('with_occlusion') or {}
                occ_without['tp'] += int(wout.get('tp', 0))
                occ_without['fp'] += int(wout.get('fp', 0))
                occ_without['fn'] += int(wout.get('fn', 0))
                occ_with['tp'] += int(wth.get('tp', 0))
                occ_with['fp'] += int(wth.get('fp', 0))
                occ_with['fn'] += int(wth.get('fn', 0))

                occ_cat = p.get('occlusion_metrics_category')
                if occ_cat:
                    cwout = occ_cat.get('without_occlusion') or {}
                    cwth = occ_cat.get('with_occlusion') or {}
                    occ_cat_without['tp'] += int(cwout.get('tp', 0))
                    occ_cat_without['fp'] += int(cwout.get('fp', 0))
                    occ_cat_without['fn'] += int(cwout.get('fn', 0))
                    occ_cat_with['tp'] += int(cwth.get('tp', 0))
                    occ_cat_with['fp'] += int(cwth.get('fp', 0))
                    occ_cat_with['fn'] += int(cwth.get('fn', 0))

            def _prf(c):
                return calculate_detection_metrics(c['tp'], c['fp'], c['fn'])

            if any(v > 0 for v in occ_without.values()) or any(v > 0 for v in occ_with.values()):
                p0, r0, f0 = _prf(occ_without)
                p1, r1, f1 = _prf(occ_with)
                summary['overall_metrics']['occlusion'] = {
                    'without_occlusion': {
                        'tp': occ_without['tp'], 'fp': occ_without['fp'], 'fn': occ_without['fn'],
                        'precision': float(p0), 'recall': float(r0), 'f1_score': float(f0)
                    },
                    'with_occlusion': {
                        'tp': occ_with['tp'], 'fp': occ_with['fp'], 'fn': occ_with['fn'],
                        'precision': float(p1), 'recall': float(r1), 'f1_score': float(f1)
                    }
                }

            if any(v > 0 for v in occ_cat_without.values()) or any(v > 0 for v in occ_cat_with.values()):
                cp0, cr0, cf0 = _prf(occ_cat_without)
                cp1, cr1, cf1 = _prf(occ_cat_with)
                summary['overall_metrics']['occlusion_category'] = {
                    'without_occlusion': {
                        'tp': occ_cat_without['tp'], 'fp': occ_cat_without['fp'], 'fn': occ_cat_without['fn'],
                        'precision': float(cp0), 'recall': float(cr0), 'f1_score': float(cf0)
                    },
                    'with_occlusion': {
                        'tp': occ_cat_with['tp'], 'fp': occ_cat_with['fp'], 'fn': occ_cat_with['fn'],
                        'precision': float(cp1), 'recall': float(cr1), 'f1_score': float(cf1)
                    }
                }
        
        return summary
    
    def _generate_confusion_matrices(self, predictions: List[Dict], output_dir: Path):
        """Generate confusion matrices."""
        if not predictions or 'ground_truth_sequence' not in predictions[0]:
            return
        
        num_classes = 105
        
        all_gt = []
        all_pred = []
        for p in predictions:
            all_gt.extend(p['ground_truth_sequence'])
            all_pred.extend(p['predicted_sequence'])
        
        global_cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pred in zip(all_gt, all_pred):
            if 0 <= gt < num_classes and 0 <= pred < num_classes:
                global_cm[gt, pred] += 1
        
        cm_path = output_dir / 'confusion_matrix_global.json'
        with open(cm_path, 'w', encoding='utf-8') as f:
            json.dump(global_cm.tolist(), f)
        
        per_signer_cm = defaultdict(lambda: np.zeros((num_classes, num_classes), dtype=int))
        per_strategy_cm = defaultdict(lambda: np.zeros((num_classes, num_classes), dtype=int))
        
        for p in predictions:
            signer = p.get('signer')
            strategy = p.get('strategy')
            
            for gt, pred in zip(p['ground_truth_sequence'], p['predicted_sequence']):
                if 0 <= gt < num_classes and 0 <= pred < num_classes:
                    if signer:
                        per_signer_cm[signer][gt, pred] += 1
                    if strategy:
                        per_strategy_cm[strategy][gt, pred] += 1
        
        if per_signer_cm:
            signer_cm_data = {s: cm.tolist() for s, cm in per_signer_cm.items()}
            with open(output_dir / 'confusion_matrix_per_signer.json', 'w', encoding='utf-8') as f:
                json.dump(signer_cm_data, f)
        
        if per_strategy_cm:
            strategy_cm_data = {s: cm.tolist() for s, cm in per_strategy_cm.items()}
            with open(output_dir / 'confusion_matrix_per_strategy.json', 'w', encoding='utf-8') as f:
                json.dump(strategy_cm_data, f)


def main():
    parser = argparse.ArgumentParser(description='CTC Continuous Sign Language Prediction')
    
    parser.add_argument('--model', choices=['transformer_ctc', 'mediapipe_gru_ctc'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory with continuous sequence NPZ files')
    parser.add_argument('--ground-truth-dir', type=Path, help='Directory with ground truth JSON files')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory for predictions')
    parser.add_argument('--decode-method', choices=['greedy', 'beam_search'], default='greedy')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--fps', type=int, default=30, help='FPS for timestamp estimation')
    parser.add_argument('--temporal-tolerance', type=int, default=500, help='Tolerance in ms for temporal alignment')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CTC CONTINUOUS SIGN PREDICTION")
    print("=" * 80)
    print(f"Model:              {args.model}")
    print(f"Checkpoint:         {args.checkpoint}")
    print(f"Input directory:    {args.input_dir}")
    print(f"Ground truth dir:   {args.ground_truth_dir}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Device:             {device}")
    print(f"Decode method:      {args.decode_method}")
    print(f"FPS:                {args.fps}")
    print()
    
    predictor = CTCPredictor(args.model, str(args.checkpoint), device=device)
    
    results = predictor.predict_batch(
        args.input_dir,
        args.ground_truth_dir,
        args.output_dir,
        args.decode_method,
        args.beam_width,
        args.fps,
        args.temporal_tolerance
    )
    
    summary = results['summary']
    
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total sequences:    {summary['total_sequences']}")
    
    # Print detection metrics summary
    if 'overall_metrics' in summary:
        om = summary['overall_metrics']
        print(f"\n=== Detection Metrics Summary ===")
        print(f"Overall Precision:  {om['overall_precision']:.4f} ({om['overall_precision']*100:.2f}%)")
        print(f"Overall Recall:     {om['overall_recall']:.4f} ({om['overall_recall']*100:.2f}%)")
        print(f"Overall F1-Score:   {om['overall_f1_score']:.4f} ({om['overall_f1_score']*100:.2f}%)")
        print(f"Total TP:           {om['total_tp']}")
        print(f"Total FP:           {om['total_fp']}")
        print(f"Total FN:           {om['total_fn']}")
        # Mean IoU no longer reported in summary
        
        if summary.get('per_signer_metrics'):
            print(f"\nPer-signer F1-Score:")
            for signer, metrics in sorted(summary['per_signer_metrics'].items()):
                print(f"  {signer}: {metrics['f1_score']:.4f} (P:{metrics['precision']:.3f}, R:{metrics['recall']:.3f})")
        
        if summary.get('per_strategy_metrics'):
            print(f"\nPer-strategy F1-Score:")
            for strategy, metrics in sorted(summary['per_strategy_metrics'].items()):
                print(f"  {strategy}: {metrics['f1_score']:.4f} (P:{metrics['precision']:.3f}, R:{metrics['recall']:.3f})")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
