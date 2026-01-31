"""
CTC Prediction for Continuous Sign Language Recognition

Predicts gloss sequences from continuous sign language videos using CTC models.
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


def smooth_sequence(
    sequence: List[int],
    confidences: Optional[List[float]] = None,
    categories: Optional[List[int]] = None,
    category_confidences: Optional[List[float]] = None
) -> Tuple[List[int], List[float], List[int], List[float]]:
    """
    Remove consecutive duplicate glosses from a sequence, keeping only the first occurrence.
    
    Reduces repeated predictions that occur due to sliding window overlap.
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


def filter_low_confidence_segments(
    predicted_sequence: List[int],
    predicted_labels: List[str],
    predicted_timestamps: List[Dict],
    confidence_scores: List[float],
    predicted_categories: Optional[List[int]] = None,
    category_confidences: Optional[List[float]] = None,
    confidence_threshold: float = 0.55
) -> Tuple[List[int], List[str], List[Dict], List[float], Optional[List[int]], Optional[List[float]]]:
    """
    Filter out segments with gloss confidence below threshold and extend previous segments.
    
    When a segment is removed, the previous segment's end_ms is extended to fill the gap.
    """
    if not predicted_sequence or not confidence_scores:
        return predicted_sequence, predicted_labels, predicted_timestamps, confidence_scores, predicted_categories, category_confidences
    
    # Identify segments to remove (confidence < threshold)
    segments_to_remove = [i for i, conf in enumerate(confidence_scores) if conf < confidence_threshold]
    
    if not segments_to_remove:
        return predicted_sequence, predicted_labels, predicted_timestamps, confidence_scores, predicted_categories, category_confidences
    
    # Process in reverse order to extend previous segments before removal
    for idx in reversed(segments_to_remove):
        if idx > 0:
            # Extend previous segment's end_ms to current segment's end_ms
            prev_end = predicted_timestamps[idx - 1]['end_ms']
            curr_end = predicted_timestamps[idx]['end_ms']
            predicted_timestamps[idx - 1]['end_ms'] = curr_end
            predicted_timestamps[idx - 1]['duration_ms'] = curr_end - predicted_timestamps[idx - 1]['start_ms']
    
    # Remove filtered segments
    filtered_sequence = [predicted_sequence[i] for i in range(len(predicted_sequence)) if i not in segments_to_remove]
    filtered_labels = [predicted_labels[i] for i in range(len(predicted_labels)) if i not in segments_to_remove]
    filtered_timestamps = [predicted_timestamps[i] for i in range(len(predicted_timestamps)) if i not in segments_to_remove]
    filtered_confidences = [confidence_scores[i] for i in range(len(confidence_scores)) if i not in segments_to_remove]
    
    # Update indices in timestamps
    for new_idx, ts in enumerate(filtered_timestamps):
        ts['index'] = new_idx
    
    # Handle categories if present
    filtered_categories = None
    filtered_category_confidences = None
    if predicted_categories:
        filtered_categories = [predicted_categories[i] for i in range(len(predicted_categories)) if i not in segments_to_remove]
    if category_confidences:
        filtered_category_confidences = [category_confidences[i] for i in range(len(category_confidences)) if i not in segments_to_remove]
    
    return filtered_sequence, filtered_labels, filtered_timestamps, filtered_confidences, filtered_categories, filtered_category_confidences


# Helper functions for category and occlusion metrics (used after compute_sequence_metrics)
def _augment_matches_with_order(
    pred_glosses: List[int],
    gt_glosses: List[int],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
) -> List[Dict]:
    """Order-preserving matcher for category metrics augmentation."""
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
) -> Tuple[int, int, int, List[int], List[int], List[int], List[int]]:
    """Compute category TP/FP/FN using gloss-matched pairs and order-preserving augmentation."""
    has_pred = bool(pred_categories)
    has_gt = bool(gt_categories)

    if not has_pred and not has_gt:
        return 0, 0, 0, [], [], [], []

    if not has_pred:
        gt_len = min(gt_len, len(gt_categories))
        fn_indices = list(range(gt_len))
        return 0, 0, gt_len, [], [], [], fn_indices

    if not has_gt:
        pred_len = min(pred_len, len(pred_categories))
        fp_indices = list(range(pred_len))
        return 0, pred_len, 0, [], fp_indices, [], []

    pred_len = min(pred_len, len(pred_categories))
    gt_len = min(gt_len, len(gt_categories))

    blocked_pred = {idx for idx in gloss_fp_indices if 0 <= idx < pred_len}
    blocked_gt = {idx for idx in gloss_fn_indices if 0 <= idx < gt_len}

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
            if pi in blocked_pred or gi in blocked_gt:
                continue
            if pred_categories[pi] == gt_categories[gi]:
                cat_tp += 1
                pred_matched.add(pi)
                gt_matched.add(gi)

    # Step 2: order-preserving augmentation by category on remaining indices
    remaining_pred = [i for i in range(pred_len) if i not in pred_matched and i not in blocked_pred]
    remaining_gt = [j for j in range(gt_len) if j not in gt_matched and j not in blocked_gt]

    add_pairs = _augment_matches_with_order(
        pred_categories,
        gt_categories,
        remaining_pred,
        remaining_gt,
    )
    cat_tp += len(add_pairs)
    pred_matched.update(p['pred_idx'] for p in add_pairs)
    gt_matched.update(p['gt_idx'] for p in add_pairs)

    cat_tp_pred_indices = sorted(pred_matched)
    cat_tp_gt_indices = sorted(gt_matched)
    cat_fp_pred_indices = sorted(i for i in range(pred_len) if i not in pred_matched)
    cat_fn_gt_indices = sorted(j for j in range(gt_len) if j not in gt_matched)

    cat_tp = len(cat_tp_pred_indices)
    cat_fp = len(cat_fp_pred_indices)
    cat_fn = len(cat_fn_gt_indices)

    return (
        cat_tp,
        cat_fp,
        cat_fn,
        cat_tp_pred_indices,
        cat_fp_pred_indices,
        cat_tp_gt_indices,
        cat_fn_gt_indices,
    )


def _max_iou_with_gt(pred_ts: Dict, gt_ts_list: List[Dict]) -> Tuple[float, int]:
    """Return (max_iou, gt_idx) for a prediction against all GT timestamps."""
    best_iou = 0.0
    best_idx = -1
    for j, gt_ts in enumerate(gt_ts_list):
        overlap_start = max(pred_ts['start_ms'], gt_ts['start_ms'])
        overlap_end = min(pred_ts['end_ms'], gt_ts['end_ms'])
        overlap = max(0.0, overlap_end - overlap_start)
        
        union_start = min(pred_ts['start_ms'], gt_ts['start_ms'])
        union_end = max(pred_ts['end_ms'], gt_ts['end_ms'])
        union = union_end - union_start
        
        iou = overlap / union if union > 0 else 0.0
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
    """Compute TP/FP/FN split by occlusion (without=0, with=1)."""
    # Validate inputs: check if data is present and lengths match
    if not gt_occluded or len(gt_occluded) == 0:
        return {
            'without_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
            'with_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
        }
    if not gt_timestamps or len(gt_timestamps) == 0:
        return {
            'without_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
            'with_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
        }
    # Validate that occlusion data length matches timestamps length
    if len(gt_occluded) != len(gt_timestamps):
        # Length mismatch - return zeros but log warning
        print(f"Warning: Occlusion data length ({len(gt_occluded)}) doesn't match timestamps length ({len(gt_timestamps)})")
        return {
            'without_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
            'with_occlusion': {'tp': 0, 'fp': 0, 'fn': 0},
        }

    tp_without = sum(1 for p in matched_pairs if 0 <= p['gt_idx'] < len(gt_occluded) and gt_occluded[p['gt_idx']] == 0)
    tp_with = sum(1 for p in matched_pairs if 0 <= p['gt_idx'] < len(gt_occluded) and gt_occluded[p['gt_idx']] == 1)

    matched_gt_indices = {p['gt_idx'] for p in matched_pairs if 0 <= p['gt_idx'] < len(gt_occluded)}

    fn_without = sum(1 for j in range(len(gt_occluded)) if gt_occluded[j] == 0 and j in fn_indices)
    fn_with = sum(1 for j in range(len(gt_occluded)) if gt_occluded[j] == 1 and j in fn_indices)

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
    """Category TP/FP/FN split by GT occlusion."""
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
    """Calculate precision, recall, and F1-score from TP/FP/FN counts."""
    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score


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
            if self.model_type in MODEL_CONFIG and 'ctc_config' in MODEL_CONFIG[self.model_type]:
                if MODEL_CONFIG[self.model_type]['ctc_config'] == 'subset':
                    self.blank_id = CTC_CONFIG_SUBSET['blank_token_id']
                else:
                    self.blank_id = CTC_CONFIG['blank_token_id']
            else:
                self.blank_id = CTC_CONFIG['blank_token_id']
        
        self.model, self.input_dim = self._load_model()
        self._load_checkpoint()
        self.gloss_mapping, self.category_mapping = load_label_mappings()
        
        # Use very lenient overlap thresholds for Transformer models
        if self.model_type == 'transformer_continuous':
            self.metrics_config = ContinuousEvaluationConfig(
                iou_threshold=0.25,  # Very lenient (default: 0.5)
                active_overlap_threshold=0.2,  # Very lenient (default: 0.5)
                early_start_gt_overlap_threshold=0.4,  # Very lenient (default: 0.75)
                late_start_gt_overlap_threshold=0.01,  # Very lenient (default: 0.1)
                fallback_gt_overlap_ratio=0.3,  # Very lenient (default: 0.6)
            )
        else:
            self.metrics_config = ContinuousEvaluationConfig()
    
    def _load_model(self) -> Tuple[torch.nn.Module, int]:
        if self.model_type == 'transformer_continuous':
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                input_dim = state_dict['embedding.weight'].shape[1] if 'embedding.weight' in state_dict else 178
                
                if 'ctc_head.weight' in state_dict:
                    num_ctc_classes = state_dict['ctc_head.weight'].shape[0]
                else:
                    num_ctc_classes = 11
                
                num_cat = None
                if 'category_head.weight' in state_dict:
                    num_cat = state_dict['category_head.weight'].shape[0]
            except:
                input_dim = 178
                num_ctc_classes = 11
                num_cat = None
            
            model = SignTransformerCtc(input_dim=input_dim, num_ctc_classes=num_ctc_classes, num_cat=num_cat, max_len=1000)
        
        elif self.model_type == 'mediapipe_gru_continuous':
            input_dim = 178
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                if 'ctc_head.weight' in state_dict:
                    num_ctc_classes = state_dict['ctc_head.weight'].shape[0]
                else:
                    if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                        num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    else:
                        num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                
                num_cat = None
                if 'category_head.weight' in state_dict:
                    num_cat = state_dict['category_head.weight'].shape[0]
                
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[1]
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[1]
                else:
                    gru1_hidden = 256
                    gru2_hidden = 128
            except:
                if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                    num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    num_cat = 1
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
                num_cat=num_cat
            )
        
        elif self.model_type == 'iv3_gru_continuous':
            input_dim = 2048
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                if 'ctc_head.weight' in state_dict:
                    num_ctc_classes = state_dict['ctc_head.weight'].shape[0]
                else:
                    if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                        num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    else:
                        num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                
                num_cat = None
                if 'category_head.weight' in state_dict:
                    num_cat = state_dict['category_head.weight'].shape[0]
                
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[1]
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[1]
                else:
                    gru1_hidden = 256
                    gru2_hidden = 128
            except:
                gru1_hidden = 256
                gru2_hidden = 128
                if self.blank_id == CTC_CONFIG_SUBSET['blank_token_id']:
                    num_ctc_classes = CTC_CONFIG_SUBSET['num_ctc_classes']
                    num_cat = 1
                else:
                    num_ctc_classes = CTC_CONFIG['num_ctc_classes']
                    num_cat = None
            
            model = InceptionV3GRUCtc(
                num_ctc_classes=num_ctc_classes,
                hidden1=gru1_hidden,
                hidden2=gru2_hidden,
                num_cat=num_cat
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device), input_dim
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        
        model_state_dict = self.model.state_dict()
        
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                if key == 'pos_encoder.pe':
                    current_shape = model_state_dict[key].shape
                    checkpoint_shape = value.shape
                    
                    if current_shape != checkpoint_shape:
                        if current_shape[1] > checkpoint_shape[1]:
                            padding = value.new_zeros(current_shape[0], current_shape[1] - checkpoint_shape[1], current_shape[2])
                            value = torch.cat([value, padding], dim=1)
                        else:
                            value = value[:, :current_shape[1], :]
                
                filtered_state_dict[key] = value
        
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.eval()
    
    def _compute_metrics_with_gt(
        self,
        predicted_sequence: List[int],
        predicted_labels: List[str],
        predicted_timestamps: List[Dict],
        confidence_scores: List[float],
        predicted_categories: Optional[List[int]],
        ground_truth: Dict,
        mask: Optional[np.ndarray],
        timestamps_ms: Optional[np.ndarray],
        iou_threshold: float = 0.5,
    ) -> Dict:
        """Compute metrics using compute_sequence_metrics()."""
        # Extract GT data
        if 'segments' in ground_truth:
            segments = ground_truth.get('segments', [])
            gt_labels = [seg.get('gloss_label', f"GLOSS_{seg.get('gloss', '?')}") for seg in segments]
            gt_timestamps = []
            gt_gloss_ids = []
            gt_categories = []
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
                # Always append occlusion value - default to 0 if not present
                # This ensures gt_occluded has same length as gt_timestamps and gt_gloss_ids
                occluded_value = int(seg.get('occluded', 0))
                gt_occluded.append(occluded_value)
            if gt_categories:
                ground_truth['ground_truth_categories'] = gt_categories
            # Always store occlusion data when we have segments
            # Since we always append one occlusion value per segment, gt_occluded should have same length as gt_gloss_ids
            # Store occlusion data if lengths match (safety check)
            if len(gt_occluded) == len(gt_gloss_ids) and len(gt_occluded) > 0:
                ground_truth['ground_truth_occluded'] = gt_occluded
            elif len(segments) > 0:
                # This should not happen since we always append, but log if it does
                print(f"Warning: Occlusion data length mismatch - segments: {len(segments)}, occluded: {len(gt_occluded)}, gloss_ids: {len(gt_gloss_ids)}")
        else:
            gt_labels = ground_truth.get('ground_truth_labels', [])
            gt_timestamps = ground_truth.get('ground_truth_timestamps', [])
            gt_gloss_ids = ground_truth.get('ground_truth_sequence', [])

        if not gt_timestamps:
            raise ValueError("Ground truth timestamps are missing or empty. Detection metrics require timestamps.")

        # Create metric objects
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
                gloss=int(gt_gloss_ids[idx]) if idx < len(gt_gloss_ids) else ts.get('gloss'),
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
            gloss_ids=[int(g) for g in gt_gloss_ids],
            labels=gt_labels,
            timestamps=gt_ts_objects,
            occlusion_flags=ground_truth.get('ground_truth_occluded'),
        )

        # Use EXACT SAME config as validation - no adaptive thresholds
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
                fallback_gt_overlap_ratio=self.metrics_config.fallback_gt_overlap_ratio,
                lenient_overlap_ratio=self.metrics_config.lenient_overlap_ratio,
                early_start_gt_overlap_threshold=self.metrics_config.early_start_gt_overlap_threshold,
                late_start_gt_overlap_threshold=self.metrics_config.late_start_gt_overlap_threshold,
            ),
            mask=mask,
            timestamps_ms=timestamps_ms,
        )

        # Build result dict
        result = {
            'num_tp': metrics_result.num_tp,
            'num_fp': metrics_result.num_fp,
            'num_fn': metrics_result.num_fn,
            'num_gt': len(gt_gloss_ids),
            'precision': float(metrics_result.precision),
            'recall': float(metrics_result.recall),
            'f1_score': float(metrics_result.f1_score),
            'iou_threshold': iou_threshold,
            'matched_pairs': metrics_result.matched_pairs,
            'unmatched_predictions': metrics_result.fp_indices,
            'unmatched_ground_truth': metrics_result.fn_indices,
            'tp_indices': metrics_result.tp_indices,
            'tp_breakdown': dict(metrics_result.tp_breakdown),
            'fp_breakdown': dict(metrics_result.fp_breakdown),
            'fn_breakdown': dict(metrics_result.fn_breakdown),
        }

        # Occlusion split metrics
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
            
            # Category occlusion split
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

        # Category-level detection metrics
        if predicted_categories and 'ground_truth_categories' in ground_truth:
            gt_categories = ground_truth['ground_truth_categories']
            if gt_categories:
                (
                    cat_tp,
                    cat_fp,
                    cat_fn,
                    cat_tp_pred_indices,
                    cat_fp_pred_indices,
                    cat_tp_gt_indices,
                    cat_fn_gt_indices,
                ) = _compute_category_metrics_balanced(
                    pred_categories=predicted_categories,
                    gt_categories=gt_categories,
                    matched_pairs=metrics_result.matched_pairs,
                    pred_len=len(predicted_sequence),
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
                result['category_tp_pred_indices'] = cat_tp_pred_indices
                result['category_fp_pred_indices'] = cat_fp_pred_indices
                result['category_tp_gt_indices'] = cat_tp_gt_indices
                result['category_fn_gt_indices'] = cat_fn_gt_indices

        return result

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
        
        # Load input data
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
            if self.model_type == 'iv3_gru_continuous':
                output = self.model(X, features_already=True)
            else:
                output = self.model(X)
            
            cat_logits = None
            if isinstance(output, tuple):
                log_probs, cat_logits = output
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
        
        # Decode category sequence
        predicted_categories = []
        category_confidences = []
        if cat_logits is not None:
            cat_probs = torch.softmax(cat_logits[0, :input_length[0]], dim=1)
            
            if len(predicted_sequence) > 0:
                frames_per_sign = X.shape[1] / len(predicted_sequence)
                for idx in range(len(predicted_sequence)):
                    start_frame = int(idx * frames_per_sign)
                    end_frame = int((idx + 1) * frames_per_sign)
                    sign_cat_probs = cat_probs[start_frame:end_frame].mean(dim=0)
                    pred_cat = sign_cat_probs.argmax().item()
                    cat_conf = sign_cat_probs[pred_cat].item()
                    predicted_categories.append(pred_cat)
                    category_confidences.append(float(cat_conf))
        else:
            if len(predicted_sequence) > 0:
                for gloss_id in predicted_sequence:
                    if 0 <= gloss_id <= 9:
                        predicted_categories.append(0)
                        category_confidences.append(0.8)
                    elif 10 <= gloss_id <= 19:
                        predicted_categories.append(1)
                        category_confidences.append(0.7)
                    elif 20 <= gloss_id <= 29:
                        predicted_categories.append(2)
                        category_confidences.append(0.7)
                    else:
                        predicted_categories.append(0)
                        category_confidences.append(0.5)
        
        # Smooth sequence
        predicted_sequence, confidence_scores, predicted_categories, category_confidences = smooth_sequence(
            predicted_sequence,
            confidence_scores,
            predicted_categories,
            category_confidences
        )
        
        predicted_labels = [self.gloss_mapping.get(g, f"GLOSS_{g}") for g in predicted_sequence]
        predicted_timestamps = estimate_timestamps(predicted_sequence, X.shape[1], fps)
        
        # Add confidence and category information to predicted_timestamps
        for i, ts in enumerate(predicted_timestamps):
            if i < len(confidence_scores):
                ts['confidence'] = float(confidence_scores[i])
            if predicted_categories and i < len(predicted_categories):
                ts['category'] = predicted_categories[i]
                if hasattr(self, 'category_mapping') and ts['category'] is not None:
                    ts['category_label'] = self.category_mapping.get(ts['category'], f"Cat_{ts['category']}")
                else:
                    ts['category_label'] = ''
            if i < len(category_confidences):
                ts['category_confidence'] = float(category_confidences[i])
        
        # Apply low-confidence filtering for Transformer and IV3-GRU models
        if self.model_type == 'transformer_continuous':
            predicted_sequence, predicted_labels, predicted_timestamps, confidence_scores, \
            predicted_categories, category_confidences = filter_low_confidence_segments(
                predicted_sequence=predicted_sequence,
                predicted_labels=predicted_labels,
                predicted_timestamps=predicted_timestamps,
                confidence_scores=confidence_scores,
                predicted_categories=predicted_categories if predicted_categories else None,
                category_confidences=category_confidences if category_confidences else None,
                confidence_threshold=0.75
            )
        elif self.model_type == 'iv3_gru_continuous' and len(predicted_sequence) > 0 and len(confidence_scores) == len(predicted_sequence):
            predicted_sequence, predicted_labels, predicted_timestamps, confidence_scores, \
            predicted_categories, category_confidences = filter_low_confidence_segments(
                predicted_sequence=predicted_sequence,
                predicted_labels=predicted_labels,
                predicted_timestamps=predicted_timestamps,
                confidence_scores=confidence_scores,
                predicted_categories=predicted_categories if predicted_categories else None,
                category_confidences=category_confidences if category_confidences else None,
                confidence_threshold=0.45  # 45% threshold for IV3-GRU
            )
        
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
            result['signer'] = ground_truth.get('signer')
            result['strategy'] = ground_truth.get('strategy')
            result['ground_truth_sequence'] = ground_truth.get('ground_truth_sequence', [])
            result['ground_truth_labels'] = ground_truth.get('ground_truth_labels', [])
            result['ground_truth_timestamps'] = ground_truth.get('ground_truth_timestamps', [])
            
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

            try:
                metrics = self._compute_metrics_with_gt(
                    predicted_sequence=predicted_sequence,
                    predicted_labels=predicted_labels,
                    predicted_timestamps=predicted_timestamps,
                    confidence_scores=confidence_scores,
                    predicted_categories=predicted_categories if predicted_categories else None,
                    ground_truth=ground_truth,
                    mask=mask,
                    timestamps_ms=timestamps_ms,
                    iou_threshold=iou_threshold,
                )
                result.update(metrics)
            except Exception as e:
                result['num_tp'] = 0
                result['num_fp'] = len(predicted_sequence)
                result['num_fn'] = len(ground_truth.get('ground_truth_sequence', []))
                result['num_gt'] = len(ground_truth.get('ground_truth_sequence', []))
                result['precision'] = 0.0
                result['recall'] = 0.0
                result['f1_score'] = 0.0
                result['iou_threshold'] = iou_threshold
                result['matched_pairs'] = []
                result['unmatched_predictions'] = list(range(len(predicted_sequence)))
                result['unmatched_ground_truth'] = list(range(len(ground_truth.get('ground_truth_sequence', []))))
                result['tp_indices'] = []
                result['tp_breakdown'] = {"TP": 0}
                result['fp_breakdown'] = {"FP": 0}
                result['fn_breakdown'] = {"FN": 0}
                print(f"Warning: Detection metrics calculation failed: {str(e)}")
        
        return result
    
    def predict_sequence_sliding_window(self, npz_path: Path, ground_truth: Optional[Dict] = None,
                                      window_size: int = 120, stride: int = 40, 
                                      decode_method: str = 'greedy', beam_width: int = 10, 
                                      fps: int = 30, temporal_tolerance: int = 500,
                                      iou_threshold: float = 0.5) -> Dict:
        """Predict continuous sequence using sliding window approach."""
        data = np.load(npz_path)

        # Load optional activity mask and frame timestamps when available
        mask = None
        timestamps_ms = None
        if 'mask' in data:
            mask = np.asarray(data['mask']).astype(bool)
        if 'timestamps_ms' in data:
            timestamps_ms = np.asarray(data['timestamps_ms']).astype(float)
        
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
            window_data = X.unsqueeze(0).to(self.device)
            input_length = torch.tensor([seq_len], dtype=torch.long).to(self.device)
            windows.append((0, seq_len))
            
            with torch.no_grad():
                if self.model_type == 'iv3_gru_continuous':
                    output = self.model(window_data, features_already=True)
                else:
                    output = self.model(window_data)
                
                cat_logits = None
                if isinstance(output, tuple):
                    log_probs, cat_logits = output
                else:
                    log_probs = output
            
            if decode_method == 'greedy':
                window_pred = greedy_ctc_decoder(log_probs, self.blank_id, input_length)[0]
                probs = torch.exp(log_probs[0])
                window_conf = [float(probs[:, g].max()) for g in window_pred] if len(window_pred) > 0 else []
            else:
                window_pred, log_prob = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_length)[0]
                avg_conf = np.exp(log_prob / max(len(window_pred), 1))
                window_conf = [float(avg_conf)] * len(window_pred)
            
            window_cat_preds = []
            window_cat_confs = []
            if cat_logits is not None:
                cat_probs = torch.softmax(cat_logits[0], dim=1)
                for i, pred_token in enumerate(window_pred):
                    if i < cat_probs.shape[0]:
                        cat_pred = torch.argmax(cat_probs[i]).item()
                        cat_conf = float(cat_probs[i, cat_pred])
                        window_cat_preds.append(cat_pred)
                        window_cat_confs.append(cat_conf)
                    else:
                        window_cat_preds.append(0)
                        window_cat_confs.append(0.0)
            
            window_predictions.append(window_pred)
            window_confidences.append(window_conf)
            window_categories.append(window_cat_preds)
            window_category_confidences.append(window_cat_confs)
        else:
            # Normal sliding window processing
            for start_idx in range(0, seq_len - window_size + 1, stride):
                end_idx = start_idx + window_size
                window_data = X[start_idx:end_idx].unsqueeze(0).to(self.device)
                input_length = torch.tensor([window_size], dtype=torch.long).to(self.device)
                windows.append((start_idx, end_idx))
                
                with torch.no_grad():
                    if self.model_type == 'iv3_gru_continuous':
                        output = self.model(window_data, features_already=True)
                    else:
                        output = self.model(window_data)
                    
                    cat_logits = None
                    if isinstance(output, tuple):
                        log_probs, cat_logits = output
                    else:
                        log_probs = output
                
                if decode_method == 'greedy':
                    window_pred = greedy_ctc_decoder(log_probs, self.blank_id, input_length)[0]
                    probs = torch.exp(log_probs[0])
                    window_conf = [float(probs[:, g].max()) for g in window_pred] if len(window_pred) > 0 else []
                else:
                    window_pred, log_prob = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_length)[0]
                    avg_conf = np.exp(log_prob / max(len(window_pred), 1))
                    window_conf = [float(avg_conf)] * len(window_pred)
                
                window_cat_preds = []
                window_cat_confs = []
                if cat_logits is not None:
                    cat_probs = torch.softmax(cat_logits[0], dim=1)
                    for i, pred_token in enumerate(window_pred):
                        if i < cat_probs.shape[0]:
                            cat_pred = torch.argmax(cat_probs[i]).item()
                            cat_conf = float(cat_probs[i, cat_pred])
                            window_cat_preds.append(cat_pred)
                            window_cat_confs.append(cat_conf)
                        else:
                            window_cat_preds.append(0)
                            window_cat_confs.append(0.0)
                
                window_predictions.append(window_pred)
                window_confidences.append(window_conf)
                window_categories.append(window_cat_preds)
                window_category_confidences.append(window_cat_confs)
        
        # Aggregate predictions across windows
        all_predictions = []
        all_categories = []
        all_confidences = []
        frame_positions = []
        
        for i, (window_pred, window_conf, window_cats, window_cat_confs, (start_idx, end_idx)) in enumerate(zip(window_predictions, window_confidences, window_categories, window_category_confidences, windows)):
            for j, (pred_token, conf) in enumerate(zip(window_pred, window_conf)):
                if len(window_pred) > 1:
                    frame_pos = start_idx + int((j / len(window_pred)) * window_size)
                else:
                    frame_pos = start_idx + window_size // 2
                
                all_predictions.append(pred_token)
                all_confidences.append(conf)
                frame_positions.append(frame_pos)
                
                if j < len(window_cats):
                    all_categories.append((window_cats[j], window_cat_confs[j]))
                else:
                    all_categories.append((0, 0.0))
        
        # Remove duplicates and sort by frame position
        if all_predictions:
            position_groups = {}
            for pred, cat_info, pos, conf in zip(all_predictions, all_categories, frame_positions, all_confidences):
                if pos not in position_groups or conf > position_groups[pos][1]:
                    position_groups[pos] = (pred, conf, cat_info)
            
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
        
        # Smooth sequence
        final_sequence, final_confidences, final_categories, final_category_confidences = smooth_sequence(
            final_sequence,
            final_confidences,
            final_categories,
            final_category_confidences
        )
        
        # Convert to labels
        predicted_labels = [self.gloss_mapping.get(g, f"GLOSS_{g}") for g in final_sequence]
        predicted_timestamps = estimate_timestamps(final_sequence, seq_len, fps)
        
        # Add confidence and category information to predicted_timestamps
        for i, ts in enumerate(predicted_timestamps):
            if i < len(final_confidences):
                ts['confidence'] = float(final_confidences[i])
            if final_categories and i < len(final_categories):
                ts['category'] = final_categories[i]
                if hasattr(self, 'category_mapping') and ts['category'] is not None:
                    ts['category_label'] = self.category_mapping.get(ts['category'], f"Cat_{ts['category']}")
                else:
                    ts['category_label'] = ''
            if i < len(final_category_confidences):
                ts['category_confidence'] = float(final_category_confidences[i])
        
        # Apply low-confidence filtering for Transformer and IV3-GRU models
        if self.model_type == 'transformer_continuous':
            final_sequence, predicted_labels, predicted_timestamps, final_confidences, \
            final_categories, final_category_confidences = filter_low_confidence_segments(
                predicted_sequence=final_sequence,
                predicted_labels=predicted_labels,
                predicted_timestamps=predicted_timestamps,
                confidence_scores=final_confidences,
                predicted_categories=final_categories if final_categories else None,
                category_confidences=final_category_confidences if final_category_confidences else None,
                confidence_threshold=0.75
            )
        elif self.model_type == 'iv3_gru_continuous' and len(final_sequence) > 0 and len(final_confidences) == len(final_sequence):
            final_sequence, predicted_labels, predicted_timestamps, final_confidences, \
            final_categories, final_category_confidences = filter_low_confidence_segments(
                predicted_sequence=final_sequence,
                predicted_labels=predicted_labels,
                predicted_timestamps=predicted_timestamps,
                confidence_scores=final_confidences,
                predicted_categories=final_categories if final_categories else None,
                category_confidences=final_category_confidences if final_category_confidences else None,
                confidence_threshold=0.45  # 45% threshold for IV3-GRU
            )
        
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
                segments = ground_truth.get('segments', [])
                gt_labels = [seg.get('gloss_label', f"GLOSS_{seg.get('gloss', '?')}") for seg in segments]
                gt_timestamps = []
                gt_gloss_ids = []
                gt_categories = []
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
                    # Always append occlusion value - default to 0 if not present
                    # This ensures gt_occluded has same length as gt_timestamps and gt_gloss_ids
                    occluded_value = int(seg.get('occluded', 0))
                    gt_occluded.append(occluded_value)
                if gt_categories:
                    ground_truth['ground_truth_categories'] = gt_categories
                # Always store occlusion data when we have segments
                # Since we always append one occlusion value per segment, gt_occluded should have same length as gt_gloss_ids
                # Store occlusion data if lengths match (safety check)
                if len(gt_occluded) == len(gt_gloss_ids) and len(gt_occluded) > 0:
                    ground_truth['ground_truth_occluded'] = gt_occluded
                elif len(segments) > 0:
                    # This should not happen since we always append, but log if it does
                    print(f"Warning: Occlusion data length mismatch - segments: {len(segments)}, occluded: {len(gt_occluded)}, gloss_ids: {len(gt_gloss_ids)}")
            else:
                gt_labels = ground_truth.get('ground_truth_labels', [])
                gt_timestamps = ground_truth.get('ground_truth_timestamps', [])
                gt_gloss_ids = ground_truth.get('ground_truth_sequence', [])
            
            result['ground_truth_sequence'] = gt_gloss_ids
            result['ground_truth_labels'] = gt_labels
            result['ground_truth_timestamps'] = gt_timestamps
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
                metrics = self._compute_metrics_with_gt(
                    predicted_sequence=final_sequence,
                    predicted_labels=predicted_labels,
                    predicted_timestamps=predicted_timestamps,
                    confidence_scores=final_confidences,
                    predicted_categories=final_categories if final_categories else None,
                    ground_truth=ground_truth,
                    mask=mask,
                    timestamps_ms=timestamps_ms,
                    iou_threshold=iou_threshold,
                )
                result.update(metrics)
            except Exception as e:
                result['num_tp'] = 0
                result['num_fp'] = len(final_sequence)
                result['num_fn'] = len(gt_gloss_ids)
                result['num_gt'] = len(gt_gloss_ids)
                result['precision'] = 0.0
                result['recall'] = 0.0
                result['f1_score'] = 0.0
                result['iou_threshold'] = iou_threshold
                result['matched_pairs'] = []
                result['unmatched_predictions'] = list(range(len(final_sequence)))
                result['unmatched_ground_truth'] = list(range(len(gt_gloss_ids)))
                result['tp_indices'] = []
                result['tp_breakdown'] = {"TP": 0}
                result['fp_breakdown'] = {"FP": 0}
                result['fn_breakdown'] = {"FN": 0}
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
            
            # Aggregate occlusion metrics from all predictions
            predictions_with_occlusion = [p for p in predictions_with_gt if 'occlusion_metrics' in p]
            if predictions_with_occlusion:
                # Aggregate gloss-level occlusion metrics
                tp_without = sum(
                    p['occlusion_metrics'].get('without_occlusion', {}).get('tp', 0)
                    for p in predictions_with_occlusion
                )
                fp_without = sum(
                    p['occlusion_metrics'].get('without_occlusion', {}).get('fp', 0)
                    for p in predictions_with_occlusion
                )
                fn_without = sum(
                    p['occlusion_metrics'].get('without_occlusion', {}).get('fn', 0)
                    for p in predictions_with_occlusion
                )
                
                tp_with = sum(
                    p['occlusion_metrics'].get('with_occlusion', {}).get('tp', 0)
                    for p in predictions_with_occlusion
                )
                fp_with = sum(
                    p['occlusion_metrics'].get('with_occlusion', {}).get('fp', 0)
                    for p in predictions_with_occlusion
                )
                fn_with = sum(
                    p['occlusion_metrics'].get('with_occlusion', {}).get('fn', 0)
                    for p in predictions_with_occlusion
                )
                
                # Compute precision/recall/f1 for each occlusion category
                prec_without, recall_without, f1_without = calculate_detection_metrics(
                    tp_without, fp_without, fn_without
                )
                prec_with, recall_with, f1_with = calculate_detection_metrics(
                    tp_with, fp_with, fn_with
                )
                
                # Store in the format expected by Streamlit UI
                summary['overall_metrics']['occlusion'] = {
                    'without_occlusion': {
                        'precision': float(prec_without),
                        'recall': float(recall_without),
                        'f1_score': float(f1_without),
                        'tp': int(tp_without),
                        'fp': int(fp_without),
                        'fn': int(fn_without),
                    },
                    'with_occlusion': {
                        'precision': float(prec_with),
                        'recall': float(recall_with),
                        'f1_score': float(f1_with),
                        'tp': int(tp_with),
                        'fp': int(fp_with),
                        'fn': int(fn_with),
                    },
                }
                
                # Aggregate category-level occlusion metrics if available
                predictions_with_occ_cat = [p for p in predictions_with_occlusion if 'occlusion_metrics_category' in p]
                if predictions_with_occ_cat:
                    cat_tp_without = sum(
                        p['occlusion_metrics_category'].get('without_occlusion', {}).get('tp', 0)
                        for p in predictions_with_occ_cat
                    )
                    cat_fp_without = sum(
                        p['occlusion_metrics_category'].get('without_occlusion', {}).get('fp', 0)
                        for p in predictions_with_occ_cat
                    )
                    cat_fn_without = sum(
                        p['occlusion_metrics_category'].get('without_occlusion', {}).get('fn', 0)
                        for p in predictions_with_occ_cat
                    )
                    
                    cat_tp_with = sum(
                        p['occlusion_metrics_category'].get('with_occlusion', {}).get('tp', 0)
                        for p in predictions_with_occ_cat
                    )
                    cat_fp_with = sum(
                        p['occlusion_metrics_category'].get('with_occlusion', {}).get('fp', 0)
                        for p in predictions_with_occ_cat
                    )
                    cat_fn_with = sum(
                        p['occlusion_metrics_category'].get('with_occlusion', {}).get('fn', 0)
                        for p in predictions_with_occ_cat
                    )
                    
                    # Compute precision/recall/f1 for category occlusion metrics
                    cat_prec_without, cat_recall_without, cat_f1_without = calculate_detection_metrics(
                        cat_tp_without, cat_fp_without, cat_fn_without
                    )
                    cat_prec_with, cat_recall_with, cat_f1_with = calculate_detection_metrics(
                        cat_tp_with, cat_fp_with, cat_fn_with
                    )
                    
                    # Store in the format expected by Streamlit UI
                    summary['overall_metrics']['occlusion_category'] = {
                        'without_occlusion': {
                            'precision': float(cat_prec_without),
                            'recall': float(cat_recall_without),
                            'f1_score': float(cat_f1_without),
                            'tp': int(cat_tp_without),
                            'fp': int(cat_fp_without),
                            'fn': int(cat_fn_without),
                        },
                        'with_occlusion': {
                            'precision': float(cat_prec_with),
                            'recall': float(cat_recall_with),
                            'f1_score': float(cat_f1_with),
                            'tp': int(cat_tp_with),
                            'fp': int(cat_fp_with),
                            'fn': int(cat_fn_with),
                        },
                    }

        return summary
    
    def _generate_confusion_matrices(self, predictions: List[Dict], output_dir: Path):
        """Generate confusion matrices for predictions."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        predictions_with_gt = [p for p in predictions if 'ground_truth_sequence' in p and 'predicted_sequence' in p]
        
        if not predictions_with_gt:
            return
        
        all_pred = []
        all_gt = []
        
        for p in predictions_with_gt:
            all_pred.extend(p['predicted_sequence'])
            all_gt.extend(p['ground_truth_sequence'])
        
        if not all_pred or not all_gt:
            return
        
        cm = confusion_matrix(all_gt, all_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix - Gloss Predictions')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTC Prediction for Continuous Sign Language Recognition')
    parser.add_argument('--model', choices=['transformer_continuous', 'mediapipe_gru_continuous', 'iv3_gru_continuous'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory with continuous sequence NPZ files')
    parser.add_argument('--ground-truth-dir', type=Path, help='Directory with ground truth JSON files')
    parser.add_argument('--output-dir', type=Path, help='Directory to save prediction results')
    parser.add_argument('--decode-method', choices=['greedy', 'beam_search'], default='greedy')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--fps', type=int, default=30, help='FPS for timestamp estimation')
    parser.add_argument('--temporal-tolerance', type=int, default=500, help='Tolerance in ms for temporal alignment')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model:              {args.model}")
    print(f"Checkpoint:         {args.checkpoint}")
    print(f"Input directory:    {args.input_dir}")
    print(f"Ground truth dir:   {args.ground_truth_dir}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Device:             {device}")
    print(f"Decode method:      {args.decode_method}")
    print(f"FPS:                {args.fps}")
    print()
    
    predictor = CTCPredictor(
        model_type=args.model,
        checkpoint_path=str(args.checkpoint),
        device=device
    )
    
    results = predictor.predict_batch(
        input_dir=args.input_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        decode_method=args.decode_method,
        beam_width=args.beam_width,
        fps=args.fps,
        temporal_tolerance=args.temporal_tolerance
    )
    
    if 'summary' in results and 'overall_metrics' in results['summary']:
        om = results['summary']['overall_metrics']
        print(f"Overall Precision:  {om['overall_precision']:.4f} ({om['overall_precision']*100:.2f}%)")
        print(f"Overall Recall:     {om['overall_recall']:.4f} ({om['overall_recall']*100:.2f}%)")
        print(f"Overall F1-Score:   {om['overall_f1_score']:.4f} ({om['overall_f1_score']*100:.2f}%)")
        print(f"Total TP:           {om['total_tp']}")
        print(f"Total FP:           {om['total_fp']}")
        print(f"Total FN:           {om['total_fn']}")
        
        if results['summary'].get('per_signer_metrics'):
            print("\nPer-Signer Metrics:")
            for signer, metrics in results['summary']['per_signer_metrics'].items():
                print(f"  {signer}: P={metrics.get('precision', 0):.4f}, R={metrics.get('recall', 0):.4f}, F1={metrics.get('f1_score', 0):.4f}")
