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
from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder, calculate_wer
from streamlit_app.core.config import CTC_CONFIG
from data.labels.label_mapping import load_label_mappings


def load_ground_truth_json(json_path: Path) -> Dict:
    """Load ground truth from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_wer_with_details(reference: List[int], hypothesis: List[int]) -> Tuple[float, int, int, int]:
    """
    Calculate WER with detailed error breakdown.
    
    Returns:
        Tuple of (wer, num_insertions, num_deletions, num_substitutions)
    """
    if len(reference) == 0:
        return (0.0 if len(hypothesis) == 0 else float('inf'), 0, 0, len(hypothesis))
    
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    ops = [[None] * (hyp_len + 1) for _ in range(ref_len + 1)]
    
    for i in range(ref_len + 1):
        dp[i][0] = i
        ops[i][0] = 'D'
    for j in range(hyp_len + 1):
        dp[0][j] = j
        ops[0][j] = 'I'
    ops[0][0] = None
    
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'M'
            else:
                sub_cost = dp[i-1][j-1] + 1
                del_cost = dp[i-1][j] + 1
                ins_cost = dp[i][j-1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                
                if min_cost == sub_cost:
                    ops[i][j] = 'S'
                elif min_cost == del_cost:
                    ops[i][j] = 'D'
                else:
                    ops[i][j] = 'I'
    
    i, j = ref_len, hyp_len
    insertions = deletions = substitutions = 0
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'M':
            i -= 1
            j -= 1
        elif op == 'S':
            substitutions += 1
            i -= 1
            j -= 1
        elif op == 'D':
            deletions += 1
            i -= 1
        elif op == 'I':
            insertions += 1
            j -= 1
        else:
            break
    
    wer = dp[ref_len][hyp_len] / ref_len
    return wer, insertions, deletions, substitutions


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


class CTCPredictor:
    """CTC-based sign language recognition predictor."""
    
    def __init__(self, model_type: str, checkpoint_path: str, blank_id: Optional[int] = None, device: Optional[torch.device] = None):
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.blank_id = blank_id if blank_id is not None else CTC_CONFIG['blank_token_id']
        
        self.model, self.input_dim = self._load_model()
        self._load_checkpoint()
        self.gloss_mapping, _ = load_label_mappings()
    
    def _load_model(self) -> Tuple[torch.nn.Module, int]:
        if self.model_type == 'transformer_ctc':
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                input_dim = state_dict['embedding.weight'].shape[1] if 'embedding.weight' in state_dict else 156
            except:
                input_dim = 156
            
            model = SignTransformerCtc(input_dim=input_dim, num_ctc_classes=CTC_CONFIG['num_ctc_classes'])
        
        elif self.model_type == 'mediapipe_gru_ctc':
            input_dim = 156
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3 // 2
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3 // 2
                else:
                    gru1_hidden = 256
                    gru2_hidden = 128
            except:
                gru1_hidden = 256
                gru2_hidden = 128
            
            model = MediaPipeGRUCtc(input_dim=input_dim, num_ctc_classes=CTC_CONFIG['num_ctc_classes'],
                                   hidden1=gru1_hidden, hidden2=gru2_hidden)
        
        elif self.model_type == 'iv3_gru_ctc':
            # InceptionV3GRUCtc uses 2048-D features
            input_dim = 2048
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                # Try to extract hidden dimensions from checkpoint
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    # For bidirectional GRU: weight_hh_l0 has shape [3*hidden_size*2, hidden_size]
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3 // 2
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3 // 2
                else:
                    gru1_hidden = 256
                    gru2_hidden = 128
            except:
                gru1_hidden = 256
                gru2_hidden = 128
            
            model = InceptionV3GRUCtc(
                num_ctc_classes=CTC_CONFIG['num_ctc_classes'],
                hidden1=gru1_hidden,
                hidden2=gru2_hidden
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device), input_dim
    
    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def predict_sequence(
        self,
        npz_path: Path,
        ground_truth: Optional[Dict] = None,
        decode_method: str = 'greedy',
        beam_width: int = 10,
        fps: int = 30,
        temporal_tolerance: int = 500
    ) -> Dict:
        """Predict single continuous sequence with full metrics."""
        data = np.load(npz_path)
        
        if self.input_dim == 2048:
            if 'X2048' not in data:
                raise ValueError(f"NPZ file missing 'X2048' key")
            X = torch.from_numpy(data['X2048']).float().unsqueeze(0)
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
        
        predicted_labels = [self.gloss_mapping.get(g, f"GLOSS_{g}") for g in predicted_sequence]
        predicted_timestamps = estimate_timestamps(predicted_sequence, X.shape[1], fps)
        
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
            
            # Ground truth categories if available
            if 'ground_truth_categories' in ground_truth:
                result['ground_truth_categories'] = ground_truth['ground_truth_categories']
            
            wer, insertions, deletions, substitutions = calculate_wer_with_details(
                ground_truth['ground_truth_sequence'],
                predicted_sequence
            )
            
            result['wer'] = float(wer)
            result['cer'] = float(wer)
            result['correct'] = wer == 0.0
            result['num_insertions'] = insertions
            result['num_deletions'] = deletions
            result['num_substitutions'] = substitutions
            
            # Category accuracy (if both predicted and ground truth available)
            if predicted_categories and 'ground_truth_categories' in ground_truth:
                gt_cats = ground_truth['ground_truth_categories']
                if len(predicted_categories) == len(gt_cats):
                    correct_cats = sum(1 for p, g in zip(predicted_categories, gt_cats) if p == g)
                    result['category_accuracy'] = correct_cats / len(gt_cats)
                else:
                    result['category_accuracy'] = 0.0
            
            if wer == 0.0:
                result['temporal_alignment_accuracy'] = calculate_temporal_alignment_accuracy(
                    predicted_timestamps,
                    ground_truth['ground_truth_timestamps'],
                    temporal_tolerance
                )
            else:
                result['temporal_alignment_accuracy'] = 0.0
        
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
        """Generate summary statistics."""
        if not predictions:
            return {'total_sequences': 0}
        
        has_gt = 'wer' in predictions[0]
        has_categories = 'predicted_categories' in predictions[0] and len(predictions[0]['predicted_categories']) > 0
        
        summary = {
            'total_sequences': len(predictions),
            'model_type': self.model_type,
            'decode_method': predictions[0].get('wer') is not None,
            'has_category_predictions': has_categories
        }
        
        if has_gt:
            wers = [p['wer'] for p in predictions]
            correct = sum(1 for p in predictions if p['correct'])
            
            summary['mean_wer'] = float(np.mean(wers))
            summary['mean_cer'] = summary['mean_wer']
            summary['sequence_accuracy'] = correct / len(predictions)
            summary['mean_temporal_alignment'] = float(np.mean([p.get('temporal_alignment_accuracy', 0.0) for p in predictions]))
            
            # Category accuracy statistics
            if has_categories:
                cat_accs = [p['category_accuracy'] for p in predictions if 'category_accuracy' in p]
                if cat_accs:
                    summary['mean_category_accuracy'] = float(np.mean(cat_accs))
                    summary['median_category_accuracy'] = float(np.median(cat_accs))
            
            per_signer = defaultdict(list)
            per_strategy = defaultdict(list)
            
            for p in predictions:
                if 'signer' in p:
                    per_signer[p['signer']].append(p['wer'])
                if 'strategy' in p:
                    per_strategy[p['strategy']].append(p['wer'])
            
            summary['per_signer_wer'] = {s: float(np.mean(wers)) for s, wers in per_signer.items()}
            summary['per_strategy_wer'] = {s: float(np.mean(wers)) for s, wers in per_strategy.items()}
            
            total_ins = sum(p['num_insertions'] for p in predictions)
            total_del = sum(p['num_deletions'] for p in predictions)
            total_sub = sum(p['num_substitutions'] for p in predictions)
            
            summary['total_insertions'] = total_ins
            summary['total_deletions'] = total_del
            summary['total_substitutions'] = total_sub
        
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
        with open(cm_path, 'w') as f:
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
            with open(output_dir / 'confusion_matrix_per_signer.json', 'w') as f:
                json.dump(signer_cm_data, f)
        
        if per_strategy_cm:
            strategy_cm_data = {s: cm.tolist() for s, cm in per_strategy_cm.items()}
            with open(output_dir / 'confusion_matrix_per_strategy.json', 'w') as f:
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
    
    if 'mean_wer' in summary:
        print(f"Mean WER:           {summary['mean_wer']:.4f} ({summary['mean_wer']*100:.2f}%)")
        print(f"Sequence accuracy:  {summary['sequence_accuracy']:.4f} ({summary['sequence_accuracy']*100:.2f}%)")
        print(f"Temporal alignment: {summary['mean_temporal_alignment']:.4f}")
        print(f"\nError breakdown:")
        print(f"  Insertions:       {summary['total_insertions']}")
        print(f"  Deletions:        {summary['total_deletions']}")
        print(f"  Substitutions:    {summary['total_substitutions']}")
        
        if summary.get('per_signer_wer'):
            print(f"\nPer-signer WER:")
            for signer, wer in sorted(summary['per_signer_wer'].items()):
                print(f"  {signer}: {wer:.4f}")
        
        if summary.get('per_strategy_wer'):
            print(f"\nPer-strategy WER:")
            for strategy, wer in sorted(summary['per_strategy_wer'].items()):
                print(f"  {strategy}: {wer:.4f}")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
