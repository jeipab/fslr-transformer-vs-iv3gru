"""
CTC Model Evaluation Script for Continuous Sign Language Recognition

This script provides comprehensive evaluation metrics for CTC-based continuous
sign language recognition models, including Word Error Rate (WER), 
sequence-level accuracy, per-signer/strategy performance, and a confusion matrix.

Usage:
    python evaluate_ctc.py --model transformer_ctc \\
        --checkpoint path/to/model.pt \\
        --test-data path/to/continuous_npz_folder \\
        --ground-truth-dir path/to/ground_truth_json_folder \\
        --output-dir path/to/evaluation_report_folder
        
    # With beam search decoding
    python evaluate_ctc.py --model transformer_ctc \\
        --checkpoint path/to/model.pt \\
        --test-data path/to/continuous_npz_folder \\
        --ground-truth-dir path/to/ground_truth_json_folder \\
        --output-dir path/to/evaluation_report_folder \\
        --decode-method beam_search \\
        --beam-width 10
"""

# Standard library imports
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from models import SignTransformerCtc, MediaPipeGRUCtc
from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder, calculate_wer_and_errors
from evaluation.prediction.predict_ctc import filter_low_confidence_segments, estimate_timestamps
from streamlit_app.core.config import CTC_CONFIG

class ContinuousSignDataset(Dataset):
    """
    Dataset for loading continuous sign language NPZ files and ground truth JSONs.
    """
    def __init__(self, npz_dir: Path, ground_truth_dir: Path, kp_key='X'):
        self.npz_dir = npz_dir
        self.kp_key = kp_key
        self.ground_truth_files = sorted(ground_truth_dir.glob("*.json"))
        
        if not self.ground_truth_files:
            raise FileNotFoundError(f"No ground truth JSON files found in {ground_truth_dir}")

    def __len__(self):
        return len(self.ground_truth_files)

    def __getitem__(self, idx):
        json_path = self.ground_truth_files[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        npz_path = self.npz_dir / meta['file_name']
        if not npz_path.exists():
            # Handle cases where file might have a different extension or name
            npz_path = self.npz_dir / Path(meta['file_name']).stem
            if not npz_path.exists():
                 raise FileNotFoundError(f"NPZ file not found for {meta['file_name']}")

        keypoints = np.load(npz_path)[self.kp_key]
        
        ground_truth_sequence = meta['ground_truth_sequence']
        
        return {
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'ground_truth': torch.tensor(ground_truth_sequence, dtype=torch.long),
            'meta': meta
        }

def collate_continuous_for_ctc(batch: List[Dict]) -> Tuple:
    """Collator for continuous data to prepare batches for CTC model."""
    keypoints = [item['keypoints'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    metas = [item['meta'] for item in batch]

    # Pad keypoints
    input_lengths = torch.tensor([len(kp) for kp in keypoints], dtype=torch.long)
    padded_keypoints = torch.nn.utils.rnn.pad_sequence(keypoints, batch_first=True, padding_value=0.0)

    # Concatenate ground truths
    target_lengths = torch.tensor([len(gt) for gt in ground_truths], dtype=torch.long)
    targets = torch.cat(ground_truths)
    
    return padded_keypoints, targets, input_lengths, target_lengths, metas


class CTCEvaluator:
    """
    Evaluator for CTC-based continuous sign language recognition models.
    """
    def __init__(self, model, blank_id, device, model_type=None):
        self.model = model
        self.blank_id = blank_id
        self.device = device
        self.model_type = model_type
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataloader, decode_method='greedy', beam_width=10):
        """
        Evaluate the model on a dataset of continuous sign sequences.
        """
        all_results = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                X, targets, input_lengths, target_lengths, metas = batch
                X = X.to(self.device)
                input_lengths = input_lengths.to(self.device)

                output = self.model(X)
                
                # Handle dual-task models (CTC + Category)
                if isinstance(output, tuple):
                    log_probs, _ = output  # Extract CTC predictions, ignore category
                else:
                    log_probs = output

                if decode_method == 'greedy':
                    predicted_sequences = greedy_ctc_decoder(log_probs, self.blank_id, input_lengths)
                    # Extract confidence scores for greedy decoding
                    probs = torch.exp(log_probs)
                    confidence_scores_list = []
                    for j, seq in enumerate(predicted_sequences):
                        if len(seq) > 0:
                            actual_length = input_lengths[j].item()
                            seq_probs = probs[j, :actual_length]
                            # Get confidence for each predicted token
                            # For CTC, we need to map decoded sequence back to frame-level predictions
                            # Use max probability of the predicted token across frames
                            confidences = []
                            frames_per_token = actual_length / len(seq) if len(seq) > 0 else 1
                            for token_idx, gloss_id in enumerate(seq):
                                start_frame = int(token_idx * frames_per_token)
                                end_frame = int((token_idx + 1) * frames_per_token)
                                if end_frame > actual_length:
                                    end_frame = actual_length
                                if start_frame < end_frame:
                                    token_conf = float(seq_probs[start_frame:end_frame, gloss_id].max())
                                else:
                                    token_conf = float(seq_probs[min(start_frame, actual_length-1), gloss_id])
                                confidences.append(token_conf)
                            confidence_scores_list.append(confidences)
                        else:
                            confidence_scores_list.append([])
                else:
                    beam_results = beam_search_ctc_decoder(log_probs, self.blank_id, beam_width, input_lengths)
                    predicted_sequences = [seq for seq, score in beam_results]
                    # For beam search, use average confidence from beam score
                    confidence_scores_list = []
                    for j, (seq, log_score) in enumerate(beam_results):
                        if len(seq) > 0:
                            avg_conf = float(np.exp(log_score / max(len(seq), 1)))
                            confidence_scores_list.append([avg_conf] * len(seq))
                        else:
                            confidence_scores_list.append([])

                reference_sequences = self._parse_targets(targets, target_lengths)

                for j in range(len(predicted_sequences)):
                    pred_seq = predicted_sequences[j]
                    ref_seq = reference_sequences[j]
                    meta = metas[j]
                    confidence_scores = confidence_scores_list[j]
                    
                    # Apply low-confidence filtering for Transformer models only
                    if self.model_type == 'transformer_continuous' and len(pred_seq) > 0 and len(confidence_scores) == len(pred_seq):
                        # Estimate timestamps for filtering
                        actual_length = input_lengths[j].item()
                        predicted_timestamps = estimate_timestamps(pred_seq, actual_length, fps=30)
                        predicted_labels = [f"GLOSS_{g}" for g in pred_seq]
                        
                        # Apply filtering
                        filtered_seq, filtered_labels, filtered_timestamps, filtered_confidences, _, _ = \
                            filter_low_confidence_segments(
                                predicted_sequence=pred_seq,
                                predicted_labels=predicted_labels,
                                predicted_timestamps=predicted_timestamps,
                                confidence_scores=confidence_scores,
                                confidence_threshold=0.50
                            )
                        pred_seq = filtered_seq
                    
                    wer, errors = calculate_wer_and_errors(ref_seq, pred_seq)
                    
                    all_results.append({
                        'file_name': meta['file_name'],
                        'signer': meta.get('signer', 'N/A'),
                        'strategy': meta.get('strategy', 'N/A'),
                        'ground_truth': ref_seq,
                        'prediction': pred_seq,
                        'wer': wer,
                        'exact_match': 1 if pred_seq == ref_seq else 0,
                        'substitutions': errors['S'],
                        'deletions': errors['D'],
                        'insertions': errors['I'],
                    })
        return all_results

    def _parse_targets(self, targets, target_lengths):
        """Parse concatenated targets back into individual sequences."""
        sequences = []
        start_idx = 0
        for length in target_lengths:
            end_idx = start_idx + length
            sequences.append(targets[start_idx:end_idx].cpu().numpy().tolist())
            start_idx = end_idx
        return sequences

def analyze_results(results: List[Dict], output_dir: Path, num_classes: int):
    """
    Calculates and prints summary metrics and generates reports.
    """
    df = pd.DataFrame(results)
    
    # Overall metrics
    overall_wer = df['wer'].mean()
    sequence_accuracy = df['exact_match'].mean()
    total_substitutions = df['substitutions'].sum()
    total_deletions = df['deletions'].sum()
    total_insertions = df['insertions'].sum()

    # Per-signer metrics
    signer_wer = df.groupby('signer')['wer'].mean().to_dict()

    # Per-strategy metrics
    strategy_wer = df.groupby('strategy')['wer'].mean().to_dict()

    # Generate Confusion Matrix
    all_gt = []
    all_pred = []
    for _, row in df.iterrows():
        # Simple alignment for confusion matrix: pad shorter sequence
        gt = row['ground_truth']
        pred = row['prediction']
        max_len = max(len(gt), len(pred))
        gt.extend([-1] * (max_len - len(gt))) # Pad with -1 for non-existent gloss
        pred.extend([-1] * (max_len - len(pred)))
        all_gt.extend(gt)
        all_pred.extend(pred)
        
    cm = confusion_matrix(all_gt, all_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_df, annot=False, cmap='viridis')
    plt.title("Gloss Confusion Matrix")
    plt.xlabel("Predicted Gloss ID")
    plt.ylabel("True Gloss ID")
    plt.savefig(output_dir / "confusion_matrix.png")

    # Generate report
    with open(output_dir / "report.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("CTC MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Overall Word Error Rate (WER): {overall_wer:.4f}\n")
        f.write(f"Overall Sequence Accuracy: {sequence_accuracy:.4f}\n\n")

        f.write("--- Error Analysis ---\n")
        f.write(f"Total Substitutions: {total_substitutions}\n")
        f.write(f"Total Deletions: {total_deletions}\n")
        f.write(f"Total Insertions: {total_insertions}\n\n")

        f.write("--- Per-Signer WER ---\n")
        for signer, wer in sorted(signer_wer.items()):
            f.write(f"  {signer}: {wer:.4f}\n")
        f.write("\n")

        f.write("--- Per-Strategy WER ---\n")
        for strategy, wer in sorted(strategy_wer.items()):
            f.write(f"  {strategy}: {wer:.4f}\n")
        f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("Detailed results saved to detailed_results.csv\n")
        f.write("Confusion matrix saved to confusion_matrix.csv and confusion_matrix.png\n")

    # Save detailed results
    df.to_csv(output_dir / "detailed_results.csv", index=False)

    print(f"✅ Evaluation complete. Report saved to {output_dir}")

def load_model(model_type, checkpoint_path, device, num_classes):
    """Loads a CTC model from a checkpoint."""
    if model_type == 'transformer_continuous':
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
            input_dim = state_dict.get('embedding.weight', state_dict.get('input_projection.weight')).shape[1]
        except Exception:
            input_dim = 156 # Default if detection fails
        model = SignTransformerCtc(input_dim=input_dim, num_ctc_classes=num_classes, max_len=1000)
    elif model_type == 'mediapipe_gru_continuous':
        model = MediaPipeGRUCtc(num_ctc_classes=num_classes)
    elif model_type == 'iv3_gru_continuous':
        from models.iv3_gru import InceptionV3GRUCtc
        model = InceptionV3GRUCtc(num_ctc_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Continuous CTC Sign Language Models")
    parser.add_argument('--model', choices=['transformer_continuous', 'mediapipe_gru_continuous', 'iv3_gru_continuous'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True, help='Directory with continuous .npz files')
    parser.add_argument('--ground-truth-dir', type=str, required=True, help='Directory with ground truth .json files')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save evaluation reports')
    parser.add_argument('--decode-method', choices=['greedy', 'beam_search'], default='beam_search')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--kp-key', type=str, default='X')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')
    
    print_config(args, device)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, args.checkpoint, device, CTC_CONFIG['num_ctc_classes'])
    print("✓ Model loaded successfully.")

    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset = ContinuousSignDataset(Path(args.test_data), Path(args.ground_truth_dir), args.kp_key)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_continuous_for_ctc
    )
    print(f"✓ Dataset loaded with {len(dataset)} samples.")

    # Evaluate
    print("\nStarting evaluation...")
    evaluator = CTCEvaluator(model, CTC_CONFIG['blank_token_id'], device, model_type=args.model)
    results = evaluator.evaluate(
        dataloader,
        decode_method=args.decode_method,
        beam_width=args.beam_width
    )
    
    # Analyze and save results
    print("\nAnalyzing results and generating reports...")
    analyze_results(results, output_dir, CTC_CONFIG['num_ctc_classes'])

def print_config(args, device):
    print("="*60)
    print("CONTINUOUS CTC EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Data: {args.test_data}")
    print(f"Ground Truth: {args.ground_truth_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Decode Method: {args.decode_method}")
    if args.decode_method == 'beam_search':
        print(f"Beam Width: {args.beam_width}")
    print("="*60)

if __name__ == "__main__":
    main()