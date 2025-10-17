"""
CTC Model Evaluation Script

This script provides comprehensive evaluation metrics for CTC-based continuous
sign language recognition models, including Word Error Rate (WER), Character Error Rate (CER),
and sequence-level accuracy.

Usage:
    python evaluate_ctc.py --model transformer_ctc \\
        --checkpoint path/to/model.pt \\
        --test-data path/to/test_folder \\
        --test-labels path/to/test.csv
        
    # With beam search decoding
    python evaluate_ctc.py --model transformer_ctc \\
        --checkpoint path/to/model.pt \\
        --test-data path/to/test_folder \\
        --test-labels path/to/test.csv \\
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

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from models import SignTransformerCtc, MediaPipeGRUCtc
from training.train import FSLKeypointFileDataset, FSLFeatureFileDataset, collate_for_ctc
from evaluation.ctc_utils import greedy_ctc_decoder, beam_search_ctc_decoder, calculate_wer
from streamlit_app.core.config import CTC_CONFIG


class CTCEvaluator:
    """
    Evaluator for CTC-based sign language recognition models.
    
    Computes comprehensive metrics including:
    - Word Error Rate (WER)
    - Character Error Rate (CER)
    - Sequence-level accuracy
    - Per-class performance
    """
    
    def __init__(self, model, blank_id, device):
        """
        Initialize the CTC evaluator.
        
        Args:
            model: Trained CTC model
            blank_id (int): Blank token ID for CTC
            device: Torch device
        """
        self.model = model
        self.blank_id = blank_id
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(
        self,
        dataloader,
        decode_method='greedy',
        beam_width=10,
        verbose=True
    ):
        """
        Evaluate model on a complete dataset.
        
        Args:
            dataloader: DataLoader with CTC-formatted batches
            decode_method (str): 'greedy' or 'beam_search'
            beam_width (int): Beam width for beam search
            verbose (bool): Print progress information
            
        Returns:
            dict: Evaluation metrics including WER, CER, accuracy
        """
        all_predictions = []
        all_references = []
        total_samples = 0
        exact_matches = 0
        
        print(f"\nEvaluating on {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Unpack CTC batch
                X, targets, input_lengths, target_lengths, _ = batch
                
                # Move to device
                X = X.to(self.device)
                input_lengths = input_lengths.to(self.device)
                
                # Model inference
                log_probs = self.model(X)  # [B, T, C]
                
                # Decode predictions
                if decode_method == 'greedy':
                    predicted_sequences = greedy_ctc_decoder(
                        log_probs, self.blank_id, input_lengths
                    )
                else:
                    beam_results = beam_search_ctc_decoder(
                        log_probs, self.blank_id, beam_width, input_lengths
                    )
                    predicted_sequences = [seq for seq, score in beam_results]
                
                # Parse reference sequences from concatenated targets
                batch_size = X.size(0)
                reference_sequences = self._parse_targets(targets, target_lengths, batch_size)
                
                # Store predictions and references
                for pred, ref in zip(predicted_sequences, reference_sequences):
                    all_predictions.append(pred)
                    all_references.append(ref)
                    
                    # Check for exact match
                    if pred == ref:
                        exact_matches += 1
                    
                    total_samples += 1
                
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
        
        # ================================================================
        # CALCULATE METRICS
        # ================================================================
        
        # Word Error Rate (WER)
        wer_scores = [
            calculate_wer(ref, pred)
            for ref, pred in zip(all_references, all_predictions)
        ]
        avg_wer = np.mean(wer_scores)
        
        # Sequence-level accuracy (exact match)
        sequence_accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
        
        # Calculate per-sample metrics
        per_sample_metrics = []
        for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
            per_sample_metrics.append({
                'sample_id': i,
                'predicted': pred,
                'reference': ref,
                'wer': wer_scores[i],
                'exact_match': pred == ref
            })
        
        return {
            'wer': avg_wer,
            'cer': avg_wer,  # Alias for consistency
            'sequence_accuracy': sequence_accuracy,
            'total_samples': total_samples,
            'exact_matches': exact_matches,
            'decode_method': decode_method,
            'per_sample_metrics': per_sample_metrics
        }
    
    def _parse_targets(self, targets, target_lengths, batch_size):
        """
        Parse concatenated targets back into individual sequences.
        
        Args:
            targets: Concatenated target tensor [sum(target_lengths)]
            target_lengths: Target lengths tensor [B]
            batch_size: Number of sequences in batch
            
        Returns:
            List of reference sequences
        """
        targets_np = targets.cpu().numpy()
        lengths_np = target_lengths.cpu().numpy()
        
        sequences = []
        start_idx = 0
        
        for length in lengths_np:
            end_idx = start_idx + length
            seq = targets_np[start_idx:end_idx].tolist()
            sequences.append(seq)
            start_idx = end_idx
        
        return sequences


def load_model(model_type, checkpoint_path, device):
    """
    Load a CTC model from checkpoint.
    
    Args:
        model_type (str): 'transformer_ctc' or 'mediapipe_gru_ctc'
        checkpoint_path (str): Path to checkpoint file
        device: Torch device
        
    Returns:
        Loaded model in eval mode
    """
    if model_type == 'transformer_ctc':
        # Auto-detect input dim
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
            if 'embedding.weight' in state_dict:
                input_dim = state_dict['embedding.weight'].shape[1]
            else:
                input_dim = 156
        except:
            input_dim = 156
        
        model = SignTransformerCtc(
            input_dim=input_dim,
            num_ctc_classes=CTC_CONFIG['num_ctc_classes']
        )
    
    elif model_type == 'mediapipe_gru_ctc':
        model = MediaPipeGRUCtc(
            num_ctc_classes=CTC_CONFIG['num_ctc_classes']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CTC Sign Language Recognition Models")
    parser.add_argument('--model', choices=['transformer_ctc', 'mediapipe_gru_ctc'],
                       required=True, help='CTC model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Directory with test .npz files')
    parser.add_argument('--test-labels', type=str, required=True,
                       help='CSV file with test labels')
    parser.add_argument('--decode-method', choices=['greedy', 'beam_search'], default='greedy',
                       help='CTC decoding method')
    parser.add_argument('--beam-width', type=int, default=10,
                       help='Beam width for beam search')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--kp-key', type=str, default='X',
                       help='Key for keypoints in NPZ files')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"CTC Model Evaluation")
    print(f"="*60)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Device: {device}")
    print(f"Decode method: {args.decode_method}")
    if args.decode_method == 'beam_search':
        print(f"Beam width: {args.beam_width}")
    print(f"="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, args.checkpoint, device)
    print(f"✓ Model loaded successfully")
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = FSLKeypointFileDataset(
        keypoints_dir=args.test_data,
        labels_csv=args.test_labels,
        kp_key=args.kp_key,
        augment=False,
        mode='ctc'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_for_ctc,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Evaluate
    print("\nEvaluating model...")
    evaluator = CTCEvaluator(model, CTC_CONFIG['blank_token_id'], device)
    results = evaluator.evaluate_dataset(
        test_loader,
        decode_method=args.decode_method,
        beam_width=args.beam_width,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Word Error Rate (WER): {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"Sequence Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_accuracy']*100:.2f}%)")
    print(f"Exact matches: {results['exact_matches']}/{results['total_samples']}")
    print("="*60)
    
    # Save results if output path specified
    if args.output:
        output_data = {
            'model_type': args.model,
            'checkpoint': args.checkpoint,
            'test_data': args.test_data,
            'decode_method': args.decode_method,
            'metrics': {
                'wer': results['wer'],
                'cer': results['cer'],
                'sequence_accuracy': results['sequence_accuracy'],
                'total_samples': results['total_samples'],
                'exact_matches': results['exact_matches']
            },
            'per_sample_results': results['per_sample_metrics']
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

