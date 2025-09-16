#!/usr/bin/env python3
"""
Sign Language Recognition Model Validation Script

This script provides comprehensive validation of trained Sign Language Recognition models
on the validation dataset. It supports both Transformer and IV3-GRU models and provides
detailed performance analysis including occlusion-based evaluation.

For detailed usage instructions and examples, see VALIDATION_GUIDE.md

Usage:
    python validate.py --model <model_type> --checkpoint <checkpoint_path> [options]

Example:
    python validate.py --model transformer --checkpoint transformer/model.pt --batch-size 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SignTransformer, InceptionV3GRU
from label_mapping import load_label_mappings, format_prediction_results

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ValidationDataset:
    """Dataset class for loading validation data efficiently."""
    
    def __init__(self, data_dir: str, labels_csv: str, model_type: str):
        """
        Initialize validation dataset.
        
        Args:
            data_dir: Directory containing NPZ files
            labels_csv: Path to labels CSV file
            model_type: 'transformer' or 'iv3_gru'
        """
        self.data_dir = Path(data_dir)
        self.labels_csv = labels_csv
        self.model_type = model_type
        
        # Load labels with proper encoding handling
        try:
            self.labels_df = pd.read_csv(labels_csv, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.labels_df = pd.read_csv(labels_csv, encoding='latin-1')
            except UnicodeDecodeError:
                self.labels_df = pd.read_csv(labels_csv, encoding='cp1252')
        self.labels_df['file'] = self.labels_df['file'].str.replace('.npz', '')
        
        # Filter files that exist
        self.valid_files = []
        for _, row in self.labels_df.iterrows():
            npz_path = self.data_dir / f"{row['file']}.npz"
            if npz_path.exists():
                self.valid_files.append({
                    'file': row['file'],
                    'gloss': int(row['gloss']),
                    'cat': int(row['cat']),
                    'occluded': int(row['occluded']),
                    'npz_path': str(npz_path)
                })
        
        print(f"Loaded {len(self.valid_files)} valid samples from {len(self.labels_df)} total labels")
        
        if len(self.valid_files) == 0:
            raise ValueError(f"No valid NPZ files found in {data_dir}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """Load a single sample."""
        sample = self.valid_files[idx]
        
        # Load NPZ data
        data = np.load(sample['npz_path'])
        
        if self.model_type == 'transformer':
            if 'X' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X' key for transformer")
            X = torch.from_numpy(data['X']).float()
            
            # Handle sequence length truncation
            if X.shape[0] > 300:
                X = X[:300, :]
            
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file']
        
        elif self.model_type == 'iv3_gru':
            if 'X2048' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X2048' key for IV3-GRU")
            X = torch.from_numpy(data['X2048']).float()
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file']
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class ModelValidator:
    """Main validation class for comprehensive model evaluation."""
    
    def __init__(self, model_type: str, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize the validator.
        
        Args:
            model_type: 'transformer' or 'iv3_gru'
            checkpoint_path: Path to model checkpoint
            device: Device to use for inference
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Load model and checkpoint
        self.model = self._load_model()
        self._load_checkpoint()
        
        # Load label mappings
        self.gloss_mapping, self.category_mapping = load_label_mappings()
        
        print(f"✓ Initialized {self.model_type} validator on {self.device}")
    
    def _load_model(self):
        """Load the appropriate model architecture."""
        if self.model_type == 'transformer':
            model = SignTransformer(
                input_dim=156,
                emb_dim=256,
                n_heads=8,
                n_layers=4,
                num_gloss=105,
                num_cat=10,
                dropout=0.1,
                max_len=300,
                pooling_method='mean'
            )
        elif self.model_type == 'iv3_gru':
            model = InceptionV3GRU(
                num_gloss=105,
                num_cat=10,
                hidden1=16,
                hidden2=12,
                dropout=0.3,
                pretrained_backbone=True,
                freeze_backbone=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Loaded checkpoint from {self.checkpoint_path}")
    
    def predict_batch(self, batch_data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch of data.
        
        Args:
            batch_data: List of tensors (one per sample)
            
        Returns:
            Tuple of (gloss_logits, category_logits)
        """
        if self.model_type == 'transformer':
            # Pad sequences to same length
            max_len = max(x.shape[0] for x in batch_data)
            padded_batch = []
            masks = []
            
            for x in batch_data:
                if x.shape[0] < max_len:
                    pad_len = max_len - x.shape[0]
                    padded_x = torch.cat([x, torch.zeros(pad_len, x.shape[1])], dim=0)
                    mask = torch.cat([torch.ones(x.shape[0]), torch.zeros(pad_len)], dim=0)
                else:
                    padded_x = x
                    mask = torch.ones(x.shape[0])
                
                padded_batch.append(padded_x)
                masks.append(mask)
            
            X = torch.stack(padded_batch).to(self.device)
            mask = torch.stack(masks).bool().to(self.device)
            
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, mask)
        
        elif self.model_type == 'iv3_gru':
            # Get lengths and pad sequences
            lengths = torch.tensor([x.shape[0] for x in batch_data], dtype=torch.long)
            max_len = max(x.shape[0] for x in batch_data)
            
            padded_batch = []
            for x in batch_data:
                if x.shape[0] < max_len:
                    pad_len = max_len - x.shape[0]
                    padded_x = torch.cat([x, torch.zeros(pad_len, x.shape[1])], dim=0)
                else:
                    padded_x = x
                padded_batch.append(padded_x)
            
            X = torch.stack(padded_batch).to(self.device)
            lengths = lengths.to(self.device)
            
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, lengths, features_already=True)
        
        return gloss_logits, cat_logits
    
    def validate(self, dataset: ValidationDataset, batch_size: int = 32, 
                save_predictions: bool = False, output_dir: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the dataset.
        
        Args:
            dataset: ValidationDataset instance
            batch_size: Batch size for evaluation
            save_predictions: Whether to save individual predictions
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing all validation results
        """
        print(f"\n{'='*60}")
        print(f"VALIDATING {self.model_type.upper()} MODEL")
        print(f"{'='*60}")
        print(f"Dataset: {len(dataset)} samples")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        
        # Initialize results storage
        all_predictions = []
        all_ground_truth = []
        all_occlusions = []
        all_files = []
        
        # Process in batches
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        with tqdm(total=len(dataset), desc="Validating") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                
                # Load batch data
                batch_data = []
                batch_gloss = []
                batch_cat = []
                batch_occluded = []
                batch_files = []
                
                for i in range(start_idx, end_idx):
                    X, gloss, cat, occluded, file = dataset[i]
                    batch_data.append(X)
                    batch_gloss.append(gloss)
                    batch_cat.append(cat)
                    batch_occluded.append(occluded)
                    batch_files.append(file)
                
                # Make predictions
                gloss_logits, cat_logits = self.predict_batch(batch_data)
                
                # Get predictions
                gloss_preds = gloss_logits.argmax(dim=1).cpu().numpy()
                cat_preds = cat_logits.argmax(dim=1).cpu().numpy()
                
                # Get probabilities
                gloss_probs = F.softmax(gloss_logits, dim=1).cpu().numpy()
                cat_probs = F.softmax(cat_logits, dim=1).cpu().numpy()
                
                # Store results
                for i in range(len(batch_data)):
                    all_predictions.append({
                        'file': batch_files[i],
                        'gloss_pred': int(gloss_preds[i]),
                        'cat_pred': int(cat_preds[i]),
                        'gloss_gt': batch_gloss[i],
                        'cat_gt': batch_cat[i],
                        'occluded': batch_occluded[i],
                        'gloss_prob': float(gloss_probs[i][gloss_preds[i]]),
                        'cat_prob': float(cat_probs[i][cat_preds[i]]),
                        'gloss_top5': [(int(j), float(gloss_probs[i][j])) 
                                     for j in np.argsort(gloss_probs[i])[-5:][::-1]],
                        'cat_top3': [(int(j), float(cat_probs[i][j])) 
                                   for j in np.argsort(cat_probs[i])[-3:][::-1]]
                    })
                
                all_ground_truth.extend(list(zip(batch_gloss, batch_cat)))
                all_occlusions.extend(batch_occluded)
                all_files.extend(batch_files)
                
                pbar.update(end_idx - start_idx)
        
        # Convert to numpy arrays for analysis
        gloss_preds = np.array([p['gloss_pred'] for p in all_predictions])
        cat_preds = np.array([p['cat_pred'] for p in all_predictions])
        gloss_gts = np.array([p['gloss_gt'] for p in all_predictions])
        cat_gts = np.array([p['cat_gt'] for p in all_predictions])
        occlusions = np.array(all_occlusions)
        
        # Compute comprehensive metrics
        results = self._compute_metrics(
            gloss_preds, cat_preds, gloss_gts, cat_gts, 
            occlusions, all_predictions, save_predictions, output_dir
        )
        
        return results
    
    def _compute_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                        gloss_gts: np.ndarray, cat_gts: np.ndarray,
                        occlusions: np.ndarray, all_predictions: List[Dict],
                        save_predictions: bool, output_dir: str) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        
        # Overall metrics
        overall_results = self._compute_overall_metrics(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Occlusion-based metrics
        occluded_mask = occlusions == 1
        non_occluded_mask = occlusions == 0
        
        occluded_results = self._compute_overall_metrics(
            gloss_preds[occluded_mask], cat_preds[occluded_mask],
            gloss_gts[occluded_mask], cat_gts[occluded_mask]
        )
        
        non_occluded_results = self._compute_overall_metrics(
            gloss_preds[non_occluded_mask], cat_preds[non_occluded_mask],
            gloss_gts[non_occluded_mask], cat_gts[non_occluded_mask]
        )
        
        # Per-class metrics
        per_class_results = self._compute_per_class_metrics(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Confusion matrices
        confusion_matrices = self._compute_confusion_matrices(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Save individual predictions if requested
        if save_predictions and output_dir:
            self._save_individual_predictions(all_predictions, output_dir)
        
        # Compile final results
        results = {
            'model_info': {
                'model_type': self.model_type,
                'checkpoint_path': self.checkpoint_path,
                'device': str(self.device),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dataset_info': {
                'total_samples': len(gloss_preds),
                'occluded_samples': int(np.sum(occluded_mask)),
                'non_occluded_samples': int(np.sum(non_occluded_mask))
            },
            'overall_results': overall_results,
            'occluded_results': occluded_results,
            'non_occluded_results': non_occluded_results,
            'per_class_results': per_class_results,
            'confusion_matrices': confusion_matrices
        }
        
        return results
    
    def _compute_overall_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                               gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute overall metrics for given predictions."""
        if len(gloss_preds) == 0:
            return {'error': 'No samples to evaluate'}
        
        # Accuracy
        gloss_acc = accuracy_score(gloss_gts, gloss_preds)
        cat_acc = accuracy_score(cat_gts, cat_preds)
        
        # Precision, Recall, F1
        gloss_prec, gloss_rec, gloss_f1, _ = precision_recall_fscore_support(
            gloss_gts, gloss_preds, average='weighted', zero_division=0
        )
        cat_prec, cat_rec, cat_f1, _ = precision_recall_fscore_support(
            cat_gts, cat_preds, average='weighted', zero_division=0
        )
        
        return {
            'gloss_accuracy': float(gloss_acc),
            'category_accuracy': float(cat_acc),
            'gloss_precision': float(gloss_prec),
            'gloss_recall': float(gloss_rec),
            'gloss_f1_score': float(gloss_f1),
            'category_precision': float(cat_prec),
            'category_recall': float(cat_rec),
            'category_f1_score': float(cat_f1),
            'num_samples': int(len(gloss_preds))
        }
    
    def _compute_per_class_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                                 gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute per-class metrics."""
        gloss_report = classification_report(
            gloss_gts, gloss_preds, output_dict=True, zero_division=0
        )
        cat_report = classification_report(
            cat_gts, cat_preds, output_dict=True, zero_division=0
        )
        
        return {
            'gloss_per_class': gloss_report,
            'category_per_class': cat_report
        }
    
    def _compute_confusion_matrices(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                                  gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute confusion matrices."""
        gloss_cm = confusion_matrix(gloss_gts, gloss_preds)
        cat_cm = confusion_matrix(cat_gts, cat_preds)
        
        return {
            'gloss_confusion_matrix': gloss_cm.tolist(),
            'category_confusion_matrix': cat_cm.tolist()
        }
    
    def _save_individual_predictions(self, predictions: List[Dict], output_dir: str):
        """Save individual predictions to JSON files."""
        pred_dir = Path(output_dir) / 'individual_predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        for pred in predictions:
            # Format prediction results
            formatted_pred = {
                'file': pred['file'],
                'ground_truth': {
                    'gloss': f"{self.gloss_mapping.get(pred['gloss_gt'], 'Unknown')} ({pred['gloss_gt']})",
                    'category': f"{self.category_mapping.get(pred['cat_gt'], 'Unknown')} ({pred['cat_gt']})",
                    'occluded': bool(pred['occluded'])
                },
                'prediction': {
                    'gloss': f"{self.gloss_mapping.get(pred['gloss_pred'], 'Unknown')} ({pred['gloss_pred']})",
                    'category': f"{self.category_mapping.get(pred['cat_pred'], 'Unknown')} ({pred['cat_pred']})",
                    'gloss_probability': pred['gloss_prob'],
                    'category_probability': pred['cat_prob']
                },
                'gloss_top5': [
                    [f"{self.gloss_mapping.get(gloss_id, 'Unknown')} ({gloss_id})", prob]
                    for gloss_id, prob in pred['gloss_top5']
                ],
                'category_top3': [
                    [f"{self.category_mapping.get(cat_id, 'Unknown')} ({cat_id})", prob]
                    for cat_id, prob in pred['cat_top3']
                ],
                'correct': {
                    'gloss': pred['gloss_pred'] == pred['gloss_gt'],
                    'category': pred['cat_pred'] == pred['cat_gt']
                }
            }
            
            # Save to file
            output_file = pred_dir / f"{pred['file']}_validation.json"
            with open(output_file, 'w') as f:
                json.dump(formatted_pred, f, indent=2)
        
        print(f"✓ Saved {len(predictions)} individual predictions to {pred_dir}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of validation results."""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        # Model info
        model_info = results['model_info']
        dataset_info = results['dataset_info']
        
        print(f"Model: {model_info['model_type'].upper()}")
        print(f"Checkpoint: {model_info['checkpoint_path']}")
        print(f"Total Samples: {dataset_info['total_samples']}")
        print(f"Occluded: {dataset_info['occluded_samples']} ({dataset_info['occluded_samples']/dataset_info['total_samples']*100:.1f}%)")
        print(f"Non-Occluded: {dataset_info['non_occluded_samples']} ({dataset_info['non_occluded_samples']/dataset_info['total_samples']*100:.1f}%)")
        
        # Overall results
        overall = results['overall_results']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Gloss Accuracy: {overall['gloss_accuracy']:.4f}")
        print(f"  Category Accuracy: {overall['category_accuracy']:.4f}")
        print(f"  Gloss F1-Score: {overall['gloss_f1_score']:.4f}")
        print(f"  Category F1-Score: {overall['category_f1_score']:.4f}")
        
        # Occlusion comparison
        occluded = results['occluded_results']
        non_occluded = results['non_occluded_results']
        
        print(f"\nOCCLUSION IMPACT:")
        print(f"  Occluded Gloss Accuracy: {occluded['gloss_accuracy']:.4f}")
        print(f"  Non-Occluded Gloss Accuracy: {non_occluded['gloss_accuracy']:.4f}")
        print(f"  Accuracy Difference: {non_occluded['gloss_accuracy'] - occluded['gloss_accuracy']:+.4f}")
        
        print(f"  Occluded Category Accuracy: {occluded['category_accuracy']:.4f}")
        print(f"  Non-Occluded Category Accuracy: {non_occluded['category_accuracy']:.4f}")
        print(f"  Category Accuracy Difference: {non_occluded['category_accuracy'] - occluded['category_accuracy']:+.4f}")


def save_results(results: Dict[str, Any], output_dir: str):
    """Save validation results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual result files
    files_to_save = [
        ('overall_results.json', results['overall_results']),
        ('occluded_results.json', results['occluded_results']),
        ('non_occluded_results.json', results['non_occluded_results']),
        ('per_class_results.json', results['per_class_results']),
        ('confusion_matrices.json', results['confusion_matrices'])
    ]
    
    for filename, data in files_to_save:
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Save complete results
    complete_filepath = output_path / 'complete_validation_results.json'
    with open(complete_filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Sign Language Recognition Model Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="For detailed usage instructions and examples, see VALIDATION_GUIDE.md"
    )
    
    # Required arguments
    parser.add_argument('--model', choices=['transformer', 'iv3_gru'], required=True,
                       help='Model type to validate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    
    # Optional arguments
    parser.add_argument('--data-dir', type=str, 
                       default='../data/processed/seq prepro_30 fps_09-13',
                       help='Directory containing validation NPZ files')
    parser.add_argument('--labels-csv', type=str, 
                       default='../data/processed/val_labels.csv',
                       help='Path to validation labels CSV')
    parser.add_argument('--output-dir', type=str, 
                       default='results-validate',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for inference')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save individual predictions to JSON files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = ModelValidator(args.model, args.checkpoint, args.device)
        
        # Load dataset
        dataset = ValidationDataset(args.data_dir, args.labels_csv, args.model)
        
        # Perform validation
        results = validator.validate(
            dataset, 
            batch_size=args.batch_size,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        
        # Save results
        save_results(results, args.output_dir)
        
        # Print summary
        validator.print_summary(results)
        
        return 0
        
    except Exception as e:
        print(f"Error during validation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
