"""Validation manager for model validation and results processing."""

import io
import json
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import time
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..core.config import MODEL_CONFIG, is_ctc_model, get_model_config
from ..components.utils import detect_file_type


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
            # Transformer uses 178-dimensional keypoint features
            if 'X' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X' key for transformer model (expected 178-D keypoints)")
            X = torch.from_numpy(data['X']).float()
            
            # Truncate sequence if too long
            if X.shape[0] > 300:
                X = X[:300, :]
            
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file']
        
        elif self.model_type == 'iv3_gru':
            # IV3-GRU uses 2048-dimensional InceptionV3 features
            if 'X2048' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X2048' key for IV3-GRU model (expected 2048-D features)")
            X = torch.from_numpy(data['X2048']).float()
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file']
        
        elif self.model_type == 'mediapipe_gru':
            # MediaPipe-GRU uses 178-dimensional keypoint features
            if 'X' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X' key for MediaPipe-GRU model (expected 178-D keypoints)")
            X = torch.from_numpy(data['X']).float()
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
        self.gloss_mapping, self.category_mapping = self._load_label_mappings()
        
        print(f"✓ Initialized {self.model_type} validator on {self.device}")
    
    def _load_model(self):
        """Load the appropriate model architecture using Streamlit config."""
        # Get model configuration from Streamlit config
        model_config = get_model_config(self.model_type)
        if not model_config:
            raise ValueError(f"Model type '{self.model_type}' not found in Streamlit config")
        
        if self.model_type == 'transformer':
            from models.transformer import SignTransformer
            
            # Use input dimension from Streamlit config (178 for keypoints)
            input_dim = model_config.get('input_dim', 178)
            print(f"Using input dimension from config: {input_dim}")
            
            model = SignTransformer(
                input_dim=input_dim,
                emb_dim=256,
                n_heads=8,
                n_layers=4,
                num_gloss=model_config['num_gloss_classes'],    # From Streamlit config
                num_cat=model_config['num_category_classes'],  # From Streamlit config
                dropout=0.1,
                max_len=300,
                pooling_method='mean'
            )
        elif self.model_type == 'iv3_gru':
            from models.iv3_gru import InceptionV3GRU
            
            # Try to determine hidden sizes from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Detect GRU hidden sizes from checkpoint weights
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    # GRU weight_hh has shape [3*hidden, hidden] for each layer
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3
                    print(f"Detected GRU hidden sizes from checkpoint: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
                else:
                    gru1_hidden = 16  # Fallback to default
                    gru2_hidden = 12  # Fallback to default
                    print(f"Warning: Could not detect GRU hidden sizes from checkpoint, using defaults: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
                    
            except Exception as e:
                gru1_hidden = 16  # Fallback to default
                gru2_hidden = 12  # Fallback to default
                print(f"Warning: Could not load checkpoint to detect GRU hidden sizes, using defaults: {e}")
            
            model = InceptionV3GRU(
                num_gloss=model_config['num_gloss_classes'],    # From Streamlit config
                num_cat=model_config['num_category_classes'],  # From Streamlit config
                hidden1=gru1_hidden,
                hidden2=gru2_hidden,
                dropout=0.3,
                pretrained_backbone=True,
                freeze_backbone=True
            )
        elif self.model_type == 'mediapipe_gru':
            from models.mediapipe_gru import MediaPipeGRU
            
            # Try to determine hidden sizes from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                
                # Detect GRU hidden sizes from checkpoint weights
                if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                    # GRU weight_hh has shape [3*hidden, hidden] for each layer
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3
                    print(f"Detected GRU hidden sizes from checkpoint: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
                else:
                    gru1_hidden = 256  # Fallback to default
                    gru2_hidden = 128  # Fallback to default
                    print(f"Warning: Could not detect GRU hidden sizes from checkpoint, using defaults: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
                    
            except Exception as e:
                gru1_hidden = 256  # Fallback to default
                gru2_hidden = 128  # Fallback to default
                print(f"Warning: Could not load checkpoint to detect GRU hidden sizes, using defaults: {e}")
            
            model = MediaPipeGRU(
                num_gloss=model_config['num_gloss_classes'],    # From Streamlit config
                num_cat=model_config['num_category_classes'],   # From Streamlit config
                input_dim=model_config['input_dim'],           # From Streamlit config
                hidden1=gru1_hidden,
                hidden2=gru2_hidden,
                dropout=0.3,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        if not Path(self.checkpoint_path).exists():
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
    
    def _load_label_mappings(self):
        """Load label mappings."""
        try:
            from data.labels.label_mapping import load_label_mappings
            return load_label_mappings()
        except Exception as e:
            print(f"Warning: Could not load label mappings: {e}")
            return ({}, {})
    
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
        
        elif self.model_type == 'mediapipe_gru':
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
                gloss_logits, cat_logits = self.model(X, lengths)
        
        return gloss_logits, cat_logits
    
    def validate(self, dataset: ValidationDataset, batch_size: int = 32, 
                progress_callback=None) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the dataset.
        
        Args:
            dataset: ValidationDataset instance
            batch_size: Batch size for evaluation
            progress_callback: Optional callback for progress updates
            
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
        all_gloss_probs = []  # For top-k accuracy computation
        all_cat_probs = []    # For top-k accuracy computation
        
        # Process in batches
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
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
            gloss_probs = torch.softmax(gloss_logits, dim=1).cpu().numpy()
            cat_probs = torch.softmax(cat_logits, dim=1).cpu().numpy()
            
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
                    'gloss_top10': [(int(j), float(gloss_probs[i][j])) 
                                  for j in np.argsort(gloss_probs[i])[-10:][::-1]],
                    'cat_top5': [(int(j), float(cat_probs[i][j])) 
                               for j in np.argsort(cat_probs[i])[-5:][::-1]]
                })
                
                # Store probabilities for top-k accuracy computation
                all_gloss_probs.append(gloss_probs[i])
                all_cat_probs.append(cat_probs[i])
            
            all_ground_truth.extend(list(zip(batch_gloss, batch_cat)))
            all_occlusions.extend(batch_occluded)
            all_files.extend(batch_files)
            
            # Update progress
            if progress_callback:
                progress_callback(batch_idx + 1, num_batches)
        
        # Convert to numpy arrays for analysis
        gloss_preds = np.array([p['gloss_pred'] for p in all_predictions])
        cat_preds = np.array([p['cat_pred'] for p in all_predictions])
        gloss_gts = np.array([p['gloss_gt'] for p in all_predictions])
        cat_gts = np.array([p['cat_gt'] for p in all_predictions])
        occlusions = np.array(all_occlusions)
        gloss_probs = np.array(all_gloss_probs)
        cat_probs = np.array(all_cat_probs)
        
        # Compute comprehensive metrics
        results = self._compute_metrics(
            gloss_preds, cat_preds, gloss_gts, cat_gts, 
            occlusions, gloss_probs, cat_probs, all_predictions
        )
        
        return results
    
    def _compute_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                        gloss_gts: np.ndarray, cat_gts: np.ndarray,
                        occlusions: np.ndarray, gloss_probs: np.ndarray, cat_probs: np.ndarray,
                        all_predictions: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        
        # Overall metrics
        overall_results = self._compute_overall_metrics(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Top-k accuracy for overall
        overall_topk = self._compute_topk_accuracy(gloss_probs, cat_probs, gloss_gts, cat_gts)
        overall_results.update(overall_topk)
        
        # Occlusion-based metrics
        occluded_mask = occlusions == 1
        non_occluded_mask = occlusions == 0
        
        occluded_results = self._compute_overall_metrics(
            gloss_preds[occluded_mask], cat_preds[occluded_mask],
            gloss_gts[occluded_mask], cat_gts[occluded_mask]
        )
        occluded_topk = self._compute_topk_accuracy(
            gloss_probs[occluded_mask], cat_probs[occluded_mask],
            gloss_gts[occluded_mask], cat_gts[occluded_mask]
        )
        occluded_results.update(occluded_topk)
        
        non_occluded_results = self._compute_overall_metrics(
            gloss_preds[non_occluded_mask], cat_preds[non_occluded_mask],
            gloss_gts[non_occluded_mask], cat_gts[non_occluded_mask]
        )
        non_occluded_topk = self._compute_topk_accuracy(
            gloss_probs[non_occluded_mask], cat_probs[non_occluded_mask],
            gloss_gts[non_occluded_mask], cat_gts[non_occluded_mask]
        )
        non_occluded_results.update(non_occluded_topk)
        
        # Per-class metrics
        per_class_results = self._compute_per_class_metrics(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Confusion matrices
        confusion_matrices = self._compute_confusion_matrices(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
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
            'confusion_matrices': confusion_matrices,
            'detailed_predictions': all_predictions
        }
        
        return results
    
    def _compute_overall_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                               gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute overall metrics for given predictions."""
        if len(gloss_preds) == 0:
            return {'error': 'No samples to evaluate'}
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
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
    
    def _compute_topk_accuracy(self, gloss_probs: np.ndarray, cat_probs: np.ndarray,
                               gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """
        Compute top-k accuracy for gloss and category predictions.
        
        Args:
            gloss_probs: Probability distributions for gloss predictions [num_samples, num_gloss_classes]
            cat_probs: Probability distributions for category predictions [num_samples, num_cat_classes]
            gloss_gts: Ground truth gloss labels [num_samples]
            cat_gts: Ground truth category labels [num_samples]
            
        Returns:
            Dictionary containing top-k accuracy metrics
        """
        if len(gloss_probs) == 0:
            return {}
        
        results = {}
        
        # Compute top-k accuracy for gloss (top-1, top-5, top-10)
        for k in [1, 5, 10]:
            # Get top-k predictions
            top_k_preds = np.argsort(gloss_probs, axis=1)[:, -k:]
            # Check if ground truth is in top-k
            correct = np.array([gt in top_k_preds[i] for i, gt in enumerate(gloss_gts)])
            results[f'gloss_top{k}_accuracy'] = float(np.mean(correct))
        
        # Compute top-k accuracy for category (top-1, top-5)
        # (only top-5 since there are 10 categories total)
        for k in [1, 5]:
            # Get top-k predictions
            top_k_preds = np.argsort(cat_probs, axis=1)[:, -k:]
            # Check if ground truth is in top-k
            correct = np.array([gt in top_k_preds[i] for i, gt in enumerate(cat_gts)])
            results[f'category_top{k}_accuracy'] = float(np.mean(correct))
        
        return results
    
    def _compute_per_class_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                                 gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute per-class metrics."""
        from sklearn.metrics import classification_report
        
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
        from sklearn.metrics import confusion_matrix
        
        gloss_cm = confusion_matrix(gloss_gts, gloss_preds)
        cat_cm = confusion_matrix(cat_gts, cat_preds)
        
        return {
            'gloss_confusion_matrix': gloss_cm.tolist(),
            'category_confusion_matrix': cat_cm.tolist()
        }


def create_validation_dataset_from_folder(npz_folder_path: str, labels_csv_file, model_type: str) -> ValidationDataset:
    """Create validation dataset from NPZ folder path and uploaded labels CSV."""
    # Create temporary directory for labels CSV
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    # Save labels CSV to temporary directory
    labels_csv_file.seek(0)
    labels_content = labels_csv_file.read()
    labels_csv_path = temp_dir_path / "labels.csv"
    with open(labels_csv_path, 'wb') as f:
        f.write(labels_content)
    
    # Create validation dataset using the provided folder path
    dataset = ValidationDataset(npz_folder_path, str(labels_csv_path), model_type)
    
    return dataset


def create_validation_dataset_from_uploads(npz_files: List, labels_csv_file, model_type: str) -> ValidationDataset:
    """Create validation dataset from uploaded NPZ files and labels CSV."""
    # Create temporary directory for NPZ files
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    # Save NPZ files to temporary directory
    for uploaded_file in npz_files:
        # Reset file pointer
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        
        # Save to temporary directory
        temp_file_path = temp_dir_path / uploaded_file.name
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
    
    # Save labels CSV to temporary directory
    labels_csv_file.seek(0)
    labels_content = labels_csv_file.read()
    labels_csv_path = temp_dir_path / "labels.csv"
    with open(labels_csv_path, 'wb') as f:
        f.write(labels_content)
    
    # Create validation dataset
    dataset = ValidationDataset(str(temp_dir_path), str(labels_csv_path), model_type)
    
    return dataset


def run_validation_from_folder(model_type: str, npz_folder_path: str, labels_csv_file, 
                              batch_size: int = 32, progress_callback=None) -> Dict[str, Any]:
    """
    Run validation on NPZ files from a folder path.
    
    Args:
        model_type: 'transformer' or 'iv3_gru'
        npz_folder_path: Path to folder containing NPZ files
        labels_csv_file: Uploaded labels CSV file
        batch_size: Batch size for evaluation
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary containing validation results
    """
    # Get model configuration
    config = MODEL_CONFIG.get(model_type)
    if not config or not config['enabled']:
        raise ValueError(f"Model {model_type} is not available")
    
    # Create validation dataset
    dataset = create_validation_dataset_from_folder(npz_folder_path, labels_csv_file, model_type)
    
    # Run validation
    validator = ModelValidator(
        model_type=model_type,
        checkpoint_path=config['checkpoint_path']
    )
    
    results = validator.validate(dataset, progress_callback=progress_callback)
    
    return results


def run_validation(model_type: str, npz_files: List, labels_csv_file, 
                  batch_size: int = 32, progress_callback=None) -> Dict[str, Any]:
    """
    Run validation on uploaded files.
    
    Args:
        model_type: 'transformer' or 'iv3_gru'
        npz_files: List of uploaded NPZ files
        labels_csv_file: Uploaded labels CSV file
        batch_size: Batch size for evaluation
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary containing validation results
    """
    # Get model configuration
    config = MODEL_CONFIG.get(model_type)
    if not config or not config['enabled']:
        raise ValueError(f"Model {model_type} is not available")
    
    # Create validation dataset
    dataset = create_validation_dataset_from_uploads(npz_files, labels_csv_file, model_type)
    
    # Initialize validator
    validator = ModelValidator(model_type, config['checkpoint_path'])
    
    # Run validation
    results = validator.validate(dataset, batch_size, progress_callback)
    
    return results


def run_ctc_validation_with_sliding_window(
    model_type: str,
    npz_folder_path: str,
    ground_truth_folder: str,
    decode_method: str = 'greedy',
    beam_width: int = 10,
    window_size: int = 150,
    stride: int = 50,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run CTC validation on continuous sequences using sliding window approach.
    
    Args:
        model_type: CTC model type ('transformer_ctc' or 'iv3_gru_ctc')
        npz_folder_path: Path to folder containing continuous sequence NPZ files
        ground_truth_folder: Path to folder containing ground truth JSON files
        decode_method: Decoding method ('greedy' or 'beam_search')
        beam_width: Beam width for beam search
        window_size: Size of sliding window in frames
        stride: Stride between windows in frames
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary containing CTC validation results
    """
    # Verify model is CTC model
    if not is_ctc_model(model_type):
        raise ValueError(f"{model_type} is not a CTC model")
    
    # Get model configuration
    config = MODEL_CONFIG.get(model_type)
    if not config or not config['enabled']:
        raise ValueError(f"Model {model_type} is not available")
    
    try:
        # Import CTC predictor
        from evaluation.prediction.predict_ctc import CTCPredictor
        from ..core.config import CTC_CONFIG, CTC_CONFIG_SUBSET
        
        # Get blank_id from config
        # For CTC models, blank_id should be num_gloss_classes (e.g., 10 for subset, 105 for full)
        blank_id = config.get('blank_token_id', config['num_gloss_classes'])
        
        # Initialize predictor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = CTCPredictor(
            model_type=config['model_type'],
            checkpoint_path=config['checkpoint_path'],
            blank_id=blank_id,
            device=device
        )
        
        # Run batch prediction with sliding window by calling sliding window method directly
        npz_files = sorted(Path(npz_folder_path).glob('*.npz'))
        predictions = []
        
        for npz_path in npz_files:
            # Load ground truth if available
            ground_truth = None
            if ground_truth_folder:
                import json
                
                # Try both naming conventions
                gt_path_1 = Path(ground_truth_folder) / (npz_path.stem + '_gt.json')
                gt_path_2 = Path(ground_truth_folder) / (npz_path.stem + '.json')
                
                if gt_path_1.exists():
                    with open(gt_path_1, 'r', encoding='utf-8') as f:
                        ground_truth = json.load(f)
                elif gt_path_2.exists():
                    with open(gt_path_2, 'r', encoding='utf-8') as f:
                        ground_truth = json.load(f)
            
            # Use sliding window prediction
            result = predictor.predict_sequence_sliding_window(
                npz_path=npz_path,
                ground_truth=ground_truth,
                window_size=window_size,
                stride=stride,
                decode_method=decode_method,
                beam_width=beam_width,
                fps=30,
                temporal_tolerance=500
            )
            predictions.append(result)
        
        # Generate summary
        summary = predictor._generate_summary(predictions)
        
        results = {
            'predictions': predictions,
            'summary': summary
        }
        
        # Add model info
        from datetime import datetime
        results['model_info'] = {
            'model_type': model_type,
            'checkpoint_path': config['checkpoint_path'],
            'device': str(device),
            'decode_method': decode_method,
            'beam_width': beam_width,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results
    
    except Exception as e:
        raise RuntimeError(f"CTC validation failed: {str(e)}")


def cleanup_temp_files():
    """Clean up temporary files."""
    import shutil
    import tempfile
    
    # Clean up any remaining temporary directories
    temp_dir = Path(tempfile.gettempdir())
    for item in temp_dir.iterdir():
        if item.is_dir() and item.name.startswith('tmp'):
            try:
                shutil.rmtree(item)
            except:
                pass
