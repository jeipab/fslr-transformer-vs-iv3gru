"""
Sign Language Recognition Model Validation Script

This script validates trained Sign Language Recognition models on validation data.
Supports both Transformer and IV3-GRU models with comprehensive performance analysis.

Usage:
    python validate.py --model <model_type> --checkpoint <checkpoint_path> [options]

Example:
    python validate.py --model transformer --checkpoint transformer/model.pt --batch-size 32
"""

# Standard library imports
import argparse, json, os, sys, time, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Project imports - add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SignTransformer, InceptionV3GRU, MediaPipeGRU
from data.labels.label_mapping import load_label_mappings, format_prediction_results
from streamlit_app.core.config import MODEL_CONFIG, get_model_config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ValidationDataset:
    """
    Dataset class for loading and managing validation data.
    
    This class handles loading NPZ files and labels CSV, filtering valid samples,
    and providing data in the format expected by the models.
    """
    
    def __init__(self, data_dir: str, labels_csv: str, model_type: str, model=None, 
                 signer_filter: List[str] = None, category_filter: List[int] = None):
        """
        Initialize validation dataset by loading labels and filtering valid NPZ files.
        
        Steps:
        1. Load labels CSV with multiple encoding fallbacks
        2. Clean file names (remove .npz extension)
        3. Apply signer and category filters if specified
        4. Filter to only include files that actually exist
        5. Store metadata for each valid sample
        
        Args:
            data_dir: Directory containing NPZ files
            labels_csv: Path to labels CSV file
            model_type: 'transformer' or 'iv3_gru' (determines data format)
            model: Model instance (needed to check expected input dimensions)
            signer_filter: List of signer IDs to include (None for all)
            category_filter: List of category IDs to include (None for all)
        """
        self.data_dir = Path(data_dir)
        self.labels_csv = labels_csv
        self.model_type = model_type
        self.model = model
        self.signer_filter = signer_filter
        self.category_filter = category_filter
        
        # Step 1: Load labels CSV with encoding fallbacks
        # Try UTF-8 first, then fallback encodings for compatibility
        try:
            self.labels_df = pd.read_csv(labels_csv, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.labels_df = pd.read_csv(labels_csv, encoding='latin-1')
            except UnicodeDecodeError:
                self.labels_df = pd.read_csv(labels_csv, encoding='cp1252')
        
        # Step 2: Clean file names (remove .npz extension if present)
        self.labels_df['file'] = self.labels_df['file'].str.replace('.npz', '')
        
        # Step 3: Apply filters if specified
        if signer_filter is not None:
            self.labels_df = self.labels_df[self.labels_df['signer'].isin(signer_filter)]
            print(f"Filtered to signers: {signer_filter}")
        
        if category_filter is not None:
            self.labels_df = self.labels_df[self.labels_df['cat'].isin(category_filter)]
            print(f"Filtered to categories: {category_filter}")
        
        # Step 4: Filter to only include files that actually exist
        self.valid_files = []
        for _, row in self.labels_df.iterrows():
            npz_path = self.data_dir / f"{row['file']}.npz"
            if npz_path.exists():
                # Store all metadata needed for validation
                self.valid_files.append({
                    'file': row['file'],           # File identifier
                    'gloss': int(row['gloss']),    # Ground truth gloss label
                    'cat': int(row['cat']),        # Ground truth category label
                    'occluded': int(row['occluded']),  # Occlusion flag (0/1)
                    'signer': str(row['signer']),  # Signer ID
                    'duration': float(row['duration']),  # Duration in seconds
                    'npz_path': str(npz_path)      # Full path to NPZ file
                })
        
        print(f"Loaded {len(self.valid_files)} valid samples from {len(self.labels_df)} total labels")
        
        if len(self.valid_files) == 0:
            raise ValueError(f"No valid NPZ files found in {data_dir}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """
        Load a single sample from the dataset.
        
        Steps:
        1. Get sample metadata from valid_files list
        2. Load NPZ data from disk
        3. Extract appropriate features based on model type
        4. Handle sequence length limits
        5. Return data tensor and labels
        
        Args:
            idx: Index of sample to load
            
        Returns:
            Tuple of (features_tensor, gloss_label, category_label, occlusion_flag, filename)
        """
        sample = self.valid_files[idx]
        
        # Step 1: Load NPZ data from disk
        data = np.load(sample['npz_path'])
        
        if self.model_type == 'transformer':
            # Step 2: Extract features for transformer model (178-D keypoints)
            if 'X' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X' key for transformer model (expected 178-D keypoints)")
            X = torch.from_numpy(data['X']).float()
            
            # Step 3: Handle sequence length truncation (max 300 frames)
            if X.shape[0] > 300:
                X = X[:300, :]
            
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file'], sample['signer'], sample['duration']
        
        elif self.model_type == 'iv3_gru':
            # Step 2: Extract features for IV3-GRU model (requires 2048-D InceptionV3 features)
            if 'X2048' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X2048' key for IV3-GRU model (expected 2048-D features)")
            X = torch.from_numpy(data['X2048']).float()
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file'], sample['signer'], sample['duration']
        
        elif self.model_type == 'mediapipe_gru':
            # Step 2: Extract features for MediaPipe-GRU model (requires 178-D keypoint features)
            if 'X' not in data:
                raise ValueError(f"NPZ file {sample['npz_path']} missing 'X' key for MediaPipe-GRU model (expected 178-D keypoints)")
            X = torch.from_numpy(data['X']).float()
            return X, sample['gloss'], sample['cat'], sample['occluded'], sample['file'], sample['signer'], sample['duration']
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class ModelValidator:
    """
    Main validation class for comprehensive model evaluation.
    
    This class handles model loading, checkpoint restoration, and provides
    methods for batch prediction and comprehensive validation analysis.
    """
    
    def __init__(self, model_type: str, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize the validator by loading model architecture and checkpoint.
        
        Steps:
        1. Set device (GPU if available, otherwise CPU)
        2. Load model architecture with auto-detected parameters
        3. Load trained weights from checkpoint
        4. Load label mappings for human-readable results
        5. Set model to evaluation mode
        
        Args:
            model_type: 'transformer' or 'iv3_gru'
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        
        # Step 1: Set device (GPU if available and not forced to CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Step 2: Load model architecture with auto-detected parameters
        self.model = self._load_model()
        
        # Step 3: Load trained weights from checkpoint
        self._load_checkpoint()
        
        # Step 4: Load label mappings for human-readable results
        self.gloss_mapping, self.category_mapping = load_label_mappings()
        
        print(f"✓ Initialized {self.model_type} validator on {self.device}")
    
    def _load_model(self):
        """
        Load the appropriate model architecture using Streamlit config parameters.
        
        Steps:
        1. Get model configuration from Streamlit config
        2. Load checkpoint to inspect model parameters
        3. Auto-detect architecture parameters from checkpoint weights
        4. Create model instance with detected parameters
        5. Move model to target device
        
        Returns:
            Model instance ready for inference
        """
        # Step 1: Get model configuration from Streamlit config
        model_config = get_model_config(self.model_type)
        if not model_config:
            raise ValueError(f"Model type '{self.model_type}' not found in Streamlit config")
        
        # Step 2: Load checkpoint to inspect parameters
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        
        if self.model_type == 'transformer':
            # Step 3: Use input dimension from Streamlit config
            input_dim = model_config.get('input_dim', 178)
            print(f"Using input_dim={input_dim} from Streamlit config")
            
            # Step 4: Create transformer model with config parameters
            model = SignTransformer(
                input_dim=input_dim,                    # From Streamlit config
                emb_dim=256,                          # Fixed architecture parameter
                n_heads=8,                           # Fixed architecture parameter
                n_layers=4,                          # Fixed architecture parameter
                num_gloss=model_config['num_gloss_classes'],    # From Streamlit config
                num_cat=model_config['num_category_classes'],  # From Streamlit config
                dropout=0.1,                         # Fixed architecture parameter
                max_len=300,                         # Fixed: maximum sequence length
                pooling_method='mean'                # Fixed: pooling method
            )
            
        elif self.model_type == 'iv3_gru':
            # Step 3: Auto-detect GRU hidden sizes from checkpoint weights
            if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                # GRU weight_hh has shape [3*hidden, hidden] for each layer
                gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3
                gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3
                print(f"Detected GRU hidden sizes from checkpoint: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
            else:
                gru1_hidden = 16  # Default fallback
                gru2_hidden = 12  # Default fallback
                print(f"Warning: Could not detect GRU hidden sizes from checkpoint, using defaults: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
            
            # Step 4: Create IV3-GRU model with config parameters
            model = InceptionV3GRU(
                num_gloss=model_config['num_gloss_classes'],    # From Streamlit config
                num_cat=model_config['num_category_classes'],   # From Streamlit config
                hidden1=gru1_hidden,                           # Auto-detected from checkpoint
                hidden2=gru2_hidden,                           # Auto-detected from checkpoint
                dropout=0.3,                                   # Fixed architecture parameter
                pretrained_backbone=True,                      # Fixed: use pretrained InceptionV3
                freeze_backbone=True                           # Fixed: freeze backbone weights
            )
            
        elif self.model_type == 'mediapipe_gru':
            # Step 3: Auto-detect GRU hidden sizes from checkpoint weights
            if 'gru1.weight_hh_l0' in state_dict and 'gru2.weight_hh_l0' in state_dict:
                # GRU weight_hh has shape [3*hidden, hidden] for each layer
                gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3
                gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3
                print(f"Detected GRU hidden sizes from checkpoint: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
            else:
                gru1_hidden = 256  # Default fallback
                gru2_hidden = 128  # Default fallback
                print(f"Warning: Could not detect GRU hidden sizes from checkpoint, using defaults: hidden1={gru1_hidden}, hidden2={gru2_hidden}")
            
            # Step 4: Create MediaPipe-GRU model with config parameters
            model = MediaPipeGRU(
                num_gloss=model_config['num_gloss_classes'],    # From Streamlit config
                num_cat=model_config['num_category_classes'],   # From Streamlit config
                input_dim=model_config['input_dim'],           # From Streamlit config
                hidden1=gru1_hidden,                           # Auto-detected from checkpoint
                hidden2=gru2_hidden,                           # Auto-detected from checkpoint
                dropout=0.3,                                   # Fixed architecture parameter
                bidirectional=False                            # Fixed: unidirectional GRU
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Step 5: Move model to target device
        return model.to(self.device)
    
    def _load_checkpoint(self):
        """
        Load trained model weights from checkpoint file.
        
        Steps:
        1. Verify checkpoint file exists
        2. Load checkpoint data to device
        3. Handle different checkpoint formats (PyTorch Lightning, custom, etc.)
        4. Load weights into model
        5. Set model to evaluation mode
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        # Step 1: Verify checkpoint file exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Step 2: Load checkpoint data to device
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Step 3: Handle different checkpoint formats
        # PyTorch Lightning format: {'model_state_dict': ...}
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        # Custom format: {'state_dict': ...}
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        # Direct format: {'model': ...}
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        # Raw format: direct state dict
        else:
            self.model.load_state_dict(checkpoint)
        
        # Step 4: Set model to evaluation mode (disable dropout, batch norm updates)
        self.model.eval()
        print(f"✓ Loaded checkpoint from {self.checkpoint_path}")
    
    def predict_batch(self, batch_data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch of variable-length sequences.
        
        Steps:
        1. Handle variable-length sequences by padding to same length
        2. Create attention masks (transformer) or length tensors (GRU)
        3. Move data to device
        4. Run inference with no gradient computation
        5. Return prediction logits
        
        Args:
            batch_data: List of tensors with shape [seq_len, features] (one per sample)
            
        Returns:
            Tuple of (gloss_logits, category_logits) with shape [batch_size, num_classes]
        """
        if self.model_type == 'transformer':
            # Step 1: Pad sequences to same length for transformer
            max_len = max(x.shape[0] for x in batch_data)
            padded_batch = []
            masks = []
            
            for x in batch_data:
                if x.shape[0] < max_len:
                    # Pad with zeros to max length
                    pad_len = max_len - x.shape[0]
                    padded_x = torch.cat([x, torch.zeros(pad_len, x.shape[1])], dim=0)
                    # Create attention mask: 1 for real data, 0 for padding
                    mask = torch.cat([torch.ones(x.shape[0]), torch.zeros(pad_len)], dim=0)
                else:
                    padded_x = x
                    mask = torch.ones(x.shape[0])
                
                padded_batch.append(padded_x)
                masks.append(mask)
            
            # Step 2: Stack into batch tensors and move to device
            X = torch.stack(padded_batch).to(self.device)  # [batch_size, max_len, features]
            mask = torch.stack(masks).bool().to(self.device)  # [batch_size, max_len]
            
            # Step 3: Run inference
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, mask)
        
        elif self.model_type == 'iv3_gru':
            # Step 1: Get sequence lengths and pad sequences for GRU
            lengths = torch.tensor([x.shape[0] for x in batch_data], dtype=torch.long)
            max_len = max(x.shape[0] for x in batch_data)
            
            padded_batch = []
            for x in batch_data:
                if x.shape[0] < max_len:
                    # Pad with zeros to max length
                    pad_len = max_len - x.shape[0]
                    padded_x = torch.cat([x, torch.zeros(pad_len, x.shape[1])], dim=0)
                else:
                    padded_x = x
                padded_batch.append(padded_x)
            
            # Step 2: Stack into batch tensors and move to device
            X = torch.stack(padded_batch).to(self.device)  # [batch_size, max_len, features]
            lengths = lengths.to(self.device)  # [batch_size]
            
            # Step 3: Run inference (features_already=True means input is already InceptionV3 features)
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, lengths, features_already=True)
        
        elif self.model_type == 'mediapipe_gru':
            # Step 1: Get sequence lengths and pad sequences for GRU
            lengths = torch.tensor([x.shape[0] for x in batch_data], dtype=torch.long)
            max_len = max(x.shape[0] for x in batch_data)
            
            padded_batch = []
            for x in batch_data:
                if x.shape[0] < max_len:
                    # Pad with zeros to max length
                    pad_len = max_len - x.shape[0]
                    padded_x = torch.cat([x, torch.zeros(pad_len, x.shape[1])], dim=0)
                else:
                    padded_x = x
                padded_batch.append(padded_x)
            
            # Step 2: Stack into batch tensors and move to device
            X = torch.stack(padded_batch).to(self.device)  # [batch_size, max_len, features]
            lengths = lengths.to(self.device)  # [batch_size]
            
            # Step 3: Run inference
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X, lengths)
        
        return gloss_logits, cat_logits
    
    def validate(self, dataset: ValidationDataset, batch_size: int = 32, 
                save_predictions: bool = False, output_dir: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the dataset.
        
        Steps:
        1. Initialize validation session and display progress
        2. Process dataset in batches for memory efficiency
        3. Load batch data and make predictions
        4. Extract predictions, probabilities, and top-k results
        5. Store detailed results for each sample
        6. Convert to numpy arrays for analysis
        7. Compute comprehensive metrics
        
        Args:
            dataset: ValidationDataset instance
            batch_size: Batch size for evaluation (affects memory usage)
            save_predictions: Whether to save individual predictions to JSON files
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing all validation results and metrics
        """
        # Step 1: Initialize validation session
        print(f"\n{'='*60}")
        print(f"VALIDATING {self.model_type.upper()} MODEL")
        print(f"{'='*60}")
        print(f"Dataset: {len(dataset)} samples")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        
        # Initialize results storage
        all_predictions = []      # Detailed prediction results for each sample
        all_ground_truth = []     # Ground truth labels
        all_occlusions = []       # Occlusion flags
        all_files = []            # File names
        all_signers = []          # Signer IDs
        all_durations = []        # Duration values
        all_gloss_probs = []      # All gloss probabilities for top-k accuracy
        all_cat_probs = []        # All category probabilities for top-k accuracy
        
        # Step 2: Process dataset in batches for memory efficiency
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        with tqdm(total=len(dataset), desc="Validating") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                
                # Step 3: Load batch data from dataset
                batch_data = []      # Feature tensors
                batch_gloss = []     # Ground truth gloss labels
                batch_cat = []       # Ground truth category labels
                batch_occluded = []  # Occlusion flags
                batch_files = []     # File names
                batch_signers = []   # Signer IDs
                batch_durations = [] # Duration values
                
                for i in range(start_idx, end_idx):
                    X, gloss, cat, occluded, file, signer, duration = dataset[i]
                    batch_data.append(X)
                    batch_gloss.append(gloss)
                    batch_cat.append(cat)
                    batch_occluded.append(occluded)
                    batch_files.append(file)
                    batch_signers.append(signer)
                    batch_durations.append(duration)
                
                # Step 4: Make predictions on batch
                gloss_logits, cat_logits = self.predict_batch(batch_data)
                
                # Step 5: Extract predictions and probabilities
                gloss_preds = gloss_logits.argmax(dim=1).cpu().numpy()  # Predicted gloss classes
                cat_preds = cat_logits.argmax(dim=1).cpu().numpy()      # Predicted category classes
                
                # Convert logits to probabilities
                gloss_probs = F.softmax(gloss_logits, dim=1).cpu().numpy()
                cat_probs = F.softmax(cat_logits, dim=1).cpu().numpy()
                
                # Step 6: Store detailed results for each sample
                for i in range(len(batch_data)):
                    all_predictions.append({
                        'file': batch_files[i],                    # File identifier
                        'gloss_pred': int(gloss_preds[i]),         # Predicted gloss class
                        'cat_pred': int(cat_preds[i]),             # Predicted category class
                        'gloss_gt': batch_gloss[i],                # Ground truth gloss class
                        'cat_gt': batch_cat[i],                    # Ground truth category class
                        'occluded': batch_occluded[i],             # Occlusion flag
                        'signer': batch_signers[i],                # Signer ID
                        'duration': batch_durations[i],            # Duration in seconds
                        'gloss_prob': float(gloss_probs[i][gloss_preds[i]]),  # Prediction confidence
                        'cat_prob': float(cat_probs[i][cat_preds[i]]),        # Prediction confidence
                        'gloss_top10': [(int(j), float(gloss_probs[i][j]))    # Top 10 gloss predictions
                                      for j in np.argsort(gloss_probs[i])[-10:][::-1]],
                        'cat_top5': [(int(j), float(cat_probs[i][j]))         # Top 5 category predictions
                                   for j in np.argsort(cat_probs[i])[-5:][::-1]]
                    })
                    
                    # Store probabilities for top-k accuracy computation
                    all_gloss_probs.append(gloss_probs[i])
                    all_cat_probs.append(cat_probs[i])
                
                # Store batch metadata
                all_ground_truth.extend(list(zip(batch_gloss, batch_cat)))
                all_occlusions.extend(batch_occluded)
                all_files.extend(batch_files)
                all_signers.extend(batch_signers)
                all_durations.extend(batch_durations)
                
                pbar.update(end_idx - start_idx)
        
        # Step 7: Convert to numpy arrays for analysis
        gloss_preds = np.array([p['gloss_pred'] for p in all_predictions])
        cat_preds = np.array([p['cat_pred'] for p in all_predictions])
        gloss_gts = np.array([p['gloss_gt'] for p in all_predictions])
        cat_gts = np.array([p['cat_gt'] for p in all_predictions])
        occlusions = np.array(all_occlusions)
        gloss_probs = np.array(all_gloss_probs)
        cat_probs = np.array(all_cat_probs)
        
        # Step 8: Compute comprehensive metrics
        results = self._compute_metrics(
            gloss_preds, cat_preds, gloss_gts, cat_gts, 
            occlusions, gloss_probs, cat_probs,
            all_predictions, all_signers, all_durations,
            save_predictions, output_dir
        )
        
        return results
    
    def _compute_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                        gloss_gts: np.ndarray, cat_gts: np.ndarray,
                        occlusions: np.ndarray, gloss_probs: np.ndarray, cat_probs: np.ndarray,
                        all_predictions: List[Dict], all_signers: List[str], all_durations: List[float],
                        save_predictions: bool, output_dir: str) -> Dict[str, Any]:
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
        
        # Per-signer metrics
        per_signer_results = self._compute_per_signer_metrics(gloss_preds, cat_preds, gloss_gts, cat_gts, all_signers)
        
        # Per-category metrics
        per_category_results = self._compute_per_category_metrics(gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Duration analysis
        duration_analysis = self._compute_duration_analysis(all_durations, gloss_preds, cat_preds, gloss_gts, cat_gts)
        
        # Confusion matrices with proper TP, FP, TN, FN calculations
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
                'non_occluded_samples': int(np.sum(non_occluded_mask)),
                'unique_signers': len(set(all_signers)),
                'signers': list(set(all_signers)),
                'duration_stats': {
                    'mean': float(np.mean(all_durations)),
                    'std': float(np.std(all_durations)),
                    'min': float(np.min(all_durations)),
                    'max': float(np.max(all_durations))
                }
            },
            'overall_results': overall_results,
            'occluded_results': occluded_results,
            'non_occluded_results': non_occluded_results,
            'per_class_results': per_class_results,
            'per_signer_results': per_signer_results,
            'per_category_results': per_category_results,
            'duration_analysis': duration_analysis,
            'confusion_matrices': confusion_matrices,
            'detailed_predictions': all_predictions
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
        gloss_report = classification_report(
            gloss_gts, gloss_preds, output_dict=True, zero_division=0
        )
        cat_report = classification_report(
            cat_gts, cat_preds, output_dict=True, zero_division=0
        )
        
        # Transform reports to include actual labels and rename support to occurrences
        gloss_per_class_with_labels = self._transform_report_with_labels(gloss_report, self.gloss_mapping)
        cat_per_class_with_labels = self._transform_report_with_labels(cat_report, self.category_mapping)
        
        return {
            'gloss_per_class': gloss_per_class_with_labels,
            'category_per_class': cat_per_class_with_labels
        }
    
    def _transform_report_with_labels(self, report: Dict[str, Any], label_mapping: Dict[int, str]) -> Dict[str, Any]:
        """
        Transform classification report to include actual labels and rename support to occurrences.
        
        Args:
            report: Classification report from sklearn
            label_mapping: Dictionary mapping class IDs to label names
            
        Returns:
            Transformed report with labels and renamed support column
        """
        transformed_report = {}
        
        for class_id_str, metrics in report.items():
            if class_id_str.isdigit():  # Only numeric class IDs
                class_id = int(class_id_str)
                label_name = label_mapping.get(class_id, f'Unknown_{class_id}')
                
                # Keep numeric key but add label information and rename support to occurrences
                transformed_report[class_id_str] = {
                    'class': f"{label_name} ({class_id})",  # Format like individual predictions
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'occurrences': metrics['support']  # Rename support to occurrences
                }
            else:
                # Keep non-numeric keys (like 'accuracy', 'macro avg', 'weighted avg') as-is
                transformed_report[class_id_str] = metrics
        
        return transformed_report
    
    def _compute_confusion_matrices(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                                  gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute confusion matrices with proper TP, FP, TN, FN calculations."""
        gloss_cm = confusion_matrix(gloss_gts, gloss_preds)
        cat_cm = confusion_matrix(cat_gts, cat_preds)
        
        # Calculate TP, FP, TN, FN for each class
        def calculate_class_metrics(cm):
            metrics = {}
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[i] = {
                    'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                    'Precision': float(precision), 'Recall': float(recall), 'F1': float(f1)
                }
            return metrics
        
        return {
            'gloss_confusion_matrix': gloss_cm.tolist(),
            'category_confusion_matrix': cat_cm.tolist(),
            'gloss_class_metrics': calculate_class_metrics(gloss_cm),
            'category_class_metrics': calculate_class_metrics(cat_cm)
        }
    
    def _compute_per_signer_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                                   gloss_gts: np.ndarray, cat_gts: np.ndarray, 
                                   all_signers: List[str]) -> Dict[str, Any]:
        """Compute per-signer accuracy metrics."""
        unique_signers = list(set(all_signers))
        per_signer_results = {}
        
        for signer in unique_signers:
            signer_mask = np.array([s == signer for s in all_signers])
            if np.sum(signer_mask) == 0:
                continue
                
            signer_gloss_preds = gloss_preds[signer_mask]
            signer_cat_preds = cat_preds[signer_mask]
            signer_gloss_gts = gloss_gts[signer_mask]
            signer_cat_gts = cat_gts[signer_mask]
            
            # Compute metrics for this signer
            signer_metrics = self._compute_overall_metrics(
                signer_gloss_preds, signer_cat_preds, signer_gloss_gts, signer_cat_gts
            )
            signer_metrics['num_samples'] = int(np.sum(signer_mask))
            
            per_signer_results[signer] = signer_metrics
        
        return per_signer_results
    
    def _compute_per_category_metrics(self, gloss_preds: np.ndarray, cat_preds: np.ndarray,
                                    gloss_gts: np.ndarray, cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute per-category accuracy metrics."""
        unique_categories = list(set(cat_gts))
        per_category_results = {}
        
        for category in unique_categories:
            cat_mask = cat_gts == category
            if np.sum(cat_mask) == 0:
                continue
                
            cat_gloss_preds = gloss_preds[cat_mask]
            cat_gloss_gts = gloss_gts[cat_mask]
            
            # Compute gloss accuracy for this category
            cat_gloss_acc = accuracy_score(cat_gloss_gts, cat_gloss_preds)
            
            # Get category name from mapping
            cat_name = self.category_mapping.get(category, f'Category_{category}')
            
            per_category_results[category] = {
                'category_name': cat_name,
                'gloss_accuracy': float(cat_gloss_acc),
                'num_samples': int(np.sum(cat_mask))
            }
        
        return per_category_results
    
    def _compute_duration_analysis(self, all_durations: List[float], gloss_preds: np.ndarray, 
                                 cat_preds: np.ndarray, gloss_gts: np.ndarray, 
                                 cat_gts: np.ndarray) -> Dict[str, Any]:
        """Compute duration-based analysis."""
        durations = np.array(all_durations)
        
        # Duration bins for analysis
        duration_bins = [0, 1, 2, 3, 5, 10, float('inf')]
        bin_labels = ['0-1s', '1-2s', '2-3s', '3-5s', '5-10s', '10s+']
        
        duration_analysis = {
            'overall_stats': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
                'median': float(np.median(durations))
            },
            'bin_analysis': {}
        }
        
        # Analyze performance by duration bins
        for i in range(len(duration_bins) - 1):
            bin_min = duration_bins[i]
            bin_max = duration_bins[i + 1]
            
            if bin_max == float('inf'):
                bin_mask = durations >= bin_min
            else:
                bin_mask = (durations >= bin_min) & (durations < bin_max)
            
            if np.sum(bin_mask) == 0:
                continue
                
            bin_gloss_preds = gloss_preds[bin_mask]
            bin_cat_preds = cat_preds[bin_mask]
            bin_gloss_gts = gloss_gts[bin_mask]
            bin_cat_gts = cat_gts[bin_mask]
            
            bin_metrics = self._compute_overall_metrics(
                bin_gloss_preds, bin_cat_preds, bin_gloss_gts, bin_cat_gts
            )
            
            duration_analysis['bin_analysis'][bin_labels[i]] = {
                'duration_range': f"{bin_min}-{bin_max}s" if bin_max != float('inf') else f"{bin_min}s+",
                'num_samples': int(np.sum(bin_mask)),
                'metrics': bin_metrics
            }
        
        return duration_analysis
    
    def _save_individual_predictions(self, predictions: List[Dict], output_dir: str):
        """Save individual predictions to JSON files."""
        pred_dir = Path(output_dir) / 'individual_predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        for pred in predictions:
            # Format prediction results
            formatted_pred = {
                'file': pred['file'],
                'signer': pred['signer'],
                'duration': pred['duration'],
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
                'gloss_top10': [
                    [f"{self.gloss_mapping.get(gloss_id, 'Unknown')} ({gloss_id})", prob]
                    for gloss_id, prob in pred['gloss_top10']
                ],
                'category_top5': [
                    [f"{self.category_mapping.get(cat_id, 'Unknown')} ({cat_id})", prob]
                    for cat_id, prob in pred['cat_top5']
                ],
                'correct': {
                    'gloss': pred['gloss_pred'] == pred['gloss_gt'],
                    'category': pred['cat_pred'] == pred['cat_gt']
                }
            }
            
            # Save to file
            output_file = pred_dir / f"{pred['file']}_validation.json"
            with open(output_file, 'w', encoding='utf-8') as f:
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
        
        # Per-signer results
        per_signer = results['per_signer_results']
        print(f"\nPER-SIGNER PERFORMANCE:")
        for signer, metrics in per_signer.items():
            print(f"  Signer {signer}: Gloss Acc={metrics['gloss_accuracy']:.4f}, "
                  f"Cat Acc={metrics['category_accuracy']:.4f} ({metrics['num_samples']} samples)")
        
        # Duration analysis
        duration_stats = results['dataset_info']['duration_stats']
        print(f"\nDURATION ANALYSIS:")
        print(f"  Mean Duration: {duration_stats['mean']:.2f}s")
        print(f"  Duration Range: {duration_stats['min']:.2f}s - {duration_stats['max']:.2f}s")
        
        # Per-category results
        per_category = results['per_category_results']
        print(f"\nPER-CATEGORY PERFORMANCE:")
        for cat_id, metrics in per_category.items():
            print(f"  {metrics['category_name']}: Gloss Acc={metrics['gloss_accuracy']:.4f} "
                  f"({metrics['num_samples']} samples)")


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
        ('per_signer_results.json', results['per_signer_results']),
        ('per_category_results.json', results['per_category_results']),
        ('duration_analysis.json', results['duration_analysis']),
        ('confusion_matrices.json', results['confusion_matrices'])
    ]
    
    for filename, data in files_to_save:
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    # Save complete results
    complete_filepath = output_path / 'complete_validation_results.json'
    with open(complete_filepath, 'w', encoding='utf-8') as f:
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
    parser.add_argument('--model', choices=['transformer', 'iv3_gru', 'mediapipe_gru'], required=True,
                       help='Model type to validate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    
    # Optional arguments
    parser.add_argument('--data-dir', type=str, 
                       default='data/processed/fsl_val',
                       help='Directory containing validation NPZ files')
    parser.add_argument('--labels-csv', type=str, 
                       default='data/processed/fsl_val.csv',
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
    parser.add_argument('--signer-filter', type=str, nargs='+', default=None,
                       help='Filter by specific signer(s) (e.g., --signer-filter S1 S2)')
    parser.add_argument('--category-filter', type=int, nargs='+', default=None,
                       help='Filter by specific category(ies) (e.g., --category-filter 0 1 2)')
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = ModelValidator(args.model, args.checkpoint, args.device)
        
        # Load dataset
        dataset = ValidationDataset(args.data_dir, args.labels_csv, args.model, validator.model,
                                  signer_filter=args.signer_filter, category_filter=args.category_filter)
        
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
