"""
Training entrypoint for sign language recognition models.

This comprehensive training module supports:

1. MULTI-TASK TRAINING:
   - Joint gloss and category classification with configurable loss weights
   - Advanced loss weighting strategies (static, grid-search, uncertainty, gradnorm)
   - Curriculum learning with different strategies (gloss-first, category-first, dynamic)

2. MODEL SUPPORT:
   - SignTransformer: Multi-head attention transformer for keypoint sequences
   - InceptionV3GRU: Hybrid CNN-RNN model for visual features
   - Automatic model compilation and parallel processing optimization

3. DATA HANDLING:
   - File-based datasets from preprocessed .npz files
   - Support for keypoints [T, 178], features [T, 2048], or combined [T, 2226]
   - Combined mode: concatenates keypoints + features for richer representations
   - Temporal data augmentation (noise, masking)
   - Variable-length sequence padding and batching

4. TRAINING FEATURES:
   - Automatic Mixed Precision (AMP) for faster training
   - Learning rate scheduling (plateau, cosine, warmup-cosine)
   - Early stopping and checkpointing
   - Resume training from checkpoints
   - Automatic logging: CSV metrics + console logs saved with timestamps
   - Exponential Moving Average (EMA) for model stability

5. ADVANCED LOSS FUNCTIONS:
   - Standard CrossEntropy
   - Focal Loss for class imbalance
   - Label Smoothing for better generalization

Usage:
    # Basic training (automatic logging with timestamps)
    python training/train.py --model transformer --epochs 50 \\
        --output-dir trained_models/transformer/run1
    
    # Advanced training with curriculum learning
    python training/train.py --model iv3_gru --curriculum gloss-first --epochs 100 \\
        --output-dir trained_models/iv3_gru/run1
    
    # Smoke test
    python training/train.py --smoke-test
"""

# Standard library imports
import os, csv, random, argparse, time, sys, platform
from datetime import datetime
from typing import Tuple, Callable

# Third-party imports
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Local imports
from models import InceptionV3GRU, SignTransformer, MediaPipeGRU, SignTransformerCtc, MediaPipeGRUCtc, InceptionV3GRUCtc
from streamlit_app.core.config import CTC_CONFIG

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class TeeLogger:
    """
    Utility class to duplicate stdout to both console and a log file.
    
    This allows capturing all printed output during training to a file while
    still displaying it on the console in real-time.
    
    Usage:
        sys.stdout = TeeLogger('training.log')
        print("This goes to both console and file")
        sys.stdout.close()  # Close the file when done
    """
    def __init__(self, log_path):
        """
        Initialize the TeeLogger.
        
        Args:
            log_path (str): Path to the log file to write to
        """
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', encoding='utf-8')
        
    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file
        
    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        """Close the log file."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# ============================================================================
# DATASET CLASSES
# ============================================================================

class FSLFeatureFileDataset(Dataset):
    """
    PyTorch Dataset for precomputed visual features from InceptionV3 backbone.
    
    This dataset loads pre-extracted features with shape [T, 2048] from .npz files,
    where T is the temporal dimension (variable length sequences) and 2048 is the
    feature dimension from InceptionV3's final layer.
    
    The dataset expects:
    - A directory of .npz files containing feature arrays
    - A CSV file mapping filenames to gloss, category, occlusion, signer, and duration labels
    - Optional temporal augmentation for training data
    
    Data Flow:
    1. Load CSV to build filename -> (gloss, category, occluded, signer, duration) mapping
    2. For each sample, load corresponding .npz file
    3. Extract feature array using specified key (default: 'X2048')
    4. Apply temporal augmentation if enabled and in training mode
    5. Return data based on mode (classification or CTC)

    Args:
        features_dir (str): Directory containing .npz feature files
        labels_csv (str): CSV file with columns: file, gloss, cat, occluded, signer, duration
        feature_key (str): Key in .npz files containing [T, 2048] features
        augment (bool): Enable temporal data augmentation
        augment_params (dict): Augmentation parameters (noise_std, mask_prob, etc.)
        mode (str): Dataset mode - 'classification' or 'ctc'
        signer_filter (str, optional): Filter dataset to only include samples from this signer
        return_metadata (bool): Whether to return signer and duration in __getitem__

    Returns:
        Classification mode: (features[T,2048] float32, gloss long, cat long, length long)
        CTC mode: (features[T,2048] float32, gloss_seq[1] long, input_length long, target_length long, cat long)
        
    Raises:
        ValueError: If CSV format is invalid or required columns missing
        FileNotFoundError: If feature file doesn't exist
        KeyError: If expected feature key not found in .npz file
    """
    def __init__(self, features_dir, labels_csv, feature_key='X2048', augment=False, augment_params=None, mode='classification', signer_filter=None, return_metadata=False):
        """
        Initialize the feature dataset.
        
        This method sets up the dataset by:
        1. Storing configuration parameters
        2. Setting up augmentation if enabled
        3. Loading and parsing the labels CSV file
        4. Building an index of (filename_stem, gloss_id, category_id, occluded, signer, duration) tuples
        5. Optionally filtering by signer
        
        Args:
            features_dir (str): Directory containing .npz feature files
            labels_csv (str): CSV file with columns: file, gloss, cat, occluded, signer, duration
            feature_key (str): Key to extract features from .npz files
            augment (bool): Whether to apply temporal augmentation
            augment_params (dict): Augmentation parameters
            mode (str): Dataset mode - 'classification' or 'ctc'
            signer_filter (str, optional): Filter dataset to only include samples from this signer
            return_metadata (bool): Whether to return signer and duration in __getitem__
        """
        # Store dataset configuration
        self.features_dir = features_dir  # Directory containing .npz feature files
        self.feature_key = feature_key    # Key to extract features from .npz files
        self.index = []                   # List of (stem, gloss, cat, occluded, signer, duration) tuples for indexing
        self.augment = augment            # Whether to apply temporal augmentation
        self.training = True              # Training mode flag (set by DataLoader)
        self.mode = mode                  # Dataset mode: 'classification' or 'ctc'
        self.signer_filter = signer_filter  # Optional signer filter
        self.return_metadata = return_metadata  # Whether to return metadata in __getitem__
        
        # Validate mode parameter
        if mode not in ['classification', 'ctc']:
            raise ValueError(f"mode must be 'classification' or 'ctc', got '{mode}'")
        
        # Initialize temporal augmentation if enabled
        if augment and augment_params:
            # Use custom augmentation parameters
            self.augmentation = TemporalAugmentation(**augment_params)
        elif augment:
            # Use default augmentation parameters
            self.augmentation = TemporalAugmentation()

        # Validate that labels CSV is provided
        if labels_csv is None:
            raise ValueError("labels_csv must be provided for feature dataset")

        # Load and parse the labels CSV file
        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            
            # Validate CSV structure - must have required columns
            required = {'file', 'gloss', 'cat', 'occluded', 'signer', 'duration'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"labels_csv must have columns: {required}")
            
            # Parse each row and build the dataset index
            for row in reader:
                try:
                    # Extract filename stem (without extension) for flexibility
                    # This allows CSV to have filenames with or without .npz extension
                    stem = os.path.splitext(row['file'])[0]
                    
                    # Convert labels to appropriate types
                    gloss = int(row['gloss'])  # Gloss class ID
                    cat = int(row['cat'])       # Category class ID
                    occluded = int(row['occluded'])  # Occlusion flag
                    signer = row['signer']      # Signer identifier
                    duration = float(row['duration'])  # Duration in seconds
                    
                    # Apply signer filter if specified
                    if signer_filter is not None and signer != signer_filter:
                        continue
                    
                    # Add to index for later retrieval
                    self.index.append((stem, gloss, cat, occluded, signer, duration))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid data in row {row}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        This method:
        1. Gets the filename and labels from the index
        2. Constructs the full path to the .npz file
        3. Loads and validates the feature data
        4. Applies temporal augmentation if enabled
        5. Returns tensors in the expected format based on mode
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Classification mode: (features[T,2048] float32, gloss_label long, cat_label long, length long) or with metadata
            CTC mode: (features[T,2048] float32, gloss_seq[1] long, input_length long, target_length long, cat long) or with metadata
        """
        # Get filename stem and labels from the pre-built index
        stem, gloss, cat, occluded, signer, duration = self.index[idx]
        
        # Construct full path to the .npz feature file
        path = os.path.join(self.features_dir, stem + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        
        # Load feature data from .npz file and convert to PyTorch tensor
        data = torch.from_numpy(self._load_npz_features(path))  # Shape: [T, 2048]
        input_length = data.shape[0]  # Temporal dimension (sequence length)
        
        # Apply temporal augmentation if enabled and in training mode
        # Augmentation helps improve model generalization by adding noise/variations
        if self.augment and self.training and hasattr(self, 'augmentation'):
            data = self.augmentation(data)
        
        # Return tensors based on mode
        if self.mode == 'ctc':
            gloss_label_seq = torch.tensor([gloss], dtype=torch.long)
            cat_label_seq = torch.tensor([cat], dtype=torch.long)
            target_length = torch.tensor([1], dtype=torch.long)
            
            base_return = (
                data.float(),                                    # Features as float32 [T, 2048]
                gloss_label_seq,                                 # Gloss sequence [1]
                torch.tensor(input_length, dtype=torch.long),    # Input sequence length
                target_length,                                   # Target sequence length [1]
                cat_label_seq                                    # Category sequence [1]
            )
            
            # Add metadata if requested
            if self.return_metadata:
                return base_return + (signer, duration)
            else:
                return base_return
        else:
            # Classification mode: original format
            base_return = (
                data.float(),                                    # Features as float32
                torch.tensor(gloss, dtype=torch.long),          # Gloss label as int64
                torch.tensor(cat, dtype=torch.long),            # Category label as int64
                torch.tensor(input_length, dtype=torch.long)    # Sequence length as int64
            )
            
            # Add metadata if requested
            if self.return_metadata:
                return base_return + (signer, duration)
            else:
                return base_return

    def _load_npz_features(self, path):
        """
        Load feature array from .npz file with validation.
        
        This method:
        1. Opens the .npz file safely
        2. Tries to load features using the specified key
        3. Falls back to 'X' key if primary key not found
        4. Validates the array shape and dimensions
        5. Returns the validated feature array
        
        Args:
            path (str): Path to the .npz file
            
        Returns:
            np.ndarray: Feature array with shape [T, 2048]
            
        Raises:
            KeyError: If neither the specified key nor 'X' key is found
            ValueError: If array doesn't have expected shape [T, 2048]
        """
        # Load .npz file with allow_pickle=True for compatibility
        with np.load(path, allow_pickle=True) as npz:
            # Try to load features using the specified key first
            if self.feature_key in npz:
                X = np.array(npz[self.feature_key])
            # Fall back to 'X' key if primary key not found
            elif 'X' in npz:
                X = np.array(npz['X'])
            else:
                raise KeyError(f"Neither '{self.feature_key}' nor 'X' found in {path}")
        
        # Validate array dimensions - must be 2D with 2048 features
        if X.ndim != 2 or X.shape[-1] != 2048:
            raise ValueError(f"Expected [T,2048] features in {path}, got shape {X.shape}")
        
        return X

class FSLKeypointFileDataset(Dataset):
    """
    PyTorch Dataset for precomputed keypoint sequences from pose estimation.
    
    This dataset loads keypoint sequences with shape [T, 178] from .npz files,
    where T is the temporal dimension (variable length sequences) and 178 represents
    the flattened keypoint coordinates (89 keypoints × 2 coordinates = 178).
    
    The dataset supports both raw keypoints and processed features:
    - Raw keypoints [T, 178]: Direct pose estimation output (89 keypoints: 25 pose + 21 left hand + 21 right hand + 22 face)
    - Processed features [T, 2048]: Keypoints processed through feature extraction
    
    Data Flow:
    1. Load CSV to build filename -> (gloss, category, occluded, signer, duration) mapping
    2. For each sample, load corresponding .npz file
    3. Extract keypoint array using specified key (default: 'X')
    4. Validate array dimensions based on key type
    5. Apply temporal augmentation if enabled and in training mode
    6. Return data based on mode (classification or CTC)

    Args:
        keypoints_dir (str): Directory containing .npz keypoint files
        labels_csv (str): CSV file with columns: file, gloss, cat, occluded, signer, duration
        kp_key (str): Key in .npz files containing keypoint data
        augment (bool): Enable temporal data augmentation
        augment_params (dict): Augmentation parameters (noise_std, mask_prob, etc.)
        mode (str): Dataset mode - 'classification' or 'ctc'
        signer_filter (str, optional): Filter dataset to only include samples from this signer
        return_metadata (bool): Whether to return signer and duration in __getitem__

    Returns:
        Classification mode: (keypoints[T,D] float32, gloss long, cat long, length long)
        CTC mode: (keypoints[T,D] float32, gloss_seq[1] long, input_length long, target_length long, cat long)
        where D is 178 for raw keypoints or 2048 for processed features
        
    Raises:
        ValueError: If CSV format is invalid or array dimensions don't match key type
        FileNotFoundError: If keypoint file doesn't exist
        KeyError: If expected keypoint key not found in .npz file
    """
    def __init__(self, keypoints_dir, labels_csv, kp_key='X', augment=False, augment_params=None, mode='classification', signer_filter=None, return_metadata=False):
        self.keypoints_dir = keypoints_dir
        self.kp_key = kp_key
        self.index = []  # list of (stem, gloss, cat, occluded, signer, duration)
        self.augment = augment
        self.training = True  # Will be set by DataLoader
        self.mode = mode
        self.signer_filter = signer_filter  # Optional signer filter
        self.return_metadata = return_metadata  # Whether to return metadata in __getitem__
        
        # Validate mode parameter
        if mode not in ['classification', 'ctc']:
            raise ValueError(f"mode must be 'classification' or 'ctc', got '{mode}'")
        
        if augment and augment_params:
            self.augmentation = TemporalAugmentation(**augment_params)
        elif augment:
            self.augmentation = TemporalAugmentation()

        if labels_csv is None:
            raise ValueError("labels_csv must be provided for keypoint dataset")

        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            required = {'file', 'gloss', 'cat', 'occluded', 'signer', 'duration'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"labels_csv must have columns: {required}")
            for row in reader:
                try:
                    stem = os.path.splitext(row['file'])[0]
                    gloss = int(row['gloss'])
                    cat = int(row['cat'])
                    occluded = int(row['occluded'])
                    signer = row['signer']
                    duration = float(row['duration'])
                    
                    # Apply signer filter if specified
                    if signer_filter is not None and signer != signer_filter:
                        continue
                    
                    self.index.append((stem, gloss, cat, occluded, signer, duration))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid data in row {row}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stem, gloss, cat, occluded, signer, duration = self.index[idx]
        path = os.path.join(self.keypoints_dir, stem + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Keypoint file not found: {path}")
        data = torch.from_numpy(self._load_npz_keypoints(path))  # [T, 178] or [T, 2048]
        input_length = data.shape[0]
        
        # Apply augmentation if enabled and in training mode
        if self.augment and self.training and hasattr(self, 'augmentation'):
            data = self.augmentation(data)
        
        # Return tensors based on mode
        if self.mode == 'ctc':
            gloss_label_seq = torch.tensor([gloss], dtype=torch.long)
            cat_label_seq = torch.tensor([cat], dtype=torch.long)
            target_length = torch.tensor([1], dtype=torch.long)
            
            base_return = (
                data.float(),
                gloss_label_seq,
                torch.tensor(input_length, dtype=torch.long),
                target_length,
                cat_label_seq
            )
            
            # Add metadata if requested
            if self.return_metadata:
                return base_return + (signer, duration)
            else:
                return base_return
        else:
            # Classification mode: original format
            base_return = (
                data.float(),
                torch.tensor(gloss, dtype=torch.long),
                torch.tensor(cat, dtype=torch.long),
                torch.tensor(input_length, dtype=torch.long)
            )
            
            # Add metadata if requested
            if self.return_metadata:
                return base_return + (signer, duration)
            else:
                return base_return

    def _load_npz_keypoints(self, path):
        with np.load(path, allow_pickle=True) as npz:
            if self.kp_key in npz:
                X = np.array(npz[self.kp_key])
            else:
                raise KeyError(f"Key '{self.kp_key}' not found in {path}")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D data in {path}, got shape {X.shape}")
        
        # Validate dimension based on the key being used
        if self.kp_key == "X2048" and X.shape[-1] != 2048:
            raise ValueError(f"Expected [T,2048] features in {path}, got shape {X.shape}")
        elif self.kp_key == "X" and X.shape[-1] != 178:
            raise ValueError(f"Expected [T,178] keypoints in {path}, got shape {X.shape}")
        return X

class FSLCombinedFileDataset(Dataset):
    """
    PyTorch Dataset that combines keypoints and features into single input.
    
    This dataset loads both keypoints [T, 178] and features [T, 2048] from the same
    .npz files and concatenates them to create combined input [T, 2226].
    
    The combined approach leverages both:
    - Raw keypoint information (178-dim): Direct pose landmarks for interpretability (89 keypoints × 2)
    - Learned visual features (2048-dim): Rich InceptionV3 representations
    
    Data Flow:
    1. Load CSV to build filename -> (gloss, category, occluded, signer, duration) mapping
    2. For each sample, load corresponding .npz file
    3. Extract both keypoint array (X) and feature array (X2048)
    4. Concatenate along feature dimension: [T, 178] + [T, 2048] = [T, 2226]
    5. Apply temporal augmentation if enabled and in training mode
    6. Return data based on mode (classification or CTC)
    
    Args:
        data_dir (str): Directory containing .npz files with both X and X2048 keys
        labels_csv (str): CSV file with columns: file, gloss, cat, occluded, signer, duration
        kp_key (str): Key for keypoints in .npz files (default: 'X')
        feature_key (str): Key for features in .npz files (default: 'X2048')
        augment (bool): Enable temporal data augmentation
        augment_params (dict): Augmentation parameters (noise_std, mask_prob, etc.)
        mode (str): Dataset mode - 'classification' or 'ctc'
        signer_filter (str, optional): Filter dataset to only include samples from this signer
        return_metadata (bool): Whether to return signer and duration in __getitem__
    
    Returns:
        Classification mode: (combined[T,2226] float32, gloss long, cat long, length long)
        CTC mode: (combined[T,2226] float32, gloss_seq[1] long, input_length long, target_length long, cat long)
        
    Raises:
        ValueError: If CSV format is invalid or array dimensions don't match
        FileNotFoundError: If data file doesn't exist
        KeyError: If expected keys not found in .npz file
    """
    def __init__(self, data_dir, labels_csv, kp_key='X', feature_key='X2048', augment=False, augment_params=None, mode='classification', signer_filter=None, return_metadata=False):
        self.data_dir = data_dir
        self.kp_key = kp_key
        self.feature_key = feature_key
        self.index = []  # list of (stem, gloss, cat, occluded, signer, duration)
        self.augment = augment
        self.training = True  # Will be set by DataLoader
        self.mode = mode
        self.signer_filter = signer_filter  # Optional signer filter
        self.return_metadata = return_metadata  # Whether to return metadata in __getitem__
        
        # Validate mode parameter
        if mode not in ['classification', 'ctc']:
            raise ValueError(f"mode must be 'classification' or 'ctc', got '{mode}'")
        
        # Initialize temporal augmentation if enabled
        if augment and augment_params:
            self.augmentation = TemporalAugmentation(**augment_params)
        elif augment:
            self.augmentation = TemporalAugmentation()
        
        if labels_csv is None:
            raise ValueError("labels_csv must be provided for combined dataset")
        
        # Load and parse the labels CSV file
        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            required = {'file', 'gloss', 'cat', 'occluded', 'signer', 'duration'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"labels_csv must have columns: {required}")
            for row in reader:
                try:
                    stem = os.path.splitext(row['file'])[0]
                    gloss = int(row['gloss'])
                    cat = int(row['cat'])
                    occluded = int(row['occluded'])
                    signer = row['signer']
                    duration = float(row['duration'])
                    
                    # Apply signer filter if specified
                    if signer_filter is not None and signer != signer_filter:
                        continue
                    
                    self.index.append((stem, gloss, cat, occluded, signer, duration))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid data in row {row}: {e}")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        stem, gloss, cat, occluded, signer, duration = self.index[idx]
        path = os.path.join(self.data_dir, stem + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load both keypoints and features
        keypoints, features = self._load_combined_data(path)
        
        # Concatenate along feature dimension: [T, 178] + [T, 2048] = [T, 2226]
        combined = torch.cat([keypoints, features], dim=1)
        input_length = combined.shape[0]
        
        # Apply augmentation if enabled and in training mode
        if self.augment and self.training and hasattr(self, 'augmentation'):
            combined = self.augmentation(combined)
        
        # Return tensors based on mode
        if self.mode == 'ctc':
            # CTC mode: return sequence format for CTCLoss
            gloss_label_seq = torch.tensor([gloss], dtype=torch.long)
            cat_label_seq = torch.tensor([cat], dtype=torch.long)
            target_length = torch.tensor([1], dtype=torch.long)
            
            base_return = (
                combined.float(),
                gloss_label_seq,
                torch.tensor(input_length, dtype=torch.long),
                target_length,
                cat_label_seq
            )
            
            # Add metadata if requested
            if self.return_metadata:
                return base_return + (signer, duration)
            else:
                return base_return
        else:
            # Classification mode: original format
            base_return = (
                combined.float(),
                torch.tensor(gloss, dtype=torch.long),
                torch.tensor(cat, dtype=torch.long),
                torch.tensor(input_length, dtype=torch.long)
            )
            
            # Add metadata if requested
            if self.return_metadata:
                return base_return + (signer, duration)
            else:
                return base_return
    
    def _load_combined_data(self, path):
        """Load and validate both keypoints and features from .npz file."""
        with np.load(path, allow_pickle=True) as npz:
            # Load keypoints
            if self.kp_key not in npz:
                raise KeyError(f"Keypoint key '{self.kp_key}' not found in {path}")
            keypoints = np.array(npz[self.kp_key])
            
            # Load features
            if self.feature_key not in npz:
                raise KeyError(f"Feature key '{self.feature_key}' not found in {path}")
            features = np.array(npz[self.feature_key])
        
        # Validate shapes
        if keypoints.ndim != 2 or keypoints.shape[-1] != 178:
            raise ValueError(f"Expected [T,178] keypoints in {path}, got shape {keypoints.shape}")
        if features.ndim != 2 or features.shape[-1] != 2048:
            raise ValueError(f"Expected [T,2048] features in {path}, got shape {features.shape}")
        
        # Ensure temporal dimensions match
        if keypoints.shape[0] != features.shape[0]:
            raise ValueError(f"Temporal dimension mismatch in {path}: keypoints {keypoints.shape[0]} != features {features.shape[0]}")
        
        return torch.from_numpy(keypoints), torch.from_numpy(features)

def collate_features_with_padding(batch):
    """
    Collate function to batch variable-length feature sequences with padding.
    
    This function is used by PyTorch DataLoader to combine multiple samples into batches.
    Since sequences have different lengths, we need to pad shorter sequences to match
    the longest sequence in the batch.
    
    Process:
    1. Separate sequences, labels, and lengths from batch items
    2. Find the maximum sequence length in the batch
    3. Create a padded tensor with shape [batch_size, max_length, feature_dim]
    4. Copy each sequence into the padded tensor
    5. Stack labels and lengths into tensors

    Args:
        batch: List of tuples, each containing (features[T,2048], gloss_label, cat_label, length)

    Returns:
        tuple: (padded_features[B,Tmax,2048], gloss_labels[B], cat_labels[B], lengths[B])
            - padded_features: Batch of padded feature sequences
            - gloss_labels: Batch of gloss class labels
            - cat_labels: Batch of category class labels  
            - lengths: Original sequence lengths (needed for attention masking)
    """
    # Unzip the batch to separate sequences, labels, and lengths
    sequences, gloss, cat, lengths = zip(*batch)
    
    # Convert lengths to tensor and find maximum sequence length in batch
    lengths = torch.stack(lengths, dim=0)  # Shape: [batch_size]
    B = len(sequences)                     # Batch size
    Tmax = int(max(l.item() for l in lengths))  # Maximum sequence length
    D = sequences[0].shape[-1]             # Feature dimension (2048 for features)
    
    # Create padded tensor with zeros - shape [batch_size, max_length, feature_dim]
    X_pad = torch.zeros((B, Tmax, D), dtype=sequences[0].dtype)
    
    # Copy each sequence into the padded tensor
    for i, seq in enumerate(sequences):
        t = seq.shape[0]  # Actual length of this sequence
        X_pad[i, :t] = seq  # Copy sequence data, leaving remainder as zeros
    
    # Stack labels into tensors and return
    return (
        X_pad,                           # Padded features [B, Tmax, D]
        torch.stack(gloss, dim=0),       # Gloss labels [B]
        torch.stack(cat, dim=0),         # Category labels [B]
        lengths                          # Original lengths [B]
    )

def collate_keypoints_with_padding(batch):
    """
    Pad variable-length keypoint sequences [T, 178] to the max length in batch.

    Args:
        batch: Iterable of (X[T,178], gloss, cat, length) items.

    Returns:
        tuple: (X_pad [B,Tmax,178], gloss [B], cat [B], lengths [B])
    """
    sequences, gloss, cat, lengths = zip(*batch)
    lengths = torch.stack(lengths, dim=0)
    B = len(sequences)
    Tmax = int(max(l.item() for l in lengths))
    D = sequences[0].shape[-1]
    X_pad = torch.zeros((B, Tmax, D), dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        t = seq.shape[0]
        X_pad[i, :t] = seq
    return X_pad, torch.stack(gloss, dim=0), torch.stack(cat, dim=0), lengths

def collate_features_with_metadata(batch):
    """
    Collate function for feature sequences with optional metadata.
    
    This function handles both standard format and metadata-enhanced format:
    - Standard: (features[T,2048], gloss, cat, length)
    - With metadata: (features[T,2048], gloss, cat, length, signer, duration)
    
    Args:
        batch: List of tuples from dataset with optional metadata
        
    Returns:
        tuple: Standard format + optional metadata lists
    """
    # Check if metadata is present by looking at tuple length
    if len(batch[0]) == 6:  # With metadata
        sequences, gloss, cat, lengths, signers, durations = zip(*batch)
        
        # Convert lengths to tensor and find maximum sequence length in batch
        lengths = torch.stack(lengths, dim=0)
        B = len(sequences)
        Tmax = int(max(l.item() for l in lengths))
        D = sequences[0].shape[-1]
        
        # Create padded tensor
        X_pad = torch.zeros((B, Tmax, D), dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            t = seq.shape[0]
            X_pad[i, :t] = seq
        
        return (
            X_pad,                           # Padded features [B, Tmax, D]
            torch.stack(gloss, dim=0),       # Gloss labels [B]
            torch.stack(cat, dim=0),         # Category labels [B]
            lengths,                         # Original lengths [B]
            list(signers),                   # Signer identifiers [B]
            list(durations)                  # Duration values [B]
        )
    else:  # Standard format without metadata
        return collate_features_with_padding(batch)

def collate_keypoints_with_metadata(batch):
    """
    Collate function for keypoint sequences with optional metadata.
    
    This function handles both standard format and metadata-enhanced format:
    - Standard: (keypoints[T,D], gloss, cat, length)
    - With metadata: (keypoints[T,D], gloss, cat, length, signer, duration)
    
    Args:
        batch: List of tuples from dataset with optional metadata
        
    Returns:
        tuple: Standard format + optional metadata lists
    """
    # Check if metadata is present by looking at tuple length
    if len(batch[0]) == 6:  # With metadata
        sequences, gloss, cat, lengths, signers, durations = zip(*batch)
        
        # Convert lengths to tensor and find maximum sequence length in batch
        lengths = torch.stack(lengths, dim=0)
        B = len(sequences)
        Tmax = int(max(l.item() for l in lengths))
        D = sequences[0].shape[-1]
        
        # Create padded tensor
        X_pad = torch.zeros((B, Tmax, D), dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            t = seq.shape[0]
            X_pad[i, :t] = seq
        
        return (
            X_pad,                           # Padded keypoints [B, Tmax, D]
            torch.stack(gloss, dim=0),       # Gloss labels [B]
            torch.stack(cat, dim=0),         # Category labels [B]
            lengths,                         # Original lengths [B]
            list(signers),                   # Signer identifiers [B]
            list(durations)                  # Duration values [B]
        )
    else:  # Standard format without metadata
        return collate_keypoints_with_padding(batch)

def collate_for_ctc(batch):
    """
    Collate function for CTC training with variable-length sequences.
    
    This function batches sequences for CTC loss computation by:
    1. Padding input sequences to the maximum length in the batch
    2. Concatenating label sequences (targets) for CTCLoss
    3. Stacking input and target lengths
    
    CTC Loss requires:
    - log_probs: [T, B, C] - model outputs (will be permuted from [B, T, C])
    - targets: [sum(target_lengths)] - concatenated target sequences
    - input_lengths: [B] - actual lengths of each input sequence
    - target_lengths: [B] - actual lengths of each target sequence
    
    Args:
        batch: List of tuples from CTC-mode dataset, each containing:
               (data[T,D], gloss_seq[N], input_length, target_length[N], cat_seq[N]) or with metadata
               
    Returns:
        tuple: (X_pad, targets, input_lengths, target_lengths, cat_targets) where:
            - X_pad: [B, Tmax, D] - padded input sequences
            - targets: [sum(target_lengths)] - concatenated gloss sequences
            - input_lengths: [B] - input sequence lengths
            - target_lengths: [B] - target sequence lengths
            - cat_targets: [sum(target_lengths)] - concatenated category sequences
    
    Example:
        For a batch of 2 samples with sequences:
        - Sample 1: input_len=50, target=[3]
        - Sample 2: input_len=75, target=[17]
        
        Output:
        - X_pad: [2, 75, D] (padded to max length 75)
        - targets: [3, 17] (concatenated targets)
        - input_lengths: [50, 75]
        - target_lengths: [1, 1]
    """
    # Unzip the batch to separate components
    sequences, gloss_label_seqs, input_lengths, target_lengths, cat_label_seqs = zip(*batch)
    
    # Get batch dimensions
    B = len(sequences)                                      # Batch size
    Tmax = max(seq.shape[0] for seq in sequences)         # Maximum input sequence length
    D = sequences[0].shape[-1]                              # Feature dimension
    
    # Create padded tensor for input sequences
    X_pad = torch.zeros((B, Tmax, D), dtype=sequences[0].dtype)
    
    # Copy each sequence into the padded tensor
    for i, seq in enumerate(sequences):
        t = seq.shape[0]  # Actual length of this sequence
        X_pad[i, :t] = seq  # Copy sequence data, pad remainder with zeros
    
    # Concatenate all target label sequences for CTCLoss
    targets = torch.cat(gloss_label_seqs, dim=0)  # Shape: [sum(target_lengths)]
    
    # Stack input and target lengths into tensors
    input_lengths = torch.stack(input_lengths, dim=0)       # Shape: [B]
    target_lengths = torch.cat(target_lengths, dim=0)       # Shape: [B]
    
    # Concatenate all category label sequences (parallel to gloss targets)
    cat_targets = torch.cat(cat_label_seqs, dim=0)  # Shape: [sum(target_lengths)]
    
    return X_pad, targets, input_lengths, target_lengths, cat_targets

def _make_dataloader(dataset, batch_size, shuffle, args, collate_fn=None):
    """
    Build an optimized DataLoader with performance enhancements.
    
    This function creates a DataLoader with optimized settings for better training
    performance, including automatic worker detection, memory pinning, and prefetching.
    
    Args:
        dataset: PyTorch Dataset to load from
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle data each epoch
        args: Training arguments containing DataLoader configuration
        collate_fn (callable, optional): Function to collate batches
        
    Returns:
        DataLoader: Optimized PyTorch DataLoader
    """
    # ============================================================================
    # WORKER CONFIGURATION
    # ============================================================================
    # Auto-detect optimal number of workers for data loading
    # More workers = faster data loading, but also more memory usage
    num_workers = args.num_workers
    if args.auto_workers:
        # Calculate optimal worker count based on CPU cores
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
        # Use 1/2 of CPU cores, but cap between 2 and 8 for stability
        num_workers = min(8, max(2, cpu_count // 2))
        print(f"Auto-detected {num_workers} DataLoader workers (from {cpu_count} CPU cores)")
    else:
        # Respect the user's num_workers setting (default: 0 for single-process loading)
        if num_workers > 0:
            print(f"Using {num_workers} DataLoader workers (user-specified)")
    
    # ============================================================================
    # MEMORY PINNING CONFIGURATION
    # ============================================================================
    # pin_memory=True enables faster CPU-GPU data transfer by keeping data in pinned memory
    pin_memory = args.pin_memory
    if not hasattr(args, 'pin_memory') or args.pin_memory is None:
        # Auto-enable pin_memory only if CUDA is available
        pin_memory = torch.cuda.is_available()
    
    # ============================================================================
    # DATALOADER CONFIGURATION
    # ============================================================================
    
    # Build DataLoader configuration dictionary
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,  # Keep workers alive between epochs for efficiency
    }
    
    # Add collate function if provided (needed for variable-length sequences)
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
    
    # Configure prefetching for better data loading performance
    # prefetch_factor determines how many batches each worker prefetches
    if num_workers > 0:
        prefetch_factor = getattr(args, 'prefetch_factor', None)
        if prefetch_factor is None:
            # Auto-set prefetch factor based on device type and available memory
            if torch.cuda.is_available():
                kwargs['prefetch_factor'] = 2  # Conservative for GPU (memory limited)
            else:
                kwargs['prefetch_factor'] = 4  # More aggressive for CPU (memory abundant)
        elif isinstance(prefetch_factor, int) and prefetch_factor > 0:
            kwargs['prefetch_factor'] = prefetch_factor
    
    # Create DataLoader with error handling for resource constraints
    try:
        return DataLoader(dataset, **kwargs)
    except (BlockingIOError, OSError, RuntimeError) as e:
        # Handle resource exhaustion errors (common in containerized environments)
        if num_workers > 0:
            print(f"\n⚠️  WARNING: Failed to create DataLoader with {num_workers} workers")
            print(f"   Error: {type(e).__name__}: {e}")
            print(f"   Falling back to single-process loading (num_workers=0)")
            print(f"   This is common in resource-constrained environments (containers, cloud VMs)")
            
            # Retry with 0 workers
            kwargs['num_workers'] = 0
            kwargs['persistent_workers'] = False
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            
            return DataLoader(dataset, **kwargs)
        else:
            # If already using 0 workers, re-raise the error
            raise

def save_checkpoint(state: dict, is_best: bool, output_dir: str, model_name: str) -> None:
    """Save training state to disk, keeping both last and best checkpoints.

    Args:
        state: Serializable checkpoint dict (model, optimizer, etc.).
        is_best: Whether this state is the current best by validation metric.
        output_dir: Directory path to store checkpoints.
        model_name: Base model name used for file naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    last_path = os.path.join(output_dir, f"{model_name}_last.pt")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pt")
        torch.save(state, best_path)

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for training with comprehensive optimizations.
    
    This function automatically selects the best available device (CUDA > MPS > CPU)
    and applies device-specific optimizations for maximum performance.
    
    CUDA Optimizations:
    - Enables cuDNN benchmark mode for optimal convolution performance
    - Sets memory allocation strategy to reduce fragmentation
    
    Returns:
        torch.device: The optimal device for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Enable CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN for optimal performance
        torch.backends.cudnn.enabled = True    # Ensure cuDNN is enabled
        
        # Set memory allocation strategy for better memory management
        # max_split_size_mb limits the size of memory chunks to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Metal Performance Shaders (M1/M2 Macs)
        return torch.device("mps")
    else:
        # Fallback to CPU
        return torch.device("cpu")

def print_device_info(device: torch.device) -> None:
    """
    Print comprehensive device information for training setup.
    
    This function displays detailed information about the selected device,
    including hardware specifications and current memory usage.
    
    Args:
        device (torch.device): The device to display information for
    """
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        # CUDA-specific information
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA memory: {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA compute capability: {props.major}.{props.minor}")
        print(f"CUDA multiprocessors: {props.multi_processor_count}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    elif device.type == 'mps':
        # Apple Metal Performance Shaders
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        # CPU information
        print("Using CPU")
        print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"Available RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")


def optimize_model_for_parallel(model, device):
    """Optimize model for parallel processing if multiple GPUs available."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        return model
    return model


def calculate_optimal_batch_size(model, device, base_batch_size=32):
    """
    Calculate optimal batch size based on available hardware resources.
    
    This function analyzes the available memory and processing power to determine
    the best batch size for training efficiency.
    
    Args:
        model: The model being trained (for parameter estimation)
        device (torch.device): The device to optimize for
        base_batch_size (int): Default batch size to use as reference
        
    Returns:
        int: Optimal batch size for the given hardware
    """
    if device.type == 'cuda':
        # GPU memory-based batch size optimization
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Scale batch size based on available GPU memory
        # More memory allows larger batches, which can improve training efficiency
        if gpu_memory_gb > 16:
            optimal_batch_size = base_batch_size * 2  # High-memory GPU: 64
        elif gpu_memory_gb > 8:
            optimal_batch_size = base_batch_size      # Mid-range GPU: 32
        elif gpu_memory_gb > 4:
            optimal_batch_size = base_batch_size // 2 # Low-memory GPU: 16
        else:
            optimal_batch_size = base_batch_size // 4 # Very low memory: 8
        
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB, Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    else:
        # CPU training - optimize based on core count
        # More cores can handle larger batches more efficiently
        cpu_count = psutil.cpu_count(logical=False)
        optimal_batch_size = min(base_batch_size, cpu_count * 4)
        print(f"CPU cores: {cpu_count}, Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

def log_comprehensive_config(args, device, model=None):
    """Log comprehensive training configuration and system information."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING CONFIGURATION")
    print("="*80)
    
    # Session Information
    print(f"Session Information:")
    print(f"  - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Working directory: {os.getcwd()}")
    print(f"  - Command: {' '.join(sys.argv)}")
    
    # System Information
    print(f"\nSystem Information:")
    print(f"  - Platform: {platform.platform()}")
    print(f"  - Python version: {sys.version}")
    print(f"  - PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  - NumPy version: {np.__version__}")
    
    # Core Training Parameters
    print(f"\nCore Training Parameters:")
    print(f"  - Model: {args.model}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Gradient clipping: {args.grad_clip}")
    print(f"  - Loss weights - Alpha: {args.alpha}, Beta: {args.beta}")
    
    # Curriculum Training Parameters
    if args.curriculum is not None:
        print(f"\nCurriculum Training Parameters:")
        print(f"  - Strategy: {args.curriculum}")
        print(f"  - Curriculum epochs: {args.curriculum_epochs}")
        if args.curriculum == "dynamic":
            print(f"  - Warmup epochs: {args.curriculum_warmup}")
        print(f"  - Min weight: {args.curriculum_min_weight}")
        print(f"  - Schedule: {args.curriculum_schedule}")
    else:
        print(f"\nCurriculum Training: Disabled")
    
    # Loss Weighting Parameters
    print(f"\nLoss Weighting Parameters:")
    print(f"  - Strategy: {args.loss_weighting}")
    if args.loss_weighting == "grid-search":
        weight_combinations = parse_grid_search_weights(args.grid_search_weights)
        print(f"  - Weight combinations: {weight_combinations}")
        print(f"  - Epochs per combination: {max(1, args.epochs // len(weight_combinations))}")
    elif args.loss_weighting == "uncertainty":
        print(f"  - Initial uncertainty: {args.uncertainty_init}")
    elif args.loss_weighting == "gradnorm":
        print(f"  - Alpha: {args.gradnorm_alpha}")
        print(f"  - Update frequency: {args.gradnorm_update_freq}")
    else:
        print(f"  - Alpha: {args.alpha}")
        print(f"  - Beta: {args.beta}")
    
    # Model-Specific Parameters
    if args.model == "iv3_gru":
        print(f"\nIV3-GRU Model Parameters:")
        print(f"  - Hidden1: {args.hidden1}")
        print(f"  - Hidden2: {args.hidden2}")
        print(f"  - Dropout: {args.dropout}")
        print(f"  - Pretrained backbone: {args.pretrained_backbone}")
        print(f"  - Freeze backbone: {args.freeze_backbone}")
    
    # Data Configuration
    print(f"\nData Configuration:")
    print(f"  - Gloss classes: {args.num_gloss}")
    print(f"  - Category classes: {args.num_cat}")
    
    # Data Source Information
    if args.features_train or args.keypoints_train:
        print(f"  - Data source: Real data files")
        if args.model == "iv3_gru":
            print(f"  - Training folder: {args.features_train}")
            print(f"  - Validation folder: {args.features_val}")
            print(f"  - Feature key: {args.feature_key}")
        elif args.model in ["transformer", "transformer_ctc", "mediapipe_gru", "mediapipe_gru_ctc"]:
            print(f"  - Training folder: {args.keypoints_train}")
            print(f"  - Validation folder: {args.keypoints_val}")
            if args.combine_features and args.model == "transformer":
                print(f"  - Mode: Combined (Keypoints + Features)")
                print(f"  - Keypoint key: {args.kp_key} (178-dim)")
                print(f"  - Feature key: {args.feature_key} (2048-dim)")
                print(f"  - Combined input: 2226-dim")
            else:
                print(f"  - Keypoint key: {args.kp_key}")
    else:
        print(f"  - Data source: Synthetic data")
        print(f"  - Training samples: {args.train_samples}")
        print(f"  - Validation samples: {args.val_samples}")
        print(f"  - Sequence length: {args.seq_length}")
    
    # Training Control
    print(f"\nTraining Control:")
    print(f"  - Scheduler: {args.scheduler}")
    print(f"  - Scheduler patience: {args.scheduler_patience}")
    print(f"  - Early stopping: {args.early_stop}")
    print(f"  - Resume from: {args.resume}")
    
    # Performance Settings
    print(f"\nPerformance Settings:")
    print(f"  - AMP (Mixed Precision): {args.amp}")
    print(f"  - Model compilation: {args.compile_model}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - DataLoader workers: {args.num_workers}")
    print(f"  - Auto workers: {args.auto_workers}")
    print(f"  - Pin memory: {args.pin_memory}")
    print(f"  - Prefetch factor: {args.prefetch_factor}")
    
    # Reproducibility
    print(f"\nReproducibility:")
    print(f"  - Random seed: {args.seed}")
    print(f"  - Deterministic mode: {args.deterministic}")
    
    # Output Configuration
    print(f"\nOutput Configuration:")
    print(f"  - Checkpoint directory: {args.output_dir}")
    print(f"  - CSV log file: {args.log_csv}")
    
    # Smoke Test Configuration
    if args.smoke_test:
        print(f"\nSmoke Test Configuration:")
        print(f"  - Smoke test mode: {args.smoke_test}")
        print(f"  - Smoke batch size: {args.smoke_batch_size}")
        print(f"  - Smoke sequence length: {args.smoke_T}")
    
    # Model Information (if model is provided)
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Information:")
        print(f"  - Model type: {model.__class__.__name__}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    print("="*80)

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set global RNG seeds across Python, NumPy, and PyTorch.

    Optionally configures deterministic CuDNN for reproducibility at the
    expense of performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

class CurriculumScheduler:
    """
    Curriculum learning scheduler for multi-task training.
    
    Supports three strategies:
    - gloss-first: Train gloss classification first, then gradually add category
    - category-first: Train category classification first, then gradually add gloss  
    - dynamic: Start with one task dominant and gradually balance both tasks
    """
    
    def __init__(self, strategy: str, curriculum_epochs: int, warmup_epochs: int = 0, 
                 min_weight: float = 0.1, schedule_type: str = "linear"):
        """
        Initialize curriculum scheduler.
        
        Args:
            strategy: Curriculum strategy ("gloss-first", "category-first", "dynamic")
            curriculum_epochs: Number of epochs for curriculum phase
            warmup_epochs: Number of warmup epochs before curriculum starts (for dynamic)
            min_weight: Minimum weight for secondary task (0.0-1.0)
            schedule_type: Weight scheduling function ("linear", "cosine", "exponential")
        """
        self.strategy = strategy
        self.curriculum_epochs = curriculum_epochs
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.schedule_type = schedule_type
        
        # Validate inputs
        if strategy not in ["gloss-first", "category-first", "dynamic"]:
            raise ValueError(f"Invalid strategy: {strategy}")
        if schedule_type not in ["linear", "cosine", "exponential"]:
            raise ValueError(f"Invalid schedule_type: {schedule_type}")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError(f"min_weight must be between 0.0 and 1.0, got {min_weight}")
    
    def get_weights(self, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Get alpha and beta weights for current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
            
        Returns:
            Tuple of (alpha, beta) weights for gloss and category tasks
        """
        if self.strategy is None:
            return 0.5, 0.5  # Default balanced weights
        
        # Calculate progress through curriculum
        if self.strategy == "dynamic":
            # Dynamic: warmup -> curriculum -> balanced
            if epoch < self.warmup_epochs:
                # Warmup phase: start with balanced weights
                return 0.5, 0.5
            elif epoch < self.warmup_epochs + self.curriculum_epochs:
                # Curriculum phase: gradually balance
                progress = (epoch - self.warmup_epochs) / self.curriculum_epochs
                weight = self._schedule_weight(progress)
                return weight, 1.0 - weight
            else:
                # Balanced phase
                return 0.5, 0.5
        else:
            # gloss-first or category-first
            if epoch < self.curriculum_epochs:
                # Curriculum phase: focus on primary task
                progress = epoch / self.curriculum_epochs
                secondary_weight = self._schedule_weight(progress)
                primary_weight = 1.0 - secondary_weight
                
                if self.strategy == "gloss-first":
                    return primary_weight, secondary_weight  # alpha, beta
                else:  # category-first
                    return secondary_weight, primary_weight  # alpha, beta
            else:
                # Balanced phase
                return 0.5, 0.5
    
    def _schedule_weight(self, progress: float) -> float:
        """
        Calculate secondary task weight based on progress and schedule type.
        
        Args:
            progress: Progress through curriculum (0.0 to 1.0)
            
        Returns:
            Weight for secondary task (min_weight to 0.5)
        """
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))
        
        if self.schedule_type == "linear":
            # Linear interpolation from min_weight to 0.5
            return self.min_weight + (0.5 - self.min_weight) * progress
        elif self.schedule_type == "cosine":
            # Cosine annealing from min_weight to 0.5
            return self.min_weight + (0.5 - self.min_weight) * (1 - np.cos(np.pi * progress)) / 2
        elif self.schedule_type == "exponential":
            # Exponential growth from min_weight to 0.5
            return self.min_weight + (0.5 - self.min_weight) * (np.exp(2 * progress) - 1) / (np.exp(2) - 1)
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
    
    def get_phase_info(self, epoch: int) -> str:
        """
        Get human-readable information about current curriculum phase.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            String describing current phase
        """
        if self.strategy is None:
            return "Balanced training (no curriculum)"
        
        if self.strategy == "dynamic":
            if epoch < self.warmup_epochs:
                return f"Warmup phase (epoch {epoch+1}/{self.warmup_epochs})"
            elif epoch < self.warmup_epochs + self.curriculum_epochs:
                return f"Dynamic curriculum phase (epoch {epoch+1-self.warmup_epochs}/{self.curriculum_epochs})"
            else:
                return "Balanced phase"
        else:
            if epoch < self.curriculum_epochs:
                primary_task = "gloss" if self.strategy == "gloss-first" else "category"
                return f"{primary_task.title()}-first curriculum phase (epoch {epoch+1}/{self.curriculum_epochs})"
            else:
                return "Balanced phase"

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Focal Loss = -alpha * (1-pt)^gamma * log(pt)
    where pt is the predicted probability for the true class.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing CrossEntropy Loss for better generalization.
    
    Combines hard target loss with uniform distribution loss.
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, x, target):
        """
        Args:
            x: (N, C) logits
            target: (N,) class indices
        """
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TemporalAugmentation:
    """
    Temporal augmentation for sequence data to improve generalization.
    """
    
    def __init__(self, noise_std=0.01, time_mask_prob=0.1, time_mask_ratio=0.1):
        self.noise_std = noise_std
        self.time_mask_prob = time_mask_prob
        self.time_mask_ratio = time_mask_ratio
    
    def __call__(self, sequence):
        """
        Apply temporal augmentation to a sequence.
        
        Args:
            sequence: (T, D) tensor
            
        Returns:
            Augmented sequence of same shape
        """
        # Add Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(sequence) * self.noise_std
            sequence = sequence + noise
        
        # Time masking (mask random frames)
        if random.random() < self.time_mask_prob:
            seq_len = sequence.shape[0]
            mask_len = max(1, int(seq_len * self.time_mask_ratio))
            start_idx = random.randint(0, max(0, seq_len - mask_len))
            sequence[start_idx:start_idx + mask_len] = 0
        
        return sequence

class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup followed by cosine annealing.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        
        # Cosine scheduler for remaining epochs
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=self.min_lr
        )
    
    def step(self, epoch):
        """Step the scheduler for the given epoch."""
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
    
    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'cosine_scheduler': self.cosine_scheduler.state_dict(),
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load the state of the scheduler."""
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.base_lr = state_dict['base_lr']
        self.min_lr = state_dict['min_lr']

class EMA:
    """
    Exponential Moving Average for model parameters to improve stability.
    """
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.registered = False
    
    def register(self):
        """Register parameters for EMA."""
        if not self.registered:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
            self.registered = True
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        if not self.registered:
            self.register()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        if not self.registered:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters."""
        if not self.registered:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

class LossWeightingStrategy:
    """
    Base class for loss weighting strategies.
    
    All loss weighting strategies should inherit from this class and implement
    the get_weights method to return alpha and beta weights for each batch.
    """
    
    def __init__(self, **kwargs):
        """Initialize the loss weighting strategy."""
        pass
    
    def get_weights(self, epoch: int, batch_idx: int, loss_gloss: float, loss_cat: float, 
                   model=None, optimizer=None) -> Tuple[float, float]:
        """
        Get alpha and beta weights for current batch.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            loss_gloss: Current gloss loss value
            loss_cat: Current category loss value
            model: The model being trained (for strategies that need model access)
            optimizer: The optimizer (for strategies that need optimizer access)
            
        Returns:
            Tuple of (alpha, beta) weights
        """
        raise NotImplementedError("Subclasses must implement get_weights method")
    
    def update_weights(self, epoch: int, losses: dict, model=None, optimizer=None):
        """
        Update internal state for adaptive weighting strategies.
        
        Args:
            epoch: Current epoch number
            losses: Dictionary of loss values
            model: The model being trained
            optimizer: The optimizer
        """
        pass

class StaticWeighting(LossWeightingStrategy):
    """Static loss weighting with fixed alpha and beta values."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def get_weights(self, epoch: int, batch_idx: int, loss_gloss: float, loss_cat: float, 
                   model=None, optimizer=None) -> Tuple[float, float]:
        return self.alpha, self.beta

class GridSearchWeighting(LossWeightingStrategy):
    """Grid search over multiple weight combinations."""
    
    def __init__(self, weight_combinations: list, epochs_per_combination: int = 10):
        super().__init__()
        self.weight_combinations = weight_combinations
        self.epochs_per_combination = epochs_per_combination
        self.current_combination_idx = 0
    
    def get_weights(self, epoch: int, batch_idx: int, loss_gloss: float, loss_cat: float, 
                   model=None, optimizer=None) -> Tuple[float, float]:
        # Calculate which combination to use based on epoch
        combination_idx = min(epoch // self.epochs_per_combination, len(self.weight_combinations) - 1)
        alpha, beta = self.weight_combinations[combination_idx]
        return alpha, beta

class UncertaintyWeighting(LossWeightingStrategy):
    """
    Uncertainty weighting based on Kendall et al. 2018.
    
    Learns log variance parameters for each task and weights losses by 1/exp(log_var).
    """
    
    def __init__(self, init_uncertainty: float = 1.0, device: str = "cpu"):
        super().__init__()
        self.device = device
        # Initialize log variance parameters (higher = more uncertainty = lower weight)
        self.log_var_gloss = torch.tensor(np.log(init_uncertainty), device=device, requires_grad=True)
        self.log_var_cat = torch.tensor(np.log(init_uncertainty), device=device, requires_grad=True)
    
    def get_weights(self, epoch: int, batch_idx: int, loss_gloss: float, loss_cat: float, 
                   model=None, optimizer=None) -> Tuple[float, float]:
        # Convert to weights: 1 / exp(log_var) = exp(-log_var)
        alpha = torch.exp(-self.log_var_gloss).item()
        beta = torch.exp(-self.log_var_cat).item()
        return alpha, beta
    
    def get_uncertainty_loss(self, loss_gloss: torch.Tensor, loss_cat: torch.Tensor) -> torch.Tensor:
        """
        Compute the uncertainty-weighted loss.
        
        Args:
            loss_gloss: Gloss loss tensor
            loss_cat: Category loss tensor
            
        Returns:
            Total uncertainty-weighted loss
        """
        # Uncertainty weighting: 1/(2*sigma^2) * loss + 1/2 * log(sigma^2)
        # where sigma^2 = exp(log_var)
        alpha = torch.exp(-self.log_var_gloss)
        beta = torch.exp(-self.log_var_cat)
        
        weighted_loss = alpha * loss_gloss + beta * loss_cat
        uncertainty_penalty = 0.5 * (self.log_var_gloss + self.log_var_cat)
        
        return weighted_loss + uncertainty_penalty

class GradNormWeighting(LossWeightingStrategy):
    """
    GradNorm weighting based on Chen et al. 2018.
    
    Adjusts task weights so that each task's gradients have similar magnitudes.
    """
    
    def __init__(self, alpha: float = 1.5, update_freq: int = 1, device: str = "cpu"):
        super().__init__()
        self.alpha = alpha
        self.update_freq = update_freq
        self.device = device
        self.initial_losses = None
        self.weights = torch.tensor([1.0, 1.0], device=device, requires_grad=True)
        self.last_updated_epoch = -1
    
    def get_weights(self, epoch: int, batch_idx: int, loss_gloss: float, loss_cat: float, 
                   model=None, optimizer=None) -> Tuple[float, float]:
        return self.weights[0].item(), self.weights[1].item()
    
    def update_weights(self, epoch: int, losses: dict, model=None, optimizer=None):
        """Update weights using GradNorm algorithm."""
        if epoch % self.update_freq != 0 or model is None or optimizer is None:
            return
        
        # Store initial losses on first update
        if self.initial_losses is None:
            self.initial_losses = {
                'gloss': losses['gloss'].detach().clone(),
                'cat': losses['cat'].detach().clone()
            }
            return
        
        # Compute relative inverse training rates
        current_losses = {
            'gloss': losses['gloss'].detach(),
            'cat': losses['cat'].detach()
        }
        
        # Compute gradients of weighted losses w.r.t. shared parameters
        # This is a simplified version - in practice, you'd need to compute
        # gradients of each task loss w.r.t. shared parameters
        try:
            # For now, we'll use a simplified update based on loss ratios
            gloss_ratio = current_losses['gloss'] / self.initial_losses['gloss']
            cat_ratio = current_losses['cat'] / self.initial_losses['cat']
            
            # Update weights based on relative progress
            # If one task is progressing faster, increase its weight
            if gloss_ratio < cat_ratio:
                self.weights[0] = self.weights[0] * (1 + self.alpha * (cat_ratio - gloss_ratio))
            else:
                self.weights[1] = self.weights[1] * (1 + self.alpha * (gloss_ratio - cat_ratio))
            
            # Normalize weights to prevent them from growing too large
            total_weight = self.weights.sum()
            if total_weight > 2.0:  # Prevent weights from growing too large
                self.weights = self.weights / total_weight * 2.0
            
            self.last_updated_epoch = epoch
            
        except Exception as e:
            print(f"Warning: GradNorm update failed: {e}")

def create_loss_weighting_strategy(strategy: str, **kwargs) -> LossWeightingStrategy:
    """
    Factory function to create loss weighting strategies.
    
    Args:
        strategy: Strategy name ("static", "grid-search", "uncertainty", "gradnorm")
        **kwargs: Additional arguments for the strategy
        
    Returns:
        LossWeightingStrategy instance
    """
    if strategy == "static":
        return StaticWeighting(alpha=kwargs.get('alpha', 0.5), beta=kwargs.get('beta', 0.5))
    elif strategy == "grid-search":
        weight_combinations = kwargs.get('weight_combinations', [(0.5, 0.5)])
        epochs_per_combination = kwargs.get('epochs_per_combination', 10)
        return GridSearchWeighting(weight_combinations, epochs_per_combination)
    elif strategy == "uncertainty":
        init_uncertainty = kwargs.get('uncertainty_init', 1.0)
        device = kwargs.get('device', 'cpu')
        return UncertaintyWeighting(init_uncertainty, device)
    elif strategy == "gradnorm":
        alpha = kwargs.get('gradnorm_alpha', 1.5)
        update_freq = kwargs.get('gradnorm_update_freq', 1)
        device = kwargs.get('device', 'cpu')
        return GradNormWeighting(alpha, update_freq, device)
    else:
        raise ValueError(f"Unknown loss weighting strategy: {strategy}")

def parse_grid_search_weights(weight_string: str) -> list:
    """
    Parse grid search weight combinations from string format.
    
    Args:
        weight_string: String in format "a1,b1;a2,b2;..." 
        
    Returns:
        List of (alpha, beta) tuples
    """
    combinations = []
    for pair in weight_string.split(';'):
        if pair.strip():
            try:
                alpha, beta = map(float, pair.split(','))
                combinations.append((alpha, beta))
            except ValueError:
                raise ValueError(f"Invalid weight format: {pair}. Expected 'alpha,beta'")
    return combinations

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    forward_fn,
    epochs=20,
    alpha=0.5,
    beta=0.5,
    output_dir="data/processed",
    lr=1e-4,
    weight_decay=0.0,
    use_amp=False,
    grad_clip=None,
    scheduler_type=None,
    scheduler_patience=5,
    warmup_epochs=5,
    early_stop_patience=None,
    resume_path=None,
    log_csv_path=None,
    gradient_accumulation_steps=1,
    compile_model=False,
    curriculum_strategy=None,
    curriculum_epochs=10,
    curriculum_warmup=5,
    curriculum_min_weight=0.1,
    curriculum_schedule="linear",
    loss_weighting_strategy="static",
    grid_search_weights="0.5,0.5;0.7,0.3;0.3,0.7",
    uncertainty_init=1.0,
    gradnorm_alpha=1.5,
    gradnorm_update_freq=1,
    loss_type="ce",
    focal_gamma=2.0,
    focal_alpha=1.0,
    label_smoothing=0.1,
    use_ema=False,
    ema_decay=0.999,
):
    """
    Train a model with multi-task loss on gloss and category predictions.

    Args:
        model: The model to train (e.g., `SignTransformer`, `InceptionV3GRU`).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Torch device to run on.
        forward_fn: Callable(model, X, lengths) -> (gloss_logits, cat_logits).
        epochs: Number of training epochs.
        alpha: Weight for gloss loss component (used as fallback if no curriculum).
        beta: Weight for category loss component (used as fallback if no curriculum).
        output_dir: Directory to save checkpoints.
        lr: Learning rate for Adam optimizer.
        weight_decay: Weight decay for Adam optimizer.
        use_amp: Enable automatic mixed precision if True.
        grad_clip: Max norm for gradient clipping (None to disable).
        scheduler_type: LR scheduler type (None, 'plateau', or 'cosine').
        scheduler_patience: Patience for ReduceLROnPlateau.
        early_stop_patience: Stop if no improvement for this many epochs.
        resume_path: Path to checkpoint to resume from.
        log_csv_path: Path to append per-epoch metrics as CSV.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        compile_model: Whether to compile the model for better performance.
        curriculum_strategy: Curriculum strategy ("gloss-first", "category-first", "dynamic", None).
        curriculum_epochs: Number of epochs for curriculum phase.
        curriculum_warmup: Number of warmup epochs before curriculum starts (for dynamic).
        curriculum_min_weight: Minimum weight for secondary task during curriculum.
        curriculum_schedule: Weight scheduling function ("linear", "cosine", "exponential").
        loss_weighting_strategy: Loss weighting strategy ("static", "grid-search", "uncertainty", "gradnorm").
        grid_search_weights: Grid search weight combinations (format: "a1,b1;a2,b2;...").
        uncertainty_init: Initial uncertainty for uncertainty weighting.
        gradnorm_alpha: Alpha parameter for GradNorm weighting.
        gradnorm_update_freq: Update frequency for GradNorm (every N epochs).

    Returns:
        None

    Side effects:
        Saves checkpoints to `output_dir` as `{ModelName}_last.pt` (each epoch)
        and `{ModelName}_best.pt` (best validation metric). Appends metrics to
        `log_csv_path` if provided.
    """
    # ============================================================================
    # INITIAL SETUP AND CONFIGURATION
    # ============================================================================
    
    # Clear GPU memory cache to ensure clean start and prevent memory issues
    clear_gpu_memory()
    
    # ============================================================================
    # CURRICULUM LEARNING SETUP
    # ============================================================================
    # Curriculum learning gradually introduces tasks to improve training stability
    # and performance. This is especially useful for multi-task learning where
    # balancing gloss and category classification can be challenging.
    
    curriculum_scheduler = None
    if curriculum_strategy is not None:
        # Initialize curriculum scheduler with specified strategy
        curriculum_scheduler = CurriculumScheduler(
            strategy=curriculum_strategy,        # "gloss-first", "category-first", or "dynamic"
            curriculum_epochs=curriculum_epochs, # How many epochs to spend on curriculum
            warmup_epochs=curriculum_warmup,     # Warmup period for dynamic strategy
            min_weight=curriculum_min_weight,    # Minimum weight for secondary task
            schedule_type=curriculum_schedule    # "linear", "cosine", or "exponential"
        )
        print(f"✓ Curriculum training enabled: {curriculum_strategy}")
        print(f"  - Curriculum epochs: {curriculum_epochs}")
        if curriculum_strategy == "dynamic":
            print(f"  - Warmup epochs: {curriculum_warmup}")
        print(f"  - Min weight: {curriculum_min_weight}")
        print(f"  - Schedule: {curriculum_schedule}")
    
    # ============================================================================
    # LOSS WEIGHTING STRATEGY SETUP
    # ============================================================================
    # Advanced loss weighting strategies help balance multiple tasks during training.
    # This is crucial for multi-task learning where gloss and category classification
    # may have different difficulty levels and learning dynamics.
    
    loss_weighting = None
    if loss_weighting_strategy == "grid-search":
        # Grid search: Try different weight combinations over epochs
        weight_combinations = parse_grid_search_weights(grid_search_weights)
        loss_weighting = create_loss_weighting_strategy(
            loss_weighting_strategy,
            weight_combinations=weight_combinations,
            epochs_per_combination=max(1, epochs // len(weight_combinations))
        )
        print(f"✓ Grid search weighting enabled")
        print(f"  - Weight combinations: {weight_combinations}")
        print(f"  - Epochs per combination: {max(1, epochs // len(weight_combinations))}")
    else:
        # Static, uncertainty, or gradnorm weighting strategies
        loss_weighting = create_loss_weighting_strategy(
            loss_weighting_strategy,
            alpha=alpha,                    # Static gloss weight
            beta=beta,                      # Static category weight
            uncertainty_init=uncertainty_init,  # Initial uncertainty for uncertainty weighting
            gradnorm_alpha=gradnorm_alpha,      # Alpha parameter for gradnorm
            gradnorm_update_freq=gradnorm_update_freq,  # How often to update gradnorm weights
            device=str(device)              # Device for uncertainty/gradnorm parameters
        )
        print(f"✓ Loss weighting strategy: {loss_weighting_strategy}")
        if loss_weighting_strategy == "uncertainty":
            print(f"  - Initial uncertainty: {uncertainty_init}")
        elif loss_weighting_strategy == "gradnorm":
            print(f"  - Alpha: {gradnorm_alpha}")
            print(f"  - Update frequency: {gradnorm_update_freq}")
    
    # ============================================================================
    # MODEL OPTIMIZATION AND LOSS FUNCTION SETUP
    # ============================================================================
    
    # Model compilation (PyTorch 2.0+) for significant performance improvements
    # This optimizes the model graph for faster execution, especially on modern GPUs
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✓ Model compiled for better performance")
        except Exception as e:
            print(f"⚠ Model compilation failed: {e}")
    
    # Initialize loss function based on specified type
    # Different loss functions are suited for different scenarios:
    # - CrossEntropy: Standard choice for most classification tasks
    # - Focal Loss: Better for imbalanced datasets, focuses on hard examples
    # - Label Smoothing: Improves generalization by preventing overconfidence
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"✓ Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    elif loss_type == "label_smoothing":
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        print(f"✓ Using Label Smoothing CrossEntropy (smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("✓ Using standard CrossEntropy Loss")
    
    # ============================================================================
    # OPTIMIZER AND AUTOMATIC MIXED PRECISION SETUP
    # ============================================================================
    
    # Set up optimizer with model parameters
    # For uncertainty weighting, we also need to optimize the uncertainty parameters
    if loss_weighting_strategy == "uncertainty" and isinstance(loss_weighting, UncertaintyWeighting):
        # Include uncertainty parameters (log variance) in optimization
        optimizer = optim.Adam(
            list(model.parameters()) + [loss_weighting.log_var_gloss, loss_weighting.log_var_cat],
            lr=lr, weight_decay=weight_decay
        )
    else:
        # Standard optimizer with only model parameters
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Automatic Mixed Precision (AMP) for faster training and lower memory usage
    # AMP uses float16 for forward pass and float32 for backward pass
    # Only enable on CUDA devices to avoid CPU-only compatibility issues
    amp_enabled = bool(use_amp and getattr(device, "type", "cpu") == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)  # Handles gradient scaling for AMP
    
    # Print training configuration
    print(f"Training Configuration:")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  - AMP enabled: {amp_enabled}")
    print(f"  - Model compiled: {compile_model}")
    if device.type == 'cuda':
        print(f"  - CUDA memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # ============================================================================
    # LEARNING RATE SCHEDULER SETUP
    # ============================================================================
    # Learning rate scheduling helps improve training stability and final performance
    # Different schedulers are suited for different training scenarios

    if scheduler_type == "plateau":
        # Reduce LR when validation metric plateaus - good for stable training
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience)
        print(f"✓ Using ReduceLROnPlateau scheduler (patience={scheduler_patience})")
    elif scheduler_type == "cosine":
        # Cosine annealing - smooth LR decay over training
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        print(f"✓ Using CosineAnnealingLR scheduler")
    elif scheduler_type == "warmup_cosine":
        # Warmup + cosine - gradual LR increase then cosine decay
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, lr)
        print(f"✓ Using WarmupCosineScheduler (warmup_epochs={warmup_epochs})")
    else:
        scheduler = None
        print("✓ No learning rate scheduler")

    # ============================================================================
    # EXPONENTIAL MOVING AVERAGE (EMA) SETUP
    # ============================================================================
    # EMA maintains a running average of model parameters, which often leads to
    # better generalization and more stable training. The EMA model is used for
    # validation and final evaluation.
    
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)  # Higher decay = slower parameter updates
        ema.register()  # Initialize EMA with current model parameters
        print(f"✓ EMA enabled (decay={ema_decay})")

    # ============================================================================
    # RESUME TRAINING SETUP
    # ============================================================================
    # Support for resuming training from a checkpoint. This loads the model state,
    # optimizer state, scheduler state, and training progress.
    
    start_epoch = 0
    best_metric = -float('inf')
    if resume_path is not None and os.path.isfile(resume_path):
        print(f"Loading checkpoint from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        
        # Restore model parameters
        model.load_state_dict(ckpt['model'])
        
        # Restore optimizer state (includes momentum, etc.)
        optimizer.load_state_dict(ckpt['optimizer'])
        
        # Restore AMP scaler state if using AMP
        if 'scaler' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler'])
        
        # Restore scheduler state if scheduler exists
        if 'scheduler' in ckpt and scheduler is not None and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        
        # Restore training progress
        start_epoch = ckpt.get('epoch', 0)
        best_metric = ckpt.get('best_metric', best_metric)
        print(f"Resumed from {resume_path} at epoch {start_epoch} (best_metric={best_metric:.4f})")

    # ============================================================================
    # CSV LOGGING SETUP
    # ============================================================================
    # Set up CSV logging for tracking training metrics over time
    # This allows for easy analysis and visualization of training progress
    
    csv_fh = None
    if log_csv_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_csv_path) or '.', exist_ok=True)
        new_file = not os.path.exists(log_csv_path)
        csv_fh = open(log_csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_fh)
        
        if new_file:
            # Write configuration header as comments for reference
            config_header = [
                f"# Training Configuration: epochs={epochs}, batch_size={batch_size}",
                f"# Learning Rate: {lr}, Weight Decay: {weight_decay}, Alpha: {alpha}, Beta: {beta}",
                f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            for header_line in config_header:
                csv_writer.writerow([header_line])
            csv_writer.writerow([])  # Empty line separator
            
            # Write column headers based on training configuration
            if curriculum_scheduler is not None:
                csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved", "alpha", "beta", "curriculum_phase"])
            elif loss_weighting is not None and loss_weighting_strategy != "static":
                csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved", "alpha", "beta", "loss_weighting_strategy"])
            else:
                csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved"]) 

    # ============================================================================
    # TRAINING LOOP START
    # ============================================================================

    print(f"Training for {epochs} epochs...")
    if curriculum_scheduler is not None:
        print(f"Curriculum training: {curriculum_scheduler.get_phase_info(0)}")
    else:
        print(f"Loss weights - Gloss: {alpha}, Category: {beta}")
    print("-" * 60)

    epochs_to_run = epochs
    patience_counter = 0  # Counter for early stopping

    # Main training loop - iterate through epochs
    for epoch in range(start_epoch, start_epoch + epochs_to_run):
        # ========================================================================
        # EPOCH INITIALIZATION
        # ========================================================================
        
        # Get current loss weights based on curriculum or static strategy
        if curriculum_scheduler is not None:
            # Dynamic weights from curriculum learning
            current_alpha, current_beta = curriculum_scheduler.get_weights(epoch, epochs)
            phase_info = curriculum_scheduler.get_phase_info(epoch)
        else:
            # Static weights or weights from loss weighting strategy
            current_alpha, current_beta = alpha, beta
            phase_info = "Balanced training"
        
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        model.train()
        
        # Initialize epoch-level tracking variables
        total_loss = 0        # Accumulator for total loss across all batches
        num_batches = 0       # Counter for number of batches processed
        epoch_start_time = time.time()  # Track epoch duration

        # ========================================================================
        # TRAINING PHASE - GRADIENT ACCUMULATION
        # ========================================================================
        # Gradient accumulation allows us to use larger effective batch sizes
        # by accumulating gradients over multiple mini-batches before updating parameters
        
        optimizer.zero_grad(set_to_none=True)  # Clear gradients from previous epoch
        
        # Iterate through training batches
        for batch_idx, batch in enumerate(train_loader):
            # Parse batch data - handle both 3-tuple and 4-tuple formats
            if len(batch) == 4:
                X, gloss, cat, lengths = batch
                lengths = lengths.to(device, non_blocking=True)  # Sequence lengths for attention masking
            else:
                X, gloss, cat = batch
                lengths = None  # No length information available
            
            # Move tensors to device with non_blocking=True for better performance
            # non_blocking=True allows CPU-GPU transfer to overlap with computation
            X = X.to(device, non_blocking=True)      # Input features/keypoints
            gloss = gloss.to(device, non_blocking=True)  # Gloss class labels
            cat = cat.to(device, non_blocking=True)      # Category class labels

            # Forward pass with automatic mixed precision if enabled
            with torch.amp.autocast(device_type=getattr(device, "type", "cpu"), enabled=amp_enabled):
                # Model forward pass - get predictions for both tasks
                gloss_pred, cat_pred = forward_fn(model, X, lengths)
                
                # Calculate individual task losses
                loss_gloss = criterion(gloss_pred, gloss)  # Gloss classification loss
                loss_cat = criterion(cat_pred, cat)        # Category classification loss
                
                # Get dynamic weights from loss weighting strategy if available
                if loss_weighting is not None:
                    dynamic_alpha, dynamic_beta = loss_weighting.get_weights(
                        epoch, batch_idx, loss_gloss.item(), loss_cat.item(), model, optimizer
                    )
                    # Priority: curriculum > loss weighting > static
                    if curriculum_scheduler is not None:
                        # Curriculum learning takes precedence over loss weighting
                        loss = current_alpha * loss_gloss + current_beta * loss_cat
                    else:
                        # Use loss weighting strategy
                        if loss_weighting_strategy == "uncertainty":
                            # Uncertainty weighting includes uncertainty penalty
                            loss = loss_weighting.get_uncertainty_loss(loss_gloss, loss_cat)
                        else:
                            # Static or adaptive weighting
                            loss = dynamic_alpha * loss_gloss + dynamic_beta * loss_cat
                else:
                    # Use curriculum weights or static weights
                    loss = current_alpha * loss_gloss + current_beta * loss_cat
                
                # Scale loss by gradient accumulation steps to maintain correct gradient magnitude
                # This ensures that the effective learning rate remains consistent
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling for AMP
            scaler.scale(loss).backward()
            
            # Gradient accumulation: only update parameters after accumulating N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # Update model parameters
                scaler.step(optimizer)  # Apply gradients
                scaler.update()         # Update AMP scaler
                optimizer.zero_grad(set_to_none=True)  # Clear gradients for next accumulation
        
        # ========================================================================
        # END-OF-EPOCH PROCESSING
        # ========================================================================
        
        # Handle remaining gradients if last batch doesn't align with accumulation steps
        # This ensures all gradients are processed even if the total number of batches
        # is not divisible by gradient_accumulation_steps
        if len(train_loader) % gradient_accumulation_steps != 0:
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        # Update loss tracking (scale back up to get true loss magnitude)
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update Exponential Moving Average if enabled
        # EMA maintains a smoothed version of model parameters for better generalization
        if ema is not None:
            ema.update()
        
        # Clear intermediate variables to save GPU memory
        # This helps prevent memory accumulation over long training runs
        del X, gloss, cat, gloss_pred, cat_pred, loss, loss_gloss, loss_cat
        if lengths is not None:
            del lengths

        # Handle edge case where training dataloader yields zero batches
        if num_batches == 0:
            print("No training batches were provided. Check your dataset and DataLoader settings.")
            if csv_fh is not None:
                csv_fh.close()
            return

        # Calculate average training loss across all batches
        avg_train_loss = total_loss / num_batches
        
        # Update loss weighting strategy if it has adaptive components
        if loss_weighting is not None and hasattr(loss_weighting, 'update_weights'):
            loss_weighting.update_weights(
                epoch, 
                {'gloss': torch.tensor(avg_train_loss), 'cat': torch.tensor(avg_train_loss)}, 
                model, 
                optimizer
            )
        
        # ========================================================================
        # VALIDATION PHASE
        # ========================================================================
        
        # Clear GPU memory cache before validation to ensure accurate memory reporting
        clear_gpu_memory()
        
        # Apply EMA parameters for validation if EMA is enabled
        # EMA parameters often give better validation performance
        if ema is not None:
            ema.apply_shadow()
        
        # Run validation evaluation
        val_start_time = time.time()
        val_loss, val_gloss_acc, val_cat_acc = evaluate_with_forward(
            model, val_loader, criterion, device, forward_fn, 
            alpha=current_alpha, beta=current_beta
        )
        
        # Restore original model parameters after validation
        if ema is not None:
            ema.restore()
        val_time = time.time() - val_start_time
        
        epoch_time = time.time() - epoch_start_time
        
        # Get current weights for logging
        if loss_weighting is not None and curriculum_scheduler is None:
            current_alpha, current_beta = loss_weighting.get_weights(epoch, 0, 0.0, 0.0, model, optimizer)
        
        # Print epoch results with performance metrics and weighting info
        if curriculum_scheduler is not None:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Gloss Acc: {val_gloss_acc:.3f} | "
                  f"Val Cat Acc: {val_cat_acc:.3f} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"  Curriculum: {phase_info} | Weights: α={current_alpha:.3f}, β={current_beta:.3f}")
        elif loss_weighting is not None and loss_weighting_strategy != "static":
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Gloss Acc: {val_gloss_acc:.3f} | "
                  f"Val Cat Acc: {val_cat_acc:.3f} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"  Loss Weighting: {loss_weighting_strategy} | Weights: α={current_alpha:.3f}, β={current_beta:.3f}")
        else:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Gloss Acc: {val_gloss_acc:.3f} | "
                  f"Val Cat Acc: {val_cat_acc:.3f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Print GPU memory usage if available
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # Scheduler step (and then read the effective LR)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_gloss_acc)
            elif isinstance(scheduler, WarmupCosineScheduler):
                scheduler.step(epoch)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # CSV log with performance metrics
        if csv_fh is not None:
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0.0
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9 if device.type == 'cuda' else 0.0
            if curriculum_scheduler is not None:
                csv_writer.writerow([epoch + 1, avg_train_loss, val_loss, val_gloss_acc, val_cat_acc, current_lr, epoch_time, gpu_mem_alloc, gpu_mem_reserved, current_alpha, current_beta, phase_info])
            elif loss_weighting is not None and loss_weighting_strategy != "static":
                csv_writer.writerow([epoch + 1, avg_train_loss, val_loss, val_gloss_acc, val_cat_acc, current_lr, epoch_time, gpu_mem_alloc, gpu_mem_reserved, current_alpha, current_beta, loss_weighting_strategy])
            else:
                csv_writer.writerow([epoch + 1, avg_train_loss, val_loss, val_gloss_acc, val_cat_acc, current_lr, epoch_time, gpu_mem_alloc, gpu_mem_reserved])
            csv_fh.flush()

        # Checkpointing on best metric (gloss accuracy)
        metric = val_gloss_acc
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            patience_counter = 0
        else:
            patience_counter += 1

        save_state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if amp_enabled else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_metric': best_metric,
            'args': None,
        }
        save_checkpoint(save_state, is_best=is_best, output_dir=output_dir, model_name=model.__class__.__name__)

        # Early stopping
        if early_stop_patience is not None and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best gloss acc: {best_metric:.4f}")
            break

    print("-" * 60)
    print("Training completed!")
    
    if csv_fh is not None:
        csv_fh.close()

def evaluate_with_forward(model, dataloader, criterion, device, forward_fn: Callable, alpha: float = 1.0, beta: float = 1.0) -> Tuple[float, float, float]:
    """
    Evaluate model on a dataloader using a provided forward adapter.

    Args:
        model: Trained model under evaluation.
        dataloader: DataLoader providing batches.
        criterion: Loss function (cross-entropy expected).
        device: Torch device.
        forward_fn: Callable(model, X, lengths) -> (gloss_logits, cat_logits).
        alpha: Weight for gloss loss.
        beta: Weight for category loss.

    Returns:
        tuple: (avg_loss, gloss_accuracy, category_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_gloss = 0
    correct_cat = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                X, gloss, cat, lengths = batch
                lengths = lengths.to(device)
            else:
                X, gloss, cat = batch
                lengths = None
            X, gloss, cat = X.to(device), gloss.to(device), cat.to(device)
            gloss_pred, cat_pred = forward_fn(model, X, lengths)
            loss_gloss = criterion(gloss_pred, gloss)
            loss_cat = criterion(cat_pred, cat)
            batch_loss = alpha * loss_gloss + beta * loss_cat
            cat_preds = cat_pred.argmax(dim=1)
            correct_cat += (cat_preds == cat).sum().item()

            gloss_preds = gloss_pred.argmax(dim=1)
            correct_gloss += (gloss_preds == gloss).sum().item()
            total_samples += gloss.size(0)
            total_loss += batch_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    gloss_accuracy = correct_gloss / total_samples if total_samples > 0 else 0.0
    cat_accuracy = correct_cat / total_samples if total_samples > 0 else 0.0
    return avg_loss, gloss_accuracy, cat_accuracy

def train_ctc(
    model,
    train_loader,
    val_loader,
    device,
    blank_id,
    epochs=50,
    output_dir=".",
    lr=1e-4,
    weight_decay=0.0,
    use_amp=False,
    grad_clip=None,
    scheduler_type=None,
    scheduler_patience=5,
    warmup_epochs=5,
    early_stop_patience=None,
    resume_path=None,
    log_csv_path=None,
    gradient_accumulation_steps=1,
    compile_model=False,
    use_ema=False,
    ema_decay=0.999,
    alpha=0.7,
    beta=0.3,
):
    """
    Train a CTC model for continuous sign language recognition with optional category learning.
    
    This function implements CTC-based training for sequence-to-sequence learning
    without requiring frame-level alignment. Supports both CTC-only and dual-task modes:
    - CTC-only: gloss transcription only (beta=0)
    - Dual-task: gloss transcription + category classification (beta>0)
    
    Args:
        model: CTC model to train (SignTransformerCtc or MediaPipeGRUCtc)
        train_loader: DataLoader with CTC-formatted batches (uses collate_for_ctc)
        val_loader: DataLoader for validation data
        device: Torch device (cuda/cpu)
        blank_id: ID of the blank token for CTC (typically num_gloss_classes)
        epochs: Number of training epochs
        output_dir: Directory to save checkpoints
        lr: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        use_amp: Enable automatic mixed precision
        grad_clip: Gradient clipping max norm (None to disable)
        scheduler_type: LR scheduler ('plateau', 'cosine', 'warmup_cosine', or None)
        scheduler_patience: Patience for plateau scheduler
        warmup_epochs: Warmup epochs for warmup_cosine scheduler
        early_stop_patience: Stop if no improvement for this many epochs
        resume_path: Path to checkpoint to resume from
        log_csv_path: Path to CSV log file for metrics
        gradient_accumulation_steps: Number of steps to accumulate gradients
        compile_model: Whether to compile model (PyTorch 2.0+)
        use_ema: Enable Exponential Moving Average
        ema_decay: EMA decay rate
    
    Returns:
        None (saves checkpoints to output_dir)
    """
    # ============================================================================
    # INITIAL SETUP
    # ============================================================================
    
    # Clear GPU memory for clean start
    clear_gpu_memory()
    
    print("\n" + "="*80)
    print("CTC TRAINING CONFIGURATION")
    print("="*80)
    print(f"Training Mode: Continuous Sign Language Recognition (CTC)")
    print(f"Model: {model.__class__.__name__}")
    print(f"Blank Token ID: {blank_id}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Gradient Accumulation: {gradient_accumulation_steps}")
    print(f"AMP Enabled: {use_amp and device.type == 'cuda'}")
    print("="*80 + "\n")
    
    # ============================================================================
    # MODEL OPTIMIZATION
    # ============================================================================
    
    # Model compilation for performance (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✓ Model compiled for better performance")
        except Exception as e:
            print(f"⚠ Model compilation failed: {e}")
    
    # ============================================================================
    # CTC LOSS AND OPTIMIZER SETUP
    # ============================================================================
    
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    print(f"✓ Using CTCLoss (blank={blank_id}, zero_infinity=True)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Automatic Mixed Precision (AMP)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    
    print(f"✓ Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
    print(f"✓ AMP enabled: {amp_enabled}")
    
    # ============================================================================
    # LEARNING RATE SCHEDULER
    # ============================================================================
    
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=scheduler_patience
        )
        print(f"✓ Scheduler: ReduceLROnPlateau (patience={scheduler_patience})")
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"✓ Scheduler: CosineAnnealingLR")
    elif scheduler_type == "warmup_cosine":
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, lr)
        print(f"✓ Scheduler: WarmupCosineScheduler (warmup={warmup_epochs})")
    else:
        scheduler = None
        print("✓ No learning rate scheduler")
    
    # ============================================================================
    # EXPONENTIAL MOVING AVERAGE (EMA)
    # ============================================================================
    
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        ema.register()
        print(f"✓ EMA enabled (decay={ema_decay})")
    
    # ============================================================================
    # RESUME TRAINING
    # ============================================================================
    
    start_epoch = 0
    best_metric = float('inf')  # For CTC, lower loss is better
    
    if resume_path is not None and os.path.isfile(resume_path):
        print(f"\nLoading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        
        if 'scaler' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler'])
        if 'scheduler' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        
        start_epoch = ckpt.get('epoch', 0)
        best_metric = ckpt.get('best_metric', best_metric)
        print(f"✓ Resumed from epoch {start_epoch} (best_loss={best_metric:.4f})")
    
    # ============================================================================
    # CSV LOGGING
    # ============================================================================
    
    csv_fh = None
    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path) or '.', exist_ok=True)
        new_file = not os.path.exists(log_csv_path)
        csv_fh = open(log_csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_fh)
        
        if new_file:
            # Write header
            csv_writer.writerow([
                "epoch", "train_loss", "val_loss", "lr", "epoch_time",
                "gpu_memory_allocated", "gpu_memory_reserved"
            ])
        print(f"✓ Logging metrics to: {log_csv_path}")
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    
    print(f"\n{'='*80}")
    print(f"STARTING CTC TRAINING - {epochs} epochs")
    print(f"{'='*80}\n")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # ========================================================================
        # TRAINING PHASE
        # ========================================================================
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack CTC batch: (X, targets, input_lengths, target_lengths, cat_targets)
            X, targets, input_lengths, target_lengths, cat_targets = batch
            
            # Move to device
            X = X.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            cat_targets = cat_targets.to(device, non_blocking=True)
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                # Model returns log_probs [B,T,C] or (log_probs, cat_logits) for dual-task
                # Handle iv3_gru_ctc model which needs features_already=True
                if hasattr(model, '__class__') and 'InceptionV3GRUCtc' in model.__class__.__name__:
                    output = model(X, features_already=True)
                else:
                    output = model(X)
                
                if isinstance(output, tuple):
                    # Dual-task mode: CTC + Category
                    log_probs, cat_logits = output  # [B,T,num_ctc], [B,T,num_cat]
                    log_probs = log_probs.permute(1, 0, 2)  # [T, B, C] for CTC loss
                    loss_ctc = criterion(log_probs, targets, input_lengths, target_lengths)
                    
                    # Per-frame category loss: flatten and align with cat_targets
                    B, T, num_cat = cat_logits.shape
                    cat_logits_flat = cat_logits.reshape(B * T, num_cat)  # [B*T, num_cat]
                    cat_targets_expanded = cat_targets.unsqueeze(1).expand(B, T).reshape(B * T)  # [B*T]
                    loss_cat = nn.functional.cross_entropy(cat_logits_flat, cat_targets_expanded, reduction='mean')
                    
                    loss = alpha * loss_ctc + beta * loss_cat
                else:
                    # CTC-only mode
                    log_probs = output.permute(1, 0, 2)  # [T, B, C]
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation: update every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
        
        # Handle remaining gradients
        if len(train_loader) % gradient_accumulation_steps != 0:
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Average training loss
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # ========================================================================
        # VALIDATION PHASE
        # ========================================================================
        
        clear_gpu_memory()
        
        # Apply EMA for validation
        if ema is not None:
            ema.apply_shadow()
        
        val_loss, val_gloss_acc, val_cat_acc = evaluate_ctc(
            model, val_loader, criterion, device, blank_id, alpha, beta
        )
        
        # Restore original parameters
        if ema is not None:
            ema.restore()
        
        epoch_time = time.time() - epoch_start_time
        
        # ========================================================================
        # LOGGING AND CHECKPOINTING
        # ========================================================================
        
        # Print epoch results
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        gloss_str = f" | Val Gloss Acc: {val_gloss_acc:.3f}"
        cat_str = f" | Val Cat Acc: {val_cat_acc:.3f}" if beta > 0 else ""
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}{gloss_str}{cat_str} | "
              f"Time: {epoch_time:.1f}s")
        
        # Print GPU memory if available
        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(0) / 1e9
            mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, WarmupCosineScheduler):
                scheduler.step(epoch)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # CSV logging
        if csv_fh is not None:
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0.0
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9 if device.type == 'cuda' else 0.0
            csv_writer.writerow([
                epoch + 1, avg_train_loss, val_loss, current_lr,
                epoch_time, gpu_mem_alloc, gpu_mem_reserved
            ])
            csv_fh.flush()
        
        # Checkpointing (lower loss is better for CTC)
        is_best = val_loss < best_metric
        if is_best:
            best_metric = val_loss
            patience_counter = 0
            print(f"  ✓ New best validation loss: {best_metric:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if amp_enabled else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_metric': best_metric,
            'blank_id': blank_id,
        }
        save_checkpoint(save_state, is_best, output_dir, model.__class__.__name__)
        
        # Early stopping
        if early_stop_patience is not None and patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            print(f"Best validation loss: {best_metric:.4f}")
            break
    
    # ============================================================================
    # TRAINING COMPLETE
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("CTC TRAINING COMPLETED!")
    print(f"Best validation loss: {best_metric:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    if csv_fh is not None:
        csv_fh.close()

def evaluate_ctc(model, dataloader, criterion, device, blank_id, alpha=1.0, beta=0.0):
    """
    Evaluate a CTC model on a validation dataset.
    
    Args:
        model: CTC model to evaluate
        dataloader: DataLoader with CTC-formatted batches
        criterion: CTCLoss criterion
        device: Torch device
        alpha: Weight for CTC loss (dual-task mode)
        beta: Weight for category loss (dual-task mode)
    
    Returns:
        tuple: (avg_loss, gloss_accuracy, cat_accuracy)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_gloss_seq = 0
    total_sequences = 0
    correct_cat = 0
    total_samples = 0
    has_cat_head = beta > 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack CTC batch
            X, targets, input_lengths, target_lengths, cat_targets = batch
            
            # Move to device
            X = X.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            cat_targets = cat_targets.to(device)
            
            # Forward pass
            # Handle iv3_gru_ctc model which needs features_already=True
            if hasattr(model, '__class__') and 'InceptionV3GRUCtc' in model.__class__.__name__:
                output = model(X, features_already=True)
            else:
                output = model(X)
            
            if isinstance(output, tuple):
                # Dual-task mode
                log_probs, cat_logits = output  # [B,T,num_ctc], [B,T,num_cat]
                log_probs = log_probs.permute(1, 0, 2)  # [T, B, C] for CTC loss
                loss_ctc = criterion(log_probs, targets, input_lengths, target_lengths)
                
                # Per-frame category loss
                B, T, num_cat = cat_logits.shape
                cat_logits_flat = cat_logits.reshape(B * T, num_cat)
                cat_targets_expanded = cat_targets.unsqueeze(1).expand(B, T).reshape(B * T)
                loss_cat = nn.functional.cross_entropy(cat_logits_flat, cat_targets_expanded, reduction='mean')
                
                loss = alpha * loss_ctc + beta * loss_cat
                
                # Track category accuracy (per-sequence: majority vote)
                if has_cat_head:
                    for i in range(B):
                        seq_len = input_lengths[i]
                        cat_pred_seq = cat_logits[i, :seq_len].argmax(dim=1)  # [seq_len]
                        # Majority vote for sequence-level category
                        pred_cat = cat_pred_seq.mode().values.item()
                        true_cat = cat_targets[i].item()
                        correct_cat += (pred_cat == true_cat)
                    total_samples += B
            else:
                # CTC-only mode
                log_probs = output.permute(1, 0, 2)  # [T, B, C]
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            total_loss += loss.item()
            num_batches += 1

            # -------- Gloss sequence accuracy via greedy decoding --------
            # Decode per sequence and compare to ground truth targets
            T_total, B_lp, C = log_probs.shape
            # Prepare CPU copies of lengths for safe Python iteration
            input_lens_list = input_lengths.detach().cpu().tolist()
            target_lens_list = target_lengths.detach().cpu().tolist()
            # Offset into concatenated targets
            tgt_offset = 0
            pred_ids = log_probs.argmax(dim=2)  # [T, B]
            for i in range(B_lp):
                t_len = int(input_lens_list[i])
                # Greedy decode with collapse and blank removal
                prev_id = None
                decoded = []
                for t in range(t_len):
                    idx = int(pred_ids[t, i].item())
                    if idx == blank_id:
                        prev_id = idx
                        continue
                    if prev_id is None or idx != prev_id:
                        decoded.append(idx)
                    prev_id = idx
                # Ground truth slice
                g_len = int(target_lens_list[i])
                gt_seq = targets[tgt_offset:tgt_offset + g_len].detach().cpu().tolist()
                tgt_offset += g_len
                # Presence-based accuracy: correct if ground truth gloss appears in decoded sequence
                correct_gloss_seq += int(gt_seq[0] in decoded) if len(gt_seq) > 0 else 0
                total_sequences += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    gloss_accuracy = correct_gloss_seq / total_sequences if total_sequences > 0 else 0.0
    cat_accuracy = correct_cat / total_samples if total_samples > 0 else 1.0
    return avg_loss, gloss_accuracy, cat_accuracy

def load_data(n_train_samples=100, n_val_samples=20, seq_length=150, input_dim=178, num_gloss=105, num_cat=10, seed=42):
    """
    Load training and validation data for sign language recognition.
    
    Placeholder function that generates dummy data for testing.
    Replace with actual data loading from preprocessed .npz files.
    
    Returns:
        tuple: (train_X, train_gloss, train_cat, val_X, val_gloss, val_cat)
            - train_X: Training sequences [N_train, T, 178]
            - train_gloss: Training gloss labels [N_train]
            - train_cat: Training category labels [N_train]
            - val_X: Validation sequences [N_val, T, 178]
            - val_gloss: Validation gloss labels [N_val]
            - val_cat: Validation category labels [N_val]
    """
    # Dummy data configuration (override via parameters)
    rng = np.random.default_rng(seed)
    
    # Generate random training data
    train_X = rng.standard_normal((n_train_samples, seq_length, input_dim), dtype=np.float32)
    train_gloss = rng.integers(0, num_gloss, n_train_samples)
    train_cat = rng.integers(0, num_cat, n_train_samples)
    
    # Generate random validation data
    val_X = rng.standard_normal((n_val_samples, seq_length, input_dim), dtype=np.float32)
    val_gloss = rng.integers(0, num_gloss, n_val_samples)
    val_cat = rng.integers(0, num_cat, n_val_samples)
    
    return train_X, train_gloss, train_cat, val_X, val_gloss, val_cat


def parse_args():
    """
    Parse command-line arguments for comprehensive training configuration.
    
    This function sets up all command-line arguments needed for training sign language
    recognition models, including model selection, data configuration, training parameters,
    optimization settings, and advanced features like curriculum learning.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train Sign Language Recognition models with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (logs auto-created in output-dir with timestamp)
  python training/train.py --model transformer --epochs 50 \\
    --output-dir trained_models/transformer/run1
  
  # Training with custom log file name
  python training/train.py --model transformer --epochs 50 \\
    --log-file trained_models/transformer/my_training.log \\
    --output-dir trained_models/transformer/run1
  
  # Advanced training with curriculum learning
  python training/train.py --model iv3_gru --curriculum gloss-first --epochs 100 \\
    --output-dir trained_models/iv3_gru/curriculum
  
  # Quick smoke test
  python training/train.py --smoke-test
        """
    )
    # ============================================================================
    # BASIC TRAINING CONFIGURATION
    # ============================================================================
    parser.add_argument("--model", choices=["transformer", "mediapipe_gru", "iv3_gru", "transformer_ctc", "mediapipe_gru_ctc", "iv3_gru_ctc"], default="transformer", 
                       help="Model architecture: 'transformer' (keypoints), 'mediapipe_gru' (keypoints, lightweight), 'iv3_gru' (features, offline baseline), 'transformer_ctc' (CTC), 'mediapipe_gru_ctc' (CTC), 'iv3_gru_ctc' (CTC)")
    parser.add_argument("--training-mode", type=str, choices=["classification", "ctc"], default="classification",
                       help="Training mode: 'classification' (isolated signs) or 'ctc' (continuous recognition)")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of training epochs to run")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Number of samples per training batch")
    parser.add_argument("--alpha", type=float, default=0.5, 
                       help="Weight for gloss classification loss in multi-task training (classification mode only)")
    parser.add_argument("--beta", type=float, default=0.5, 
                       help="Weight for category classification loss in multi-task training (classification mode only)")
    
    # Class configuration
    parser.add_argument("--num-gloss", type=int, default=105, 
                       help="Number of gloss classes in the dataset")
    parser.add_argument("--num-cat", type=int, default=10, 
                       help="Number of category classes in the dataset (classification mode only)")
    parser.add_argument("--num-ctc-classes", type=int, default=106,
                       help="Number of CTC classes including blank token (CTC mode only, default: num_gloss + 1)")
    parser.add_argument("--ctc-blank-id", type=int, default=None,
                       help="Blank token ID for CTC (default: num_gloss)")
    # ============================================================================
    # DATA CONFIGURATION - IV3-GRU FEATURES
    # ============================================================================
    parser.add_argument("--features-train", type=str, default=None, 
                       help="Directory containing training .npz files with 2048-dimensional features")
    parser.add_argument("--features-val", type=str, default=None, 
                       help="Directory containing validation .npz files with 2048-dimensional features")
    parser.add_argument("--labels-train-csv", type=str, default=None, 
                       help="CSV file with columns: file,gloss,cat for training data labels")
    parser.add_argument("--labels-val-csv", type=str, default=None, 
                       help="CSV file with columns: file,gloss,cat for validation data labels")
    parser.add_argument("--feature-key", type=str, default="X2048", 
                       help="Key name in .npz files containing [T,2048] feature arrays")
    # ============================================================================
    # DATA CONFIGURATION - TRANSFORMER KEYPOINTS
    # ============================================================================
    parser.add_argument("--keypoints-train", type=str, default=None, 
                       help="Directory containing training .npz files with keypoint sequences [T,178]")
    parser.add_argument("--keypoints-val", type=str, default=None, 
                       help="Directory containing validation .npz files with keypoint sequences [T,178]")
    parser.add_argument("--kp-key", type=str, default="X", 
                       help="Key name in .npz files containing [T,178] keypoint arrays")
    parser.add_argument("--combine-features", action="store_true",
                       help="Combine keypoints [T,178] and features [T,2048] into single input [T,2226] (Transformer only)")
    # IV3-GRU hyperparameters
    parser.add_argument("--hidden1", type=int, default=16, help="IV3-GRU first GRU hidden size")
    parser.add_argument("--hidden2", type=int, default=12, help="IV3-GRU second GRU hidden size")
    parser.add_argument("--dropout", type=float, default=0.3, help="IV3-GRU dropout rate")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use ImageNet-pretrained InceptionV3")
    parser.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze InceptionV3 weights")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    # Optimizer & training controls
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping max norm")
    parser.add_argument("--scheduler", type=str, default=None, choices=["plateau", "cosine", "warmup_cosine"], help="LR scheduler type")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="Patience for plateau scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs for warmup_cosine scheduler")
    parser.add_argument("--early-stop", type=int, default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-csv", type=str, default=None, help="Path to CSV log file for metrics (auto-created if not specified)")
    parser.add_argument("--log-file", type=str, default=None, help="Path to text file for console logs (auto-created in output-dir with timestamp if not specified)")
    # DataLoader performance
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor (worker>0)")
    # Sequence length for synthetic data (kept for compatibility)
    parser.add_argument("--seq-length", type=int, default=150, help="Sequence length (T) - for synthetic data only")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA ops (slower)")
    # Smoke test
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick forward/backward/save/load test and exit")
    parser.add_argument("--smoke-batch-size", type=int, default=4, help="Smoke test batch size")
    parser.add_argument("--smoke-T", type=int, default=30, help="Smoke test sequence length T")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save model checkpoints")
    # Performance optimization arguments
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--compile-model", action="store_true", help="Compile model for better performance (PyTorch 2.0+)")
    parser.add_argument("--auto-workers", action="store_true", help="Auto-detect optimal number of DataLoader workers")
    parser.add_argument("--auto-batch-size", action="store_true", help="Auto-calculate optimal batch size based on available memory")
    parser.add_argument("--enable-parallel", action="store_true", help="Enable DataParallel for multiple GPUs")
    # Curriculum training arguments
    parser.add_argument("--curriculum", type=str, default=None, choices=["gloss-first", "category-first", "dynamic"], 
                       help="Curriculum training strategy: gloss-first, category-first, or dynamic weighting")
    parser.add_argument("--curriculum-epochs", type=int, default=10, 
                       help="Number of epochs for curriculum phase (when to start balancing tasks)")
    parser.add_argument("--curriculum-warmup", type=int, default=5, 
                       help="Number of warmup epochs before starting curriculum (for dynamic strategy)")
    parser.add_argument("--curriculum-min-weight", type=float, default=0.1, 
                       help="Minimum weight for secondary task during curriculum (0.0-1.0)")
    parser.add_argument("--curriculum-schedule", type=str, default="linear", choices=["linear", "cosine", "exponential"], 
                       help="Curriculum weight scheduling function: linear, cosine, or exponential")
    # Loss weighting strategy arguments
    parser.add_argument("--loss-weighting", type=str, default="static", 
                       choices=["static", "grid-search", "uncertainty", "gradnorm"], 
                       help="Loss weighting strategy: static, grid-search, uncertainty, or gradnorm")
    parser.add_argument("--grid-search-weights", type=str, default="0.5,0.5;0.7,0.3;0.3,0.7;0.8,0.2;0.2,0.8", 
                       help="Grid search weight combinations (format: 'a1,b1;a2,b2;...')")
    parser.add_argument("--uncertainty-init", type=float, default=1.0, 
                       help="Initial uncertainty for uncertainty weighting")
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5, 
                       help="Alpha parameter for GradNorm weighting")
    parser.add_argument("--gradnorm-update-freq", type=int, default=1, 
                       help="Update frequency for GradNorm (every N epochs)")
    # Advanced loss functions
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal", "label_smoothing"], 
                       help="Loss function type: ce (CrossEntropy), focal, or label_smoothing")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--focal-alpha", type=float, default=1.0, help="Focal loss alpha parameter")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor (0.0-1.0)")
    # Data augmentation
    parser.add_argument("--augment", action="store_true", help="Enable temporal data augmentation")
    parser.add_argument("--augment-noise-std", type=float, default=0.01, help="Standard deviation for noise augmentation")
    parser.add_argument("--augment-mask-prob", type=float, default=0.1, help="Probability of time masking")
    parser.add_argument("--augment-mask-ratio", type=float, default=0.1, help="Ratio of sequence length to mask")
    # EMA (Exponential Moving Average)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate (0.0-1.0)")
    parser.add_argument("--use-ema", action="store_true", help="Enable Exponential Moving Average")
    # Export for Android (Full Runtime .pt)
    parser.add_argument("--export-mobile", action="store_true", help="Export TorchScript .pt for Android after training")
    parser.add_argument("--export-only", action="store_true", help="Skip training and only export using --resume checkpoint or best in output-dir")
    parser.add_argument("--export-output", type=str, default="android_artifacts", help="Directory to write exported artifacts")
    parser.add_argument("--export-example-T", type=int, default=120, help="Representative T if tracing is required during export")
    parser.add_argument("--window-hint", type=int, default=120, help="Metadata: window size hint for Android")
    parser.add_argument("--stride-hint", type=int, default=40, help="Metadata: stride hint for Android")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    
    # Setup log files automatically if not specified
    original_stdout = None
    
    # Auto-create log file in output directory if not explicitly specified
    if args.log_file is None and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = os.path.join(args.output_dir, f"training_{timestamp}.log")
    
    # Setup logging if log file is specified (either manually or auto-created)
    if args.log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # If CSV log not specified, automatically create one in same directory as log file
        if args.log_csv is None:
            log_filename = os.path.splitext(os.path.basename(args.log_file))[0]
            args.log_csv = os.path.join(log_dir if log_dir else '.', f"{log_filename}_metrics.csv")
        
        # Redirect stdout to both console and log file
        original_stdout = sys.stdout
        sys.stdout = TeeLogger(args.log_file)
        print(f"{'='*60}")
        print(f"TRAINING LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Console output will be saved to: {args.log_file}")
        print(f"Metrics CSV will be saved to: {args.log_csv}\n")
    
    # Optimized device setup
    device = get_optimal_device()
    print_device_info(device)

    # Optional smoke test (only if comparable path exists for Transformer already)
    if args.smoke_test:
        torch.manual_seed(args.seed)
        if args.model == "iv3_gru":
            # Random [B, T, 2048] features
            B = args.smoke_batch_size
            T = args.smoke_T
            X = torch.randn(B, T, 2048, dtype=torch.float32, device=device)
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
            model = InceptionV3GRU(
                num_gloss=args.num_gloss,
                num_cat=args.num_cat,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                dropout=args.dropout,
                pretrained_backbone=args.pretrained_backbone,
                freeze_backbone=args.freeze_backbone,
            ).to(device)
            model.train()
            gloss_logits, cat_logits = model(X, lengths=lengths, features_already=True)
            assert gloss_logits.shape == (B, args.num_gloss)
            assert cat_logits.shape == (B, args.num_cat)
            loss = (gloss_logits.mean() + cat_logits.mean())
            loss.backward()
            ckpt_dir = os.path.join("data", "processed")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, f"{model.__class__.__name__}.pt")
            torch.save(model.state_dict(), ckpt)
            _ = model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"✓ IV3-GRU smoke test passed. Saved and loaded: {ckpt}")
            exit(0)
        else:
            # Transformer smoke (uses existing forward contract on [B, T, 178])
            B = args.smoke_batch_size
            T = args.smoke_T
            X = torch.randn(B, T, 178, dtype=torch.float32, device=device)
            model = SignTransformer(num_gloss=args.num_gloss, num_cat=args.num_cat).to(device)
            model.train()
            gloss_logits, cat_logits = model(X)
            assert gloss_logits.shape == (B, args.num_gloss)
            assert cat_logits.shape == (B, args.num_cat)
            loss = (gloss_logits.mean() + cat_logits.mean())
            loss.backward()
            ckpt_dir = os.path.join("data", "processed")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, f"{model.__class__.__name__}.pt")
            torch.save(model.state_dict(), ckpt)
            _ = model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"✓ Transformer smoke test passed. Saved and loaded: {ckpt}")
            exit(0)

    # Export-only quick path
    if args.export_only:
        from training.export_mobile import export_model_for_android
        ckpt_path = args.resume or None
        if ckpt_path is None:
            # Guess best from output-dir based on model name
            def _guess_best(output_dir: str, model_name: str) -> str:
                mapping = {
                    'transformer_ctc': 'SignTransformerCtc',
                    'mediapipe_gru_ctc': 'MediaPipeGRUCtc',
                }
                stem = mapping.get(model_name)
                if not stem:
                    return ''
                candidate = os.path.join(output_dir, f"{stem}_best.pt")
                return candidate if os.path.exists(candidate) else ''
            guessed = _guess_best(args.output_dir, args.model)
            if not guessed:
                raise FileNotFoundError("--export-only requires --resume or an existing *_best.pt in --output-dir")
            ckpt_path = guessed
        export_model_for_android(
            model_name=args.model,
            checkpoint_path=ckpt_path,
            output_dir=args.export_output,
            input_dim=178 if args.kp_key != 'X2048' else 2048,
            num_cat=args.num_cat,
            window_hint=args.window_hint,
            stride_hint=args.stride_hint,
            example_T=args.export_example_T,
        )
        # Close logger before exiting
        if args.log_file and original_stdout is not None:
            print(f"\n{'='*60}")
            print(f"Export completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log saved to: {args.log_file}")
            print(f"{'='*60}")
            sys.stdout.close()
            sys.stdout = original_stdout
        exit(0)

    # Data loading
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    
    try:
        # If dataset directories are provided, use file-based datasets; otherwise synthetic
        use_feature_files = (
            args.model in ["iv3_gru", "iv3_gru_ctc"] and args.features_train is not None and args.features_val is not None
        )
        use_keypoint_files = (
            args.model in ["transformer", "transformer_ctc", "mediapipe_gru", "mediapipe_gru_ctc"] and 
            args.keypoints_train is not None and args.keypoints_val is not None
        )
        if not (use_feature_files or use_keypoint_files):
            raise ValueError(
                "No data files provided. Please specify either:\n"
                "  - --features-train/--features-val for IV3-GRU model, OR\n"
                "  - --keypoints-train/--keypoints-val for Transformer/MediaPipeGRU models"
            )
        print(f"✓ Loaded data successfully")
        
        # Check for combined features mode (Transformer only)
        use_combined_features = (
            args.model == "transformer" and args.combine_features and 
            args.keypoints_train is not None and args.keypoints_val is not None
        )
        
        # Log dataset source information
        if use_combined_features:
            print(f"  - Dataset type: Transformer Combined (Keypoints + Features)")
            print(f"  - Training folder: {args.keypoints_train}")
            print(f"  - Validation folder: {args.keypoints_val}")
            print(f"  - Keypoint key: {args.kp_key} (178-dim)")
            print(f"  - Feature key: {args.feature_key} (2048-dim)")
            print(f"  - Combined dimension: 2226 (178 + 2048)")
        elif use_feature_files:
            print(f"  - Dataset type: IV3-GRU Features")
            print(f"  - Training folder: {args.features_train}")
            print(f"  - Validation folder: {args.features_val}")
            print(f"  - Feature key: {args.feature_key}")
        elif use_keypoint_files:
            model_name = args.model.replace("_", "-").upper()
            print(f"  - Dataset type: {model_name} Keypoints")
            print(f"  - Training folder: {args.keypoints_train}")
            print(f"  - Validation folder: {args.keypoints_val}")
            print(f"  - Keypoint key: {args.kp_key}")
        
        print(f"  - Gloss classes: {args.num_gloss}")
        print(f"  - Category classes: {args.num_cat}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        exit(1)

    # Dataset preparation
    print("\n" + "="*60)
    print("DATASET PREPARATION")
    print("="*60)
    
    batch_size = args.batch_size
    
    # Optimize batch size if requested BEFORE creating datasets
    if args.auto_batch_size:
        batch_size = calculate_optimal_batch_size(model, device, args.batch_size)
        print(f"✓ Auto-calculated optimal batch size: {batch_size}")

    # Prepare augmentation parameters (shared for both datasets)
    augment_params = None
    if args.augment:
        augment_params = {
            'noise_std': args.augment_noise_std,
            'time_mask_prob': args.augment_mask_prob,
            'time_mask_ratio': args.augment_mask_ratio
        }

    # Auto-detect training mode from model name if not explicitly specified
    if args.training_mode == "classification" and args.model in ["transformer_ctc", "mediapipe_gru_ctc"]:
        args.training_mode = "ctc"
        print(f"⚠ Auto-detected CTC training mode from model name: {args.model}")
    
    # Set blank token ID for CTC
    if args.training_mode == "ctc":
        if args.ctc_blank_id is None:
            args.ctc_blank_id = args.num_gloss
        
        print(f"✓ CTC Mode: blank_id={args.ctc_blank_id}, num_ctc_classes={args.num_ctc_classes}")
        
        # Validate configuration
        try:
            from data.labels.label_mapping import validate_ctc_config
            validate_ctc_config(args.num_gloss, args.num_ctc_classes, args.ctc_blank_id)
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)
        except ImportError:
            pass
    
    # Select appropriate collate function based on mode
    if args.training_mode == "ctc":
        collate_fn = collate_for_ctc
        dataset_mode = "ctc"
    else:
        collate_fn = collate_keypoints_with_padding if not use_feature_files else collate_features_with_padding
        dataset_mode = "classification"
    
    if use_combined_features:
        # Combined mode: Load both keypoints and features
        if args.labels_train_csv is None or not os.path.exists(args.labels_train_csv):
            raise FileNotFoundError(f"Training labels CSV not found: {args.labels_train_csv}")
        if args.labels_val_csv is None or not os.path.exists(args.labels_val_csv):
            raise FileNotFoundError(f"Validation labels CSV not found: {args.labels_val_csv}")
        
        train_dataset = FSLCombinedFileDataset(
            data_dir=args.keypoints_train,
            labels_csv=args.labels_train_csv,
            kp_key=args.kp_key,
            feature_key=args.feature_key,
            augment=args.augment,
            augment_params=augment_params,
            mode=dataset_mode,
        )
        val_dataset = FSLCombinedFileDataset(
            data_dir=args.keypoints_val,
            labels_csv=args.labels_val_csv,
            kp_key=args.kp_key,
            feature_key=args.feature_key,
            augment=False,  # No augmentation for validation
            augment_params=None,
            mode=dataset_mode,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_fn)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_fn)
    elif use_feature_files:
        # Validate CSV files exist
        if args.labels_train_csv is None or not os.path.exists(args.labels_train_csv):
            raise FileNotFoundError(f"Training labels CSV not found: {args.labels_train_csv}")
        if args.labels_val_csv is None or not os.path.exists(args.labels_val_csv):
            raise FileNotFoundError(f"Validation labels CSV not found: {args.labels_val_csv}")
        
        train_dataset = FSLFeatureFileDataset(
            features_dir=args.features_train,
            labels_csv=args.labels_train_csv,
            feature_key=args.feature_key,
            augment=args.augment,
            augment_params=augment_params,
            mode=dataset_mode,
        )
        val_dataset = FSLFeatureFileDataset(
            features_dir=args.features_val,
            labels_csv=args.labels_val_csv,
            feature_key=args.feature_key,
            augment=False,  # No augmentation for validation
            augment_params=None,
            mode=dataset_mode,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_fn)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_fn)
    elif use_keypoint_files:
        # Validate CSV files exist
        if args.labels_train_csv is None or not os.path.exists(args.labels_train_csv):
            raise FileNotFoundError(f"Training labels CSV not found: {args.labels_train_csv}")
        if args.labels_val_csv is None or not os.path.exists(args.labels_val_csv):
            raise FileNotFoundError(f"Validation labels CSV not found: {args.labels_val_csv}")
        
        train_dataset = FSLKeypointFileDataset(
            keypoints_dir=args.keypoints_train,
            labels_csv=args.labels_train_csv,
            kp_key=args.kp_key,
            augment=args.augment,
            augment_params=augment_params,
            mode=dataset_mode,
        )
        val_dataset = FSLKeypointFileDataset(
            keypoints_dir=args.keypoints_val,
            labels_csv=args.labels_val_csv,
            kp_key=args.kp_key,
            augment=False,  # No augmentation for validation
            augment_params=None,
            mode=dataset_mode,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_fn)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_fn)
    
    print(f"✓ Created datasets and data loaders")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    if args.augment:
        print(f"  - Data augmentation: Enabled (noise_std={args.augment_noise_std}, mask_prob={args.augment_mask_prob})")
    
    # Log dataset details
    if use_combined_features:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 2226] combined (178 keypoints + 2048 features)")
    elif use_feature_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 2048] features")
    elif use_keypoint_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 178] keypoints")

    # Model selection
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    print("Available models:")
    print("Classification models (isolated signs):")
    print("  - transformer: Multi-head attention transformer (keypoints)")
    print("  - mediapipe_gru: Lightweight GRU (keypoints, mobile-friendly)")
    print("  - iv3_gru: InceptionV3 + GRU hybrid (features, offline baseline)")
    print("CTC models (continuous recognition):")
    print("  - transformer_ctc: Transformer with CTC (keypoints)")
    print("  - mediapipe_gru_ctc: Lightweight GRU with CTC (keypoints)")
    print("  - iv3_gru_ctc: InceptionV3 + GRU with CTC (features, offline baseline)")
    
    if args.model == "transformer":
        # Determine input dimension based on the data mode being used
        if use_combined_features:
            input_dim = 2226  # 178 keypoints + 2048 features
        elif args.kp_key == "X2048":
            input_dim = 2048
        else:
            input_dim = 178  # Default for keypoints
        
        model = SignTransformer(
            input_dim=input_dim,
            num_gloss=args.num_gloss,
            num_cat=args.num_cat,
        ).to(device)
        print(f"✓ Using SignTransformer model (input_dim={input_dim})")
    
    elif args.model == "transformer_ctc":
        # Determine input dimension
        if use_combined_features:
            input_dim = 2226
        elif args.kp_key == "X2048":
            input_dim = 2048
        else:
            input_dim = 178
        
        model = SignTransformerCtc(
            input_dim=input_dim,
            num_ctc_classes=args.num_ctc_classes,
            num_cat=args.num_cat if hasattr(args, 'num_cat') else None,
        ).to(device)
        cat_info = f", num_cat={args.num_cat}" if hasattr(args, 'num_cat') and args.num_cat else ""
        print(f"✓ Using SignTransformerCtc model (input_dim={input_dim}, num_ctc_classes={args.num_ctc_classes}{cat_info})")
    
    elif args.model == "mediapipe_gru":
        # MediaPipeGRU always uses keypoints (178D)
        if use_feature_files or args.kp_key == "X2048":
            raise ValueError(
                "MediaPipeGRU requires keypoint data (178D), not features (2048D). "
                "Use --kp-files or ensure --kp-key='X' (default)"
            )
        
        input_dim = 178  # Keypoints only
        
        model = MediaPipeGRU(
            input_dim=input_dim,
            num_gloss=args.num_gloss,
            num_cat=args.num_cat,
            projection_dim=None,  # No projection by default
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            bidirectional=False,  # Default to unidirectional for speed
        ).to(device)
        print(f"✓ Using MediaPipeGRU model (input_dim={input_dim}, hidden1={args.hidden1}, hidden2={args.hidden2})")
    
    elif args.model == "mediapipe_gru_ctc":
        # MediaPipeGRUCtc always uses keypoints (178D)
        if use_feature_files or args.kp_key == "X2048":
            raise ValueError(
                "MediaPipeGRUCtc requires keypoint data (178D), not features (2048D). "
                "Use --keypoints-train/--keypoints-val with --kp-key='X'"
            )
        
        input_dim = 178
        
        model = MediaPipeGRUCtc(
            input_dim=input_dim,
            num_ctc_classes=args.num_ctc_classes,
            num_cat=args.num_cat if hasattr(args, 'num_cat') else None,
            projection_dim=None,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
        ).to(device)
        print(f"✓ Using MediaPipeGRUCtc model (input_dim={input_dim}, num_ctc_classes={args.num_ctc_classes})")
    
    elif args.model == "iv3_gru":
        model = InceptionV3GRU(
            num_gloss=args.num_gloss,
            num_cat=args.num_cat,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            pretrained_backbone=args.pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
        print("✓ Using InceptionV3GRU model")
    
    elif args.model == "iv3_gru_ctc":
        # InceptionV3GRUCtc uses 2048-D features (precomputed or extracted)
        if use_keypoint_files and not use_feature_files:
            raise ValueError(
                "InceptionV3GRUCtc requires feature data (2048D), not keypoints (178D). "
                "Use --features-train/--features-val with --kp-key='X2048'"
            )
        
        model = InceptionV3GRUCtc(
            num_ctc_classes=args.num_ctc_classes,
            num_cat=args.num_cat if hasattr(args, 'num_cat') else None,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            pretrained_backbone=args.pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
        print(f"✓ Using InceptionV3GRUCtc model (num_ctc_classes={args.num_ctc_classes})")
    
    else:
        raise ValueError(f"Invalid --model {args.model}")
    
    # Enable parallel processing if requested and multiple GPUs available
    if args.enable_parallel:
        model = optimize_model_for_parallel(model, device)
        print("✓ Parallel processing optimization applied")

    # Log model information with comprehensive config
    log_comprehensive_config(args, device, model)

    # Training execution
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    try:
        if args.training_mode == "ctc":
            # ============================================================
            # CTC TRAINING PATH
            # ============================================================
            print(f"Training mode: CTC (Continuous Sign Language Recognition)")
            
            train_ctc(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                blank_id=args.ctc_blank_id,
                epochs=args.epochs,
                output_dir=args.output_dir,
                lr=args.lr,
                weight_decay=args.weight_decay,
                use_amp=args.amp,
                grad_clip=args.grad_clip,
                scheduler_type=args.scheduler,
                scheduler_patience=args.scheduler_patience,
                warmup_epochs=args.warmup_epochs,
                early_stop_patience=args.early_stop,
                resume_path=args.resume,
                log_csv_path=args.log_csv,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                compile_model=args.compile_model,
                use_ema=args.use_ema,
                ema_decay=args.ema_decay,
                alpha=args.alpha if hasattr(args, 'alpha') else 1.0,
                beta=args.beta if hasattr(args, 'beta') else 0.0,
            )
        else:
            # ============================================================
            # CLASSIFICATION TRAINING PATH (existing multi-task)
            # ============================================================
            print(f"Training mode: Classification (Isolated Sign Recognition)")
            
            # Forward adapter per model (unifies calling convention)
            if args.model == "transformer":
                def forward_fn(m, X, lengths=None):
                    # Build attention mask from lengths if provided
                    if lengths is not None:
                        B, T, _ = X.shape
                        device = X.device
                        time_indices = torch.arange(T, device=device).unsqueeze(0)
                        mask = (time_indices < lengths.unsqueeze(1))
                    else:
                        mask = None
                    return m(X, mask=mask)
            
            elif args.model == "mediapipe_gru":
                def forward_fn(m, X, lengths=None):
                    # MediaPipeGRU accepts keypoint sequences directly
                    return m(X, lengths=lengths)
            
            else:  # iv3_gru
                def forward_fn(m, X, lengths=None):
                    return m(X, lengths=lengths, features_already=True)
            
            train_model(
                model,
                train_loader,
                val_loader,
                device,
                forward_fn,
                epochs=args.epochs,
                alpha=args.alpha,
                beta=args.beta,
                output_dir=args.output_dir,
                lr=args.lr,
                weight_decay=args.weight_decay,
                use_amp=args.amp,
                grad_clip=args.grad_clip,
                scheduler_type=args.scheduler,
                scheduler_patience=args.scheduler_patience,
                warmup_epochs=args.warmup_epochs,
                early_stop_patience=args.early_stop,
                resume_path=args.resume,
                log_csv_path=args.log_csv,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                compile_model=args.compile_model,
                curriculum_strategy=args.curriculum,
                curriculum_epochs=args.curriculum_epochs,
                curriculum_warmup=args.curriculum_warmup,
                curriculum_min_weight=args.curriculum_min_weight,
                curriculum_schedule=args.curriculum_schedule,
                loss_weighting_strategy=args.loss_weighting,
                grid_search_weights=args.grid_search_weights,
                uncertainty_init=args.uncertainty_init,
                gradnorm_alpha=args.gradnorm_alpha,
                gradnorm_update_freq=args.gradnorm_update_freq,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                label_smoothing=args.label_smoothing,
                use_ema=args.use_ema,
                ema_decay=args.ema_decay,
            )
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Post-training export hook
    if args.export_mobile:
        try:
            from training.export_mobile import export_model_for_android
            # Determine best checkpoint path
            mapping = {
                'transformer_ctc': 'SignTransformerCtc',
                'mediapipe_gru_ctc': 'MediaPipeGRUCtc',
            }
            stem = mapping.get(args.model)
            if stem is None:
                print(f"Export skipped: --export-mobile supports only transformer_ctc and mediapipe_gru_ctc")
            else:
                best_ckpt = os.path.join(args.output_dir, f"{stem}_best.pt")
                if not os.path.exists(best_ckpt):
                    print(f"Export skipped: best checkpoint not found at {best_ckpt}")
                else:
                    export_model_for_android(
                        model_name=args.model,
                        checkpoint_path=best_ckpt,
                        output_dir=args.export_output,
                        input_dim=178 if args.kp_key != 'X2048' else 2048,
                        num_cat=args.num_cat,
                        window_hint=args.window_hint,
                        stride_hint=args.stride_hint,
                        example_T=args.export_example_T,
                    )
        except Exception as e:
            print(f"Export failed: {e}")
        finally:
            # Restore stdout and close log file if it was opened
            if args.log_file and original_stdout is not None:
                print(f"\n{'='*60}")
                print(f"Training completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Log saved to: {args.log_file}")
                print(f"{'='*60}")
                sys.stdout.close()  # Close the TeeLogger
                sys.stdout = original_stdout  # Restore original stdout
    else:
        # Ensure logger cleanup even when not exporting
        if args.log_file and original_stdout is not None:
            print(f"\n{'='*60}")
            print(f"Training completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log saved to: {args.log_file}")
            print(f"{'='*60}")
            sys.stdout.close()
            sys.stdout = original_stdout