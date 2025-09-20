"""
Training entrypoint for sign language recognition.

This module provides:
- Multi-task training (gloss and category classification) with configurable loss weights
- Dataset preparation for file-based features/keypoints or synthetic data (smoke tests)
- Model selection (Transformer or InceptionV3+GRU), evaluation, and checkpointing
- Resume support, optional LR schedulers, AMP, early stopping, and CSV logging

Usage:
    python training/train.py
"""

import os
import csv
import random
import argparse
import time
import psutil
import sys
import platform
from datetime import datetime
from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models import InceptionV3GRU, SignTransformer

class FSLFeatureFileDataset(Dataset):
    """
    Dataset for precomputed visual features with shape [T, 2048] stored as .npz files.

    Expects a labels CSV mapping column 'file' (stem or with extension) to
    'gloss' and 'cat'. The .npz must contain the feature array under
    `feature_key` (default: 'X2048'); if missing, it falls back to 'X'.

    Args:
        features_dir: Directory containing .npz feature files.
        labels_csv: CSV file with columns: file, gloss, cat.
        feature_key: Key inside each .npz for the [T, 2048] array.

    Returns:
        __getitem__ returns (X[T,2048] float32, gloss long, cat long, length long).
    """
    def __init__(self, features_dir, labels_csv, feature_key='X2048', augment=False, augment_params=None):
        self.features_dir = features_dir
        self.feature_key = feature_key
        self.index = []  # list of (stem, gloss, cat)
        self.augment = augment
        self.training = True  # Will be set by DataLoader
        if augment and augment_params:
            self.augmentation = TemporalAugmentation(**augment_params)
        elif augment:
            self.augmentation = TemporalAugmentation()

        if labels_csv is None:
            raise ValueError("labels_csv must be provided for feature dataset")

        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            required = {'file', 'gloss', 'cat'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"labels_csv must have columns: {required}")
            for row in reader:
                try:
                    # accept values with or without extension
                    stem = os.path.splitext(row['file'])[0]
                    gloss = int(row['gloss'])
                    cat = int(row['cat'])
                    self.index.append((stem, gloss, cat))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid data in row {row}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stem, gloss, cat = self.index[idx]
        path = os.path.join(self.features_dir, stem + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        data = torch.from_numpy(self._load_npz_features(path))  # [T, 2048]
        length = data.shape[0]
        
        # Apply augmentation if enabled and in training mode
        if self.augment and self.training and hasattr(self, 'augmentation'):
            data = self.augmentation(data)
        
        return data.float(), torch.tensor(gloss, dtype=torch.long), torch.tensor(cat, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def _load_npz_features(self, path):
        with np.load(path, allow_pickle=True) as npz:
            if self.feature_key in npz:
                X = np.array(npz[self.feature_key])
            elif 'X' in npz:
                X = np.array(npz['X'])
            else:
                raise KeyError(f"Neither '{self.feature_key}' nor 'X' found in {path}")
        if X.ndim != 2 or X.shape[-1] != 2048:
            raise ValueError(f"Expected [T,2048] features in {path}, got shape {X.shape}")
        return X

class FSLMultiModalDataset(Dataset):
    """
    Dataset for multi-modal data combining keypoints [T, 156] and features [T, 2048] from .npz files.
    
    Loads both X (keypoints) and X2048 (features) from the same .npz file and concatenates them
    into a single tensor of shape [T, 2204] (156 + 2048).
    
    Args:
        data_dir: Directory containing .npz files with both X and X2048 keys.
        labels_csv: CSV file with columns: file, gloss, cat.
        augment: Whether to apply temporal augmentation.
        augment_params: Parameters for augmentation.
        
    Returns:
        __getitem__ returns (combined_features[T,2204] float32, gloss long, cat long, length long).
    """
    def __init__(self, data_dir, labels_csv, augment=False, augment_params=None):
        self.data_dir = data_dir
        self.index = []  # list of (stem, gloss, cat)
        self.augment = augment
        self.training = True  # Will be set by DataLoader
        if augment and augment_params:
            self.augmentation = TemporalAugmentation(**augment_params)
        elif augment:
            self.augmentation = TemporalAugmentation()

        if labels_csv is None:
            raise ValueError("labels_csv must be provided for multi-modal dataset")

        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            required = {'file', 'gloss', 'cat'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"labels_csv must have columns: {required}")
            for row in reader:
                try:
                    # accept values with or without extension
                    stem = os.path.splitext(row['file'])[0]
                    gloss = int(row['gloss'])
                    cat = int(row['cat'])
                    self.index.append((stem, gloss, cat))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid data in row {row}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stem, gloss, cat = self.index[idx]
        path = os.path.join(self.data_dir, stem + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Multi-modal file not found: {path}")
        
        # Load both keypoints and features
        keypoints, features = self._load_npz_multimodal(path)
        
        # Concatenate along feature dimension: [T, 156] + [T, 2048] = [T, 2204]
        combined_data = torch.cat([keypoints, features], dim=-1)
        length = combined_data.shape[0]
        
        # Apply augmentation if enabled and in training mode
        if self.augment and self.training and hasattr(self, 'augmentation'):
            combined_data = self.augmentation(combined_data)
        
        return combined_data.float(), torch.tensor(gloss, dtype=torch.long), torch.tensor(cat, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def _load_npz_multimodal(self, path):
        """Load both keypoints (X) and features (X2048) from .npz file."""
        with np.load(path, allow_pickle=True) as npz:
            # Load keypoints (X)
            if 'X' in npz:
                keypoints = np.array(npz['X'])
            else:
                raise KeyError(f"'X' (keypoints) not found in {path}")
            
            # Load features (X2048)
            if 'X2048' in npz:
                features = np.array(npz['X2048'])
            else:
                raise KeyError(f"'X2048' (features) not found in {path}")
        
        # Validate shapes
        if keypoints.ndim != 2 or keypoints.shape[-1] != 156:
            raise ValueError(f"Expected [T,156] keypoints in {path}, got shape {keypoints.shape}")
        if features.ndim != 2 or features.shape[-1] != 2048:
            raise ValueError(f"Expected [T,2048] features in {path}, got shape {features.shape}")
        
        # Ensure temporal alignment
        if keypoints.shape[0] != features.shape[0]:
            raise ValueError(f"Temporal mismatch in {path}: keypoints {keypoints.shape[0]} vs features {features.shape[0]}")
        
        return torch.from_numpy(keypoints), torch.from_numpy(features)

class FSLKeypointFileDataset(Dataset):
    """
    Dataset for precomputed keypoint sequences with shape [T, 156] stored as .npz.

    Expects a labels CSV mapping column 'file' (stem or with extension) to
    'gloss' and 'cat'. The .npz must contain the key specified by `kp_key`
    (default: 'X').

    Args:
        keypoints_dir: Directory containing .npz keypoint files.
        labels_csv: CSV with columns: file, gloss, cat.
        kp_key: Key inside each .npz for the [T, 156] array.

    Returns:
        __getitem__ returns (X[T,156] float32, gloss long, cat long, length long).
    """
    def __init__(self, keypoints_dir, labels_csv, kp_key='X', augment=False, augment_params=None):
        self.keypoints_dir = keypoints_dir
        self.kp_key = kp_key
        self.index = []  # list of (stem, gloss, cat)
        self.augment = augment
        self.training = True  # Will be set by DataLoader
        if augment and augment_params:
            self.augmentation = TemporalAugmentation(**augment_params)
        elif augment:
            self.augmentation = TemporalAugmentation()

        if labels_csv is None:
            raise ValueError("labels_csv must be provided for keypoint dataset")

        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            required = {'file', 'gloss', 'cat'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"labels_csv must have columns: {required}")
            for row in reader:
                try:
                    stem = os.path.splitext(row['file'])[0]
                    gloss = int(row['gloss'])
                    cat = int(row['cat'])
                    self.index.append((stem, gloss, cat))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid data in row {row}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stem, gloss, cat = self.index[idx]
        path = os.path.join(self.keypoints_dir, stem + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Keypoint file not found: {path}")
        data = torch.from_numpy(self._load_npz_keypoints(path))  # [T, 156] or [T, 2048]
        length = data.shape[0]
        
        # Apply augmentation if enabled and in training mode
        if self.augment and self.training and hasattr(self, 'augmentation'):
            data = self.augmentation(data)
        
        return data.float(), torch.tensor(gloss, dtype=torch.long), torch.tensor(cat, dtype=torch.long), torch.tensor(length, dtype=torch.long)

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
        elif self.kp_key == "X" and X.shape[-1] != 156:
            raise ValueError(f"Expected [T,156] keypoints in {path}, got shape {X.shape}")
        return X

def collate_features_with_padding(batch):
    """
    Pad variable-length feature sequences [T, 2048] to the max length in batch.

    Args:
        batch: Iterable of (X[T,2048], gloss, cat, length) items.

    Returns:
        tuple: (X_pad [B,Tmax,2048], gloss [B], cat [B], lengths [B])
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

def collate_keypoints_with_padding(batch):
    """
    Pad variable-length keypoint sequences [T, 156] to the max length in batch.

    Args:
        batch: Iterable of (X[T,156], gloss, cat, length) items.

    Returns:
        tuple: (X_pad [B,Tmax,156], gloss [B], cat [B], lengths [B])
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

def _make_dataloader(dataset, batch_size, shuffle, args, collate_fn=None):
    """
    Internal helper to build an optimized DataLoader with performance enhancements.
    """
    # Auto-detect optimal number of workers if not specified
    num_workers = args.num_workers
    if args.auto_workers or num_workers == 0:
        # Use more aggressive worker count for better performance
        cpu_count = psutil.cpu_count(logical=False)
        # Use more workers but cap at reasonable limit
        num_workers = min(8, max(2, cpu_count // 2))
        if args.auto_workers:
            print(f"Auto-detected {num_workers} DataLoader workers (from {cpu_count} CPU cores)")
    
    # Optimize pin_memory based on device
    pin_memory = args.pin_memory
    if not hasattr(args, 'pin_memory') or args.pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,  # Keep workers alive between epochs
    }
    
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
    
    # Set prefetch_factor for better data loading performance
    if num_workers > 0:
        prefetch_factor = getattr(args, 'prefetch_factor', None)
        if prefetch_factor is None:
            # Auto-set prefetch factor based on available memory
            if torch.cuda.is_available():
                kwargs['prefetch_factor'] = 2  # Conservative for GPU
            else:
                kwargs['prefetch_factor'] = 4  # More aggressive for CPU
        elif isinstance(prefetch_factor, int) and prefetch_factor > 0:
            kwargs['prefetch_factor'] = prefetch_factor
    
    return DataLoader(dataset, **kwargs)

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
    """Get the optimal device for training with comprehensive CUDA optimization."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Set memory allocation strategy for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def print_device_info(device: torch.device) -> None:
    """Print comprehensive device information."""
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA memory: {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA compute capability: {props.major}.{props.minor}")
        print(f"CUDA multiprocessors: {props.multi_processor_count}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    elif device.type == 'mps':
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
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
    """Calculate optimal batch size based on available memory."""
    if device.type == 'cuda':
        # Get GPU memory info
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Adjust batch size based on GPU memory
        if gpu_memory_gb > 16:
            optimal_batch_size = base_batch_size * 2  # 64
        elif gpu_memory_gb > 8:
            optimal_batch_size = base_batch_size      # 32
        elif gpu_memory_gb > 4:
            optimal_batch_size = base_batch_size // 2 # 16
        else:
            optimal_batch_size = base_batch_size // 4 # 8
        
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB, Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    else:
        # CPU training - use smaller batches
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
        elif args.model == "transformer":
            print(f"  - Training folder: {args.keypoints_train}")
            print(f"  - Validation folder: {args.keypoints_val}")
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
    # Clear GPU memory before training
    clear_gpu_memory()
    
    # Initialize curriculum scheduler if strategy is provided
    curriculum_scheduler = None
    if curriculum_strategy is not None:
        curriculum_scheduler = CurriculumScheduler(
            strategy=curriculum_strategy,
            curriculum_epochs=curriculum_epochs,
            warmup_epochs=curriculum_warmup,
            min_weight=curriculum_min_weight,
            schedule_type=curriculum_schedule
        )
        print(f"✓ Curriculum training enabled: {curriculum_strategy}")
        print(f"  - Curriculum epochs: {curriculum_epochs}")
        if curriculum_strategy == "dynamic":
            print(f"  - Warmup epochs: {curriculum_warmup}")
        print(f"  - Min weight: {curriculum_min_weight}")
        print(f"  - Schedule: {curriculum_schedule}")
    
    # Initialize loss weighting strategy
    loss_weighting = None
    if loss_weighting_strategy == "grid-search":
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
        loss_weighting = create_loss_weighting_strategy(
            loss_weighting_strategy,
            alpha=alpha,
            beta=beta,
            uncertainty_init=uncertainty_init,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_update_freq=gradnorm_update_freq,
            device=str(device)
        )
        print(f"✓ Loss weighting strategy: {loss_weighting_strategy}")
        if loss_weighting_strategy == "uncertainty":
            print(f"  - Initial uncertainty: {uncertainty_init}")
        elif loss_weighting_strategy == "gradnorm":
            print(f"  - Alpha: {gradnorm_alpha}")
            print(f"  - Update frequency: {gradnorm_update_freq}")
    
    # Compile model for better performance if supported
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✓ Model compiled for better performance")
        except Exception as e:
            print(f"⚠ Model compilation failed: {e}")
    
    # Initialize loss function based on type
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"✓ Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    elif loss_type == "label_smoothing":
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        print(f"✓ Using Label Smoothing CrossEntropy (smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("✓ Using standard CrossEntropy Loss")
    
    # Add uncertainty parameters to optimizer if using uncertainty weighting
    if loss_weighting_strategy == "uncertainty" and isinstance(loss_weighting, UncertaintyWeighting):
        optimizer = optim.Adam(
            list(model.parameters()) + [loss_weighting.log_var_gloss, loss_weighting.log_var_cat],
            lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Enable AMP only when running on CUDA to avoid CPU-only issues
    amp_enabled = bool(use_amp and getattr(device, "type", "cpu") == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    
    # Print training configuration
    print(f"Training Configuration:")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  - AMP enabled: {amp_enabled}")
    print(f"  - Model compiled: {compile_model}")
    if device.type == 'cuda':
        print(f"  - CUDA memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience)
        print(f"✓ Using ReduceLROnPlateau scheduler (patience={scheduler_patience})")
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        print(f"✓ Using CosineAnnealingLR scheduler")
    elif scheduler_type == "warmup_cosine":
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, lr)
        print(f"✓ Using WarmupCosineScheduler (warmup_epochs={warmup_epochs})")
    else:
        scheduler = None
        print("✓ No learning rate scheduler")

    # Initialize EMA if requested
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        ema.register()
        print(f"✓ EMA enabled (decay={ema_decay})")

    # Resume support
    start_epoch = 0
    best_metric = -float('inf')
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler'])
        if 'scheduler' in ckpt and scheduler is not None and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        best_metric = ckpt.get('best_metric', best_metric)
        print(f"Resumed from {resume_path} at epoch {start_epoch} (best_metric={best_metric:.4f})")

    # CSV logging
    csv_fh = None
    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path) or '.', exist_ok=True)
        new_file = not os.path.exists(log_csv_path)
        csv_fh = open(log_csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_fh)
        if new_file:
            # Write configuration header as comment
            config_header = [
                f"# Training Configuration: epochs={epochs}, batch_size={batch_size}",
                f"# Learning Rate: {lr}, Weight Decay: {weight_decay}, Alpha: {alpha}, Beta: {beta}",
                f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            for header_line in config_header:
                csv_writer.writerow([header_line])
            csv_writer.writerow([])  # Empty line separator
            if curriculum_scheduler is not None:
                csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved", "alpha", "beta", "curriculum_phase"])
            elif loss_weighting is not None and loss_weighting_strategy != "static":
                csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved", "alpha", "beta", "loss_weighting_strategy"])
            else:
                csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved"]) 

    print(f"Training for {epochs} epochs...")
    if curriculum_scheduler is not None:
        print(f"Curriculum training: {curriculum_scheduler.get_phase_info(0)}")
    else:
        print(f"Loss weights - Gloss: {alpha}, Category: {beta}")
    print("-" * 60)

    epochs_to_run = epochs
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs_to_run):
        # Get current curriculum weights
        if curriculum_scheduler is not None:
            current_alpha, current_beta = curriculum_scheduler.get_weights(epoch, epochs)
            phase_info = curriculum_scheduler.get_phase_info(epoch)
        else:
            current_alpha, current_beta = alpha, beta
            phase_info = "Balanced training"
        
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        # Training phase with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                X, gloss, cat, lengths = batch
                lengths = lengths.to(device, non_blocking=True)
            else:
                X, gloss, cat = batch
                lengths = None
            
            # Move tensors to device with non_blocking for better performance
            X = X.to(device, non_blocking=True)
            gloss = gloss.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=getattr(device, "type", "cpu"), enabled=amp_enabled):
                gloss_pred, cat_pred = forward_fn(model, X, lengths)
                loss_gloss = criterion(gloss_pred, gloss)
                loss_cat = criterion(cat_pred, cat)
                
                # Get dynamic weights from loss weighting strategy
                if loss_weighting is not None:
                    dynamic_alpha, dynamic_beta = loss_weighting.get_weights(
                        epoch, batch_idx, loss_gloss.item(), loss_cat.item(), model, optimizer
                    )
                    # Use curriculum weights if available, otherwise use loss weighting weights
                    if curriculum_scheduler is not None:
                        # Curriculum takes precedence over loss weighting
                        loss = current_alpha * loss_gloss + current_beta * loss_cat
                    else:
                        # Use loss weighting strategy
                        if loss_weighting_strategy == "uncertainty":
                            loss = loss_weighting.get_uncertainty_loss(loss_gloss, loss_cat)
                        else:
                            loss = dynamic_alpha * loss_gloss + dynamic_beta * loss_cat
                else:
                    # Use curriculum weights or static weights
                    loss = current_alpha * loss_gloss + current_beta * loss_cat
                
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        
        # Handle remaining gradients if last batch doesn't align with accumulation steps
        if len(train_loader) % gradient_accumulation_steps != 0:
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update EMA if enabled
        if ema is not None:
            ema.update()
        
        # Clear intermediate variables to save memory
        del X, gloss, cat, gloss_pred, cat_pred, loss, loss_gloss, loss_cat
        if lengths is not None:
            del lengths

        # Handle case where training dataloader yields zero batches
        if num_batches == 0:
            print("No training batches were provided. Check your dataset and DataLoader settings.")
            if csv_fh is not None:
                csv_fh.close()
            return

        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        
        # Update loss weighting strategy if needed
        if loss_weighting is not None and hasattr(loss_weighting, 'update_weights'):
            loss_weighting.update_weights(
                epoch, 
                {'gloss': torch.tensor(avg_train_loss), 'cat': torch.tensor(avg_train_loss)}, 
                model, 
                optimizer
            )
        
        # Clear memory before validation
        clear_gpu_memory()
        
        # Apply EMA for validation if enabled
        if ema is not None:
            ema.apply_shadow()
        
        # Validation
        val_start_time = time.time()
        val_loss, val_gloss_acc, val_cat_acc = evaluate_with_forward(model, val_loader, criterion, device, forward_fn, alpha=current_alpha, beta=current_beta)
        
        # Restore original parameters after validation
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

def load_data(n_train_samples=100, n_val_samples=20, seq_length=50, input_dim=156, num_gloss=105, num_cat=10, seed=42):
    """
    Load training and validation data for sign language recognition.
    
    Placeholder function that generates dummy data for testing.
    Replace with actual data loading from preprocessed .npz files.
    
    Returns:
        tuple: (train_X, train_gloss, train_cat, val_X, val_gloss, val_cat)
            - train_X: Training sequences [N_train, T, 156]
            - train_gloss: Training gloss labels [N_train]
            - train_cat: Training category labels [N_train]
            - val_X: Validation sequences [N_val, T, 156]
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
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition model (smoke-test ready)")
    parser.add_argument("--model", choices=["transformer", "iv3_gru"], default="transformer", help="Model to train")
    parser.add_argument("--multimodal", action="store_true", help="Use multi-modal input (keypoints + features)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for gloss loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for category loss")
    # Class counts
    parser.add_argument("--num-gloss", type=int, default=105, help="Number of gloss classes")
    parser.add_argument("--num-cat", type=int, default=10, help="Number of category classes")
    # IV3-GRU feature dataset options
    parser.add_argument("--features-train", type=str, default=None, help="Directory of training .npz 2048-d features")
    parser.add_argument("--features-val", type=str, default=None, help="Directory of validation .npz 2048-d features")
    parser.add_argument("--labels-train-csv", type=str, default=None, help="CSV with columns: file,gloss,cat for training")
    parser.add_argument("--labels-val-csv", type=str, default=None, help="CSV with columns: file,gloss,cat for validation")
    parser.add_argument("--feature-key", type=str, default="X2048", help="Key in .npz containing [T,2048] features")
    # Transformer keypoint dataset options
    parser.add_argument("--keypoints-train", type=str, default=None, help="Directory of training .npz keypoints [T,156]")
    parser.add_argument("--keypoints-val", type=str, default=None, help="Directory of validation .npz keypoints [T,156]")
    parser.add_argument("--kp-key", type=str, default="X", help="Key in .npz containing [T,156] keypoints")
    # Multi-modal dataset options
    parser.add_argument("--multimodal-train", type=str, default=None, help="Directory of training .npz files with both X and X2048")
    parser.add_argument("--multimodal-val", type=str, default=None, help="Directory of validation .npz files with both X and X2048")
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
    parser.add_argument("--log-csv", type=str, default=None, help="Path to CSV log file for metrics")
    # DataLoader performance
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor (worker>0)")
    # Sequence length for synthetic data (kept for compatibility)
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length (T) - for synthetic data only")
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    
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
            # Transformer smoke (uses existing forward contract on [B, T, 156])
            B = args.smoke_batch_size
            T = args.smoke_T
            X = torch.randn(B, T, 156, dtype=torch.float32, device=device)
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

    # Data loading
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    
    try:
        # If dataset directories are provided, use file-based datasets; otherwise synthetic
        use_feature_files = (
            args.model == "iv3_gru" and args.features_train is not None and args.features_val is not None and not args.multimodal
        )
        use_keypoint_files = (
            args.model == "transformer" and args.keypoints_train is not None and args.keypoints_val is not None and not args.multimodal
        )
        use_multimodal_files = (
            args.multimodal and args.multimodal_train is not None and args.multimodal_val is not None
        )
        if not (use_feature_files or use_keypoint_files or use_multimodal_files):
            raise ValueError("No data files provided. Please specify either --features-train/--features-val for IV3-GRU model, --keypoints-train/--keypoints-val for Transformer model, or --multimodal-train/--multimodal-val for multi-modal training.")
        print(f"✓ Loaded data successfully")
        
        # Log dataset source information
        if use_feature_files:
            print(f"  - Dataset type: IV3-GRU Features")
            print(f"  - Training folder: {args.features_train}")
            print(f"  - Validation folder: {args.features_val}")
            print(f"  - Feature key: {args.feature_key}")
        elif use_keypoint_files:
            print(f"  - Dataset type: Transformer Keypoints")
            print(f"  - Training folder: {args.keypoints_train}")
            print(f"  - Validation folder: {args.keypoints_val}")
            print(f"  - Keypoint key: {args.kp_key}")
        elif use_multimodal_files:
            print(f"  - Dataset type: Multi-Modal (Keypoints + Features)")
            print(f"  - Training folder: {args.multimodal_train}")
            print(f"  - Validation folder: {args.multimodal_val}")
            print(f"  - Model: {args.model}")
        
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

    if use_feature_files:
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
        )
        val_dataset = FSLFeatureFileDataset(
            features_dir=args.features_val,
            labels_csv=args.labels_val_csv,
            feature_key=args.feature_key,
            augment=False,  # No augmentation for validation
            augment_params=None,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_features_with_padding)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_features_with_padding)
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
        )
        val_dataset = FSLKeypointFileDataset(
            keypoints_dir=args.keypoints_val,
            labels_csv=args.labels_val_csv,
            kp_key=args.kp_key,
            augment=False,  # No augmentation for validation
            augment_params=None,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_keypoints_with_padding)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_keypoints_with_padding)
    elif use_multimodal_files:
        # Validate CSV files exist
        if args.labels_train_csv is None or not os.path.exists(args.labels_train_csv):
            raise FileNotFoundError(f"Training labels CSV not found: {args.labels_train_csv}")
        if args.labels_val_csv is None or not os.path.exists(args.labels_val_csv):
            raise FileNotFoundError(f"Validation labels CSV not found: {args.labels_val_csv}")
        
        train_dataset = FSLMultiModalDataset(
            data_dir=args.multimodal_train,
            labels_csv=args.labels_train_csv,
            augment=args.augment,
            augment_params=augment_params,
        )
        val_dataset = FSLMultiModalDataset(
            data_dir=args.multimodal_val,
            labels_csv=args.labels_val_csv,
            augment=False,  # No augmentation for validation
            augment_params=None,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_keypoints_with_padding)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_keypoints_with_padding)
    
    print(f"✓ Created datasets and data loaders")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    if args.augment:
        print(f"  - Data augmentation: Enabled (noise_std={args.augment_noise_std}, mask_prob={args.augment_mask_prob})")
    
    # Log dataset details
    if use_feature_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 2048] features")
    elif use_keypoint_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 156] keypoints")
    elif use_multimodal_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 2204] multi-modal (156 keypoints + 2048 features)")

    # Model selection
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    print("Available models:")
    print("- transformer: Multi-head attention transformer")
    print("- iv3_gru: InceptionV3 + GRU hybrid")
    
    if args.model == "transformer":
        # Determine input dimension based on data type
        if use_multimodal_files:
            input_dim = 2204  # Multi-modal: 156 keypoints + 2048 features
        elif use_feature_files:
            input_dim = 2048  # Features
        elif args.kp_key == "X2048":
            input_dim = 2048
        else:
            input_dim = 156  # Default for keypoints
        
        model = SignTransformer(
            input_dim=input_dim,
            num_gloss=args.num_gloss,
            num_cat=args.num_cat,
        ).to(device)
        print(f"✓ Using SignTransformer model (input_dim={input_dim})")
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
    else:
        raise ValueError(f"Invalid --model {args.model}")
    
    # Enable parallel processing if requested and multiple GPUs available
    if args.enable_parallel:
        model = optimize_model_for_parallel(model, device)
        print("✓ Parallel processing optimization applied")

    # Log model information with comprehensive config
    log_comprehensive_config(args, device, model)

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
    else:
        def forward_fn(m, X, lengths=None):
            if use_multimodal_files:
                return m(X, lengths=lengths, multimodal=True)
            else:
                return m(X, lengths=lengths, features_already=True)

    # Training execution
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
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