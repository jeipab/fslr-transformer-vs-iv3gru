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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .utils import FSLDataset
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
    def __init__(self, features_dir, labels_csv, feature_key='X2048'):
        self.features_dir = features_dir
        self.feature_key = feature_key
        self.index = []  # list of (stem, gloss, cat)

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
    def __init__(self, keypoints_dir, labels_csv, kp_key='X'):
        self.keypoints_dir = keypoints_dir
        self.kp_key = kp_key
        self.index = []  # list of (stem, gloss, cat)

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
        data = torch.from_numpy(self._load_npz_keypoints(path))  # [T, 156]
        length = data.shape[0]
        return data.float(), torch.tensor(gloss, dtype=torch.long), torch.tensor(cat, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def _load_npz_keypoints(self, path):
        with np.load(path, allow_pickle=True) as npz:
            if self.kp_key in npz:
                X = np.array(npz[self.kp_key])
            else:
                raise KeyError(f"Key '{self.kp_key}' not found in {path}")
        if X.ndim != 2 or X.shape[-1] != 156:
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
    early_stop_patience=None,
    resume_path=None,
    log_csv_path=None,
    gradient_accumulation_steps=1,
    compile_model=False,
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
        alpha: Weight for gloss loss component.
        beta: Weight for category loss component.
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

    Returns:
        None

    Side effects:
        Saves checkpoints to `output_dir` as `{ModelName}_last.pt` (each epoch)
        and `{ModelName}_best.pt` (best validation metric). Appends metrics to
        `log_csv_path` if provided.
    """
    # Clear GPU memory before training
    clear_gpu_memory()
    
    # Compile model for better performance if supported
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✓ Model compiled for better performance")
        except Exception as e:
            print(f"⚠ Model compilation failed: {e}")
    
    criterion = nn.CrossEntropyLoss()
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
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    else:
        scheduler = None

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
                f"# Training Configuration: model={args.model}, epochs={args.epochs}, batch_size={args.batch_size}",
                f"# Learning Rate: {args.lr}, Weight Decay: {args.weight_decay}, Alpha: {args.alpha}, Beta: {args.beta}",
                f"# Data: gloss_classes={args.num_gloss}, cat_classes={args.num_cat}, seed={args.seed}",
                f"# Performance: amp={args.amp}, compile={args.compile_model}, workers={args.num_workers}",
                f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            for header_line in config_header:
                csv_writer.writerow([header_line])
            csv_writer.writerow([])  # Empty line separator
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_gloss_acc", "val_cat_acc", "lr", "epoch_time", "gpu_memory_allocated", "gpu_memory_reserved"]) 

    print(f"Training for {epochs} epochs...")
    print(f"Loss weights - Gloss: {alpha}, Category: {beta}")
    print("-" * 60)

    epochs_to_run = epochs
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs_to_run):
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
                loss = alpha * loss_gloss + beta * loss_cat
                
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
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
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

        # Clear memory before validation
        clear_gpu_memory()
        
        # Validation
        val_start_time = time.time()
        val_loss, val_gloss_acc, val_cat_acc = evaluate_with_forward(model, val_loader, criterion, device, forward_fn, alpha=alpha, beta=beta)
        val_time = time.time() - val_start_time
        
        avg_train_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results with performance metrics
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
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # CSV log with performance metrics
        if csv_fh is not None:
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0.0
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9 if device.type == 'cuda' else 0.0
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
    parser.add_argument("--scheduler", type=str, default=None, choices=["plateau", "cosine"], help="LR scheduler type")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="Patience for plateau scheduler")
    parser.add_argument("--early-stop", type=int, default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-csv", type=str, default=None, help="Path to CSV log file for metrics")
    # DataLoader performance
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor (worker>0)")
    # Synthetic data controls for smoke tests
    parser.add_argument("--train-samples", type=int, default=100, help="Number of synthetic training samples")
    parser.add_argument("--val-samples", type=int, default=20, help="Number of synthetic validation samples")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length (T)")
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
            args.model == "iv3_gru" and args.features_train is not None and args.features_val is not None
        )
        use_keypoint_files = (
            args.model == "transformer" and args.keypoints_train is not None and args.keypoints_val is not None
        )
        if not (use_feature_files or use_keypoint_files):
            input_dim = 2048 if args.model == "iv3_gru" else 156
            train_X, train_gloss, train_cat, val_X, val_gloss, val_cat = load_data(
                n_train_samples=args.train_samples,
                n_val_samples=args.val_samples,
                seq_length=args.seq_length,
                input_dim=input_dim,
                num_gloss=args.num_gloss,
                num_cat=args.num_cat,
                seed=args.seed,
            )
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
        else:
            print(f"  - Dataset type: Synthetic/Dummy Data")
            print(f"  - Training samples: {len(train_X)}")
            print(f"  - Validation samples: {len(val_X)}")
            print(f"  - Sequence shape: {train_X.shape[1:]} (T, features)")
        
        print(f"  - Gloss classes: {args.num_gloss}")
        print(f"  - Category classes: {args.num_cat}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please implement the load_data() function with actual data loading logic")
        exit(1)

    # Dataset preparation
    print("\n" + "="*60)
    print("DATASET PREPARATION")
    print("="*60)
    
    batch_size = args.batch_size
    use_feature_files = (
        args.model == "iv3_gru" and args.features_train is not None and args.features_val is not None
    )
    use_keypoint_files = (
        args.model == "transformer" and args.keypoints_train is not None and args.keypoints_val is not None
    )

    if use_feature_files:
        train_dataset = FSLFeatureFileDataset(
            features_dir=args.features_train,
            labels_csv=args.labels_train_csv,
            feature_key=args.feature_key,
        )
        val_dataset = FSLFeatureFileDataset(
            features_dir=args.features_val,
            labels_csv=args.labels_val_csv,
            feature_key=args.feature_key,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_features_with_padding)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_features_with_padding)
    elif use_keypoint_files:
        train_dataset = FSLKeypointFileDataset(
            keypoints_dir=args.keypoints_train,
            labels_csv=args.labels_train_csv,
            kp_key=args.kp_key,
        )
        val_dataset = FSLKeypointFileDataset(
            keypoints_dir=args.keypoints_val,
            labels_csv=args.labels_val_csv,
            kp_key=args.kp_key,
        )
        train_loader = _make_dataloader(train_dataset, batch_size, True, args, collate_fn=collate_keypoints_with_padding)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args, collate_fn=collate_keypoints_with_padding)
    else:
        train_dataset = FSLDataset(train_X, train_gloss, train_cat)
        val_dataset = FSLDataset(val_X, val_gloss, val_cat)
        train_loader = _make_dataloader(train_dataset, batch_size, True, args)
        val_loader = _make_dataloader(val_dataset, batch_size, False, args)
    
    print(f"✓ Created datasets and data loaders")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    # Log dataset details
    if use_feature_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 2048] features")
    elif use_keypoint_files:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: [T, 156] keypoints")
    else:
        print(f"  - Training dataset size: {len(train_dataset)} samples")
        print(f"  - Validation dataset size: {len(val_dataset)} samples")
        print(f"  - Data format: Synthetic data")

    # Model selection
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    print("Available models:")
    print("- transformer: Multi-head attention transformer")
    print("- iv3_gru: InceptionV3 + GRU hybrid")
    
    if args.model == "transformer":
        model = SignTransformer(
            num_gloss=args.num_gloss,
            num_cat=args.num_cat,
        ).to(device)
        print("✓ Using SignTransformer model")
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
    
    # Optimize batch size if requested
    if args.auto_batch_size:
        batch_size = calculate_optimal_batch_size(model, device, args.batch_size)
        print(f"✓ Auto-calculated optimal batch size: {batch_size}")
    
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
        early_stop_patience=args.early_stop,
        resume_path=args.resume,
        log_csv_path=args.log_csv,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compile_model=args.compile_model,
    )