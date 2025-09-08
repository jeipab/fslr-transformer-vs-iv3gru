"""
train.py

Handles data loading, model training, validation, and model saving.
Supports multi-task learning (gloss + category classification).

Key Features:
- Multi-task learning (gloss + category classification)
- Configurable loss weighting (alpha, beta parameters)
- Comprehensive evaluation metrics
- Model selection interface
- Automatic device detection (CUDA/CPU)

Usage:
    python training/train.py
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from training.utils import FSLDataset
from models.iv3_gru import InceptionV3GRU
from models.transformer import SignTransformer
import argparse

class FSLFeatureFileDataset(Dataset):
    """
    Dataset for precomputed visual features [T, 2048] stored in .npz files.
    Expects a labels CSV mapping filename (column 'file', without extension also accepted) to 'gloss' and 'cat'.

    .npz requirements:
      - Feature array under key specified by feature_key (default: 'X2048').
        If not present, falls back to 'X' and verifies last dim == 2048.
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
                # accept values with or without extension
                stem = os.path.splitext(row['file'])[0]
                gloss = int(row['gloss'])
                cat = int(row['cat'])
                self.index.append((stem, gloss, cat))

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
        import numpy as np
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

def collate_features_with_padding(batch):
    """
    Pad variable-length [T, 2048] sequences to max T in batch. Returns:
      X_pad [B, Tmax, 2048], gloss [B], cat [B], lengths [B]
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

def train_model(model, train_loader, val_loader, device, forward_adapter, epochs=20, alpha=0.5, beta=0.5):
    """
    Train a sign language recognition model with multi-task learning.
    
    Args:
        model: Model to train (SignTransformer or InceptionV3GRU)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on (CPU/CUDA)
        epochs: Number of training epochs (default: 20)
        alpha: Weight for gloss loss (default: 0.5)
        beta: Weight for category loss (default: 0.5)
    
    Returns:
        None: Model saved automatically as {ModelName}.pt
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training for {epochs} epochs...")
    print(f"Loss weights - Gloss: {alpha}, Category: {beta}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Training phase
        for batch in train_loader:
            if len(batch) == 4:
                X, gloss, cat, lengths = batch
                lengths = lengths.to(device)
            else:
                X, gloss, cat = batch
                lengths = None
            X, gloss, cat = X.to(device), gloss.to(device), cat.to(device)
            optimizer.zero_grad()
            
            outputs = forward_adapter(model, X, lengths)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                gloss_pred, cat_pred = outputs
                loss_gloss = criterion(gloss_pred, gloss)
                loss_cat = criterion(cat_pred, cat)
                loss = alpha * loss_gloss + beta * loss_cat
            else:
                gloss_pred = outputs
                loss_gloss = criterion(gloss_pred, gloss)
                loss = loss_gloss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # Validation
        val_loss, val_gloss_acc, val_cat_acc = evaluate_with_adapter(model, val_loader, criterion, device, forward_adapter)
        
        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Gloss Acc: {val_gloss_acc:.3f} | "
              f"Val Cat Acc: {val_cat_acc:.3f}")

    print("-" * 60)
    print("Training completed!")
    
    model_filename = f"{model.__class__.__name__}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as: {model_filename}")

def evaluate_with_adapter(model, dataloader, criterion, device, forward_adapter):
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
            outputs = forward_adapter(model, X, lengths)

            if isinstance(outputs, tuple) and len(outputs) == 2:
                gloss_pred, cat_pred = outputs
                loss_gloss = criterion(gloss_pred, gloss)
                loss_cat = criterion(cat_pred, cat)
                batch_loss = loss_gloss + loss_cat
                cat_preds = cat_pred.argmax(dim=1)
                correct_cat += (cat_preds == cat).sum().item()
            else:
                gloss_pred = outputs
                batch_loss = criterion(gloss_pred, gloss)

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
    import numpy as np
    
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
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition model (smoke-test ready)")
    parser.add_argument("--model", choices=["transformer", "iv3_gru"], default="transformer", help="Model to train")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for gloss loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for category loss")
    # IV3-GRU feature dataset options
    parser.add_argument("--features-train", type=str, default=None, help="Directory of training .npz 2048-d features")
    parser.add_argument("--features-val", type=str, default=None, help="Directory of validation .npz 2048-d features")
    parser.add_argument("--labels-train-csv", type=str, default=None, help="CSV with columns: file,gloss,cat for training")
    parser.add_argument("--labels-val-csv", type=str, default=None, help="CSV with columns: file,gloss,cat for validation")
    parser.add_argument("--feature-key", type=str, default="X2048", help="Key in .npz containing [T,2048] features")
    # Synthetic data controls for smoke tests
    parser.add_argument("--train-samples", type=int, default=100, help="Number of synthetic training samples")
    parser.add_argument("--val-samples", type=int, default=20, help="Number of synthetic validation samples")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length (T)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic data")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data loading
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    
    try:
        # If feature directories are provided for iv3_gru, use file-based dataset; otherwise synthetic
        use_feature_files = (
            args.model == "iv3_gru" and args.features_train is not None and args.features_val is not None
        )
        if not use_feature_files:
            input_dim = 2048 if args.model == "iv3_gru" else 156
            train_X, train_gloss, train_cat, val_X, val_gloss, val_cat = load_data(
                n_train_samples=args.train_samples,
                n_val_samples=args.val_samples,
                seq_length=args.seq_length,
                input_dim=input_dim,
                num_gloss=105,
                num_cat=10,
                seed=args.seed,
            )
        print(f"✓ Loaded data successfully")
        if not use_feature_files:
            print(f"  - Training samples: {len(train_X)}")
            print(f"  - Validation samples: {len(val_X)}")
            print(f"  - Sequence shape: {train_X.shape[1:]} (T, features)")
        print(f"  - Gloss classes: {len(set(train_gloss))}")
        print(f"  - Category classes: {len(set(train_cat))}")
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_features_with_padding)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_features_with_padding)
    else:
        train_dataset = FSLDataset(train_X, train_gloss, train_cat)
        val_dataset = FSLDataset(val_X, val_gloss, val_cat)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Created datasets and data loaders")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")

    # Model selection
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    print("Available models:")
    print("- transformer: Multi-head attention transformer")
    print("- iv3_gru: InceptionV3 + GRU hybrid")
    
    if args.model == "transformer":
        model = SignTransformer().to(device)
        print("✓ Using SignTransformer model")
    elif args.model == "iv3_gru":
        model = InceptionV3GRU(
            num_classes=105,
            hidden1=16,
            hidden2=12,
            dropout=0.3,
            pretrained_backbone=True,
            freeze_backbone=True,
        ).to(device)
        print("✓ Using InceptionV3GRU model")
    else:
        raise ValueError(f"Invalid --model {args.model}")

    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  - Model type: {model.__class__.__name__}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024 / 1024:.1f} MB")

    # Forward adapter per model (unifies calling convention)
    if args.model == "transformer":
        def forward_adapter(m, X, lengths=None):
            return m(X)
    else:
        def forward_adapter(m, X, lengths=None):
            return m(X, lengths=lengths, features_already=True)

    # Training execution
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    train_model(model, train_loader, val_loader, device, forward_adapter, epochs=args.epochs, alpha=args.alpha, beta=args.beta)
