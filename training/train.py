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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.utils import FSLDataset, evaluate
from models.iv3_gru import IV3_GRU
from models.transformer import SignTransformer

def train_model(model, train_loader, val_loader, device, epochs=20, alpha=0.5, beta=0.5):
    """
    Train a sign language recognition model with multi-task learning.
    
    Args:
        model: Model to train (SignTransformer or IV3_GRU)
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
        for X, gloss, cat in train_loader:
            X, gloss, cat = X.to(device), gloss.to(device), cat.to(device)
            optimizer.zero_grad()
            
            gloss_pred, cat_pred = model(X)
            loss_gloss = criterion(gloss_pred, gloss)
            loss_cat = criterion(cat_pred, cat)
            loss = alpha * loss_gloss + beta * loss_cat

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # Validation
        val_loss, val_gloss_acc, val_cat_acc = evaluate(model, val_loader, criterion, device)
        
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

def load_data():
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
    
    # Dummy data configuration
    n_train_samples = 100
    n_val_samples = 20
    seq_length = 50
    input_dim = 156
    
    # Generate random training data
    train_X = np.random.randn(n_train_samples, seq_length, input_dim).astype(np.float32)
    train_gloss = np.random.randint(0, 105, n_train_samples)
    train_cat = np.random.randint(0, 10, n_train_samples)
    
    # Generate random validation data
    val_X = np.random.randn(n_val_samples, seq_length, input_dim).astype(np.float32)
    val_gloss = np.random.randint(0, 105, n_val_samples)
    val_cat = np.random.randint(0, 10, n_val_samples)
    
    return train_X, train_gloss, train_cat, val_X, val_gloss, val_cat

if __name__ == "__main__":
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
        train_X, train_gloss, train_cat, val_X, val_gloss, val_cat = load_data()
        print(f"✓ Loaded data successfully")
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
    
    train_dataset = FSLDataset(train_X, train_gloss, train_cat)
    val_dataset = FSLDataset(val_X, val_gloss, val_cat)
    
    batch_size = 32
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
    print("1. SignTransformer - Multi-head attention transformer")
    print("2. IV3_GRU - InceptionV3 + GRU hybrid (placeholder)")
    
    model_choice = input("\nSelect model (1 or 2): ").strip()
    
    if model_choice == "1":
        model = SignTransformer().to(device)
        print("✓ Using SignTransformer model")
    elif model_choice == "2":
        try:
            model = IV3_GRU().to(device)
            print("✓ Using IV3_GRU model")
        except:
            print("✗ IV3_GRU model not implemented, defaulting to SignTransformer")
            model = SignTransformer().to(device)
    else:
        print("✗ Invalid choice, defaulting to SignTransformer")
        model = SignTransformer().to(device)

    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  - Model type: {model.__class__.__name__}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024 / 1024:.1f} MB")

    # Training execution
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    train_model(model, train_loader, val_loader, device)
