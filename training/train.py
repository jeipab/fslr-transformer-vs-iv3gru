import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.utils import FSLDataset, evaluate
from models.iv3_gru import IV3_GRU
from models.transformer import SignTransformer

def train_model(model, train_loader, val_loader, device, epochs=20, alpha=0.5, beta=0.5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

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
        
        val_loss, val_gloss_acc, val_cat_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Gloss Acc: {val_gloss_acc:.2f}, Val Cat Acc: {val_cat_acc:.2f}")

    # Save trained model
    torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")

def load_data():
    """
    Placeholder function for data loading.
    Replace this with actual data loading logic.
    """
    # TODO: Implement actual data loading from preprocessed .npz files
    # For now, return dummy data for testing
    import numpy as np
    
    # Dummy data - replace with real data loading
    n_train_samples = 100
    n_val_samples = 20
    seq_length = 50
    input_dim = 156
    
    train_X = np.random.randn(n_train_samples, seq_length, input_dim).astype(np.float32)
    train_gloss = np.random.randint(0, 105, n_train_samples)
    train_cat = np.random.randint(0, 10, n_train_samples)
    
    val_X = np.random.randn(n_val_samples, seq_length, input_dim).astype(np.float32)
    val_gloss = np.random.randint(0, 105, n_val_samples)
    val_cat = np.random.randint(0, 10, n_val_samples)
    
    return train_X, train_gloss, train_cat, val_X, val_gloss, val_cat

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    try:
        train_X, train_gloss, train_cat, val_X, val_gloss, val_cat = load_data()
        print(f"Loaded data - Train: {len(train_X)} samples, Val: {len(val_X)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please implement the load_data() function with actual data loading logic")
        exit(1)

    # Create datasets
    train_dataset = FSLDataset(train_X, train_gloss, train_cat)
    val_dataset = FSLDataset(val_X, val_gloss, val_cat)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Choose model
    print("Available models:")
    print("1. SignTransformer")
    
    model_choice = input("Select model (1): ").strip()
    
    if model_choice == "1":
        model = SignTransformer().to(device)
        print("Using SignTransformer model")
    else:
        print("Invalid choice, defaulting to SignTransformer")
        model = SignTransformer().to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_model(model, train_loader, val_loader, device)
