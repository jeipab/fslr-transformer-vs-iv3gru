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
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}")

    # Save trained model
    torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (replace with real data loading)
    train_dataset = FSLDataset(train_X, train_gloss, train_cat)
    val_dataset = FSLDataset(val_X, val_gloss, val_cat)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Choose model
    model = IV3_GRU().to(device)
    # model = SignTransformer().to(device)

    train_model(model, train_loader, val_loader, device)
