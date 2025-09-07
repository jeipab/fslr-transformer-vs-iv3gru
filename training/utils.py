import torch
from torch.utils.data import Dataset

# Dataset class
class FSLDataset(Dataset):
    def __init__(self, sequences, gloss_labels, cat_labels):
        self.sequences = sequences
        self.gloss_labels = gloss_labels
        self.cat_labels = cat_labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.gloss_labels[idx], self.cat_labels[idx]

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct_gloss, total = 0, 0, 0

    with torch.no_grad():
        for X, gloss, cat in dataloader:
            X, gloss, cat = X.to(device), gloss.to(device), cat.to(device)

            gloss_pred, cat_pred = model(X)
            loss = criterion(gloss_pred, gloss) + criterion(cat_pred, cat)
            total_loss += loss.item()

            preds = gloss_pred.argmax(dim=1)
            correct_gloss += (preds == gloss).sum().item()
            total += gloss.size(0)

    return total_loss / len(dataloader), correct_gloss / total
