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
    total_loss = 0
    correct_gloss = 0
    correct_cat = 0
    total = 0

    with torch.no_grad():
        for X, gloss, cat in dataloader:
            X, gloss, cat = X.to(device), gloss.to(device), cat.to(device)

            gloss_pred, cat_pred = model(X)
            loss_gloss = criterion(gloss_pred, gloss)
            loss_cat = criterion(cat_pred, cat)
            total_loss += (loss_gloss + loss_cat).item()

            # Calculate accuracies
            gloss_preds = gloss_pred.argmax(dim=1)
            cat_preds = cat_pred.argmax(dim=1)
            
            correct_gloss += (gloss_preds == gloss).sum().item()
            correct_cat += (cat_preds == cat).sum().item()
            total += gloss.size(0)

    avg_loss = total_loss / len(dataloader)
    gloss_acc = correct_gloss / total
    cat_acc = correct_cat / total
    
    return avg_loss, gloss_acc, cat_acc
