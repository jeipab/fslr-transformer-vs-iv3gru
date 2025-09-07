"""
utils.py

This module provides essential utilities for training sign language recognition models:

Key Components:
- FSLDataset: PyTorch Dataset class for sign language sequences
- evaluate: Model evaluation with multi-task metrics (gloss + category)

Features:
- Multi-task dataset handling (gloss and category labels)
- Comprehensive model evaluation with accuracy metrics
- Dataset statistics and analysis tools
- Error handling and validation

Usage:
    from training.utils import FSLDataset, evaluate
"""

import torch
from torch.utils.data import Dataset

class FSLDataset(Dataset):
    """
    PyTorch Dataset for Filipino Sign Language sequences.
    
    Handles keypoint sequences with gloss and category labels for multi-task learning.
    """
    
    def __init__(self, sequences, gloss_labels, cat_labels):
        """
        Initialize the FSL dataset.
        
        Args:
            sequences: Input sequences [N, T, 156]
            gloss_labels: Gloss labels [N] with values in [0, 104]
            cat_labels: Category labels [N] with values in [0, 9]
        """
        if len(sequences) != len(gloss_labels) or len(sequences) != len(cat_labels):
            raise ValueError("All input arrays must have the same length")
        
        self.sequences = sequences
        self.gloss_labels = gloss_labels
        self.cat_labels = cat_labels
        
        self.n_samples = len(sequences)
        self.seq_length = sequences.shape[1] if len(sequences.shape) > 1 else 0
        self.n_features = sequences.shape[2] if len(sequences.shape) > 2 else 0
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            tuple: (sequence, gloss_label, cat_label)
        """
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")
        
        return self.sequences[idx], self.gloss_labels[idx], self.cat_labels[idx]
    
    def get_stats(self):
        """Get dataset statistics for analysis."""
        return {
            'n_samples': self.n_samples,
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'n_gloss_classes': len(set(self.gloss_labels)),
            'n_cat_classes': len(set(self.cat_labels)),
            'gloss_distribution': torch.bincount(torch.tensor(self.gloss_labels)),
            'cat_distribution': torch.bincount(torch.tensor(self.cat_labels))
        }


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate a sign language recognition model on a dataset.
    
    Args:
        model: Model to evaluate (SignTransformer or IV3_GRU)
        dataloader: DataLoader containing evaluation data
        criterion: Loss function (typically CrossEntropyLoss)
        device: Device to run evaluation on (CPU/CUDA)
    
    Returns:
        tuple: (avg_loss, gloss_accuracy, cat_accuracy)
    """
    model.eval()
    
    total_loss = 0.0
    correct_gloss = 0
    correct_cat = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for X, gloss, cat in dataloader:
            X, gloss, cat = X.to(device), gloss.to(device), cat.to(device)

            gloss_pred, cat_pred = model(X)
            loss_gloss = criterion(gloss_pred, gloss)
            loss_cat = criterion(cat_pred, cat)
            batch_loss = loss_gloss + loss_cat
            total_loss += batch_loss.item()
            num_batches += 1

            gloss_preds = gloss_pred.argmax(dim=1)
            cat_preds = cat_pred.argmax(dim=1)
            
            correct_gloss += (gloss_preds == gloss).sum().item()
            correct_cat += (cat_preds == cat).sum().item()
            total_samples += gloss.size(0)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    gloss_accuracy = correct_gloss / total_samples if total_samples > 0 else 0.0
    cat_accuracy = correct_cat / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, gloss_accuracy, cat_accuracy
