"""
Sign Language Recognition Prediction Script (Batch Processing Version)

This script provides a command-line interface for making predictions on a dataset
of preprocessed NPZ files, with rich metadata and analysis features.

Usage Examples:
    # Run predictions on all samples in a CSV file and save the results
    python predict.py --model transformer \\
        --checkpoint path/to/model.pt \\
        --input-csv path/to/labels.csv \\
        --npz-dir path/to/npz_files/ \\
        --output results.json

    # Run predictions for specific signers
    python predict.py --model transformer \\
        --checkpoint path/to/model.pt \\
        --input-csv path/to/labels.csv \\
        --npz-dir path/to/npz_files/ \\
        --output results_S0_S1.json \\
        --signer-filter S0 S1
"""

# Standard library imports
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np, torch

# Add project root to path for local imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from models import SignTransformer, InceptionV3GRU

class ModelPredictor:
    """
    Unified predictor for both Transformer and IV3-GRU models.
    """
    def __init__(self, model_type, checkpoint_path, device=None):
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.input_dim = self._load_model()
        self._load_checkpoint()

    def _load_model(self):
        # (Omitted for brevity - this function is the same as in the original script)
        if self.model_type == 'transformer':
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
                input_dim = state_dict['embedding.weight'].shape[1] if 'embedding.weight' in state_dict else 156
            except Exception:
                input_dim = 156
            model = SignTransformer(input_dim=input_dim, num_gloss=105, num_cat=10)
        elif self.model_type == 'iv3_gru':
            input_dim = 2048
            model = InceptionV3GRU(num_gloss=105, num_cat=10)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return model.to(self.device), input_dim

    def _load_checkpoint(self):
        # (Omitted for brevity - this function is the same as in the original script)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_from_npz(self, npz_path, metadata):
        """
        Make prediction from a single preprocessed NPZ file with metadata.
        """
        data = np.load(npz_path)
        
        # Determine which key to use based on model type and input dimension
        if self.model_type == 'transformer':
            feature_key = 'X' if self.input_dim == 156 else 'X2048'
            if self.input_dim == 2204: # Combined features
                 X_keypoints = torch.from_numpy(data['X']).float()
                 X_features = torch.from_numpy(data['X2048']).float()
                 X = torch.cat([X_keypoints, X_features], dim=1).unsqueeze(0)
            else:
                 X = torch.from_numpy(data[feature_key]).float().unsqueeze(0)
            
            X = X.to(self.device)
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X)

        elif self.model_type == 'iv3_gru':
            X2048 = torch.from_numpy(data['X2048']).float().unsqueeze(0).to(self.device)
            lengths = torch.tensor([X2048.shape[1]], dtype=torch.long).to(self.device)
            with torch.no_grad():
                gloss_logits, cat_logits = self.model(X2048, lengths, features_already=True)

        # Process results
        gloss_probs = torch.softmax(gloss_logits, dim=-1).squeeze(0)
        cat_probs = torch.softmax(cat_logits, dim=-1).squeeze(0)
        
        gloss_pred = torch.argmax(gloss_probs).item()
        cat_pred = torch.argmax(cat_probs).item()

        return {
            "file": metadata['file'],
            "gloss_pred": gloss_pred,
            "gloss_true": metadata['gloss'],
            "cat_pred": cat_pred,
            "cat_true": metadata['cat'],
            "signer": metadata['signer'],
            "duration": metadata['duration'],
            "confidence_gloss": gloss_probs[gloss_pred].item(),
            "confidence_cat": cat_probs[cat_pred].item(),
            "correct": bool(gloss_pred == metadata['gloss'])
        }

def analyze_predictions(predictions, output_path):
    """
    Calculates per-signer stats and generates a confusion matrix.
    """
    if not predictions:
        print("No predictions to analyze.")
        return

    df = pd.DataFrame(predictions)
    
    # Per-signer statistics
    print("\n--- Per-Signer Accuracy ---")
    signer_stats = df.groupby('signer')['correct'].value_counts(normalize=True).unstack(fill_value=0)
    signer_accuracy = signer_stats.get(True, pd.Series(0.0, index=signer_stats.index))
    for signer, acc in signer_accuracy.items():
        print(f"  {signer}: {acc:.2%}")

    # Overall accuracy
    overall_accuracy = df['correct'].mean()
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")

    # Confusion Matrix
    y_true = df['gloss_true']
    y_pred = df['gloss_pred']
    num_glosses = max(y_true.max(), y_pred.max()) + 1
    cm = confusion_matrix(y_true, y_pred, labels=range(num_glosses))
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='viridis', fmt='d')
    plt.title('Gloss Prediction Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    cm_path = output_path.with_name(output_path.stem + '_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Sign Language Recognition Batch Prediction")
    parser.add_argument('--model', choices=['transformer', 'iv3_gru'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input-csv', type=str, required=True, help='Input CSV file with labels and metadata')
    parser.add_argument('--npz-dir', type=str, required=True, help='Directory containing the NPZ files')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file for results')
    parser.add_argument('--signer-filter', nargs='+', help='Optional: filter by one or more signer IDs (e.g., S0 S1)')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and filter data
    labels_df = pd.read_csv(args.input_csv)
    if args.signer_filter:
        labels_df = labels_df[labels_df['signer'].isin(args.signer_filter)]
        print(f"Filtered to {len(labels_df)} samples for signers: {args.signer_filter}")

    if labels_df.empty:
        print("No data to predict after filtering. Exiting.")
        return 0

    # Initialize predictor
    try:
        predictor = ModelPredictor(args.model, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run predictions
    all_predictions = []
    npz_dir = Path(args.npz_dir)
    
    print(f"Starting predictions for {len(labels_df)} samples...")
    for index, row in labels_df.iterrows():
        npz_file = npz_dir / row['file']
        if not npz_file.exists():
            print(f"Warning: NPZ file not found, skipping: {npz_file}")
            continue
        
        try:
            result = predictor.predict_from_npz(npz_file, row)
            all_predictions.append(result)
        except Exception as e:
            print(f"Error predicting for {npz_file}: {e}")

    # Save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nPredictions saved to: {output_path}")

    # Analyze and report results
    analyze_predictions(all_predictions, output_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())