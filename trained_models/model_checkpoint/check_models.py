#!/usr/bin/env python3
"""Quick script to verify model checkpoint files exist and are accessible."""

from pathlib import Path
import sys

# Model paths from config
MODEL_PATHS = {
    'transformer': 'trained_models/transformer/FSL105_classification/SignTransformer_best.pt',
    'transformer_ctc': 'trained_models/transformer/FSL105_ctc/SignTransformerCtc_best.pt',
    'iv3_gru': 'trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt',
    'iv3_gru_ctc': 'trained_models/iv3_gru/FSL105_ctc/InceptionV3GRUCtc_best.pt',
    'mediapipe_gru': 'trained_models/mediapipe_gru/FSL105_classification/MediaPipeGRU_best.pt',
    'mediapipe_gru_ctc': 'trained_models/mediapipe_gru/FSL105_ctc/MediaPipeGRUCtc_best.pt',
}

def check_models():
    """Check if all model files exist."""
    project_root = Path(__file__).parent
    all_good = True
    
    print("=" * 70)
    print("MODEL CHECKPOINT VERIFICATION")
    print("=" * 70)
    
    for model_name, rel_path in MODEL_PATHS.items():
        full_path = project_root / rel_path
        exists = full_path.exists()
        size = full_path.stat().st_size if exists else 0
        size_mb = size / (1024 * 1024)
        
        status = "✓" if exists else "✗ MISSING"
        print(f"{model_name:20s} {status:10s} {rel_path}")
        if exists:
            print(f"{'':20s} {'':10s} Size: {size_mb:.2f} MB ({full_path.absolute()})")
        else:
            print(f"{'':20s} {'':10s} Path: {full_path.absolute()}")
            all_good = False
    
    print("=" * 70)
    if all_good:
        print("✓ All model checkpoints found and accessible!")
    else:
        print("✗ Some model checkpoints are missing!")
        sys.exit(1)
    print("=" * 70)
    
    return all_good

if __name__ == "__main__":
    check_models()

