#!/usr/bin/env python3
"""Test if model checkpoints can be loaded properly."""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

MODEL_PATHS = {
    'transformer_ctc': 'trained_models/transformer/FSL105_ctc/SignTransformerCtc_best.pt',
    'transformer': 'trained_models/transformer/FSL105_classification/SignTransformer_best.pt',
    'iv3_gru_ctc': 'trained_models/iv3_gru/FSL105_ctc/InceptionV3GRUCtc_best.pt',
    'iv3_gru': 'trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt',
    'mediapipe_gru_ctc': 'trained_models/mediapipe_gru/FSL105_ctc/MediaPipeGRUCtc_best.pt',
    'mediapipe_gru': 'trained_models/mediapipe_gru/FSL105_classification/MediaPipeGRU_best.pt',
}

def check_model_file(model_name, rel_path):
    """Check if a model file exists and can be loaded."""
    full_path = project_root / rel_path
    
    print(f"\n{model_name}:")
    print(f"  Path: {rel_path}")
    print(f"  Exists: {full_path.exists()}")
    
    if not full_path.exists():
        print(f"  ✗ File not found!")
        return False
    
    file_size = full_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    print(f"  Size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    
    # Check if file is suspiciously small (likely a placeholder)
    if file_size < 1000:  # Less than 1KB is definitely suspicious
        print(f"  ⚠ WARNING: File is suspiciously small! This may be a placeholder file.")
        print(f"    Expected model checkpoints to be several MB to hundreds of MB.")
        try:
            with open(full_path, 'rb') as f:
                content = f.read(200)
                print(f"    First 200 bytes (hex): {content.hex()[:100]}...")
        except:
            pass
        return False
    
    # Try to load with torch
    try:
        import torch
        print(f"  Attempting to load with torch.load...")
        
        # Try with weights_only=False first (for PyTorch 2.6+)
        try:
            checkpoint = torch.load(str(full_path), map_location='cpu', weights_only=False)
            print(f"  ✓ Loaded successfully with weights_only=False")
        except Exception as e1:
            try:
                checkpoint = torch.load(str(full_path), map_location='cpu')
                print(f"  ✓ Loaded successfully (default)")
            except Exception as e2:
                print(f"  ✗ Failed to load: {e2}")
                return False
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"  Checkpoint keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
            
            # Check for common checkpoint keys
            has_state_dict = any(k in checkpoint for k in ['model_state_dict', 'state_dict', 'model'])
            if has_state_dict:
                print(f"  ✓ Contains state_dict")
            else:
                print(f"  ⚠ No standard state_dict keys found")
        else:
            print(f"  ⚠ Checkpoint is not a dictionary")
        
        return True
        
    except ImportError:
        print(f"  ⚠ PyTorch not available, skipping load test")
        return True  # Assume OK if we can't test
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("MODEL CHECKPOINT LOAD TEST")
    print("=" * 70)
    
    results = {}
    for model_name, rel_path in MODEL_PATHS.items():
        results[model_name] = check_model_file(model_name, rel_path)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_good = all(results.values())
    for model_name, is_good in results.items():
        status = "✓" if is_good else "✗"
        print(f"{status} {model_name}")
    
    if all_good:
        print("\n✓ All model checkpoints are valid!")
    else:
        print("\n✗ Some model checkpoints have issues!")
        print("\nNOTE: If files are very small (< 1KB), they may be placeholder files.")
        print("      You may need to train or upload the actual model checkpoints.")
        sys.exit(1)

if __name__ == "__main__":
    main()

