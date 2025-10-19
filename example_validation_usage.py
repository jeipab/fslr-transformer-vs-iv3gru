#!/usr/bin/env python3
"""
Example usage of the updated validation system.

This script demonstrates how to use the new features:
- 6-column CSV format
- Signer-aware validation
- Duration analysis
- Per-category metrics
- Filtering capabilities
"""

import subprocess
import sys
from pathlib import Path

def run_validation_example():
    """Run a complete validation example with the new features."""
    
    print("=" * 60)
    print("VALIDATION SYSTEM USAGE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Basic validation with new format
    print("\n1. Basic validation with 6-column CSV format:")
    print("   python evaluation/validation/validate.py \\")
    print("       --model transformer \\")
    print("       --checkpoint trained_models/transformer/optimal/model.pt \\")
    print("       --data-dir data/processed/fsl_val \\")
    print("       --labels-csv data/processed/fsl_val.csv \\")
    print("       --output-dir results/validation_basic")
    
    # Example 2: Signer-specific validation
    print("\n2. Validate specific signer (S1):")
    print("   python evaluation/validation/validate.py \\")
    print("       --model transformer \\")
    print("       --checkpoint trained_models/transformer/optimal/model.pt \\")
    print("       --signer-filter S1 \\")
    print("       --output-dir results/validation_s1")
    
    # Example 3: Category-specific validation
    print("\n3. Validate specific categories (greeting and survival):")
    print("   python evaluation/validation/validate.py \\")
    print("       --model transformer \\")
    print("       --checkpoint trained_models/transformer/optimal/model.pt \\")
    print("       --category-filter 0 1 \\")
    print("       --output-dir results/validation_categories")
    
    # Example 4: Combined filtering
    print("\n4. Combined signer and category filtering:")
    print("   python evaluation/validation/validate.py \\")
    print("       --model transformer \\")
    print("       --checkpoint trained_models/transformer/optimal/model.pt \\")
    print("       --signer-filter S1 S2 \\")
    print("       --category-filter 0 1 2 \\")
    print("       --output-dir results/validation_combined")
    
    # Example 5: Save individual predictions
    print("\n5. Save individual predictions with new fields:")
    print("   python evaluation/validation/validate.py \\")
    print("       --model transformer \\")
    print("       --checkpoint trained_models/transformer/optimal/model.pt \\")
    print("       --save-predictions \\")
    print("       --output-dir results/validation_detailed")
    
    # Example 6: IV3-GRU model validation
    print("\n6. IV3-GRU model validation:")
    print("   python evaluation/validation/validate.py \\")
    print("       --model iv3_gru \\")
    print("       --checkpoint trained_models/iv3_gru/optimal/model.pt \\")
    print("       --signer-filter S1 \\")
    print("       --output-dir results/validation_iv3_gru")
    
    print("\n" + "=" * 60)
    print("NEW OUTPUT FILES GENERATED:")
    print("=" * 60)
    print("• per_signer_results.json - Per-signer accuracy metrics")
    print("• per_category_results.json - Per-category accuracy metrics")
    print("• duration_analysis.json - Duration-based performance analysis")
    print("• confusion_matrices.json - Enhanced with TP, FP, TN, FN")
    print("• individual_predictions/ - Now includes signer and duration")
    
    print("\n" + "=" * 60)
    print("CSV FORMAT REQUIREMENTS:")
    print("=" * 60)
    print("Your CSV file must have these 6 columns:")
    print("• file - File identifier (without .npz extension)")
    print("• gloss - Ground truth gloss label (integer)")
    print("• cat - Ground truth category label (integer)")
    print("• occluded - Occlusion flag (0 or 1)")
    print("• signer - Signer ID (string, e.g., 'S1', 'S2')")
    print("• duration - Duration in seconds (float)")
    
    print("\n" + "=" * 60)
    print("EXAMPLE CSV FORMAT:")
    print("=" * 60)
    print("file,gloss,cat,occluded,signer,duration")
    print("clip_0001,0,0,0,S1,2.5")
    print("clip_0002,1,0,1,S1,3.2")
    print("clip_0003,2,1,0,S2,1.8")
    print("clip_0004,3,1,0,S2,2.9")
    print("clip_0005,4,2,1,S1,4.1")

def show_help():
    """Show help for the validation script."""
    print("\n" + "=" * 60)
    print("VALIDATION SCRIPT HELP:")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "evaluation/validation/validate.py", "--help"
        ], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error running help: {e}")

if __name__ == "__main__":
    run_validation_example()
    show_help()
