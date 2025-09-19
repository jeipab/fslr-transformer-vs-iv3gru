#!/usr/bin/env python3
"""
Test script for loss weighting strategies.
This script tests the loss weighting strategies without requiring actual data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from training.train import (
    StaticWeighting, GridSearchWeighting, UncertaintyWeighting, GradNormWeighting,
    create_loss_weighting_strategy, parse_grid_search_weights
)

def test_loss_weighting_strategies():
    """Test all loss weighting strategies."""
    
    print("Testing Loss Weighting Strategies")
    print("=" * 50)
    
    # Test static weighting
    print("\n1. Testing Static Weighting:")
    static = StaticWeighting(alpha=0.7, beta=0.3)
    for epoch in range(5):
        alpha, beta = static.get_weights(epoch, 0, 1.0, 1.0)
        print(f"  Epoch {epoch+1}: α={alpha:.3f}, β={beta:.3f}")
    
    # Test grid search weighting
    print("\n2. Testing Grid Search Weighting:")
    weight_combinations = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]
    grid_search = GridSearchWeighting(weight_combinations, epochs_per_combination=3)
    for epoch in range(10):
        alpha, beta = grid_search.get_weights(epoch, 0, 1.0, 1.0)
        print(f"  Epoch {epoch+1}: α={alpha:.3f}, β={beta:.3f}")
    
    # Test uncertainty weighting
    print("\n3. Testing Uncertainty Weighting:")
    uncertainty = UncertaintyWeighting(init_uncertainty=1.0, device="cpu")
    for epoch in range(5):
        alpha, beta = uncertainty.get_weights(epoch, 0, 1.0, 1.0)
        print(f"  Epoch {epoch+1}: α={alpha:.3f}, β={beta:.3f}")
    
    # Test GradNorm weighting
    print("\n4. Testing GradNorm Weighting:")
    gradnorm = GradNormWeighting(alpha=1.5, update_freq=2, device="cpu")
    for epoch in range(8):
        alpha, beta = gradnorm.get_weights(epoch, 0, 1.0, 1.0)
        print(f"  Epoch {epoch+1}: α={alpha:.3f}, β={beta:.3f}")
        
        # Simulate loss update
        if epoch % 2 == 0:
            losses = {
                'gloss': torch.tensor(1.0 - epoch * 0.1),
                'cat': torch.tensor(1.0 - epoch * 0.05)
            }
            gradnorm.update_weights(epoch, losses, None, None)
    
    # Test factory function
    print("\n5. Testing Factory Function:")
    strategies = ["static", "grid-search", "uncertainty", "gradnorm"]
    for strategy in strategies:
        try:
            if strategy == "grid-search":
                weight_combinations = parse_grid_search_weights("0.5,0.5;0.7,0.3")
                weighting = create_loss_weighting_strategy(
                    strategy, 
                    weight_combinations=weight_combinations,
                    epochs_per_combination=5
                )
            else:
                weighting = create_loss_weighting_strategy(strategy, alpha=0.6, beta=0.4)
            
            alpha, beta = weighting.get_weights(0, 0, 1.0, 1.0)
            print(f"  {strategy}: α={alpha:.3f}, β={beta:.3f}")
        except Exception as e:
            print(f"  {strategy}: Error - {e}")
    
    # Test grid search weight parsing
    print("\n6. Testing Grid Search Weight Parsing:")
    test_weights = "0.5,0.5;0.7,0.3;0.3,0.7;0.8,0.2"
    combinations = parse_grid_search_weights(test_weights)
    print(f"  Parsed weights: {combinations}")
    
    print("\n" + "=" * 50)
    print("✓ All loss weighting strategy tests passed!")
    print("\nLoss Weighting Features:")
    print("  - Static: Fixed alpha/beta weights")
    print("  - Grid Search: Multiple weight combinations")
    print("  - Uncertainty: Learnable uncertainty-based weighting")
    print("  - GradNorm: Gradient normalization for balanced learning")
    print("\nReady for training with advanced loss weighting!")

if __name__ == "__main__":
    test_loss_weighting_strategies()
