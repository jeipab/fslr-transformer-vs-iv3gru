# Training Flow Diagram

## Quick Training Flow: Start to End

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING START                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION PHASE                                        │
├─────────────────────────────────────────────────────────────────┤
│ • parse_args() - Parse command-line arguments                  │
│ • set_global_seed() - Set reproducible random seeds           │
│ • get_optimal_device() - Auto-detect CUDA/MPS/CPU             │
│ • print_device_info() - Display device specifications         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SMOKE TEST (Optional)                                       │
├─────────────────────────────────────────────────────────────────┤
│ • Create dummy data                                            │
│ • Instantiate model                                            │
│ • Test forward pass                                            │
│ • Test backward pass                                           │
│ • Save/load checkpoint                                         │
│ • Exit if smoke test                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING & VALIDATION                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Validate data file paths                                     │
│ • Check CSV labels exist                                       │
│ • Determine dataset type (features vs keypoints)              │
│ • Log dataset information                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DATASET PREPARATION                                         │
├─────────────────────────────────────────────────────────────────┤
│ • calculate_optimal_batch_size() - Auto-calculate batch size   │
│ • Create FSLFeatureFileDataset or FSLKeypointFileDataset       │
│ • Setup augmentation parameters                                │
│ • Create DataLoaders with _make_dataloader()                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. MODEL SETUP                                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Instantiate SignTransformer or InceptionV3GRU               │
│ • optimize_model_for_parallel() - Multi-GPU support           │
│ • log_comprehensive_config() - Log training configuration     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. TRAINING EXECUTION                                          │
├─────────────────────────────────────────────────────────────────┤
│ • train_model() - Main training function                      │
│                                                                 │
│   ┌─ INITIAL SETUP ─────────────────────────────────────────┐  │
│   │ • Clear GPU memory                                       │  │
│   │ • Setup curriculum scheduler                             │  │
│   │ • Setup loss weighting strategy                          │  │
│   │ • Initialize loss function (CE/Focal/LabelSmoothing)    │  │
│   │ • Setup optimizer                                        │  │
│   │ • Setup AMP scaler                                       │  │
│   │ • Setup learning rate scheduler                          │  │
│   │ • Setup EMA (if enabled)                                 │  │
│   │ • Resume from checkpoint (if provided)                   │  │
│   │ • Setup CSV logging                                      │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                │                               │
│                                ▼                               │
│   ┌─ TRAINING LOOP ────────────────────────────────────────┐  │
│   │ For each epoch:                                         │  │
│   │                                                         │  │
│   │  ┌─ EPOCH INITIALIZATION ──────────────────────────┐   │  │
│   │  │ • Get curriculum weights                         │   │  │
│   │  │ • Set model.train()                              │   │  │
│   │  │ • Initialize epoch tracking variables            │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   │                                │                       │  │
│   │                                ▼                       │  │
│   │  ┌─ BATCH PROCESSING ──────────────────────────────┐   │  │
│   │  │ For each batch:                                 │   │  │
│   │  │ • Load batch data                               │   │  │
│   │  │ • Move to device                                │   │  │
│   │  │ • Forward pass with AMP                         │   │  │
│   │  │ • Calculate losses                              │   │  │
│   │  │ • Apply loss weighting                          │   │  │
│   │  │ • Backward pass with gradient scaling           │   │  │
│   │  │ • Gradient accumulation                         │   │  │
│   │  │ • Gradient clipping (if enabled)                │   │  │
│   │  │ • Update parameters                             │   │  │
│   │  │ • Update EMA                                    │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   │                                │                       │  │
│   │                                ▼                       │  │
│   │  ┌─ VALIDATION ────────────────────────────────────┐   │  │
│   │  │ • Clear GPU memory                             │   │  │
│   │  │ • Apply EMA parameters                         │   │  │
│   │  │ • evaluate_with_forward()                      │   │  │
│   │  │ • Restore original parameters                  │   │  │
│   │  │ • Calculate metrics                            │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   │                                │                       │  │
│   │                                ▼                       │  │
│   │  ┌─ CHECKPOINTING & LOGGING ──────────────────────┐   │  │
│   │  │ • Update learning rate scheduler               │   │  │
│   │  │ • Save checkpoint (last & best)                │   │  │
│   │  │ • Log to CSV                                   │   │  │
│   │  │ • Print epoch results                          │   │  │
│   │  │ • Check early stopping                         │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING END                             │
├─────────────────────────────────────────────────────────────────┤
│ • Close CSV log file                                           │
│ • Print final results                                          │
│ • Save final checkpoint                                        │
│ • Display training summary                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Training Phases

### **Phase 1: Setup & Initialization**

- Parse arguments and configure training
- Set up device and reproducibility
- Validate data and create datasets

### **Phase 2: Model Preparation**

- Instantiate model with optimizations
- Configure advanced training features
- Set up logging and monitoring

### **Phase 3: Training Loop**

- **Epoch Loop**: For each training epoch
  - **Batch Loop**: Process each batch with gradient accumulation
  - **Validation**: Evaluate on validation data
  - **Checkpointing**: Save model state and metrics

### **Phase 4: Cleanup & Finalization**

- Close logging files
- Save final checkpoint
- Display training summary

## Critical Functions in Flow

1. **`parse_args()`** - Entry point configuration
2. **`get_optimal_device()`** - Hardware optimization
3. **`_make_dataloader()`** - Data pipeline setup
4. **`train_model()`** - Core training execution
5. **`evaluate_with_forward()`** - Validation evaluation
6. **`save_checkpoint()`** - Model state persistence

## Training Features Integration

- **Curriculum Learning**: Dynamic weight adjustment during training
- **Loss Weighting**: Advanced multi-task learning strategies
- **Performance Optimization**: AMP, compilation, parallelization
- **Monitoring**: Real-time metrics and CSV logging
- **Robustness**: Early stopping, checkpointing, resume capability
