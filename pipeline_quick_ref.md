# PANSINAYAN Pipeline - Quick Reference Guide

## Visual Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    PANSINAYAN COMPLETE PIPELINE AT A GLANCE                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: UPLOAD & INPUT HANDLING                                                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Controller: streamlit_app/manager/upload_manager.py                                 │
│ Functions: initialize_upload_session_state(), render_upload_stage()                 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Inputs:  • NPZ files (.npz) → Direct to Stage 4                                     │
│          • Video files (.mp4, .mov, .avi) → Continue to Stage 2                     │
│          • Demo files (data/demo/) → Direct to Stage 4                               │
│ Config:  • Max files: 10                                                             │
│          • Max size: 500MB (.streamlit/config.toml)                                  │
│ Output:  • st.session_state.npz_files or st.session_state.video_files              │
│          • workflow_stage = 'preprocessing' or 'predictions'                         │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING (Video → Features)                                           │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Controller: streamlit_app/manager/preprocessing_manager.py                          │
│ Processor: preprocessing/core/preprocess.py → process_videos_multiprocess()         │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Feature Extractors:                                                                  │
│   1. MediaPipe Keypoints (156-D)                                                     │
│      • File: preprocessing/extractors/keypoints_features.py                          │
│      • Components: Pose (25) + Hands (42) + Face (11) = 78 keypoints × 2 coords     │
│      • Output: X [T, 156]                                                            │
│                                                                                      │
│   2. InceptionV3 Features (2048-D)                                                   │
│      • File: preprocessing/extractors/iv3_features.py                                │
│      • Pretrained CNN (ImageNet, frozen)                                             │
│      • Output: X2048 [T, 2048]                                                       │
│                                                                                      │
│   3. Occlusion Detection                                                             │
│      • File: preprocessing/core/occlusion_detection.py                               │
│      • Frame-level: <60% keypoints visible                                           │
│      • Clip-level: ≥40% frames OR ≥15 consecutive frames occluded                   │
│      • Output: occluded_flag (0 or 1)                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Processing Options (config.py):                                                      │
│   • target_fps: 30 (frame sampling rate)                                            │
│   • out_size: 256 (frame resize dimension)                                          │
│   • write_keypoints: True (extract MediaPipe)                                       │
│   • write_iv3_features: True (extract InceptionV3)                                  │
│   • occ_detailed: False (detailed metrics)                                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Performance Optimizations:                                                           │
│   • Dynamic resource detection (CPU/GPU/RAM)                                         │
│   • Optimal worker calculation                                                       │
│   • GPU acceleration (10-100x speedup)                                              │
│   • Multi-processing (30-50x speedup)                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Output NPZ Structure:                                                                │
│   • X: [T, 156] keypoints                                                           │
│   • X2048: [T, 2048] features                                                       │
│   • mask: [T, 78] visibility                                                        │
│   • timestamps_ms: [T] frame timestamps                                             │
│   • meta: JSON {fps, size, model_type, occluded_flag, ...}                         │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DATA VALIDATION                                                            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Validators: preprocessing/utils/validate_npz.py, components/utils.py                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Validation Layers:                                                                   │
│   1. File Structure: NPZ loadable, required keys present                            │
│   2. Shape Validation: Correct dimensions [T,156] / [T,2048]                        │
│   3. Content Validation: No NaN/Inf, normalized ranges                              │
│   4. Model Compatibility: Transformer needs X or X2048, IV3-GRU needs X2048         │
│   5. Metadata Validation: model_type, occluded_flag valid                           │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Compatibility Matrix:                                                                │
│   NPZ Content      │ Transformer │ IV3-GRU │                                        │
│   ────────────────────────────────────────                                          │
│   X (156-D)        │     ✓       │    ✗    │                                        │
│   X2048 (2048-D)   │     ✓       │    ✓    │                                        │
│   X + X2048        │  ✓ (2204-D) │    ✓    │                                        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Output: Compatibility dict stored in st.session_state.file_metadata                 │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: PREDICTION & INFERENCE                                                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Controller: streamlit_app/manager/prediction_manager.py                             │
│ Engine: evaluation/prediction/predict.py → ModelPredictor                           │
│ Models: models/transformer.py, models/iv3_gru.py                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Model Loading (Singleton Pattern):                                                  │
│   ModelManager → get_model() → ModelPredictor                                       │
│   • First load: ~5-10s (checkpoint loading)                                         │
│   • Cached: ~100-500ms (inference only)                                             │
│   • Memory: Single instance per model                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ SignTransformer Architecture:                                                        │
│   Input: [B, T, 156/2048/2204] → Embedding [B,T,256] → Positional Encoding        │
│   → Transformer Layers (4 layers, 8 heads) → Pooling [B,256]                       │
│   → Dual Heads: Gloss[105] + Category[10]                                          │
│   Checkpoint: trained_models/transformer/optimal/SignTransformer_best.pt            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ InceptionV3-GRU Architecture:                                                        │
│   Input: [B, T, 2048] → GRU1 (hidden=16) → Dropout                                 │
│   → GRU2 (hidden=12) → Dropout → Final Hidden [B,12]                               │
│   → Dual Heads: Gloss[105] + Category[10]                                          │
│   Checkpoint: trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Prediction Workflow:                                                                 │
│   1. get_model() → Load/retrieve cached model                                       │
│   2. Load NPZ data                                                                   │
│   3. Extract appropriate features (156-D / 2048-D / 2204-D)                         │
│   4. Forward pass through model                                                      │
│   5. Softmax → Get top-K predictions                                                │
│   6. Label mapping → Human-readable results                                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Prediction Output:                                                                   │
│   • gloss_prediction: int (0-104)                                                   │
│   • gloss_probability: float (confidence)                                           │
│   • category_prediction: int (0-9)                                                  │
│   • category_probability: float (confidence)                                        │
│   • gloss_top5: [(id, prob), ...]                                                  │
│   • category_top3: [(id, prob), ...]                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Label Categories (10):                                                               │
│   0: GREETING  │  1: SURVIVAL  │  2: NUMBER  │  3: CALENDAR  │  4: DAYS           │
│   5: FAMILY    │  6: RELATIONSHIPS  │  7: COLOR  │  8: FOOD   │  9: DRINK          │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: RESULTS & VISUALIZATION                                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Components: streamlit_app/components/visualization.py, components.py                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Visualization Features:                                                              │
│                                                                                      │
│   1. File Information Display                                                        │
│      • Name, size, frames, duration                                                  │
│      • Compatibility badges (Transformer/IV3-GRU)                                    │
│      • Occlusion status                                                              │
│      • Sequence overview chart                                                       │
│                                                                                      │
│   2. Prediction Results Display                                                      │
│      • Top gloss + confidence score + progress bar                                   │
│      • Top category + confidence score + progress bar                                │
│      • Top-5 gloss alternatives with probabilities                                   │
│      • Top-3 category alternatives with probabilities                                │
│                                                                                      │
│   3. Animated Keypoint Visualization                                                 │
│      • Interactive skeleton overlay                                                  │
│      • Frame slider for manual navigation                                            │
│      • Play/pause controls + FPS adjustment                                          │
│      • Color-coded body parts:                                                       │
│        - Red: Pose (upper body)                                                      │
│        - Blue: Left hand                                                             │
│        - Green: Right hand                                                           │
│        - Orange: Face landmarks                                                      │
│      • Visibility indicators (faded for occluded)                                    │
│      • Video export (MP4 animation)                                                  │
│                                                                                      │
│   4. Feature Analysis Charts                                                         │
│      • Body part selector (Pose/Left Hand/Right Hand/Face)                           │
│      • Trajectory plots over time                                                    │
│      • Temporal heatmaps                                                             │
│      • Line charts for individual coordinates                                        │
│      • Statistical analysis (mean/std/min/max/range)                                 │
│                                                                                      │
│   5. Export Options                                                                  │
│      • JSON: Complete prediction results                                             │
│      • CSV: Summary table                                                            │
│      • NPZ: Processed data download                                                  │
│      • ZIP: Batch export (all NPZ + summary CSV)                                     │
│      • Video: Keypoint animation (MP4)                                               │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        │                                 │
                        ↓                                 ↓
              Individual File View              Batch Summary View
         ┌────────────────────────┐      ┌──────────────────────────┐
         │ • Single file analysis │      │ • All files table        │
         │ • Detailed predictions │      │ • Predictions for all    │
         │ • Full visualization   │      │ • Occlusion status       │
         │ • Individual download  │      │ • Batch ZIP download     │
         └────────────────────────┘      └──────────────────────────┘
                                         │
                     ┌───────────────────┴────────────────────┐
                     │                                        │
                     ↓                                        ↓
            Continue Using Tool              Model Validation (Optional)
                                                      ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: MODEL VALIDATION & EVALUATION                                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Controller: streamlit_app/core/main.py → render_validation_stage()                  │
│ Manager: streamlit_app/manager/validation_manager.py                                │
│ Engine: evaluation/validation/validate.py → ModelValidator                          │
│ UI: streamlit_app/components/validation_components.py                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Validation Setup:                                                                    │
│   • Model selection: Transformer or IV3-GRU                                          │
│   • Dataset: NPZ folder path + Labels CSV                                            │
│   • Labels CSV columns: file, gloss, cat, occluded                                   │
│   • Batch size: 1-64 (default: 32)                                                   │
│   • Device: Auto (CUDA if available) or CPU                                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Validation Process:                                                                  │
│   1. Load ValidationDataset (filter existing files)                                 │
│   2. Load ModelValidator (model + checkpoint)                                       │
│   3. Batch inference with progress tracking                                          │
│   4. Collect predictions + ground truth                                              │
│   5. Compute comprehensive metrics                                                   │
│   6. Generate confusion matrices                                                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Metrics Computed:                                                                    │
│                                                                                      │
│   1. Overall Metrics                                                                 │
│      • Gloss: Accuracy, Precision, Recall, F1-score                                 │
│      • Category: Accuracy, Precision, Recall, F1-score                              │
│      • Sample count                                                                  │
│                                                                                      │
│   2. Occlusion-Based Metrics                                                         │
│      • Separate metrics for occluded samples                                         │
│      • Separate metrics for non-occluded samples                                     │
│      • Performance comparison                                                        │
│                                                                                      │
│   3. Per-Class Metrics                                                               │
│      • Precision/Recall/F1 per gloss (105 classes)                                  │
│      • Precision/Recall/F1 per category (10 classes)                                │
│      • Support counts                                                                │
│      • Identify difficult classes                                                    │
│                                                                                      │
│   4. Confusion Matrices                                                              │
│      • Gloss confusion [105×105]                                                     │
│      • Category confusion [10×10]                                                    │
│      • Error pattern analysis                                                        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Results Display:                                                                     │
│   • Metrics dashboard with cards                                                     │
│   • Occlusion comparison table                                                       │
│   • Confusion matrix heatmaps                                                        │
│   • Per-class performance breakdown                                                  │
│   • Error analysis visualizations                                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Export Options:                                                                      │
│   • JSON: Complete validation results                                                │
│   • CSV: Confusion matrices                                                          │
│   • CSV: Per-class metrics                                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Session State Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SESSION STATE LIFECYCLE                         │
└─────────────────────────────────────────────────────────────────────┘

INITIALIZATION (upload_manager.initialize_upload_session_state)
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ st.session_state = {                                                │
│   uploaded_files: [],        # All uploaded files                   │
│   npz_files: [],            # NPZ files (ready for inference)       │
│   video_files: [],          # Video files (need preprocessing)      │
│   preprocessed_files: [],   # Preprocessed NPZ files                │
│   file_status: {},          # {filename: status}                    │
│   processed_data: {},       # {filename: npz_data_dict}             │
│   file_metadata: {},        # {filename: metadata}                  │
│   original_file_data: {},   # For reset functionality               │
│   workflow_stage: 'upload', # Current stage                         │
│   current_tab: None,        # Selected file                         │
│   validation_results: None  # Validation output                     │
│ }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ UPLOAD STAGE                                                        │
│ • User uploads files                                                │
│ • file_status[filename] = 'pending'                                 │
│ • Route to npz_files or video_files                                 │
│ • Set workflow_stage = 'preprocessing' or 'predictions'             │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING STAGE (if videos)                                    │
│ • file_status[filename] = 'processing' → 'completed'/'error'        │
│ • processed_data[filename] = npz_data                               │
│ • file_metadata[filename] = {compatibility, frame_count, ...}       │
│ • original_file_data[filename] = {name, data, type, size}          │
│ • Move from video_files to preprocessed_files                       │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PREDICTIONS STAGE                                                   │
│ • All NPZ files available (npz_files + preprocessed_files)          │
│ • current_tab = selected_filename                                   │
│ • Display predictions and visualizations                            │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ VALIDATION STAGE (optional)                                         │
│ • validation_results = {metrics, confusion_matrices, ...}           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Status State Machine

```
┌─────────┐
│ pending │ ← Initial state after upload
└────┬────┘
     │ User clicks "Process/Preprocess"
     ↓
┌────────────┐
│ processing │ ← During processing (shows spinner)
└────┬───┬───┘
     │   │
 Success  Error
     │   │
     ↓   ↓
┌───────────┐  ┌───────┐
│ completed │  │ error │ ← Shows error message + retry button
└───────────┘  └───┬───┘
     │             │ User clicks "Retry"
     │             │
     └─────────────┘
           ↓
     ┌────────────┐
     │ processing │
     └────────────┘
```

---

## Workflow Stage Transitions

```
                    ┌─────────┐
             ┌──────┤ UPLOAD  ├──────┐
             │      └─────────┘      │
             │                       │
      has_npz_only           has_videos
             │                       │
             ↓                       ↓
    ┌──────────────┐       ┌─────────────────┐
    │ PREDICTIONS  │←──────┤ PREPROCESSING   │
    └──────┬───────┘       └─────────────────┘
           │                        ↑
           │                        │
           │ "Back" button          │ "Reset All" button
           ↓                        │
    ┌─────────────┐                │
    │  UPLOAD or  │────────────────┘
    │PREPROCESSING│
    └──────┬──────┘
           │
           │ Sidebar navigation
           ↓
    ┌─────────────┐
    │ VALIDATION  │ (Optional, any time)
    └─────────────┘
```

---

## Configuration Quick Reference

### Model Configuration

```python
# streamlit_app/core/config.py

MODEL_CONFIG = {
    'transformer': {
        'checkpoint_path': 'trained_models/transformer/optimal/SignTransformer_best.pt',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'input_dim': None,  # Auto-detected: 156/2048/2204
    },
    'iv3_gru': {
        'checkpoint_path': 'trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'input_dim': 2048,  # Fixed
    }
}
```

### Processing Configuration

```python
PROCESSING_CONFIG = {
    'video': {
        'target_fps': 30,           # Frame sampling rate
        'out_size': 256,            # Frame resize dimension
        'write_keypoints': True,    # Extract MediaPipe keypoints
        'write_iv3_features': True, # Extract InceptionV3 features
        'occ_detailed': False       # Detailed occlusion metrics
    }
}
```

### Streamlit Configuration

```toml
# .streamlit/config.toml

[server]
maxUploadSize = 500        # Max file size (MB)
maxMessageSize = 500       # Max WebSocket message (MB)
enableCORS = true          # Mobile support
enableWebsocketCompression = true  # Performance
```

---

## Performance Benchmarks

| Operation | Sequential | Optimized | Speedup |
|-----------|-----------|-----------|---------|
| Model Loading (first) | 5-10s | 100-500ms (cached) | 10-100x |
| Video Preprocessing (single) | 45-60s | 5-8s (GPU, parallel) | 6-12x |
| Batch Preprocessing (10 videos) | 450-600s | 60-90s | 5-10x |
| Feature Extraction | 30-45s | 3-5s (GPU, batched) | 6-9x |
| Model Inference | 2-5s | 100-500ms (GPU, cached) | 4-50x |
| NPZ File Size | 50-200 KB | 10-50 KB (compressed) | 3-5x |

---

## Error Handling Summary

| Stage | Common Errors | Recovery |
|-------|--------------|----------|
| **Upload** | File too large, unsupported format | Reject with message |
| **Preprocessing** | Video codec, MediaPipe failure, CUDA OOM | Mark error, allow retry |
| **Validation** | Shape mismatch, NaN/Inf, incompatible | Show details, allow re-upload |
| **Prediction** | Model load failure, CUDA OOM | Use dummy data or show error |
| **Export** | Disk full, permission error | Retry with delay |

---

## Key File Locations

```
PROJECT ROOT
├── run_app.py                              # Application entry point
├── .streamlit/config.toml                  # Upload limits, server config
│
├── streamlit_app/
│   ├── core/
│   │   ├── main.py                         # Main app logic, workflow routing
│   │   └── config.py                       # Application configuration
│   │
│   ├── manager/
│   │   ├── upload_manager.py               # Stage 1: Upload handling
│   │   ├── preprocessing_manager.py        # Stage 2: Video preprocessing
│   │   ├── prediction_manager.py           # Stage 4: Model inference
│   │   └── validation_manager.py           # Stage 6: Model evaluation
│   │
│   └── components/
│       ├── components.py                   # Reusable UI components
│       ├── validation_components.py        # Validation UI
│       ├── visualization.py                # Keypoint animation, charts
│       ├── data_processing.py              # Video processing backend
│       └── utils.py                        # Utility functions
│
├── preprocessing/
│   ├── core/
│   │   ├── preprocess.py                   # Main preprocessing pipeline
│   │   └── occlusion_detection.py          # Occlusion detection
│   │
│   └── extractors/
│       ├── keypoints_features.py           # MediaPipe keypoints (156-D)
│       └── iv3_features.py                 # InceptionV3 features (2048-D)
│
├── evaluation/
│   ├── prediction/
│   │   └── predict.py                      # ModelPredictor class
│   │
│   └── validation/
│       └── validate.py                     # ModelValidator class
│
├── models/
│   ├── transformer.py                      # SignTransformer architecture
│   └── iv3_gru.py                         # InceptionV3GRU architecture
│
├── trained_models/
│   ├── transformer/optimal/SignTransformer_best.pt
│   └── iv3_gru/optimal/InceptionV3GRU_best.pt
│
└── data/
    ├── labels/label_mapping.py             # Label mappings (105 glosses, 10 categories)
    ├── splitting/labels_reference.csv      # Label reference table
    └── demo/                                # Demo NPZ files
```

---

## Command Cheat Sheet

```bash
# Launch Application
streamlit run run_app.py

# Launch with custom port
streamlit run run_app.py --server.port 8502

# Check network info
python show_network_info.py

# Command-line prediction (Transformer)
python -m evaluation.prediction.predict \
  --model transformer \
  --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt \
  --input data/demo/clip_0138_nice\ to\ meet\ you.npz

# Command-line prediction (IV3-GRU)
python -m evaluation.prediction.predict \
  --model iv3_gru \
  --checkpoint trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt \
  --input data/demo/clip_1146_grandfather.npz

# Command-line validation
python -m evaluation.validation.validate \
  --model transformer \
  --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt \
  --data-dir data/processed/cmb_val \
  --labels-csv data/processed/cmb_val.csv

# Preprocessing (single video)
python -m preprocessing.core.preprocess \
  video.mp4 output_dir \
  --write-keypoints \
  --write-iv3-features \
  --target-fps 30

# Preprocessing (batch with parallel processing)
python -m preprocessing.core.preprocess \
  input_dir output_dir \
  --write-keypoints \
  --write-iv3-features \
  --workers 8 \
  --batch-size 32 \
  --target-fps 30

# Validate NPZ files
python -m preprocessing.utils.validate_npz data/processed/cmb_train
python -m preprocessing.utils.validate_npz data/processed/cmb_val --require-x2048
```

---

## Documentation Index

| Document | Description | Content |
|----------|-------------|---------|
| **PANSINAYAN_PIPELINE.md** | Part 1: Stages 1-4 | Upload, Preprocessing, Validation, Prediction |
| **PANSINAYAN_PIPELINE_PART2.md** | Part 2: Stages 5-6 + System | Visualization, Model Validation, Data Flow, Config, Performance |
| **PIPELINE_QUICK_REFERENCE.md** | This document | Visual overview, quick reference, cheat sheet |
| **system_archi_analysis.md** | System architecture | Complete technical architecture analysis |
| **streamlit_app/TOOL_GUIDE.md** | User guide | How to use PANSINAYAN application |
| **README.md** | Project overview | Quick start, features, workflow |

---

## Troubleshooting Quick Guide

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| **Upload fails** | File > 500MB | Compress video or split file |
| **Video processing slow** | CPU-only processing | Check CUDA availability |
| **Prediction shows error** | Incompatible NPZ | Check compatibility badges |
| **Model fails to load** | Checkpoint missing | Verify checkpoint path in config.py |
| **CUDA out of memory** | Batch size too large | Reduce batch size or use CPU |
| **Validation fails** | Shape mismatch | Check NPZ structure with validate_npz.py |
| **Skeleton not visible** | No keypoints (X) | Need 156-D keypoint data |
| **Reset doesn't work** | original_file_data missing | Re-upload files |

---

**Document Status**: Complete Quick Reference  
**Last Updated**: October 11, 2025  
**For Detailed Information**: See full pipeline documentation (PANSINAYAN_PIPELINE.md + PART2)

