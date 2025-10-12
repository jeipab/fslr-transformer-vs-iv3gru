# PANSINAYAN Complete Pipeline Documentation

**Consolidated Reference Guide**

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Stage 1: Upload & Input Handling](#stage-1-upload--input-handling)
- [Stage 2: Preprocessing](#stage-2-preprocessing)
- [Stage 3: Data Validation](#stage-3-data-validation)
- [Stage 4: Prediction & Inference](#stage-4-prediction--inference)
- [Stage 5: Results & Visualization](#stage-5-results--visualization)
- [Stage 6: Model Validation & Evaluation](#stage-6-model-validation--evaluation)
- [Data Flow & State Management](#data-flow--state-management)
- [Error Handling](#error-handling)
- [Configuration & Performance](#configuration--performance)

---

## Pipeline Overview

PANSINAYAN implements a 6-stage pipeline for Filipino Sign Language Recognition:

```
Stage 1: UPLOAD & INPUT HANDLING
┌──────────────────────────────────────┐
│ • File upload (NPZ/Video/Demo)       │
│ • Format detection & validation      │
│ • Session state initialization       │
│ • File routing (NPZ vs Video)        │
└────────────┬─────────────────────────┘
             │
             ├─── NPZ Files ───────────────────────┐
             │                                     │
             └─── Video Files                      │
                     ↓                             │
Stage 2: PREPROCESSING (Video → Features)          │
┌──────────────────────────────────────┐           │
│ • MediaPipe keypoint extraction      │           │
│ • InceptionV3 feature extraction     │           │
│ • Occlusion detection                │           │
│ • Multi-process GPU acceleration     │           │
│ • NPZ generation with metadata       │           │
└────────────┬─────────────────────────┘           │
             │                                     │
             └────── NPZ Files ────────────────────┘
                        ↓
Stage 3: DATA VALIDATION
┌──────────────────────────────────────┐
│ • NPZ structure verification         │
│ • Shape and content checks           │
│ • Model compatibility validation     │
│ • Metadata integrity                 │
└────────────┬─────────────────────────┘
             ↓
Stage 4: PREDICTION & INFERENCE
┌──────────────────────────────────────┐
│ • Model loading (Transformer/IV3-GRU)│
│ • Feature selection (156-D/2048-D)   │
│ • Forward pass & inference           │
│ • Top-K predictions & confidence     │
│ • Label mapping (gloss & category)   │
└────────────┬─────────────────────────┘
             ↓
Stage 5: RESULTS & VISUALIZATION
┌──────────────────────────────────────┐
│ • Prediction display (gloss/category)│
│ • Keypoint skeleton animation        │
│ • Feature trajectory plots           │
│ • Statistical analysis               │
│ • Export options (JSON/CSV/Video)    │
└────────────┬─────────────────────────┘
             │
             └─── Optional: Model Validation
                        ↓
Stage 6: MODEL VALIDATION & EVALUATION
┌──────────────────────────────────────┐
│ • Batch evaluation on validation set │
│ • Comprehensive metrics (Acc/P/R/F1) │
│ • Confusion matrix generation        │
│ • Occlusion analysis                 │
│ • Per-class performance              │
└──────────────────────────────────────┘
```

**Key Characteristics:**

- **Modular Design**: Each stage is independent with clear interfaces
- **Flexible Routing**: NPZ files skip preprocessing, videos go through full pipeline
- **State Persistence**: Session state tracks files through all stages
- **Error Recovery**: Checkpoints at each stage with rollback capability
- **Performance Optimized**: GPU acceleration, parallel processing, caching

---

## Stage 1: Upload & Input Handling

### Overview

Entry point for all user data, handling file uploads, format detection, and routing to appropriate processing stages.

### Components

- **Primary Controller**: `streamlit_app/manager/upload_manager.py`
- **Key Functions**: `initialize_upload_session_state()`, `render_upload_stage()`, `route_files_to_stages()`, `proceed_to_next_stage()`

### Supported Input Formats

| Format    | Extensions       | Processing Path           | Use Case           |
| --------- | ---------------- | ------------------------- | ------------------ |
| **NPZ**   | .npz             | Direct to Inference       | Pre-processed data |
| **Video** | .mp4, .mov, .avi | Preprocessing → Inference | Raw video clips    |
| **Demo**  | data/demo/       | Direct to Inference       | Testing & examples |

### Session State Structure

```python
st.session_state = {
    # File lists
    'uploaded_files': [],          # All uploaded files
    'npz_files': [],               # NPZ files (ready for inference)
    'video_files': [],             # Video files (need preprocessing)
    'preprocessed_files': [],      # Preprocessed NPZ files

    # File tracking
    'file_status': {},             # {filename: 'pending'|'processing'|'completed'|'error'}
    'processed_data': {},          # {filename: npz_data_dict}
    'file_metadata': {},           # {filename: metadata_dict}
    'original_file_data': {},      # For reset functionality

    # Workflow state
    'workflow_stage': 'upload',    # Current stage
    'current_tab': None,           # Selected file for visualization
    'pending_upload_files': []     # Files awaiting processing
}
```

### Upload Workflow

**Function**: `detect_file_type(uploaded_file)`

- Detects file type from extension and MIME type
- Returns: 'npz', 'video', or 'unknown'

**Function**: `route_files_to_stages(uploaded_files)`

- Separates files by type
- Stores in session state: `npz_files`, `video_files`

**Function**: `proceed_to_next_stage()`

- Determines next stage based on file types:
  - Only NPZ → go to 'predictions'
  - Has videos → go to 'preprocessing'

### Configuration

- **Max files**: 10 files per upload
- **Max file size**: 500MB (configurable in `.streamlit/config.toml`)
- **Mobile support**: CORS enabled, WebSocket compression

---

## Stage 2: Preprocessing

### Overview

Converts raw video files into feature representations suitable for model inference. Extracts both keypoint-based (156-D) and visual (2048-D) features with automatic occlusion detection.

### Components

- **Primary Controller**: `streamlit_app/manager/preprocessing_manager.py`
- **Core Processor**: `preprocessing/core/preprocess.py`
- **Feature Extractors**:
  - `preprocessing/extractors/keypoints_features.py` - MediaPipe keypoints
  - `preprocessing/extractors/iv3_features.py` - InceptionV3 CNN features
- **Occlusion Detection**: `preprocessing/core/occlusion_detection.py`

### Processing Architecture

```
Video File (.mp4, .mov, .avi)
         ↓
┌────────────────────────────────┐
│ 1. Video Loading & Frame       │
│    Extraction (OpenCV)         │
│    • Target FPS sampling       │
│    • Frame resizing (256×256)  │
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ 2. Parallel Feature Extraction │
│    (Multi-process/GPU)         │
└───┬─────────────────┬──────────┘
    │                 │
    ↓                 ↓
┌───────────┐   ┌──────────────┐
│MediaPipe  │   │InceptionV3   │
│Keypoints  │   │CNN Features  │
│[T, 156]   │   │[T, 2048]     │
└─────┬─────┘   └──────┬───────┘
      │                │
      └────────┬───────┘
               ↓
┌────────────────────────────────┐
│ 3. Post-Processing             │
│    • Gap interpolation (≤5)    │
│    • Occlusion detection       │
│    • Metadata generation       │
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ 4. NPZ File Generation         │
│    • X: [T, 156] keypoints     │
│    • X2048: [T, 2048] features │
│    • mask: [T, 78] visibility  │
│    • timestamps_ms: [T]        │
│    • meta: JSON metadata       │
└────────────────────────────────┘
```

### Feature Extraction

#### MediaPipe Keypoints (156-D)

**Distribution**:

- Pose (upper body): 25 points
- Left hand: 21 points
- Right hand: 21 points
- Face: 11 points
- **Total**: 78 keypoints × 2 coords (x, y) = 156 dimensions

**Function**: `extract_keypoints_from_frame(frame, mp_models)`

- Extracts keypoints using MediaPipe Holistic
- Returns: X [156] flattened coordinates, mask [78] visibility

**Function**: `interpolate_gaps(X, mask, max_gap=5)`

- Fills short gaps (≤5 frames) in keypoint sequences
- Uses linear interpolation for temporal continuity

#### InceptionV3 Features (2048-D)

**Class**: `BatchedInceptionV3Processor`

- Uses pretrained InceptionV3 (ImageNet), frozen backbone
- Removes classifier head, keeps feature layer

**Method**: `extract_batch(frames)`

- Extracts features from batch of frames
- Returns: [B, 2048] feature vectors
- Batch size optimized for GPU memory

#### Occlusion Detection

**Function**: `detect_frame_occlusion(mask, conf_thresh=0.6)`

- Detects occlusion in individual frames based on keypoint visibility
- Returns: [T] binary mask (1 = occluded)

**Function**: `compute_occlusion_detection(X, mask, ...)`

- Computes clip-level occlusion flag based on two conditions:
  1. ≥40% of frames are occluded, OR
  2. ≥15 consecutive frames are occluded
- Returns: `{'occluded_flag': 0 or 1, ...}`

### Processing Modes

**Function**: `preprocess_single_video(uploaded_file, filename)`

- Processes single video with default options
- Options: `target_fps=30, out_size=256, write_keypoints=True, write_iv3_features=True`

**Function**: `preprocess_multiple_videos_batch(uploaded_files)`

- Processes multiple videos with automatic resource optimization
- Uses `get_dynamic_resource_info()` and `calculate_optimal_workers()`
- Determines optimal: number of workers, batch size, processing type (GPU/CPU)

### Resource Optimization

**Function**: `get_dynamic_resource_info()`

- Returns: CPU count/usage, RAM available, CUDA availability, GPU count/memory

**Function**: `calculate_optimal_workers(resource_info, video_count)`

- Calculates optimal workers based on:
  - Memory-based limit: `available_gb / 2.5`
  - CPU-based limit: `cpu_count * (100 - cpu_percent) / 100`
  - Conservative choice: min of above, max 8
- Returns GPU workers if available, else CPU workers

### NPZ Output Format

```python
npz_data = {
    'X': np.ndarray,              # [T, 156] MediaPipe keypoints
    'X2048': np.ndarray,          # [T, 2048] InceptionV3 features
    'mask': np.ndarray,           # [T, 78] Keypoint visibility mask
    'timestamps_ms': np.ndarray,  # [T] Frame timestamps in milliseconds
    'meta': json.dumps({          # JSON metadata
        'target_fps': 30,
        'out_size': 256,
        'model_type': 'B',        # 'T' (Transformer), 'I' (IV3-GRU), 'B' (Both)
        'occluded_flag': 0,       # 0 (clean) or 1 (occluded)
        'occlusion_params': {...},
        'source_video': 'clip_0001.mp4',
        'processing_date': '2025-10-11T12:00:00'
    })
}
```

### Performance Metrics

**Single Video (30s clip, 30 FPS)**:

- Sequential CPU: ~45-60 seconds
- Sequential GPU: ~15-20 seconds
- Multi-process GPU (4 workers): ~5-8 seconds

**Batch Processing (10 videos)**:

- Sequential: ~450-600 seconds (7.5-10 min)
- Multi-process (4 workers): ~60-90 seconds
- **Speedup**: ~30-50x improvement

---

## Stage 3: Data Validation

### Overview

Comprehensive validation of NPZ files to ensure data integrity, correct shapes, and model compatibility before inference.

### Components

- **NPZ Validation**: `preprocessing/utils/validate_npz.py`
- **Data Processing Validation**: `streamlit_app/components/data_processing.py`
- **Compatibility Checks**: `streamlit_app/components/utils.py`

### Validation Layers

```
┌──────────────────────────────────────┐
│ Layer 1: File Structure Validation   │
│ • NPZ file loadable                  │
│ • Required keys present              │
│ • No corruption                      │
└────────────┬─────────────────────────┘
             ↓
┌──────────────────────────────────────┐
│ Layer 2: Shape Validation            │
│ • Correct dimensions                 │
│ • Sequence length check              │
│ • Feature dimension match            │
└────────────┬─────────────────────────┘
             ↓
┌──────────────────────────────────────┐
│ Layer 3: Content Validation          │
│ • Value ranges (normalized 0-1)      │
│ • NaN/Inf detection                  │
│ • Timestamp consistency              │
└────────────┬─────────────────────────┘
             ↓
┌──────────────────────────────────────┐
│ Layer 4: Model Compatibility         │
│ • Transformer: needs X or X2048      │
│ • IV3-GRU: needs X2048               │
│ • Feature availability check         │
└────────────┬─────────────────────────┘
             ↓
┌──────────────────────────────────────┐
│ Layer 5: Metadata Validation         │
│ • model_type present ('T'/'I'/'B')   │
│ • Occlusion flag valid (0 or 1)      │
│ • Processing parameters complete     │
└──────────────────────────────────────┘
```

### Validation Functions

**Function**: `validate_npz_structure(npz_path)`

- Checks: File exists, NPZ loadable, has required key (X or X2048)
- Returns: `(valid: bool, error_message: str)`

**Function**: `validate_shapes(npz_data, require_x2048=False)`

- Expected shapes: X [T, 156], X2048 [T, 2048], mask [T, 78], timestamps_ms [T]
- Checks dimension consistency
- Returns: `(valid: bool, errors: List[str])`

**Function**: `validate_content(npz_data)`

- Checks: No NaN/Inf values, keypoints in [0,1] range, timestamps monotonic
- Returns: `(valid: bool, errors: List[str])`

**Function**: `check_npz_compatibility(npz_data)`

- Compatibility rules:
  - Transformer: Can use X (156-D) OR X2048 (2048-D)
  - IV3-GRU: Requires X2048 (2048-D)
- Returns: `{'transformer': bool, 'iv3_gru': bool}`

**Function**: `validate_metadata(npz_data)`

- Checks: 'meta' key exists, JSON parseable, required fields present, valid values
- Returns: `(valid: bool, errors: List[str], meta_dict: dict)`

### Error Recovery

| Validation Failure | User Action                 | System Response             |
| ------------------ | --------------------------- | --------------------------- |
| Structure invalid  | Re-upload correct NPZ       | File marked as error        |
| Shape mismatch     | Check preprocessing options | Show specific error         |
| Content NaN/Inf    | Reprocess video             | Detailed error location     |
| Incompatible       | Change model selection      | Filter compatible models    |
| Metadata missing   | Continue with warnings      | Non-critical, allow proceed |

---

## Stage 4: Prediction & Inference

### Overview

Load trained models and perform inference on validated NPZ files to predict glosses (105 classes) and categories (10 classes).

### Components

- **Primary Controller**: `streamlit_app/manager/prediction_manager.py`
- **Inference Engine**: `evaluation/prediction/predict.py`
- **Model Architectures**: `models/transformer.py`, `models/iv3_gru.py`
- **Checkpoints**:
  - `trained_models/transformer/optimal/SignTransformer_best.pt`
  - `trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt`

### Model Loading Architecture

```
User Request for Prediction
         ↓
┌────────────────────────────────┐
│ ModelManager (Singleton)       │
│ • Single shared instance       │
│ • Lazy loading of models       │
│ • Cache loaded models          │
└────────────┬───────────────────┘
             ↓
      Model in cache?
      /            \
    YES            NO
     ↓              ↓
Return cached   Load Model
model           ↓
                ┌──────────────────────┐
                │ Load Checkpoint      │
                │ • Detect input_dim   │
                │ • Create architecture│
                │ • Load state_dict    │
                │ • Move to device     │
                │ • Set eval mode      │
                └───────────┬──────────┘
                            ↓
                    Cache model
                            ↓
                    Return model
```

### Key Classes and Methods

**Class**: `ModelManager` (Singleton pattern)

- **Methods**: `get_model(model_name)`, `_load_model(model_name)`, `get_label_mappings()`, `cleanup()`
- **Purpose**: Lazy-load models, cache for reuse, avoid repeated loading
- **Benefits**: First prediction ~5-10s, subsequent ~100-500ms

**Class**: `ModelPredictor`

- **Init**: `__init__(model_type, checkpoint_path, device=None)`
- **Methods**: `predict_from_npz(npz_path)`, `_load_model()`, `_load_checkpoint()`
- **Purpose**: Unified predictor for both Transformer and IV3-GRU models
- **Features**: Auto-detects input dimensions, handles different checkpoint formats

### Prediction Workflow

**Function**: `make_real_prediction(npz_data, model_name)`

- Gets or loads predictor from ModelManager
- Creates temporary NPZ file
- Calls `predictor.predict_from_npz()`
- Returns: `{'gloss_prediction', 'gloss_probability', 'category_prediction', 'category_probability', 'gloss_top5', 'category_top3'}`

**Method**: `ModelPredictor.predict_from_npz(npz_path)`

- Loads NPZ data
- Extracts appropriate features based on model type and input_dim:
  - Transformer: X (156-D), X2048 (2048-D), or concatenated (2204-D)
  - IV3-GRU: X2048 (2048-D) only
- Performs forward pass with torch.no_grad()
- Computes softmax probabilities
- Returns predictions with top-K alternatives

### Label Mapping

**Function**: `format_prediction_with_labels(results, gloss_mapping, category_mapping)`

- Converts numeric IDs to human-readable labels
- Formats: "hello (0)" instead of just "0"
- Returns formatted results for display

### Performance

**Prediction Times** (single file, 30s clip):

- First prediction (model loading): ~5-10 seconds
- Subsequent predictions (cached model): ~100-500ms
- GPU acceleration: ~50-200ms

**Memory Usage**:

- Transformer model: ~200MB
- IV3-GRU model: ~100MB
- Model caching saves ~5-10s per prediction

---

## Stage 5: Results & Visualization

### Overview

Comprehensive visualization and analysis tools for understanding model predictions, keypoint sequences, and temporal patterns.

### Components

- **Visualization Core**: `streamlit_app/components/visualization.py`
- **Results Display**: `streamlit_app/components/components.py`
- **Prediction Interface**: `streamlit_app/manager/prediction_manager.py`

### Visualization Architecture

```
Prediction Results + NPZ Data
         ↓
┌──────────────────────────────────┐
│ File Info Display                │
│ • Name, size, frames             │
│ • Compatibility badges           │
│ • Occlusion status               │
│ • Sequence overview chart        │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ Prediction Results               │
│ • Top gloss + confidence         │
│ • Top category + confidence      │
│ • Top-5 gloss alternatives       │
│ • Top-3 category alternatives    │
│ • Probability bars               │
└────────────┬─────────────────────┘
             ↓
┌────────────┴─────────────┬───────┐
│                          │       │
▼                          ▼       ▼
┌──────────────┐  ┌──────────────┐ ┌──────────────┐
│ Keypoint     │  │ Feature      │ │ Statistical  │
│ Animation    │  │ Analysis     │ │ Analysis     │
├──────────────┤  ├──────────────┤ ├──────────────┤
│• Skeleton    │  │• Trajectory  │ │• Mean/Std    │
│  overlay     │  │  plots       │ │• Min/Max     │
│• Frame slider│  │• Heatmaps    │ │• Range       │
│• Play/pause  │  │• Line charts │ │• Per body    │
│• FPS control │  │• Body-part   │ │  part        │
│• Video export│  │  breakdown   │ │• Temporal    │
└──────────────┘  └──────────────┘ └──────────────┘
```

### Visualization Components

**Function**: `render_consolidated_file_info(filename, npz_data, metadata, seq_length=150)`

- Displays: File details, compatibility badges, occlusion status, sequence overview chart
- Handles padding/truncation to sequence length
- Returns: `(X_pad, mask, meta_dict)` for downstream visualization

**Function**: `render_animated_keypoints(X_pad, mask, key_suffix='', meta_dict=None)`

- Features: Frame-by-frame skeleton overlay, play/pause controls with adjustable FPS, interactive slider
- Body part colors: Pose (Red), Left hand (Blue), Right hand (Green), Face (Orange)
- Includes video animation export capability

**Function**: `render_skeleton_frame(X_frame, mask_frame)`

- Renders skeleton for single frame
- Plots keypoints with visibility indicators
- Adds pose skeleton connections

**Function**: `render_feature_charts(X_pad, mask, key_suffix='')`

- Trajectory plots over time for selected body part
- Temporal heatmaps (frame × feature dimension)
- Statistical summaries (mean, std, min, max)
- Body part selector: Pose, Left Hand, Right Hand, Face

**Function**: `generate_keypoint_video(X_pad, mask, fps=15)`

- Generates MP4 video of keypoint animation
- Returns: video bytes for download

### Export Options

**Function**: `export_prediction_results(filename, npz_data, prediction_results, formatted_results)`

- **JSON**: Complete prediction results with metadata
- **CSV**: Summary table (filename, predictions, confidence)
- **NPZ**: Download processed NPZ file

**Function**: `create_batch_download(summary_data)`

- Creates ZIP archive with:
  - All NPZ files
  - summary_table.csv
  - predictions.json (detailed results)

---

## Stage 6: Model Validation & Evaluation

### Overview

Comprehensive model evaluation on validation datasets with detailed performance metrics, confusion matrices, and occlusion analysis.

### Components

- **Primary Controller**: `streamlit_app/core/main.py` → `render_validation_stage()`
- **Validation Manager**: `streamlit_app/manager/validation_manager.py`
- **Validation Engine**: `evaluation/validation/validate.py`
- **UI Components**: `streamlit_app/components/validation_components.py`

### Validation Architecture

```
Validation Dataset (NPZ folder + Labels CSV)
         ↓
┌────────────────────────────────────┐
│ ValidationDataset                  │
│ • Load labels CSV                  │
│ • Filter existing files            │
│ • Map file → (gloss, cat, occluded)│
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ ModelValidator                     │
│ • Load model + checkpoint          │
│ • Batch inference with progress    │
│ • Collect predictions              │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Metrics Computation                │
├────────────────────────────────────┤
│ Overall Metrics                    │
│ • Accuracy, Precision, Recall, F1  │
│ • For gloss (105 classes)          │
│ • For category (10 classes)        │
├────────────────────────────────────┤
│ Occlusion-Based Metrics            │
│ • Occluded samples only            │
│ • Non-occluded samples only        │
│ • Performance comparison           │
├────────────────────────────────────┤
│ Per-Class Metrics                  │
│ • Precision/Recall/F1 per class    │
│ • Support counts                   │
│ • Identify difficult classes       │
├────────────────────────────────────┤
│ Confusion Matrices                 │
│ • Gloss confusion [105×105]        │
│ • Category confusion [10×10]       │
│ • Error pattern analysis           │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Results Display & Export           │
│ • Metrics dashboard                │
│ • Confusion heatmaps               │
│ • Per-class breakdown              │
│ • Download JSON/CSV                │
└────────────────────────────────────┘
```

### Key Classes and Methods

**Class**: `ValidationDataset`

- **Init**: `__init__(data_dir, labels_csv, model_type)`
- **Methods**: `__len__()`, `__getitem__(idx)`
- **Purpose**: Efficient validation data loading with label mapping
- **Features**: Filters existing files, handles encoding, maps file → (gloss, cat, occluded)

**Class**: `ModelValidator`

- **Init**: `__init__(model_type, checkpoint_path, device='auto')`
- **Methods**: `validate(dataset, batch_size=32, progress_callback=None)`, `_compute_metrics()`, `_load_model()`, `_load_checkpoint()`
- **Purpose**: Comprehensive model evaluation engine
- **Features**: Batch inference, progress tracking, comprehensive metrics computation

### Validation Results

**Method**: `ModelValidator.validate(dataset, batch_size=32)`

- Returns results dictionary with:
  - `model_info`: Model type, checkpoint path, device, timestamp
  - `dataset_info`: Total samples, occluded/non-occluded counts
  - `overall_results`: Accuracy, Precision, Recall, F1 for gloss and category
  - `occluded_results`: Metrics for occluded samples only
  - `non_occluded_results`: Metrics for non-occluded samples only
  - `per_class_results`: Per-class metrics for gloss and category
  - `confusion_matrices`: Gloss [105×105] and category [10×10] confusion matrices
  - `detailed_predictions`: List of all predictions with probabilities

### UI Components

**Function**: `render_validation_summary(results)`

- Displays: Total samples, gloss accuracy, category accuracy, F1-score
- Occlusion comparison table: Occluded vs Non-occluded performance

**Function**: `render_validation_results(results)`

- Tabs: Confusion Matrices, Per-Class Performance, Error Analysis

**Function**: `render_confusion_matrices(results)`

- Gloss confusion matrix heatmap (105×105)
- Category confusion matrix heatmap (10×10)

**Function**: `render_download_results(results)`

- Download complete results as JSON
- Download confusion matrices as CSV
- Download per-class metrics as CSV

---

## Data Flow & State Management

### Complete Data Flow

```
USER INTERACTION
    ↓
┌─────────────────┐
│ File Upload     │ → st.file_uploader() → UploadedFile objects
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ FILE ROUTING                                                        │
│ detect_file_type(file) → 'npz' or 'video'                           │
│                                                                     │
│ IF npz:                        IF video:                            │
│   → st.session_state.npz_files    → st.session_state.video_files    │
│   → workflow_stage='predictions'  → workflow_stage='preprocessing'  │
└────────┬────────────────────────────────────┬───────────────────────┘
         │                                    │
         │                                    ↓
         │                           ┌──────────────────────┐
         │                           │ PREPROCESSING        │
         │                           │ process_videos()     │
         │                           │ → NPZ data           │
         │                           └──────────┬───────────┘
         │                                      │
         └──────────────────────────────────────┘
                                   ↓
                          ┌─────────────────┐
                          │ NPZ DATA        │
                          │ X: [T, 156]     │
                          │ X2048: [T,2048] │
                          │ mask: [T, 78]   │
                          │ timestamps: [T] │
                          │ meta: JSON      │
                          └────────┬────────┘
                                   ↓
                          ┌──────────────────┐
                          │ VALIDATION       │
                          │ validate_shapes()│
                          │ check_compat()   │
                          └────────┬─────────┘
                                   ↓
                    ┌──────────────┴──────────────┐
                    │ SESSION STATE STORAGE       │
                    │ processed_data[filename]    │
                    │ file_metadata[filename]     │
                    │ file_status[filename]       │
                    └──────────────┬──────────────┘
                                   ↓
                          ┌──────────────────┐
                          │ PREDICTION       │
                          │ ModelManager     │
                          │ → get_model()    │
                          │ → predict()      │
                          │ → format()       │
                          └────────┬─────────┘
                                   ↓
                          ┌──────────────────┐
                          │ RESULTS          │
                          │ gloss_prediction │
                          │ category_pred    │
                          │ top-5 glosses    │
                          │ top-3 categories │
                          └────────┬─────────┘
                                   ↓
                    ┌──────────────┴──────────────┐
                    │                             │
                    ↓                             ↓
         ┌───────────────────┐         ┌───────────────────┐
         │ VISUALIZATION     │         │ EXPORT            │
         │ • Skeleton anim   │         │ • JSON results    │
         │ • Trajectories    │         │ • CSV summaries   │
         │ • Heatmaps        │         │ • ZIP archives    │
         └───────────────────┘         └───────────────────┘
```

### Workflow State Transitions

```
UPLOAD → PREPROCESSING → PREDICTIONS → VALIDATION
  ↑          ↓              ↓             ↑
  └──────────┴──────────────┴─────────────┘
         (Navigation Buttons)

State Variable: st.session_state.workflow_stage
Values: 'upload', 'preprocessing', 'predictions', 'validation'
```

**Transition Rules**:

| From          | To            | Condition                                                     | Action                           |
| ------------- | ------------- | ------------------------------------------------------------- | -------------------------------- |
| Upload        | Preprocessing | has_video_files == True                                       | workflow_stage = 'preprocessing' |
| Upload        | Predictions   | has_npz_files == True AND has_video_files == False            | workflow_stage = 'predictions'   |
| Preprocessing | Predictions   | User clicks "Go to Inference" AND has_pending_videos == False | workflow_stage = 'predictions'   |
| Predictions   | Preprocessing | User clicks "← Back" AND has_video_files == True              | workflow_stage = 'preprocessing' |
| Predictions   | Upload        | User clicks "← Back" AND has_video_files == False             | workflow_stage = 'upload'        |
| Any Stage     | Validation    | User clicks "Model Validation" in sidebar                     | workflow_stage = 'validation'    |

### File Status State Machine

```
┌─────────┐
│ pending │ ← Initial state after upload
└────┬────┘
     │ User clicks "Process/Preprocess"
     ↓
┌────────────┐
│ processing │ ← During processing
└────┬───┬───┘
     │   │
 Success  Error
     │   │
     ↓   ↓
┌───────────┐  ┌───────┐
│ completed │  │ error │
└───────────┘  └───┬───┘
     │             │
     │    User clicks "Retry"
     │             │
     └─────────────┘
           ↓
     ┌────────────┐
     │ processing │
     └────────────┘
```

---

## Error Handling

### Error Detection Checkpoints

| Stage                       | Error Types                                                                                                      | Detection                                | Recovery                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| **Upload**                  | File size >500MB, Unsupported format, Too many files, Corrupted upload                                           | Size check, Extension/MIME check         | Error message, Reject upload                                                         |
| **Preprocessing**           | Video codec unsupported, Frame extraction failure, MediaPipe init error, InceptionV3 CUDA OOM, NPZ write failure | Exception catch, OpenCV/MediaPipe errors | Convert codec, Skip frames, Continue with zeros, Reduce batch size, Check disk space |
| **Validation**              | NPZ structure invalid, Shape mismatch, NaN/Inf values, Incompatible with model, Metadata parsing error           | Shape checks, Content validation         | Re-upload, Show specific error, Detailed error location, Filter compatible models    |
| **Prediction**              | Model loading failure, Checkpoint not found, State dict mismatch, CUDA OOM, Feature extraction error             | Load exceptions, State dict check        | Show error, Check checkpoint, Automatic retry with CPU                               |
| **Validation (Evaluation)** | Labels CSV not found, File-label mismatch, Batch processing error, Metrics computation failure                   | File checks, Label validation            | Show error, Skip missing files, Retry batch                                          |

### Error Recovery Strategies

| Error Type                 | Detection             | Recovery                       | User Feedback                          |
| -------------------------- | --------------------- | ------------------------------ | -------------------------------------- |
| **File too large**         | Upload size check     | Reject upload                  | Error message + suggestion to compress |
| **Unsupported format**     | Extension/MIME check  | Reject file                    | Warning toast                          |
| **Processing failure**     | Exception catch       | Mark as error, allow retry     | Error status + retry button            |
| **Validation failure**     | Shape/content checks  | Mark as error, show details    | Specific error message                 |
| **Model load failure**     | Checkpoint load       | Use dummy/show error           | Toast notification                     |
| **CUDA OOM**               | CUDA memory exception | Reduce batch size, retry       | Auto-retry with smaller batch          |
| **Temporary file cleanup** | Permission error      | Retry with delay (3× attempts) | Silent (cleanup is non-critical)       |

---

## Configuration & Performance

### Key Configuration Points

**File**: `streamlit_app/core/config.py`

```python
# Model Configuration
MODEL_CONFIG = {
    'transformer': {
        'checkpoint_path': 'trained_models/transformer/optimal/SignTransformer_best.pt',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'supports_keypoints': True,
        'supports_features': True
    },
    'iv3_gru': {
        'checkpoint_path': 'trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt',
        'num_gloss_classes': 105,
        'num_category_classes': 10,
        'input_dim': 2048,
        'supports_keypoints': False,
        'supports_features': True
    }
}

# Processing Configuration
PROCESSING_CONFIG = {
    'video': {
        'target_fps': 30,           # Frame sampling rate
        'out_size': 256,            # Frame resize dimension
        'conf_thresh': 0.5,         # MediaPipe confidence threshold
        'max_gap': 5,               # Max gap for interpolation
    },
    'npz': {
        'sequence_length': 150,     # Pad/truncate length
        'keypoint_dim': 156,
        'feature_dim': 2048
    }
}
```

**File**: `.streamlit/config.toml`

```toml
[server]
maxUploadSize = 500        # Max file size in MB
maxMessageSize = 500       # Max WebSocket message size
enableCORS = true          # Enable mobile support
enableWebsocketCompression = true

[browser]
gatherUsageStats = false
```

### Performance Optimizations

**1. Model Caching (Singleton Pattern)**

- Implementation: `ModelManager` class
- Benefits: First prediction ~5-10s, subsequent ~100-500ms
- Memory: Single instance per model

**2. Batch Processing (Multi-Processing)**

- Implementation: `process_videos_multiprocess()`
- Benefits: 30-50x speedup for video preprocessing
- Workers: Auto-calculated based on CPU/RAM/GPU

**3. GPU Acceleration**

- Implementation: Automatic CUDA detection
- Benefits: 10-100x speedup for InceptionV3, 5-10x for inference
- Batch optimization for GPU memory

**4. Dynamic Resource Optimization**

- Implementation: `get_dynamic_resource_info()`, `calculate_optimal_workers()`
- Benefits: Prevent OOM, optimal performance per platform
- Logic: Conservative choice based on memory and CPU availability

**5. NPZ Compression**

- Implementation: `np.savez_compressed()`
- Benefits: 3-5x file size reduction, faster I/O
- Typical sizes: 50-200 KB → 10-50 KB

**6. Pagination**

- Implementation: 5 files per page
- Benefits: Fast rendering with 100+ files, responsive UI

**7. Streamlit Caching**

- Implementation: `@st.cache_data`, `@st.cache_resource`
- Benefits: Instant page refreshes, reduced redundant computation

### Performance Summary

| Operation                           | Sequential | Optimized               | Speedup |
| ----------------------------------- | ---------- | ----------------------- | ------- |
| **Model Loading**                   | 5-10s      | 100-500ms (cached)      | 10-100x |
| **Video Preprocessing**             | 45-60s     | 5-8s (GPU, parallel)    | 6-12x   |
| **Batch Preprocessing (10 videos)** | 450-600s   | 60-90s                  | 5-10x   |
| **Feature Extraction**              | 30-45s     | 3-5s (GPU, batched)     | 6-9x    |
| **Model Inference**                 | 2-5s       | 100-500ms (GPU, cached) | 4-50x   |
| **NPZ File Size**                   | 50-200 KB  | 10-50 KB (compressed)   | 3-5x    |

---

## Summary

### Pipeline Capabilities

✅ **Complete End-to-End Pipeline**: Upload → Preprocess → Validate → Predict → Visualize  
✅ **Dual Model Support**: Transformer + IV3-GRU with automatic compatibility  
✅ **Flexible Input**: NPZ (preprocessed) or Video (raw)  
✅ **GPU Acceleration**: Automatic CUDA utilization  
✅ **Batch Processing**: Parallel multi-processing  
✅ **Comprehensive Validation**: Structure, shape, content, compatibility checks  
✅ **Rich Visualization**: Skeleton animation, trajectories, heatmaps  
✅ **Model Evaluation**: Metrics, confusion matrices, occlusion analysis  
✅ **Export Options**: JSON, CSV, ZIP, video animations  
✅ **Error Recovery**: Checkpoints, retry mechanisms, clear feedback  
✅ **Performance Optimized**: Caching, batching, dynamic resources

### Key Strengths

1. **Modularity**: Each stage is independent and reusable
2. **Robustness**: Comprehensive error handling and recovery
3. **Performance**: 30-100x speedups through optimization
4. **Usability**: Intuitive UI with clear workflow
5. **Flexibility**: Supports multiple models and input formats
6. **Scalability**: Dynamic resource optimization

---

**Document Status**: Consolidated Complete  
**Last Updated**: October 12, 2025  
**Total Pipeline Stages**: 6  
**Total Components**: 50+  
**Performance Optimizations**: 7

**Source Documents**:

- `pansinayan_pipeline.md` (Stages 1-4)
- `pansinayan_pipeline_v2.md` (Stages 5-6 + System Details)

**Related Documents**:

- System Architecture: `system_archi_analysis.md`
- Tool Guide: `streamlit_app/TOOL_GUIDE.md`
- README: `README.md`
