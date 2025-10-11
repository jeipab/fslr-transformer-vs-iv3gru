# PANSINAYAN Complete Pipeline Documentation

## Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Stage 1: Upload & Input Handling](#stage-1-upload--input-handling)
- [Stage 2: Preprocessing](#stage-2-preprocessing)
- [Stage 3: Data Validation](#stage-3-data-validation)
- [Stage 4: Prediction & Inference](#stage-4-prediction--inference)
- [Stage 5: Results & Visualization](#stage-5-results--visualization)
- [Stage 6: Model Validation & Evaluation](#stage-6-model-validation--evaluation)
- [Data Flow Architecture](#data-flow-architecture)
- [State Management](#state-management)
- [Error Handling](#error-handling)
- [Configuration Points](#configuration-points)
- [Performance Optimizations](#performance-optimizations)

---

## Pipeline Overview

PANSINAYAN implements a 6-stage pipeline for Filipino Sign Language Recognition:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PANSINAYAN PIPELINE FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

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
             └─── Video Files                     │
                     ↓                             │
Stage 2: PREPROCESSING (Video → Features)         │
┌──────────────────────────────────────┐          │
│ • MediaPipe keypoint extraction      │          │
│ • InceptionV3 feature extraction     │          │
│ • Occlusion detection                │          │
│ • Multi-process GPU acceleration     │          │
│ • NPZ generation with metadata       │          │
└────────────┬─────────────────────────┘          │
             │                                     │
             └────── NPZ Files ───────────────────┘
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
│ • Results export & download          │
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

**Primary Controller**: `streamlit_app/manager/upload_manager.py`

**Key Functions**:
```python
initialize_upload_session_state()  # Initialize session state variables
render_upload_stage()              # Main upload UI
route_files_to_stages()            # Route by file type
proceed_to_next_stage()            # Workflow transition
```

### Supported Input Formats

| Format | Extensions | Processing Path | Use Case |
|--------|-----------|-----------------|----------|
| **NPZ** | .npz | Direct to Inference | Pre-processed data |
| **Video** | .mp4, .mov, .avi | Preprocessing → Inference | Raw video clips |
| **Demo** | Located in `data/demo/` | Direct to Inference | Testing & examples |

### Upload Configuration

**File**: `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 500        # Maximum file size in MB
maxMessageSize = 500       # Maximum WebSocket message size
enableCORS = true          # Enable mobile browser support
enableWebsocketCompression = true  # Better mobile performance
```

### Session State Initialization

```python
# Core session state variables initialized on upload
st.session_state = {
    # File lists
    'uploaded_files': [],           # All uploaded files
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

**Step 1: File Upload**
```python
# streamlit_app/manager/upload_manager.py
def render_upload_stage():
    """Main upload interface."""
    uploaded_files = render_file_upload()  # Max 10 files
    
    if uploaded_files and len(uploaded_files) > 10:
        st.error("Maximum 10 files allowed.")
        return
```

**Step 2: File Type Detection**
```python
# streamlit_app/components/utils.py
def detect_file_type(uploaded_file):
    """Detect file type from extension and MIME."""
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.npz'):
        return 'npz'
    elif filename.endswith(('.mp4', '.mov', '.avi', '.webm')):
        return 'video'
    else:
        return 'unknown'
```

**Step 3: File Routing**
```python
def route_files_to_stages(uploaded_files):
    """Route files to appropriate stages based on type."""
    npz_files = []
    video_files = []
    
    for uploaded_file in uploaded_files:
        file_type = detect_file_type(uploaded_file)
        if file_type == 'npz':
            npz_files.append(uploaded_file)
        elif file_type == 'video':
            video_files.append(uploaded_file)
    
    # Store in session state
    st.session_state.npz_files = npz_files
    st.session_state.video_files = video_files
    
    return npz_files, video_files
```

**Step 4: Workflow Transition**
```python
def proceed_to_next_stage():
    """Determine next stage based on file types."""
    npz_files = st.session_state.npz_files
    video_files = st.session_state.video_files
    
    if npz_files and not video_files:
        # Only NPZ → go directly to predictions
        st.session_state.workflow_stage = 'predictions'
    elif video_files:
        # Has videos → must preprocess first
        st.session_state.workflow_stage = 'preprocessing'
    
    # Set all files to pending status
    for file in st.session_state.uploaded_files:
        st.session_state.file_status[file.name] = 'pending'
    
    st.rerun()
```

### UI Features

**File Display**:
- **NPZ Files**: Compact cards in columns (3-4 per row)
- **Video Files**: Carousel with preview thumbnails
- **File Info**: Name, size, type indicator

**Navigation**:
- **Proceed Button**: Dynamic text based on file types
  - "Proceed to Preprocessing" (if videos)
  - "Proceed to Inference" (if only NPZ)
- **File Summary**: Total files, NPZ count, video count

### Error Handling

| Error Condition | Response | Recovery |
|----------------|----------|----------|
| Too many files (>10) | Error message, reject upload | Remove excess files |
| Unsupported format | Warning toast | Upload valid formats |
| File too large (>500MB) | Error message | Compress or split file |
| Corrupted file | Validation error in next stage | Re-upload file |

### Example Usage

```python
# User uploads files
uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    type=['npz', 'mp4', 'mov', 'avi']
)

# System processes upload
if uploaded_files:
    route_files_to_stages(uploaded_files)
    # Display proceed button based on file types
    if st.button("Proceed to Next Stage"):
        proceed_to_next_stage()
```

---

## Stage 2: Preprocessing

### Overview
Converts raw video files into feature representations suitable for model inference. Extracts both keypoint-based (156-D) and visual (2048-D) features with automatic occlusion detection.

### Components

**Primary Controller**: `streamlit_app/manager/preprocessing_manager.py`

**Core Processor**: `preprocessing/core/preprocess.py`

**Feature Extractors**:
- `preprocessing/extractors/keypoints_features.py` - MediaPipe keypoints
- `preprocessing/extractors/iv3_features.py` - InceptionV3 CNN features

**Occlusion Detection**: `preprocessing/core/occlusion_detection.py`

### Processing Architecture

```
Video File (.mp4, .mov, .avi)
         ↓
┌────────────────────────────────┐
│ 1. Video Loading & Frame       │
│    Extraction (OpenCV)          │
│    • Target FPS sampling        │
│    • Frame resizing (256×256)   │
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ 2. Parallel Feature Extraction │
│    (Multi-process/GPU)          │
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
│ 3. Post-Processing              │
│    • Gap interpolation (≤5)     │
│    • Occlusion detection        │
│    • Metadata generation        │
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ 4. NPZ File Generation          │
│    • X: [T, 156] keypoints      │
│    • X2048: [T, 2048] features  │
│    • mask: [T, 78] visibility   │
│    • timestamps_ms: [T]         │
│    • meta: JSON metadata        │
└────────────────────────────────┘
```

### Feature Extraction Details

#### MediaPipe Keypoints (156-D)

**Source**: `preprocessing/extractors/keypoints_features.py`

**Components**:
```python
# Keypoint distribution
POSE_UPPER_25 = 25 points    # Upper body pose
N_HAND = 21 points × 2       # Left and right hands = 42 points
FACEMESH_11 = 11 points      # Key facial landmarks
# Total: 78 keypoints × 2 coords (x, y) = 156 dimensions
```

**Extraction Process**:
```python
def extract_keypoints_from_frame(frame, mp_models):
    """
    Extract keypoints from single frame using MediaPipe Holistic.
    
    Returns:
        X: [156] flattened x,y coordinates
        mask: [78] visibility mask (True if confident)
    """
    # Process frame through MediaPipe
    results = mp_models.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Extract pose (upper body only)
    pose_xy = extract_pose_upper(results.pose_landmarks)  # [25, 2]
    
    # Extract hands
    left_hand_xy = extract_hand(results.left_hand_landmarks)   # [21, 2]
    right_hand_xy = extract_hand(results.right_hand_landmarks) # [21, 2]
    
    # Extract face
    face_xy = extract_face(results.face_landmarks)  # [11, 2]
    
    # Concatenate and flatten
    X = np.concatenate([pose_xy, left_hand_xy, right_hand_xy, face_xy])  # [78, 2]
    X_flat = X.reshape(-1)  # [156]
    
    # Create visibility mask
    mask = (X[:, 0] != 0) & (X[:, 1] != 0)  # [78]
    
    return X_flat, mask
```

**Gap Interpolation**:
```python
def interpolate_gaps(X, mask, max_gap=5):
    """
    Fill short gaps (≤5 frames) in keypoint sequences.
    Uses linear interpolation for temporal continuity.
    """
    for kp_idx in range(78):
        # Find gaps in this keypoint
        gaps = find_gaps(mask[:, kp_idx])
        
        for gap_start, gap_end in gaps:
            gap_length = gap_end - gap_start
            
            if gap_length <= max_gap and gap_start > 0 and gap_end < len(X):
                # Linear interpolation
                X[gap_start:gap_end, kp_idx*2:(kp_idx+1)*2] = interpolate(
                    X[gap_start-1, kp_idx*2:(kp_idx+1)*2],
                    X[gap_end, kp_idx*2:(kp_idx+1)*2],
                    gap_length
                )
                mask[gap_start:gap_end, kp_idx] = True
```

#### InceptionV3 Features (2048-D)

**Source**: `preprocessing/extractors/iv3_features.py`

**Model**: Pretrained InceptionV3 (ImageNet), frozen backbone

**Extraction Process**:
```python
class BatchedInceptionV3Processor:
    """Batch processor for efficient GPU feature extraction."""
    
    def __init__(self, device='cuda', batch_size=32):
        # Load pretrained InceptionV3
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Identity()  # Remove classifier, keep features
        self.model.eval()
        self.model.to(device)
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_batch(self, frames):
        """
        Extract features from batch of frames.
        
        Args:
            frames: List of numpy arrays [H, W, 3]
        
        Returns:
            features: [B, 2048] feature vectors
        """
        # Preprocess frames
        tensors = [self.transform(Image.fromarray(f)) for f in frames]
        batch = torch.stack(tensors).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch)  # [B, 2048]
        
        return features.cpu().numpy()
```

#### Occlusion Detection

**Source**: `preprocessing/core/occlusion_detection.py`

**Two-Level Detection**:

**1. Frame-Level Occlusion**:
```python
def detect_frame_occlusion(mask, conf_thresh=0.6):
    """
    Detect occlusion in individual frames based on keypoint visibility.
    
    Args:
        mask: [T, 78] visibility mask
        conf_thresh: Minimum ratio of visible keypoints (default: 60%)
    
    Returns:
        frame_occluded: [T] binary mask (1 = occluded)
    """
    visible_ratio = mask.sum(axis=1) / mask.shape[1]  # [T]
    frame_occluded = (visible_ratio < conf_thresh).astype(int)
    
    return frame_occluded
```

**2. Clip-Level Occlusion**:
```python
def compute_occlusion_detection(X, mask, 
                               conf_thresh=0.6,
                               consec_thresh=15,
                               frac_thresh=0.4):
    """
    Compute clip-level occlusion flag based on two conditions.
    
    Clip is considered occluded if:
    1. ≥40% of frames are occluded, OR
    2. ≥15 consecutive frames are occluded
    
    Returns:
        result: {
            'occluded_flag': 0 or 1,
            'frame_occluded_mask': [T] (if detailed=True),
            'conf_thresh': float,
            'consec_thresh': int,
            'frac_thresh': float
        }
    """
    # Frame-level occlusion
    frame_occluded = detect_frame_occlusion(mask, conf_thresh)
    
    # Condition 1: Fraction threshold
    occluded_frac = frame_occluded.sum() / len(frame_occluded)
    condition1 = occluded_frac >= frac_thresh
    
    # Condition 2: Consecutive frames threshold
    max_consecutive = find_max_consecutive_run(frame_occluded)
    condition2 = max_consecutive >= consec_thresh
    
    # Clip-level flag
    occluded_flag = 1 if (condition1 or condition2) else 0
    
    return {
        'occluded_flag': occluded_flag,
        'frame_occluded_mask': frame_occluded.tolist(),
        'conf_thresh': conf_thresh,
        'consec_thresh': consec_thresh,
        'frac_thresh': frac_thresh
    }
```

### Processing Modes

#### Single Video Processing

```python
def preprocess_single_video(uploaded_file, filename):
    """Process single video with default options."""
    options = {
        'target_fps': 30,              # Frame sampling rate
        'out_size': 256,               # Frame resize dimension
        'write_keypoints': True,       # Extract MediaPipe keypoints
        'write_iv3_features': True,    # Extract InceptionV3 features
        'occ_detailed': False          # Include detailed occlusion metrics
    }
    
    # Process through unified pipeline
    with st.spinner(f"Preprocessing {filename}..."):
        processed_results = process_videos_unified(
            [uploaded_file],
            target_fps=options['target_fps'],
            out_size=options['out_size'],
            write_keypoints=options['write_keypoints'],
            write_iv3_features=options['write_iv3_features'],
            occ_detailed=options['occ_detailed']
        )
        
        npz_data = processed_results.get(Path(filename).stem, {})
    
    # Check compatibility
    compatibility = check_npz_compatibility(npz_data)
    
    # Store in session state
    st.session_state.processed_data[filename] = npz_data
    st.session_state.file_status[filename] = 'completed'
```

#### Batch Multi-Processing

```python
def preprocess_multiple_videos_batch(uploaded_files):
    """Process multiple videos with automatic resource optimization."""
    
    # Get real-time resource info
    resource_info = get_dynamic_resource_info()
    
    # Calculate optimal parameters
    optimal_workers, processing_type = calculate_optimal_workers(
        resource_info, 
        len(uploaded_files)
    )
    optimal_batch_size = calculate_optimal_batch_size(
        resource_info, 
        processing_type
    )
    
    # Process with optimized settings
    with st.spinner(f"Preprocessing {len(uploaded_files)} videos..."):
        processed_results = process_videos_unified(
            uploaded_files,
            target_fps=30,
            out_size=256,
            write_keypoints=True,
            write_iv3_features=True,
            workers=optimal_workers,
            batch_size=optimal_batch_size
        )
    
    # Store all results
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        basename = Path(filename).stem
        
        if basename in processed_results:
            npz_data = processed_results[basename]
            st.session_state.processed_data[filename] = npz_data
            st.session_state.file_status[filename] = 'completed'
```

### Resource Optimization

**Dynamic Resource Detection**:
```python
def get_dynamic_resource_info():
    """Get real-time system resources."""
    return {
        'cpu_count': mp.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_memory_free': {
            i: (torch.cuda.get_device_properties(i).total_memory - 
                torch.cuda.memory_allocated(i)) / (1024**3)
            for i in range(torch.cuda.device_count())
        } if torch.cuda.is_available() else {}
    }
```

**Optimal Worker Calculation**:
```python
def calculate_optimal_workers(resource_info, video_count):
    """Calculate optimal number of workers based on resources."""
    
    # Memory-based limit (2.5GB per video conservative estimate)
    memory_workers = max(1, int(resource_info['memory_available_gb'] / 2.5))
    
    # CPU-based limit (leave some CPU free)
    cpu_workers = max(1, int(
        resource_info['cpu_count'] * (100 - resource_info['cpu_percent']) / 100
    ))
    
    # Conservative choice
    max_workers = min(memory_workers, cpu_workers, resource_info['cpu_count'], 8)
    
    # GPU processing if available
    if resource_info['cuda_available'] and resource_info['gpu_count'] > 0:
        gpu_workers = min(resource_info['gpu_count'], max_workers)
        return gpu_workers, 'gpu'
    else:
        cpu_workers = min(max_workers, video_count)
        return cpu_workers, 'cpu'
```

### NPZ Output Format

```python
# Generated NPZ file structure
npz_data = {
    # Core features
    'X': np.ndarray,              # [T, 156] MediaPipe keypoints
    'X2048': np.ndarray,          # [T, 2048] InceptionV3 features
    'mask': np.ndarray,           # [T, 78] Keypoint visibility mask
    'timestamps_ms': np.ndarray,  # [T] Frame timestamps in milliseconds
    
    # Metadata (JSON string)
    'meta': json.dumps({
        'target_fps': 30,
        'out_size': 256,
        'conf_thresh': 0.5,
        'max_gap': 5,
        'model_type': 'B',        # 'T' (Transformer), 'I' (IV3-GRU), 'B' (Both)
        'occluded_flag': 0,       # 0 (clean) or 1 (occluded)
        'occlusion_params': {
            'conf_thresh': 0.6,
            'consec_thresh': 15,
            'frac_thresh': 0.4
        },
        'source_video': 'clip_0001.mp4',
        'processing_date': '2025-10-11T12:00:00'
    })
}
```

### UI Features

**File List Display**:
- Status indicators: ⏳ Pending, 🔄 Processing, ✅ Completed, ❌ Error
- Individual action buttons: Preprocess, View, Retry, Remove
- Download button for completed files

**Batch Operations**:
- "Preprocess All Pending" - Process all pending videos
- "Reset All" - Reset completed files to pending
- "Clear All" - Remove all files from stage
- "Download All" - ZIP archive with all NPZ files

**Progress Tracking**:
- Per-file processing status
- Overall batch progress
- Real-time completion notifications

### Error Handling

| Error Type | Detection | Recovery |
|-----------|-----------|----------|
| Video codec unsupported | OpenCV load failure | Convert to H.264 |
| Frame extraction failure | Empty frame buffer | Skip corrupted frames |
| MediaPipe failure | No landmarks detected | Continue with zeros |
| InceptionV3 OOM | CUDA out of memory | Reduce batch size |
| NPZ write failure | Disk full/permissions | Check disk space |

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

**NPZ Validation**: `preprocessing/utils/validate_npz.py`

**Data Processing Validation**: `streamlit_app/components/data_processing.py`

**Compatibility Checks**: `streamlit_app/components/utils.py`

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

#### File Structure Validation

```python
def validate_npz_structure(npz_path):
    """
    Validate basic NPZ file structure.
    
    Checks:
    - File exists and is readable
    - NPZ can be loaded
    - Has at least one required key (X or X2048)
    
    Returns:
        (valid: bool, error_message: str)
    """
    try:
        # Check file exists
        if not os.path.exists(npz_path):
            return False, f"File not found: {npz_path}"
        
        # Try to load NPZ
        data = np.load(npz_path, allow_pickle=True)
        
        # Check for required keys
        has_keypoints = 'X' in data
        has_features = 'X2048' in data
        
        if not (has_keypoints or has_features):
            return False, "Missing both 'X' and 'X2048' keys"
        
        return True, ""
        
    except Exception as e:
        return False, f"Failed to load NPZ: {str(e)}"
```

#### Shape Validation

```python
def validate_shapes(npz_data, require_x2048=False):
    """
    Validate tensor shapes in NPZ data.
    
    Expected shapes:
    - X: [T, 156] keypoints
    - X2048: [T, 2048] features
    - mask: [T, 78] visibility
    - timestamps_ms: [T] timestamps
    
    Returns:
        (valid: bool, errors: List[str])
    """
    errors = []
    
    # Check X (keypoints) if present
    if 'X' in npz_data:
        X = npz_data['X']
        if X.ndim != 2:
            errors.append(f"X should be 2D, got {X.ndim}D")
        elif X.shape[1] != 156:
            errors.append(f"X should have 156 features, got {X.shape[1]}")
    
    # Check X2048 (features)
    if 'X2048' in npz_data:
        X2048 = npz_data['X2048']
        if X2048.ndim != 2:
            errors.append(f"X2048 should be 2D, got {X2048.ndim}D")
        elif X2048.shape[1] != 2048:
            errors.append(f"X2048 should have 2048 features, got {X2048.shape[1]}")
    elif require_x2048:
        errors.append("X2048 required but not found")
    
    # Check mask if present
    if 'mask' in npz_data:
        mask = npz_data['mask']
        if mask.ndim != 2:
            errors.append(f"mask should be 2D, got {mask.ndim}D")
        elif mask.shape[1] != 78:
            errors.append(f"mask should have 78 keypoints, got {mask.shape[1]}")
    
    # Check sequence length consistency
    if 'X' in npz_data and 'X2048' in npz_data:
        if npz_data['X'].shape[0] != npz_data['X2048'].shape[0]:
            errors.append(
                f"Sequence length mismatch: X={npz_data['X'].shape[0]}, "
                f"X2048={npz_data['X2048'].shape[0]}"
            )
    
    return len(errors) == 0, errors
```

#### Content Validation

```python
def validate_content(npz_data):
    """
    Validate content values and ranges.
    
    Checks:
    - No NaN or Inf values
    - Keypoints in [0, 1] range (normalized)
    - Timestamps monotonically increasing
    
    Returns:
        (valid: bool, errors: List[str])
    """
    errors = []
    
    # Check X for NaN/Inf
    if 'X' in npz_data:
        X = npz_data['X']
        if np.isnan(X).any():
            errors.append(f"X contains NaN values: {np.isnan(X).sum()} positions")
        if np.isinf(X).any():
            errors.append(f"X contains Inf values: {np.isinf(X).sum()} positions")
        
        # Check normalized range [0, 1]
        if X.min() < -0.1 or X.max() > 1.1:
            errors.append(f"X values outside expected range [0,1]: [{X.min():.3f}, {X.max():.3f}]")
    
    # Check X2048 for NaN/Inf
    if 'X2048' in npz_data:
        X2048 = npz_data['X2048']
        if np.isnan(X2048).any():
            errors.append(f"X2048 contains NaN values: {np.isnan(X2048).sum()} positions")
        if np.isinf(X2048).any():
            errors.append(f"X2048 contains Inf values: {np.isinf(X2048).sum()} positions")
    
    # Check timestamps
    if 'timestamps_ms' in npz_data:
        timestamps = npz_data['timestamps_ms']
        if not np.all(np.diff(timestamps) >= 0):
            errors.append("Timestamps are not monotonically increasing")
    
    return len(errors) == 0, errors
```

#### Model Compatibility Validation

```python
def check_npz_compatibility(npz_data):
    """
    Check NPZ compatibility with models.
    
    Compatibility rules:
    - Transformer: Can use X (156-D) OR X2048 (2048-D)
    - IV3-GRU: Requires X2048 (2048-D)
    
    Returns:
        {
            'transformer': bool,
            'iv3_gru': bool
        }
    """
    has_keypoints = 'X' in npz_data
    has_features = 'X2048' in npz_data
    
    return {
        'transformer': has_keypoints or has_features,
        'iv3_gru': has_features
    }
```

#### Metadata Validation

```python
def validate_metadata(npz_data):
    """
    Validate metadata structure and values.
    
    Checks:
    - 'meta' key exists
    - JSON parseable
    - Required fields present
    - Valid values
    
    Returns:
        (valid: bool, errors: List[str], meta_dict: dict)
    """
    errors = []
    meta_dict = {}
    
    if 'meta' not in npz_data:
        return False, ["Missing 'meta' key"], {}
    
    try:
        # Parse JSON
        meta_str = npz_data['meta']
        if isinstance(meta_str, np.ndarray):
            meta_str = str(meta_str.item())
        
        meta_dict = json.loads(meta_str)
        
        # Check required fields
        required_fields = ['target_fps', 'out_size', 'model_type']
        for field in required_fields:
            if field not in meta_dict:
                errors.append(f"Missing required field: {field}")
        
        # Validate model_type
        if 'model_type' in meta_dict:
            if meta_dict['model_type'] not in ['T', 'I', 'B']:
                errors.append(f"Invalid model_type: {meta_dict['model_type']}")
        
        # Validate occlusion_flag if present
        if 'occluded_flag' in meta_dict:
            if meta_dict['occluded_flag'] not in [0, 1]:
                errors.append(f"Invalid occluded_flag: {meta_dict['occluded_flag']}")
        
    except json.JSONDecodeError as e:
        errors.append(f"Failed to parse JSON metadata: {str(e)}")
    except Exception as e:
        errors.append(f"Metadata validation error: {str(e)}")
    
    return len(errors) == 0, errors, meta_dict
```

### Validation in Pipeline

**After Upload (NPZ files)**:
```python
def process_single_npz_file(uploaded_file, filename):
    """Validate and load NPZ file."""
    try:
        st.session_state.file_status[filename] = 'processing'
        
        # Load NPZ file
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        file_bytes = io.BytesIO(file_content)
        npz_data = dict(np.load(file_bytes, allow_pickle=True))
        
        # Validate structure
        valid, error = validate_npz_structure_from_data(npz_data)
        if not valid:
            raise ValueError(f"Structure validation failed: {error}")
        
        # Validate shapes
        valid, errors = validate_shapes(npz_data)
        if not valid:
            raise ValueError(f"Shape validation failed: {', '.join(errors)}")
        
        # Validate content
        valid, errors = validate_content(npz_data)
        if not valid:
            raise ValueError(f"Content validation failed: {', '.join(errors)}")
        
        # Check compatibility
        compatibility = check_npz_compatibility(npz_data)
        if not any(compatibility.values()):
            raise ValueError("File incompatible with any model architecture")
        
        # Validate metadata
        valid, errors, meta_dict = validate_metadata(npz_data)
        if not valid:
            st.warning(f"Metadata validation warnings: {', '.join(errors)}")
        
        # Store processed data
        st.session_state.processed_data[filename] = npz_data
        st.session_state.file_metadata[filename] = {
            'compatibility': compatibility,
            'file_type': 'npz',
            'frame_count': npz_data['X'].shape[0] if 'X' in npz_data else npz_data['X2048'].shape[0],
            'meta': meta_dict
        }
        st.session_state.file_status[filename] = 'completed'
        
    except Exception as e:
        st.session_state.file_status[filename] = 'error'
        st.toast(f"Validation failed for {filename}: {str(e)}", icon="❌")
```

**After Preprocessing**:
```python
# Validation is automatic after preprocessing
npz_data = processed_results.get(basename, {})

# Check compatibility
compatibility = check_npz_compatibility(npz_data)
if not any(compatibility.values()):
    st.session_state.file_status[filename] = 'error'
    st.toast(f"{filename}: Incompatible output", icon="❌")
    return

# Store with metadata
st.session_state.file_metadata[filename] = {
    'compatibility': compatibility,
    'file_type': 'npz',
    'frame_count': npz_data['X'].shape[0] if 'X' in npz_data else npz_data['X2048'].shape[0],
    'source_type': 'video',
    'preprocessing_options': options
}
```

### Validation Results Display

**Compatibility Badges**:
```python
def render_compatibility_badges(compatibility):
    """Display model compatibility indicators."""
    col1, col2 = st.columns(2)
    
    with col1:
        if compatibility['transformer']:
            st.success("✓ Transformer Compatible")
        else:
            st.error("✗ Transformer Incompatible")
    
    with col2:
        if compatibility['iv3_gru']:
            st.success("✓ IV3-GRU Compatible")
        else:
            st.error("✗ IV3-GRU Incompatible")
```

### Error Recovery

| Validation Failure | User Action | System Response |
|-------------------|-------------|-----------------|
| Structure invalid | Re-upload correct NPZ | File marked as error |
| Shape mismatch | Check preprocessing options | Show specific error |
| Content NaN/Inf | Reprocess video | Detailed error location |
| Incompatible | Change model selection | Filter compatible models |
| Metadata missing | Continue with warnings | Non-critical, allow proceed |

---

## Stage 4: Prediction & Inference

### Overview
Load trained models and perform inference on validated NPZ files to predict glosses (105 classes) and categories (10 classes).

### Components

**Primary Controller**: `streamlit_app/manager/prediction_manager.py`

**Inference Engine**: `evaluation/prediction/predict.py`

**Model Architectures**:
- `models/transformer.py` - SignTransformer
- `models/iv3_gru.py` - InceptionV3GRU

**Checkpoints**:
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

#### ModelManager (Singleton Pattern)

```python
class ModelManager:
    """Singleton model manager for loading and caching prediction models."""
    
    _instance = None
    _models = {}          # Cache: {model_name: predictor}
    _label_mappings = None
    
    def __new__(cls):
        """Ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str):
        """Get a loaded model, loading it if necessary."""
        if model_name not in self._models:
            self._load_model(model_name)
        return self._models.get(model_name)
    
    def _load_model(self, model_name: str):
        """Load a model and cache it."""
        config = MODEL_CONFIG.get(model_name)
        if not config or not config['enabled']:
            return None
        
        try:
            from evaluation.prediction.predict import ModelPredictor
            
            # Load model predictor
            predictor = ModelPredictor(
                model_type=config['model_type'],
                checkpoint_path=config['checkpoint_path'],
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            # Cache the loaded model
            self._models[model_name] = predictor
            
            # Update config with detected input_dim
            if hasattr(predictor, 'input_dim') and predictor.input_dim is not None:
                update_model_input_dim(model_name, predictor.input_dim)
            
        except Exception as e:
            st.toast(f"Failed to load {model_name}: {str(e)}", icon="⚠️")
            self._models[model_name] = None
    
    def get_label_mappings(self):
        """Get label mappings, loading them if necessary."""
        if self._label_mappings is None:
            from data import load_label_mappings
            self._label_mappings = load_label_mappings()
        return self._label_mappings
    
    def cleanup(self):
        """Clean up all loaded models."""
        for model in self._models.values():
            if model is not None:
                try:
                    model.cleanup()
                except:
                    pass
        self._models.clear()
```

### Model Predictor

```python
class ModelPredictor:
    """Unified predictor for both Transformer and IV3-GRU models."""
    
    def __init__(self, model_type, checkpoint_path, device=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_type: 'transformer' or 'iv3_gru'
            checkpoint_path: Path to model checkpoint
            device: Device for inference (auto-detected if None)
        """
        self.model_type = model_type.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and detect input dimensions
        self.model, self.input_dim = self._load_model()
        self._load_checkpoint()
    
    def _load_model(self):
        """Load model architecture with auto-detection of parameters."""
        if self.model_type == 'transformer':
            # Auto-detect input_dim from checkpoint
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Extract input_dim from embedding layer shape
                if 'embedding.weight' in state_dict:
                    input_dim = state_dict['embedding.weight'].shape[1]
                else:
                    input_dim = 156  # Default
            except:
                input_dim = 156
            
            # Create Transformer model
            model = SignTransformer(
                input_dim=input_dim,      # 156, 2048, or 2204
                emb_dim=256,
                n_heads=8,
                n_layers=4,
                num_gloss=105,
                num_cat=10,
                dropout=0.1,
                max_len=300,
                pooling_method='mean'
            )
            
        elif self.model_type == 'iv3_gru':
            # IV3-GRU always uses 2048-D
            input_dim = 2048
            
            # Auto-detect GRU hidden sizes
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                if 'gru1.weight_hh_l0' in state_dict:
                    gru1_hidden = state_dict['gru1.weight_hh_l0'].shape[0] // 3
                    gru2_hidden = state_dict['gru2.weight_hh_l0'].shape[0] // 3
                else:
                    gru1_hidden, gru2_hidden = 16, 12
            except:
                gru1_hidden, gru2_hidden = 16, 12
            
            # Create IV3-GRU model
            model = InceptionV3GRU(
                num_gloss=105,
                num_cat=10,
                hidden1=gru1_hidden,
                hidden2=gru2_hidden,
                dropout=0.3,
                pretrained_backbone=True,
                freeze_backbone=True
            )
        
        return model.to(self.device), input_dim
    
    def _load_checkpoint(self):
        """Load trained weights from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def predict_from_npz(self, npz_path):
        """
        Make prediction from NPZ file.
        
        Returns:
            {
                'gloss_prediction': int,
                'gloss_probability': float,
                'category_prediction': int,
                'category_probability': float,
                'gloss_top5': [(id, prob), ...],
                'category_top3': [(id, prob), ...]
            }
        """
        # Load NPZ data
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract appropriate features based on model type and input_dim
        if self.model_type == 'transformer':
            if self.input_dim == 2048 and 'X2048' in data:
                X = torch.from_numpy(data['X2048']).float()
            elif self.input_dim == 156 and 'X' in data:
                X = torch.from_numpy(data['X']).float()
            elif self.input_dim == 2204:
                # Combined model
                X_kp = torch.from_numpy(data['X']).float()
                X_feat = torch.from_numpy(data['X2048']).float()
                X = torch.cat([X_kp, X_feat], dim=1)
            else:
                raise ValueError(f"No compatible features for input_dim={self.input_dim}")
        
        elif self.model_type == 'iv3_gru':
            if 'X2048' not in data:
                raise ValueError("IV3-GRU requires X2048 features")
            X = torch.from_numpy(data['X2048']).float()
        
        # Prepare input [1, T, D]
        X = X.unsqueeze(0).to(self.device)
        
        # Truncate if too long
        if X.shape[1] > 300:
            X = X[:, :300, :]
        
        # Create mask for Transformer
        if self.model_type == 'transformer':
            mask = torch.ones(X.shape[:2], dtype=torch.bool).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            if self.model_type == 'transformer':
                gloss_logits, cat_logits = self.model(X, mask)
            else:  # iv3_gru
                lengths = torch.tensor([X.shape[1]], dtype=torch.long).to(self.device)
                gloss_logits, cat_logits = self.model(X, lengths, features_already=True)
        
        # Get predictions and probabilities
        gloss_probs = F.softmax(gloss_logits, dim=1)[0].cpu().numpy()
        cat_probs = F.softmax(cat_logits, dim=1)[0].cpu().numpy()
        
        gloss_pred = gloss_probs.argmax()
        cat_pred = cat_probs.argmax()
        
        # Get top-k
        gloss_top5_indices = gloss_probs.argsort()[-5:][::-1]
        cat_top3_indices = cat_probs.argsort()[-3:][::-1]
        
        return {
            'gloss_prediction': int(gloss_pred),
            'gloss_probability': float(gloss_probs[gloss_pred]),
            'category_prediction': int(cat_pred),
            'category_probability': float(cat_probs[cat_pred]),
            'gloss_top5': [(int(i), float(gloss_probs[i])) for i in gloss_top5_indices],
            'category_top3': [(int(i), float(cat_probs[i])) for i in cat_top3_indices]
        }
```

### Prediction Workflow

```python
def make_real_prediction(npz_data: Dict[str, np.ndarray], model_name: str) -> Dict:
    """
    Make real prediction using the specified model.
    
    Args:
        npz_data: NPZ data dictionary
        model_name: 'transformer' or 'iv3_gru'
    
    Returns:
        Prediction results dictionary
    """
    # Get ModelManager instance (singleton)
    model_manager = get_model_manager()
    
    # Get or load predictor
    predictor = model_manager.get_model(model_name)
    if predictor is None:
        st.error(f"Failed to load {model_name} model")
        return None
    
    try:
        # Create temporary NPZ file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            np.savez_compressed(tmp_path, **npz_data)
        
        try:
            # Make prediction
            results = predictor.predict_from_npz(tmp_path)
            return results
        finally:
            # Cleanup with retry mechanism
            for attempt in range(3):
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(0.1)
    
    except Exception as e:
        st.toast(f"Prediction failed: {str(e)}", icon="⚠️")
        return None
```

### Label Mapping

```python
def format_prediction_with_labels(results, gloss_mapping, category_mapping):
    """
    Format prediction results with human-readable labels.
    
    Args:
        results: Raw prediction results
        gloss_mapping: {0: "hello", 1: "goodbye", ...}
        category_mapping: {0: "GREETING", 1: "SURVIVAL", ...}
    
    Returns:
        Formatted results with labels
    """
    gloss_id = results['gloss_prediction']
    cat_id = results['category_prediction']
    
    formatted = {
        'gloss_prediction': f"{gloss_mapping.get(gloss_id, 'Unknown')} ({gloss_id})",
        'gloss_probability': results['gloss_probability'],
        'category_prediction': f"{category_mapping.get(cat_id, 'Unknown')} ({cat_id})",
        'category_probability': results['category_probability'],
        'gloss_top5': [
            [f"{gloss_mapping.get(gid, 'Unknown')} ({gid})", prob]
            for gid, prob in results['gloss_top5']
        ],
        'category_top3': [
            [f"{category_mapping.get(cid, 'Unknown')} ({cid})", prob]
            for cid, prob in results['category_top3']
        ]
    }
    
    return formatted
```

### UI Prediction Display

```python
def render_predictions_section(cfg, npz_data, filename):
    """Display prediction results with model selection."""
    st.markdown("### Predictions")
    
    # Model selection
    selected_model_name = cfg['model_choice']  # From sidebar
    model_name = 'transformer' if selected_model_name == 'SignTransformer' else 'iv3_gru'
    
    # Check compatibility
    compatibility = st.session_state.file_metadata[filename]['compatibility']
    if not compatibility[model_name]:
        st.warning(f"File incompatible with {selected_model_name}")
        return
    
    # Make prediction
    with st.spinner("Making prediction..."):
        results = make_real_prediction(npz_data, model_name)
    
    if results is None:
        st.error("Prediction failed")
        return
    
    # Get label mappings
    model_manager = get_model_manager()
    gloss_mapping, category_mapping = model_manager.get_label_mappings()
    
    # Format with labels
    formatted = format_prediction_with_labels(results, gloss_mapping, category_mapping)
    
    # Display main predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Gloss Prediction")
        st.markdown(f"**{formatted['gloss_prediction']}**")
        st.progress(formatted['gloss_probability'])
        st.caption(f"Confidence: {formatted['gloss_probability']*100:.1f}%")
    
    with col2:
        st.markdown("#### Category Prediction")
        st.markdown(f"**{formatted['category_prediction']}**")
        st.progress(formatted['category_probability'])
        st.caption(f"Confidence: {formatted['category_probability']*100:.1f}%")
    
    # Display top-5 gloss predictions
    st.markdown("#### Top 5 Gloss Predictions")
    for i, (label_with_id, prob) in enumerate(formatted['gloss_top5'], 1):
        st.markdown(f"{i}. **{label_with_id}**: {prob*100:.1f}%")
        st.progress(prob)
    
    # Display top-3 category predictions
    st.markdown("#### Top 3 Category Predictions")
    for i, (label_with_id, prob) in enumerate(formatted['category_top3'], 1):
        st.markdown(f"{i}. **{label_with_id}**: {prob*100:.1f}%")
        st.progress(prob)
```

### Batch Prediction

```python
def render_batch_summary_tab(cfg):
    """Render batch summary with predictions for all files."""
    all_npz_files = get_all_npz_files()
    completed_files = [f for f in all_npz_files 
                      if st.session_state.file_status.get(f.name) == 'completed']
    
    if not completed_files:
        st.info("No completed files to summarize.")
        return
    
    # Summary table with predictions
    summary_data = []
    model_name = 'transformer' if cfg['model_choice'] == 'SignTransformer' else 'iv3_gru'
    
    model_manager = get_model_manager()
    gloss_mapping, category_mapping = model_manager.get_label_mappings()
    
    for uploaded_file in completed_files:
        filename = uploaded_file.name
        npz_data = st.session_state.processed_data[filename]
        
        # Make prediction
        results = make_real_prediction(npz_data, model_name)
        
        if results:
            # Format predictions
            gloss_id = results['gloss_prediction']
            cat_id = results['category_prediction']
            gloss_prob = results['gloss_probability']
            cat_prob = results['category_probability']
            
            gloss_label = gloss_mapping.get(gloss_id, f'Unknown')
            cat_label = category_mapping.get(cat_id, f'Unknown')
            
            top_gloss = f"{gloss_label} ({gloss_prob*100:.1f}%)"
            top_category = f"{cat_label} ({cat_prob*100:.1f}%)"
        else:
            top_gloss = "Prediction Failed"
            top_category = "Prediction Failed"
        
        # Extract occlusion status
        occlusion_flag = extract_occlusion_flag(npz_data)
        occlusion_status = interpret_occlusion_flag(occlusion_flag)
        
        summary_data.append({
            'File': filename,
            'Top Gloss': top_gloss,
            'Top Category': top_category,
            'Occluded': occlusion_status
        })
    
    st.dataframe(summary_data, use_container_width=True)
```

### Performance

**Prediction Times** (single file, 30s clip):
- First prediction (model loading): ~5-10 seconds
- Subsequent predictions (cached model): ~100-500ms
- GPU acceleration: ~50-200ms

**Memory Usage**:
- Transformer model: ~200MB
- IV3-GRU model: ~100MB (InceptionV3 not loaded for features)
- Model caching saves ~5-10s per prediction

---

*Continued in next section...*

For complete pipeline documentation, this file should be read with:
- [Stage 5: Results & Visualization](#stage-5-results--visualization)
- [Stage 6: Model Validation & Evaluation](#stage-6-model-validation--evaluation)
- [Data Flow Architecture](#data-flow-architecture)
- [State Management](#state-management)
- [Error Handling](#error-handling)
- [Configuration Points](#configuration-points)
- [Performance Optimizations](#performance-optimizations)

**Document Status**: Part 1 of 2 (Stages 1-4 complete)  
**Next Section**: Stages 5-6 and system-level details

See `PANSINAYAN_PIPELINE_PART2.md` for continuation.

