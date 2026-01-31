# PANSINAYAN

### _Where Every Sign Gets Attention_

**Filipino Sign Language Recognition System**

PANSINAYAN is a comprehensive deep learning system for Filipino Sign Language (FSL) recognition, comparing Multi-Head Attention Transformer and InceptionV3-GRU architectures. The system supports both isolated sign classification and continuous sign sequence recognition using CTC decoding.

**Key Features:**
- Dual architecture comparison (Transformer vs InceptionV3-GRU)
- Isolated sign classification (105 gloss classes, 10 categories)
- Continuous sign recognition with CTC decoding
- Interactive web interface (PANSINAYAN Streamlit app)
- Comprehensive preprocessing pipeline with occlusion detection
- Model training, validation, and evaluation tools

## Overview

PANSINAYAN provides a complete pipeline for Filipino Sign Language Recognition:

1. **Preprocessing**: Extract MediaPipe keypoints (178-D) and InceptionV3 features (2048-D) from raw videos
2. **Training**: Train Transformer or InceptionV3-GRU models for isolated or continuous sign recognition
3. **Evaluation**: Validate models and analyze performance metrics
4. **Inference**: Predict signs from videos or preprocessed NPZ files
5. **Visualization**: Interactive web interface for model comparison and analysis

The system supports both **isolated sign recognition** (classification) and **continuous sign recognition** (CTC-based sequence-to-sequence).

## Dataset

- **Glosses**: 105 Filipino sign words (IDs: 0-104)
- **Categories**: 10 semantic categories (IDs: 0-9): Greeting, Survival, Number, Calendar, Days, Family, Relationships, Color, Food, Drink
- **Training Data**: FSL-105 dataset
- **Models**: Pre-trained Transformer and IV3-GRU models available (see [Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md))

**Model Setup**: Model checkpoints must be placed in `trained_models/transformer/` and `trained_models/iv3_gru/` directories following the structure defined in the README.txt files in those directories. See [Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md) for details.

## Quick Start

### Setup

**Requirements**: Python 3.9-3.11

```powershell
# Clone the repository
git clone https://github.com/jeipab/fslr-transformer-vs-iv3gru.git
cd fslr-transformer-vs-iv3gru

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pyarrow  # optional, for parquet inspection
```

### Interactive Demo with PANSINAYAN

```powershell
# Run the PANSINAYAN application
streamlit run run_app.py
```

**PANSINAYAN Features**:

- Animated keypoint visualization with attention mechanism
- Real-time predictions for 105 Filipino sign words
- Support for both preprocessed `.npz` files and raw videos
- Dual model comparison (Transformer vs IV3-GRU)
- Occlusion-aware analysis and validation

### Quick Prediction

**Predict from Demo Files:**

```powershell
# Transformer model
python -m evaluation.prediction.predict ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --input data\demo\clip_0138_nice to meet you.npz

# IV3-GRU model
python -m evaluation.prediction.predict ^
  --model iv3_gru_isolated ^
  --checkpoint trained_models\iv3_gru\FSL105_classification\InceptionV3GRU_best.pt ^
  --input data\demo\clip_1146_grandfather.npz
```

**Predict from Video:**

```powershell
python -m evaluation.prediction.predict ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --input video.mp4
```

**Output Example:**

```
Gloss: NICE TO MEET YOU (6) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

## Project Structure

```
fslr-transformer-vs-iv3gru/
├── data/                       # Data management and label mapping
│   ├── demo/                   # Demo NPZ files for testing
│   ├── labels/                 # Label mappings (105 glosses, 10 categories)
│   │   └── LABEL_MAPPING_TABLE.md
│   ├── processed/              # Preprocessed NPZ files
│   │   ├── FSL105_train/      # Training set (80%)
│   │   ├── FSL105_val/        # Validation set (20%)
│   │   ├── FSL105_train.csv   # Training labels
│   │   └── FSL105_val.csv     # Validation labels
│   ├── raw/                    # Raw video files
│   ├── splitting/              # Data splitting utilities
│   └── DATA_GUIDE.md          # Data format documentation
├── evaluation/                 # Model validation and prediction
│   ├── prediction/             # Inference scripts
│   │   ├── predict.py         # Classification prediction
│   │   ├── predict_ctc.py      # CTC prediction
│   │   └── PREDICTION_GUIDE.md
│   └── validation/             # Model evaluation
│       ├── validate.py         # Classification validation
│       ├── evaluate_ctc.py     # CTC evaluation
│       └── VALIDATION_GUIDE.md
├── models/                     # Neural network architectures
│   ├── transformer.py         # SignTransformer (keypoints)
│   ├── iv3_gru.py             # InceptionV3GRU (features)
│   └── MODEL_GUIDE.md         # Architecture documentation
├── preprocessing/              # Video preprocessing and feature extraction
│   ├── core/                   # Core preprocessing modules
│   │   ├── preprocess.py       # Main preprocessing pipeline
│   │   └── occlusion_detection.py
│   ├── extractors/            # Feature extractors
│   │   ├── keypoints_features.py
│   │   └── iv3_features.py
│   ├── continuous/            # Continuous sequence generation
│   ├── docs/                  # Preprocessing documentation
│   │   ├── PREPROCESS_GUIDE.MD
│   │   ├── OCCLUSION_GUIDE.md
│   │   └── OCCLUSION_PARAMETERS_GUIDE.md
│   └── utils/                 # Preprocessing utilities
├── streamlit_app/             # Interactive web application
│   ├── core/                  # Application core
│   ├── components/            # UI components
│   ├── manager/               # Workflow managers
│   └── TOOL_GUIDE.md         # Application documentation
├── trained_models/            # Model checkpoints and weights
│   ├── transformer/          # Transformer model checkpoints
│   │   ├── FSL105_classification/  # Classification models
│   │   ├── FSL105_ctc/             # CTC models
│   │   └── README.txt              # Setup instructions
│   ├── iv3_gru/               # InceptionV3-GRU model checkpoints
│   │   ├── FSL105_classification/  # Classification models
│   │   ├── FSL105_ctc/             # CTC models
│   │   └── README.txt              # Setup instructions
│   └── TRAINED_MODEL_GUIDE.md     # Model usage guide
├── training/                   # Model training and evaluation
│   ├── train.py               # Training script
│   ├── utils.py               # Training utilities
│   └── TRAINING_GUIDE.md      # Training documentation
├── run_app.py                 # Streamlit app launcher
└── README.md                  # This file
```

## Workflow

### 1. Preprocessing

**Multi-Process (Recommended - 30-50x faster):**

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\videos ^
  data\processed\output ^
  --write-keypoints ^
  --write-iv3-features ^
  --workers 8 ^
  --batch-size 32 ^
  --target-fps 30 ^
  --disable-parquet
```

**Sequential (For small datasets):**

```powershell
python -m preprocessing.core.preprocess ^
  data\raw\videos ^
  data\processed\output ^
  --write-keypoints ^
  --write-iv3-features ^
  --target-fps 30
```

**Output**: `.npz` files with:

- Keypoints `X [T,178]` - 89 MediaPipe keypoints (pose, hands, face)
- Features `X2048 [T,2048]` - InceptionV3 features
- Visibility mask `mask [T,89]`
- Timestamps `timestamps_ms [T]`
- Metadata with occlusion detection

For detailed preprocessing instructions, see [Preprocessing Guide](preprocessing/docs/PREPROCESS_GUIDE.MD).

### 2. Data Splitting

After preprocessing, create labels and split data:

```powershell
# Assign gloss and category IDs
python data\splitting\assign.py

# Split into train/val sets
python data\splitting\data_split.py ^
  --processed-root data\processed\FSL-105 ^
  --labels data\processed\FSL-105\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir FSL105_train ^
  --val-dir FSL105_val ^
  --train-csv FSL105_train.csv ^
  --val-csv FSL105_val.csv
```

For detailed data splitting instructions, see [Data Guide](data/DATA_GUIDE.md).

### 3. Training

**Transformer Model (Keypoints):**

```powershell
python -m training.train ^
  --model transformer_isolated ^
  --keypoints-train data\processed\FSL105_train ^
  --keypoints-val data\processed\FSL105_val ^
  --labels-train-csv data\processed\FSL105_train.csv ^
  --labels-val-csv data\processed\FSL105_val.csv ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --amp ^
  --auto-workers ^
  --auto-batch-size
```

**IV3-GRU Model (InceptionV3 Features):**

```powershell
python -m training.train ^
  --model iv3_gru_isolated ^
  --features-train data\processed\FSL105_train ^
  --features-val data\processed\FSL105_val ^
  --labels-train-csv data\processed\FSL105_train.csv ^
  --labels-val-csv data\processed\FSL105_val.csv ^
  --feature-key X2048 ^
  --num-gloss 105 ^
  --num-cat 10 ^
  --epochs 100 ^
  --batch-size 32 ^
  --amp ^
  --auto-workers ^
  --auto-batch-size
```

For detailed training instructions, see [Training Guide](training/TRAINING_GUIDE.md).

### 3.1 CTC Training (Continuous Recognition)

Train models for continuous sign recognition using CTC (no frame-level alignment needed).

```powershell
# SignTransformerCtc
python training/train.py --model transformer_continuous --keypoints-train data\processed\FSL105_train --keypoints-val data\processed\FSL105_val --labels-train-csv data\processed\FSL105_train.csv --labels-val-csv data\processed\FSL105_val.csv --epochs 100 --grad-clip 1.0 --amp

# InceptionV3GRUCtc
python training/train.py --model iv3_gru_continuous --features-train data\processed\FSL105_train --features-val data\processed\FSL105_val --labels-train-csv data\processed\FSL105_train.csv --labels-val-csv data\processed\FSL105_val.csv --feature-key X2048 --epochs 100 --grad-clip 1.0 --amp
```

For detailed CTC training options, see [Training Guide](training/TRAINING_GUIDE.md#ctc-training-continuous-recognition).

### 4. Validation

**Data Validation:**

```powershell
# Validate NPZ files
python -m preprocessing.utils.validate_npz data\processed\FSL105_train
python -m preprocessing.utils.validate_npz data\processed\FSL105_val --require-x2048
```

**Model Validation:**

```powershell
# Transformer model
python -m evaluation.validation.validate ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv

# IV3-GRU model
python -m evaluation.validation.validate ^
  --model iv3_gru_isolated ^
  --checkpoint trained_models\iv3_gru\FSL105_classification\InceptionV3GRU_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv
```

**Smoke Tests:**

```powershell
python -m training.train --model transformer_isolated --smoke-test --num-gloss 105 --num-cat 10
python -m training.train --model iv3_gru_isolated --smoke-test --num-gloss 105 --num-cat 10
```

For detailed validation instructions, see [Validation Guide](evaluation/validation/VALIDATION_GUIDE.md).

### 5. CTC Prediction & Evaluation

**Predict**:

```powershell
python evaluation\prediction\predict_ctc.py --model transformer_continuous --checkpoint model.pt --input clip.npz
```

**Evaluate** (computes WER, sequence accuracy):

```powershell
python evaluation\validation\evaluate_ctc.py --model transformer_continuous --checkpoint model.pt --test-data data\processed\FSL105_val --test-labels FSL105_val.csv
```

See [Prediction Guide](evaluation/prediction/PREDICTION_GUIDE.md#ctc-prediction-continuous-recognition) and [Validation Guide](evaluation/validation/VALIDATION_GUIDE.md#ctc-evaluation-continuous-recognition).

## Models

### Classification Models (Isolated Sign Recognition)

#### Transformer (SignTransformer)

- **Input**: MediaPipe keypoints [T, 178]
- **Architecture**: Multi-head attention with positional encoding
- **Advantages**: Lighter, interpretable attention weights
- **Best for**: Keypoint-based sign recognition

#### InceptionV3-GRU

- **Input**: InceptionV3 features [T, 2048]
- **Architecture**: CNN + GRU with pretrained backbone
- **Advantages**: Transfer learning from ImageNet
- **Best for**: Visual feature-based recognition

Classification models predict:

- **Gloss**: Specific sign word (105 classes)
- **Category**: Semantic category (10 classes)

### CTC Models (Continuous Sign Language Recognition) 🆕

#### SignTransformerCtc

- **Input**: MediaPipe keypoints [T, 178]
- **Output**: Gloss sequences (variable length)
- **Architecture**: Transformer encoder + CTC head
- **Advantages**: No frame-level alignment required, attention-based
- **Best for**: Continuous sign recognition

#### InceptionV3GRUCtc

- **Input**: InceptionV3 features [T, 2048]
- **Output**: Gloss sequences (variable length)
- **Architecture**: CNN + GRU + CTC head
- **Advantages**: Transfer learning benefits, visual features
- **Best for**: Continuous recognition with visual features

**CTC Features:**

- Sequence-to-sequence learning
- Variable-length output
- No alignment required
- Supports continuous sign sentences

For architecture details, see [Model Guide](models/MODEL_GUIDE.md).

## Documentation

### Prediction & Usage

- **[Prediction Guide](evaluation/prediction/PREDICTION_GUIDE.md)** - Using trained models (classification & CTC)
- **[Validation Guide](evaluation/validation/VALIDATION_GUIDE.md)** - Model validation and evaluation (classification & CTC)
- **[Label Mapping Table](data/labels/LABEL_MAPPING_TABLE.md)** - Complete list of signs and categories
- **[Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md)** - Model checkpoints and usage
- **[Tool Guide](streamlit_app/TOOL_GUIDE.md)** - Interactive visualization app

### Development & Training

- **[Model Guide](models/MODEL_GUIDE.md)** - Architecture details (classification & CTC models)
- **[Training Guide](training/TRAINING_GUIDE.md)** - Model training (classification & CTC)
- **[Data Guide](data/DATA_GUIDE.md)** - File formats and data structures
- **[Preprocessing Guide](preprocessing/docs/PREPROCESS_GUIDE.MD)** - Video preprocessing
- **[Occlusion Guide](preprocessing/docs/OCCLUSION_GUIDE.md)** - Hand occlusion detection and handling
- **[Occlusion Parameters Guide](preprocessing/docs/OCCLUSION_PARAMETERS_GUIDE.md)** - Adjusting occlusion detection sensitivity
- **[Continuous Signs Guide](preprocessing/docs/CONTINUOUS_SIGNS_GUIDE.md)** - Generating continuous sequences for CTC evaluation

## Troubleshooting

### Common Issues

**File not found:**

- CSV `file` values must match `.npz` basenames exactly (without extension)
- Example: CSV has `clip_0315_yes`, NPZ file is `clip_0315_yes.npz`

**Wrong shapes:**

- Transformer needs `X [T,178]` keypoints
- IV3-GRU needs `X2048 [T,2048]` InceptionV3 features

**Label ranges:**

- `gloss` must be in `[0, 104]` (105 classes, 0-based)
- `cat` must be in `[0, 9]` (10 categories, 0-based)

**Port conflicts:**

- Use `streamlit run run_app.py --server.port 8502` for alternative port

**CUDA issues:**

- Auto-detects CUDA, falls back to CPU if unavailable
- Use `--device cpu` to force CPU mode

**Out of Memory (OOM):**

- Reduce `--batch-size` (try 16 or 8)
- Enable `--amp` for mixed precision
- Use `--gradient-accumulation-steps` for effective larger batches

### Mobile Upload Issues

**Problem**: Video uploads from mobile camera fail (6-10MB+), but gallery uploads work.

**Root Cause**: Default Streamlit upload size limit and mobile-specific WebSocket constraints.

**Solution**: Configuration has been updated in `.streamlit/config.toml`:

- `maxUploadSize = 500` MB (increased from 200MB default)
- `maxMessageSize = 500` MB (matches upload size)
- `enableCORS = true` (mobile browser compatibility)
- `enableWebsocketCompression = true` (better mobile network performance)

**After deploying these changes:**

1. Restart the Streamlit app
2. Test on actual mobile devices (iOS Safari, Android Chrome)

**For deployment platforms:**

- **Streamlit Cloud**: Automatically reads config from repository
- **Heroku/Railway**: May need additional platform configuration
- **Self-hosted**: Check nginx/Apache upload limits

### Performance Tips

**For Training:**

- Use `--amp` for automatic mixed precision training
- Add `--auto-workers` for optimal data loading
- Use `--auto-batch-size` for memory-efficient batch sizing
- Enable `--compile-model` for PyTorch 2.0+ optimization

**For Preprocessing:**

- Use `--workers 8` for parallel processing
- Enable `--disable-parquet` for faster I/O
- Lower `--target-fps` (15-20) for faster processing

**For Multi-GPU:**

- Enable `--enable-parallel` for automatic DataParallel
- Increase `--batch-size` to utilize multiple GPUs

## Deployment

### Local Development

```powershell
streamlit run run_app.py
```

### Vast.ai Deployment

For remote deployment on Vast.ai instances, see [Vast.ai Guide](shared/for vast ai/VAST.AI_GUIDE.md).

## Contributing

PANSINAYAN supports Filipino Sign Language Recognition research and accessibility initiatives.

## License

This project is part of academic research in Filipino Sign Language Recognition.

## Citation

If you use PANSINAYAN in your research, please cite:

```bibtex
@thesis{pansinayan2025,
  title={PANSINAYAN: Multi-Head Attention Transformer for Filipino Sign Language Recognition},
  author={Estrella, Novelle Lyn and Magtibay, Nathaniel L. and Migueh, Rica Joi C. and Pablo, Jeremias G.},
  year={2025},
  school={Polytechnic University of the Philippines}
}
```

## Links

- **Repository**: [https://github.com/jeipab/fslr-transformer-vs-iv3gru](https://github.com/jeipab/fslr-transformer-vs-iv3gru)
- **Documentation**: See individual guide files listed above
- **Demo Data**: Available in `data/demo/` directory
