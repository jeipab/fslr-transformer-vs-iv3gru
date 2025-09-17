# Filipino Sign Language Recognition (FSLR)

Multi-Head Attention Transformer for Filipino Sign Language Recognition.

## 🚀 Quick Start

### Setup

**Requirements**: Python 3.9-3.11

```bash
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

### Interactive Demo

```bash
# Run the Streamlit application
streamlit run run_app.py

# Alternative: Run from streamlit_app directory
cd streamlit_app && streamlit run main.py
```

**Features**: Animated keypoint visualization, feature analysis, real-time predictions, support for both `.npz` files and raw videos.

### Quick Prediction

```bash
# Predict from NPZ file
python -m evaluation.prediction.predict \
  --model transformer \
  --checkpoint trained_models/transformer/transformer_100_epoch/SignTransformer_best.pt \
  --input data/processed/transformer_only/clip_0089_how\ are\ you.npz

# Predict from video file
python -m evaluation.prediction.predict \
  --model transformer \
  --checkpoint trained_models/transformer/transformer_100_epoch/SignTransformer_best.pt \
  --input data/raw/videos/new_sign.mp4
```

**Output Example:**

```
Gloss: HOW ARE YOU (4) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

## 📁 Project Structure

```
fslr-transformer-vs-iv3gru/
├── 📊 data/                    # Data management and label mapping
├── 📈 evaluation/              # Model validation and prediction
├── 🧠 models/                  # Neural network architectures
├── 📓 notebooks/               # Jupyter notebooks for experiments
├── 🔧 preprocessing/           # Video preprocessing and feature extraction
├── 📚 shared/                  # Shared resources and documentation
├── 🖥️ streamlit_app/          # Interactive web application
├── 💾 trained_models/          # Model checkpoints and weights
└── 🏋️ training/               # Model training and evaluation
```

## 🔄 Workflow

### 1. Preprocessing

**Multi-Process (Recommended - 30-50x faster):**

```bash
python -m preprocessing.core.multi_preprocess \
  data/raw/videos data/processed \
  --write-keypoints --write-iv3-features \
  --workers 10 --batch-size 64 --target-fps 15 --disable-parquet
```

**Sequential (For small datasets):**

```bash
python -m preprocessing.core.preprocess \
  data/raw/videos data/processed \
  --write-keypoints --write-iv3-features \
  --target-fps 30
```

**Output**: `.npz` files with keypoints `X [T,156]`, features `X2048 [T,2048]`, visibility `mask [T,78]`, timestamps, and metadata.

### 2. Training

**Transformer Model (Keypoints):**

```bash
python -m training.train \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --amp
```

**IV3-GRU Model (Features):**

```bash
python -m training.train \
  --model iv3_gru \
  --features-train data/processed/keypoints_train \
  --features-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --feature-key X2048 \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --amp
```

### 3. Validation

**Data Validation:**

```bash
python -m preprocessing.utils.validate_npz data/processed/keypoints_train
python -m preprocessing.utils.validate_npz data/processed/keypoints_val --require-x2048
```

**Model Validation:**

```bash
python -m evaluation.validation.validate \
  --model transformer \
  --checkpoint trained_models/transformer/transformer_100_epoch/SignTransformer_best.pt \
  --data-dir data/processed
```

**Smoke Tests:**

```bash
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10
python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10
```

## 📖 Documentation

### 🎯 Prediction & Usage

- **[Prediction Guide](evaluation/prediction/PREDICTION_GUIDE.md)** - Using trained models for predictions
- **[Validation Guide](evaluation/validation/VALIDATION_GUIDE.md)** - Model validation and evaluation
- **[Label Mapping Table](data/labels/LABEL_MAPPING_TABLE.md)** - Complete list of signs and categories
- **[Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md)** - Model checkpoints and usage

### 🔧 Development & Training

- **[Data Guide](data/DATA_GUIDE.md)** - File formats and data structures
- **[Preprocessing Guide](preprocessing/docs/PREPROCESS_GUIDE.MD)** - Video to NPZ conversion
- **[Multi-Process Guide](preprocessing/docs/MULTI_PREPROCESS_GUIDE.md)** - 30-50x faster preprocessing
- **[Occlusion Guide](preprocessing/docs/OCCLUSION_GUIDE.md)** - Hand occlusion detection and handling
- **[Model Guide](models/MODEL_GUIDE.md)** - Architecture details and usage
- **[Training Guide](training/TRAINING_GUIDE.md)** - Model training instructions
- **[Tool Guide](streamlit_app/TOOL_GUIDE.md)** - Interactive visualization app
- **[Sharing Guide](shared/SHARING_GUIDE.md)** - Collaboration and sharing resources

## 🛠️ Troubleshooting

### Common Issues

- **File not found**: CSV `file` values must match `.npz` basenames exactly
- **Wrong shapes**: Transformer needs `X [T,156]`; IV3-GRU needs `X2048 [T,2048]`
- **Label ranges**: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`
- **Port conflicts**: Use `streamlit run run_app.py --server.port 8502`
- **CUDA issues**: Auto-detects CUDA, CPU fallback available

### Performance Tips

- Use `--amp` for automatic mixed precision training
- Add `--num-workers N` for faster data loading
- Use `--pin-memory` for GPU training
- Enable `--disable-parquet` for faster preprocessing

## 🤝 Contributing

This project supports Filipino Sign Language Recognition research. For collaboration guidelines, see the [Sharing Guide](shared/SHARING_GUIDE.md).

## 📄 License

This project is part of academic research in Filipino Sign Language Recognition.
