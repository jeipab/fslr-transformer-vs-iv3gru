# PANSINAYAN

### _Where Every Sign Gets Attention_

**Filipino Sign Language Recognition System**

Multi-Head Attention Transformer vs InceptionV3-GRU for Filipino Sign Language Recognition.

## 📊 Dataset

- **Glosses**: 105 Filipino sign words
- **Categories**: 10 semantic categories (Greeting, Survival, Number, Calendar, Days, Family, Relationships, Color, Food, Drink)
- **Training Data**: fsl-105 dataset
- **Models**: Pre-trained Transformer and IV3-GRU models available

## 🚀 Quick Start

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

# Check network connection info (local IP and access URLs)
python show_network_info.py
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
  --model transformer ^
  --checkpoint trained_models\transformer\optimal\SignTransformer_best.pt ^
  --input data\demo\clip_0138_nice to meet you.npz

# IV3-GRU model
python -m evaluation.prediction.predict ^
  --model iv3_gru ^
  --checkpoint trained_models\iv3_gru\optimal\InceptionV3GRU_best.pt ^
  --input data\demo\clip_1146_grandfather.npz
```

**Predict from Video:**

```powershell
python -m evaluation.prediction.predict ^
  --model transformer ^
  --checkpoint trained_models\transformer\optimal\SignTransformer_best.pt ^
  --input video.mp4
```

**Output Example:**

```
Gloss: NICE TO MEET YOU (6) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

## 📁 Project Structure

```
fslr-transformer-vs-iv3gru/
├── 📊 data/                    # Data management and label mapping
│   ├── demo/                   # Demo NPZ files for testing
│   ├── labels/                 # Label mappings (105 glosses, 10 categories)
│   ├── processed/              # Preprocessed NPZ files
│   │   ├── fsl_train/         # Training set (80%)
│   │   ├── fsl_val/           # Validation set (20%)
│   │   ├── fsl_train.csv      # Training labels
│   │   └── fsl_val.csv        # Validation labels
│   ├── raw/                    # Raw video files
│   └── splitting/              # Data splitting utilities
├── 📈 evaluation/              # Model validation and prediction
│   ├── prediction/            # Inference scripts
│   └── validation/            # Model evaluation
├── 🧠 models/                  # Neural network architectures
│   ├── transformer.py         # SignTransformer (keypoints)
│   └── iv3_gru.py            # InceptionV3GRU (features)
├── 📓 notebooks/               # Jupyter notebooks for experiments
├── 🔧 preprocessing/           # Video preprocessing and feature extraction
│   ├── core/                  # Core preprocessing modules
│   └── extractors/            # Feature extractors
├── 📚 shared/                  # Shared resources and documentation
│   └── for vast ai/           # Vast.ai deployment resources
├── 🖥️ streamlit_app/          # Interactive web application
├── 💾 trained_models/          # Model checkpoints and weights
│   ├── transformer\optimal\  # Transformer models
│   └── iv3_gru\optimal\      # IV3-GRU models
└── 🏋️ training/               # Model training and evaluation
```

## 🔄 Workflow

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

- Keypoints `X [T,156]` - 78 MediaPipe keypoints (pose, hands, face)
- Features `X2048 [T,2048]` - InceptionV3 features
- Visibility mask `mask [T,78]`
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
  --processed-root data\processed\fsl-105_10-08 ^
  --labels data\processed\fsl-105_10-08\labels.csv ^
  --out-root data\processed ^
  --copy ^
  --train-ratio 0.8 ^
  --train-dir fsl_train ^
  --val-dir fsl_val ^
  --train-csv fsl_train.csv ^
  --val-csv fsl_val.csv
```

For detailed data splitting instructions, see [Data Guide](data/DATA_GUIDE.md).

### 3. Training

**Transformer Model (Keypoints):**

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\fsl_train ^
  --keypoints-val data\processed\fsl_val ^
  --labels-train-csv data\processed\fsl_train.csv ^
  --labels-val-csv data\processed\fsl_val.csv ^
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
  --model iv3_gru ^
  --features-train data\processed\fsl_train ^
  --features-val data\processed\fsl_val ^
  --labels-train-csv data\processed\fsl_train.csv ^
  --labels-val-csv data\processed\fsl_val.csv ^
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

### 4. Validation

**Data Validation:**

```powershell
# Validate NPZ files
python -m preprocessing.utils.validate_npz data\processed\fsl_train
python -m preprocessing.utils.validate_npz data\processed\fsl_val --require-x2048
```

**Model Validation:**

```powershell
# Transformer model
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\optimal\SignTransformer_best.pt ^
  --data-dir data\processed\fsl_val ^
  --labels-csv data\processed\fsl_val.csv

# IV3-GRU model
python -m evaluation.validation.validate ^
  --model iv3_gru ^
  --checkpoint trained_models\iv3_gru\optimal\InceptionV3GRU_best.pt ^
  --data-dir data\processed\fsl_val ^
  --labels-csv data\processed\fsl_val.csv
```

**Smoke Tests:**

```powershell
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10
python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10
```

For detailed validation instructions, see [Validation Guide](evaluation/validation/VALIDATION_GUIDE.md).

## 🧠 Models

### Transformer (SignTransformer)

- **Input**: MediaPipe keypoints [T, 156]
- **Architecture**: Multi-head attention with positional encoding
- **Advantages**: Lighter, interpretable attention weights
- **Best for**: Keypoint-based sign recognition

### InceptionV3-GRU

- **Input**: InceptionV3 features [T, 2048]
- **Architecture**: CNN + GRU with pretrained backbone
- **Advantages**: Transfer learning from ImageNet
- **Best for**: Visual feature-based recognition

Both models predict:

- **Gloss**: Specific sign word (105 classes)
- **Category**: Semantic category (10 classes)

For architecture details, see [Model Guide](models/MODEL_GUIDE.md).

## 📖 Documentation

### 🎯 Prediction & Usage

- **[Prediction Guide](evaluation/prediction/PREDICTION_GUIDE.md)** - Using trained models for predictions
- **[Validation Guide](evaluation/validation/VALIDATION_GUIDE.md)** - Model validation and evaluation
- **[Label Mapping Table](data/labels/LABEL_MAPPING_TABLE.md)** - Complete list of signs and categories
- **[Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md)** - Model checkpoints and usage
- **[Tool Guide](streamlit_app/TOOL_GUIDE.md)** - Interactive visualization app

### 🔧 Development & Training

- **[Data Guide](data/DATA_GUIDE.md)** - File formats and data structures
- **[Preprocessing Guide](preprocessing/docs/PREPROCESS_GUIDE.MD)** - Video preprocessing
- **[Occlusion Guide](preprocessing/docs/OCCLUSION_GUIDE.md)** - Hand occlusion detection and handling
- **[Model Guide](models/MODEL_GUIDE.md)** - Architecture details and usage
- **[Training Guide](training/TRAINING_GUIDE.md)** - Model training instructions
- **[Sharing Guide](shared/SHARING_GUIDE.md)** - Vast.ai deployment and collaboration

## 🛠️ Troubleshooting

### Common Issues

**File not found:**

- CSV `file` values must match `.npz` basenames exactly (without extension)
- Example: CSV has `clip_0315_yes`, NPZ file is `clip_0315_yes.npz`

**Wrong shapes:**

- Transformer needs `X [T,156]` keypoints
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
3. See `.streamlit/CONFIG_NOTES.md` for detailed configuration explanation

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

## 🚀 Deployment

### Local Development

```powershell
streamlit run run_app.py
```

### Vast.ai Deployment

For remote deployment on Vast.ai instances:

1. Follow [Vast.ai Guide](shared/for vast ai/VAST.AI_GUIDE.md)
2. Replace components with Vast.ai versions
3. Use cloudflared tunnel for external access

See [Sharing Guide](shared/SHARING_GUIDE.md) for detailed deployment instructions.

## 🤝 Contributing

PANSINAYAN supports Filipino Sign Language Recognition research and accessibility initiatives. For collaboration guidelines, see the [Sharing Guide](shared/SHARING_GUIDE.md).

## 📄 License

This project is part of academic research in Filipino Sign Language Recognition.

## 📚 Citation

If you use PANSINAYAN in your research, please cite:

```bibtex
@thesis{pansinayan2025,
  title={PANSINAYAN: Multi-Head Attention Transformer for Filipino Sign Language Recognition},
  author={Estrella, Novelle Lyn and Magtibay, Nathaniel L. and Migueh, Rica Joi C. and Pablo, Jeremias G.},
  year={2025},
  school={Polytechnic University of the Philippines}
}
```

## 🔗 Links

- **Repository**: [https://github.com/jeipab/fslr-transformer-vs-iv3gru](https://github.com/jeipab/fslr-transformer-vs-iv3gru)
- **Documentation**: See individual guide files listed above
- **Demo Data**: Available in `data/demo/` directory
