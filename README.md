# Filipino Sign Language Recognition

Multi-Head Attention Transformer for Filipino Sign Language Recognition.

## Structure

- `preprocessing/` - Keypoint extraction and occlusion handling
- `models/` - Transformer and IV3-GRU architectures
- `training/` - Training scripts and evaluation
- `streamlit_app/` - Interactive demo application
- `notebooks/` - Jupyter notebooks for experiments

## Setup

**Requirements**: Python 3.9-3.11

```bash
git clone https://github.com/jeipab/fslr-transformer-vs-iv3gru.git
cd fslr-transformer-vs-iv3gru
pip install -r requirements.txt
pip install pyarrow  # optional, for parquet inspection
```

## Demo

Run the interactive Streamlit application:

```bash
# Option 1: From root directory
streamlit run run_app.py

# Option 2: From streamlit_app directory
cd streamlit_app
streamlit run main.py
```

**Features**:

- Animated keypoint visualization
- Feature analysis
- Simulated predictions
- Support for both `.npz` files and raw videos

**Port conflict**: `streamlit run run_app.py --server.port 8502`

## Preprocessing

### Multi-Process (Recommended)

**30-50x faster** for large datasets:

```bash
python preprocessing/multi_preprocess.py /path/to/videos /path/to/out_dir \
  --write-keypoints --write-iv3-features \
  --workers 10 --batch-size 64 --target-fps 15 --disable-parquet
```

**Features**: Batched GPU inference, multi-process parallelization, configurable workers

### Sequential (Original)

For small datasets or single videos:

```bash
# Single video
python preprocessing/preprocess.py --write-keypoints --write-iv3-features \
  /path/to/video.mp4 /path/to/out_dir

# Directory
python -m preprocessing.preprocess /path/to/videos /path/to/out_dir \
  --target-fps 30 --write-keypoints --write-iv3-features
```

**Output**: `.npz` files with `X [T,156]`, `X2048 [T,2048]`, `mask [T,78]`, `timestamps_ms [T]`, `meta`

**Data Structure**:

```
data/processed/
  keypoints_train/
  keypoints_val/
  train_labels.csv  # file,gloss,cat,occluded
  val_labels.csv
```

## Training

### Transformer (Keypoints)

```bash
python -m training.train \
  --model transformer \
  --keypoints-train data/processed/keypoints_train \
  --keypoints-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data/processed
```

### IV3-GRU (Features)

```bash
python -m training.train \
  --model iv3_gru \
  --features-train data/processed/keypoints_train \
  --features-val data/processed/keypoints_val \
  --labels-train-csv data/processed/train_labels.csv \
  --labels-val-csv data/processed/val_labels.csv \
  --feature-key X2048 \
  --num-gloss 105 --num-cat 10 \
  --epochs 30 --batch-size 32 --output-dir data/processed
```

**Performance**: Add `--amp`, `--num-workers N`, `--pin-memory` for faster training

## Validation

### Data Validation

```bash
python -m preprocessing.validate_npz data/processed/keypoints_train
python -m preprocessing.validate_npz data/processed/keypoints_val --require-x2048
```

### Smoke Tests

```bash
python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10
python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10 --no-pretrained-backbone
```

## Guides

- [Preprocessing Guide](preprocessing/PREPROCESS_GUIDE.md) - Video to NPZ conversion
- [Multi-Process Guide](preprocessing/MULTI_PREPROCESS_GUIDE.md) - 30-50x faster preprocessing
- [Model Guide](models/MODEL_GUIDE.md) - Architecture details and usage
- [Training Guide](training/TRAINING_GUIDE.md) - Model training instructions
- [Data Guide](data/DATA_GUIDE.md) - File formats and structures

## Troubleshooting

- **File not found**: CSV `file` values must match `.npz` basenames
- **Wrong shapes**: Transformer needs `X [T,156]`; IV3-GRU needs `X2048 [T,2048]`
- **Label ranges**: `gloss` in `[0, num_gloss-1]`, `cat` in `[0, num_cat-1]`
- **CPU vs GPU**: Auto-detects CUDA, CPU fallback available
