# Prediction Guide

This guide explains how to use the prediction script for your trained Sign Language Recognition models via command line interface (CLI).

## Quick Start

### 1. List Available Models

```bash
python predict.py --list-models
```

### 2. Predict from NPZ File (Transformer)

```bash
python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input path/to/your/data.npz
```

### 3. Predict from Video File (IV3-GRU)

```bash
python predict.py --model iv3_gru --checkpoint iv3_gru/model.pt --input path/to/your/video.mp4
```

## What You Need

- **Trained model checkpoint** (.pt file)
- **Input data**: NPZ file (preprocessed) OR video file (raw)
- **Python environment** with required dependencies (see `requirements.txt`)

## Understanding Results

The prediction script outputs human-readable results with embedded class IDs:

**Example Output:**

```
Gloss: HOW ARE YOU (4) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

- **HOW ARE YOU (4)**: The predicted sign with its class ID
- **GREETING (0)**: The predicted category with its class ID
- **Confidence**: How certain the model is (0.0 to 1.0)

For a complete list of all possible signs and categories, see [LABEL_MAPPING_TABLE.md](LABEL_MAPPING_TABLE.md).

## Command Line Arguments

### Required Arguments

- `--model`: Model type to use
  - `transformer`: For Transformer model (uses keypoint data)
  - `iv3_gru`: For IV3-GRU model (uses video features)
- `--checkpoint`: Path to model checkpoint file (.pt file)
- `--input`: Input file path (NPZ or video file)

### Optional Arguments

- `--device`: Device to use (`cpu`, `cuda`, or `auto` - default: `auto`)
- `--fps`: Target FPS for video processing (default: `30`)
- `--image-size`: Image size for video processing (default: `256`)
- `--output`: Save results to JSON file (optional)
- `--list-models`: List all available model checkpoints

## Complete Examples

### Working Example (Transformer with NPZ)

```bash
python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input "../data/processed/transformer_only/clip_0089_how are you.npz"
```

**Output:**

```
Using device: cpu
✓ Loaded transformer model from transformer/transformer_low-acc_09-15/SignTransformer_best.pt
Predicting from NPZ file: ../data/processed/transformer_only/clip_0089_how are you.npz

============================================================
PREDICTION RESULTS
============================================================
Gloss: HOW ARE YOU (4) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)

Top 5 Gloss Predictions:
  1. HOW ARE YOU (4): 0.882
  2. SLOW (18): 0.074
  3. CORRECT (17): 0.013
  4. BREAD (85): 0.007
  5. NICE TO MEET YOU (6): 0.006

Top 3 Category Predictions:
  1. GREETING (0): 0.774
  2. FOOD (8): 0.160
  3. SURVIVAL (1): 0.061
```

### Save Results to File

```bash
python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input "../data/processed/transformer_only/clip_0089_how are you.npz" --output results.json
```

### Force CPU Usage

```bash
python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input "../data/processed/transformer_only/clip_0089_how are you.npz" --device cpu
```

### Process Video File with Transformer

```bash
python predict.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt --input "../data/raw/videos/new_sign.mp4" --fps 15 --image-size 256
```

### Process Video File with IV3-GRU (when trained)

```bash
python predict.py --model iv3_gru --checkpoint iv3_gru/model.pt --input "../data/raw/videos/new_sign.mp4" --fps 15 --image-size 299
```

## Model Types & Input Compatibility

**Both Transformer and IV3-GRU models can accept either NPZ files or raw video files.**

### Transformer Model

- **NPZ Input**: Requires NPZ files with keypoint data (`X` key)
- **Video Input**: Automatically extracts 156-dimensional keypoint sequences using MediaPipe
- **Features**: 156-dimensional keypoint sequences (78 keypoints × 2 coordinates)
- **Preprocessing**: MediaPipe Holistic for keypoint extraction

### IV3-GRU Model

- **NPZ Input**: Requires NPZ files with IV3 features (`X2048` key)
- **Video Input**: Automatically extracts 2048-dimensional InceptionV3 features
- **Features**: 2048-dimensional InceptionV3 feature sequences
- **Preprocessing**: InceptionV3 backbone for feature extraction

## Input Formats

### NPZ Files (Preprocessed Data)

Your NPZ files should contain the appropriate features for your chosen model:

**For Transformer:**

- `X`: Keypoint data [T, 156] - 78 keypoints × 2 coordinates
- `mask`: Visibility mask [T, 78] (optional)

**For IV3-GRU:**

- `X2048`: InceptionV3 features [T, 2048] - 2048-dimensional features per frame

### Video Files (Raw Videos)

Supported formats: MP4, AVI, MOV, etc.

**Processing Flow:**

1. **Video → Preprocessing**: Automatically extracts appropriate features based on model type
   - Transformer: Extracts keypoints using MediaPipe
   - IV3-GRU: Extracts InceptionV3 features
2. **Features → Prediction**: Uses extracted features for model prediction

**Configurable Parameters:**

- `--fps`: Target FPS for frame extraction (default: 30)
- `--image-size`: Image size for preprocessing (default: 256)

## Output Format

The script outputs:

- Gloss prediction (sign name and confidence)
- Category prediction (category name and confidence)
- Top 5 gloss predictions with probabilities
- Top 3 category predictions with probabilities
- Additional metadata (frames extracted for videos)

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**: Check the path to your .pt file
2. **"NPZ file must contain 'X' key"**: Use NPZ files with keypoint data for Transformer
3. **"NPZ file must contain 'X2048' key"**: Use NPZ files with IV3 features for IV3-GRU
4. **CUDA out of memory**: Use `--device cpu` to run on CPU
5. **"No module named 'mediapipe'"**: Video processing requires mediapipe (NPZ processing works without it)

### Getting Help

- Check [LABEL_MAPPING_TABLE.md](LABEL_MAPPING_TABLE.md) for all possible signs and categories
- Run `python predict.py --help` for command-line help
- Ensure your virtual environment is activated: `.venv\Scripts\Activate.ps1`
