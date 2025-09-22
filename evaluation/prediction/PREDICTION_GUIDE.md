# Prediction Guide

This guide explains how to use the prediction script for Sign Language Recognition models.

## Quick Start

### List Available Models

```bash
python predict.py --list-models
```

### Predict from NPZ File (Transformer)

```bash
python predict.py --model transformer --checkpoint transformer/optimal/SignTransformer_best.pt --input data.npz
```

### Predict from Video File (IV3-GRU)

```bash
python predict.py --model iv3_gru --checkpoint iv3_gru/70-gloss_acc/InceptionV3GRU_best.pt --input video.mp4
```

## Required Arguments

- `--model`: Model type (`transformer` or `iv3_gru`)
- `--checkpoint`: Path to model checkpoint (.pt file)
- `--input`: Input file (NPZ or video file)

## Optional Arguments

- `--device`: Device to use (`cpu`, `cuda`, or `auto` - default: `auto`)
- `--fps`: Target FPS for video processing (default: `30`)
- `--image-size`: Image size for video processing (default: `256`)
- `--output`: Save results to JSON file
- `--list-models`: List all available model checkpoints

## Understanding Results

The script outputs human-readable results:

```
Gloss: HOW ARE YOU (4) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

- **HOW ARE YOU (4)**: The predicted sign with its class ID
- **GREETING (0)**: The predicted category with its class ID
- **Confidence**: How certain the model is (0.0 to 1.0)

## Examples

### Save Results to File

```bash
python predict.py --model transformer --checkpoint transformer/optimal/SignTransformer_best.pt --input data.npz --output results.json
```

### Force CPU Usage

```bash
python predict.py --model transformer --checkpoint transformer/optimal/SignTransformer_best.pt --input data.npz --device cpu
```

### Process Video File

```bash
python predict.py --model transformer --checkpoint transformer/optimal/SignTransformer_best.pt --input video.mp4 --fps 15 --image-size 256
```

## Model Types

### Transformer Model

- **NPZ Input**: Requires NPZ files with keypoint data (`X` key) or IV3 features (`X2048` key)
- **Video Input**: Automatically extracts features using MediaPipe and InceptionV3
- **Features**: 156-dimensional keypoints or 2048-dimensional IV3 features

### IV3-GRU Model

- **NPZ Input**: Requires NPZ files with IV3 features (`X2048` key)
- **Video Input**: Automatically extracts 2048-dimensional InceptionV3 features
- **Features**: 2048-dimensional InceptionV3 feature sequences

## Input Formats

### NPZ Files (Preprocessed Data)

**For Transformer:**

- `X`: Keypoint data [T, 156] - 78 keypoints × 2 coordinates
- `X2048`: IV3 features [T, 2048] - 2048-dimensional features per frame
- `mask`: Visibility mask [T, 78] (optional)

**For IV3-GRU:**

- `X2048`: InceptionV3 features [T, 2048] - 2048-dimensional features per frame

### Video Files (Raw Videos)

Supported formats: MP4, AVI, MOV, etc.

**Processing Flow:**

1. **Video → Preprocessing**: Automatically extracts appropriate features based on model type
2. **Features → Prediction**: Uses extracted features for model prediction

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

- Check [LABEL_MAPPING_TABLE.md](../../data/labels/LABEL_MAPPING_TABLE.md) for all possible signs and categories
- Run `python predict.py --help` for command-line help
- Ensure your virtual environment is activated: `.venv\Scripts\Activate.ps1`
