# Prediction Guide

This guide explains how to use the prediction script for Sign Language Recognition models.

## Quick Start

### List Available Models

```powershell
python -m evaluation.prediction.predict --list-models
```

### Predict from NPZ File (Transformer)

```powershell
python -m evaluation.prediction.predict ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --input data\demo\clip_0138_nice to meet you.npz
```

### Predict from Video File (IV3-GRU)

```powershell
python -m evaluation.prediction.predict ^
  --model iv3_gru ^
  --checkpoint trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt ^
  --input video.mp4
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
Gloss: NICE TO MEET YOU (6) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

- **NICE TO MEET YOU (6)**: The predicted sign with its gloss ID
- **GREETING (0)**: The predicted category with its category ID
- **Confidence**: How certain the model is (0.0 to 1.0)

## Examples

### Predict from Demo Files

```powershell
python -m evaluation.prediction.predict ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --input data\demo\clip_0585_nine.npz
```

### Save Results to File

```powershell
python -m evaluation.prediction.predict ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --input data\demo\clip_1493_green.npz ^
  --output results.json
```

### Force CPU Usage

```powershell
python -m evaluation.prediction.predict ^
  --model iv3_gru ^
  --checkpoint trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt ^
  --input data\demo\clip_1765_fish.npz ^
  --device cpu
```

### Process Video File

```powershell
python -m evaluation.prediction.predict ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --input video.mp4 ^
  --fps 30 ^
  --image-size 256
```

## Model Types

### Transformer Model

- **NPZ Input**: Requires NPZ files with keypoint data (`X` key)
- **Video Input**: Automatically extracts 156-dimensional keypoints using MediaPipe
- **Features**: 156-dimensional keypoints (78 points × 2 coordinates)

### IV3-GRU Model

- **NPZ Input**: Requires NPZ files with InceptionV3 features (`X2048` key)
- **Video Input**: Automatically extracts 2048-dimensional InceptionV3 features
- **Features**: 2048-dimensional InceptionV3 feature sequences

## Input Formats

### NPZ Files (Preprocessed Data)

**For Transformer:**

- `X`: Keypoint data `[T, 156]` - 78 keypoints × 2 coordinates
- `mask`: Visibility mask `[T, 78]` (optional)

**For IV3-GRU:**

- `X2048`: InceptionV3 features `[T, 2048]` - 2048-dimensional features per frame

### Video Files (Raw Videos)

Supported formats: `.mp4`, `.avi`, `.mov` (OpenCV-compatible)

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

## Dataset Information

- **Glosses**: 105 sign words (IDs: 0-104)
- **Categories**: 10 semantic categories (IDs: 0-9)
  - 0: GREETING
  - 1: SURVIVAL
  - 2: NUMBER
  - 3: CALENDAR
  - 4: DAYS
  - 5: FAMILY
  - 6: RELATIONSHIPS
  - 7: COLOR
  - 8: FOOD
  - 9: DRINK

For complete mappings, see [LABEL_MAPPING_TABLE.md](../../data/labels/LABEL_MAPPING_TABLE.md)

## Demo Files

Test predictions using demo files in `data\demo\`:

- `clip_0138_nice to meet you.npz`
- `clip_0585_nine.npz`
- `clip_1146_grandfather.npz`
- `clip_1493_green.npz`
- `clip_1765_fish.npz`
- `clip_1912_crab.npz`

## Available Models

### Transformer Models

```
trained_models\transformer\
└── cmb_optimal\
    ├── SignTransformer_best.pt   # Best validation performance
    └── SignTransformer_last.pt   # Most recent epoch
```

### IV3-GRU Models

```
trained_models\iv3_gru\
└── cmb_optimal\
    ├── InceptionV3GRU_best.pt    # Best validation performance
    └── InceptionV3GRU_last.pt    # Most recent epoch
```

All models are trained on the combined dataset (fsl-105 + sample-105).

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**: Verify the path to your .pt file exists
2. **"NPZ file must contain 'X' key"**: Transformer requires keypoint data
3. **"NPZ file must contain 'X2048' key"**: IV3-GRU requires InceptionV3 features
4. **CUDA out of memory**: Use `--device cpu` to run on CPU
5. **"No module named 'mediapipe'"**: Install mediapipe for video processing: `pip install mediapipe`

### Getting Help

```powershell
python -m evaluation.prediction.predict --help
```

### Verify Installation

```powershell
# List available models
python -m evaluation.prediction.predict --list-models

# Test with demo file
python -m evaluation.prediction.predict ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --input data\demo\clip_0138_nice to meet you.npz
```
