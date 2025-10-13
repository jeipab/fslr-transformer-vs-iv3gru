# Prediction Guide

## Quick Start

### List Available Models

```powershell
python -m evaluation.prediction.predict --list-models
```

### Predict from NPZ File

```powershell
python -m evaluation.prediction.predict --model transformer --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt --input data\demo\clip_0138_nice to meet you.npz
```

### Predict from Video File

```powershell
python -m evaluation.prediction.predict --model iv3_gru --checkpoint trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt --input video.mp4
```

**Note**: Model input dimensions (156 for keypoints, 2048 for features) are auto-detected from checkpoints.

## Arguments

### Required

- `--model`: Model type (`transformer` or `iv3_gru`)
- `--checkpoint`: Path to model checkpoint (.pt file)
- `--input`: Input file (NPZ or video)

### Optional

- `--device`: Device (`cpu`, `cuda`, `auto` - default: `auto`)
- `--fps`: Target FPS for video processing (default: `30`)
- `--image-size`: Image size for video processing (default: `256`)
- `--output`: Save results to JSON file
- `--list-models`: List available model checkpoints

## Output

Console output:

```
Gloss: NICE TO MEET YOU (6) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

Return structure (when used programmatically or with `--output`):

```python
{
  'gloss_prediction': 6,
  'gloss_probability': 0.882,
  'category_prediction': 0,
  'category_probability': 0.774,
  'gloss_top5': [(6, 0.882), (1, 0.045), ...],
  'category_top3': [(0, 0.774), (1, 0.156), (2, 0.070)]
}
```

## Examples

### NPZ File

```powershell
python -m evaluation.prediction.predict --model transformer --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt --input data\demo\clip_0585_nine.npz
```

### Save to File

```powershell
python -m evaluation.prediction.predict --model transformer --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt --input data\demo\clip_1493_green.npz --output results.json
```

### Video File

```powershell
python -m evaluation.prediction.predict --model transformer --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt --input video.mp4 --fps 30 --image-size 256
```

### CPU Only

```powershell
python -m evaluation.prediction.predict --model iv3_gru --checkpoint trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt --input data\demo\clip_1765_fish.npz --device cpu
```

## Model Types

### Transformer

- **Input**: Keypoints (156-dim) or InceptionV3 features (2048-dim)
- **NPZ keys**: `X` for keypoints, `X2048` for features
- **Video**: Automatically extracts appropriate features

### IV3-GRU

- **Input**: InceptionV3 features (2048-dim)
- **NPZ key**: `X2048`
- **Video**: Automatically extracts InceptionV3 features

## Input Formats

### NPZ Files

**Transformer:**

- `X`: Keypoints `[T, 156]` (78 keypoints × 2 coordinates)
- `X2048`: InceptionV3 features `[T, 2048]` (alternative)

**IV3-GRU:**

- `X2048`: InceptionV3 features `[T, 2048]`

### Video Files

Supported: `.mp4`, `.avi`, `.mov`

Automatically extracts features based on model requirements.

## Dataset

- **Glosses**: 105 classes (0-104)
- **Categories**: 10 classes (0-9): GREETING, SURVIVAL, NUMBER, CALENDAR, DAYS, FAMILY, RELATIONSHIPS, COLOR, FOOD, DRINK

Label mappings: [LABEL_MAPPING_TABLE.md](../../data/labels/LABEL_MAPPING_TABLE.md)

## Demo Files

Located in `data\demo\`:

- `clip_0138_nice to meet you.npz`
- `clip_0585_nine.npz`
- `clip_1146_grandfather.npz`
- `clip_1493_green.npz`
- `clip_1765_fish.npz`
- `clip_1912_crab.npz`

## Model Checkpoints

```
trained_models\
├── transformer\cmb_optimal\
│   ├── SignTransformer_best.pt
│   └── SignTransformer_last.pt
└── iv3_gru\cmb_optimal\
    ├── InceptionV3GRU_best.pt
    └── InceptionV3GRU_last.pt
```

## Troubleshooting

**Checkpoint not found**: Verify .pt file path exists

**Missing NPZ key**: Transformer needs `X` or `X2048`, IV3-GRU needs `X2048`

**CUDA out of memory**: Use `--device cpu`

**Missing mediapipe**: Install for video processing: `pip install mediapipe`

**Help**: Run `python -m evaluation.prediction.predict --help`
