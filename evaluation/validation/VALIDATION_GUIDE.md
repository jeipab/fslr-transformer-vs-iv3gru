# Sign Language Recognition Model Validation Guide

## Usage

### Basic Command

```powershell
python -m evaluation.validation.validate --model <model_type> --checkpoint <checkpoint_path> [options]
```

### Required Arguments

- `--model`: Model type (`transformer` or `iv3_gru`)
- `--checkpoint`: Path to model checkpoint (.pt file)

### Optional Arguments

- `--data-dir`: Directory with validation NPZ files (default: `data/processed/fsl_val`)
- `--labels-csv`: Path to labels CSV (default: `data/processed/fsl_val.csv`)
- `--output-dir`: Output directory (default: `results-validate`)
- `--device`: Device (`cpu`, `cuda`, `auto`) (default: `auto`)
- `--batch-size`: Batch size (default: `32`)
- `--save-predictions`: Save individual predictions to JSON files
- `--verbose`: Enable detailed output

**Note**: Model input dimensions (156 for keypoints, 2048 for features) are auto-detected from checkpoints.

### Output Files

```
results-validate/
├── overall_results.json                   # Overall metrics
├── occluded_results.json                  # Metrics for occluded samples
├── non_occluded_results.json              # Metrics for non-occluded samples
├── per_class_results.json                 # Per-class metrics
├── confusion_matrices.json                # Confusion matrices
├── complete_validation_results.json       # All results combined
└── individual_predictions/                # Individual predictions (if --save-predictions)
    ├── clip_0001_validation.json
    └── ...
```

## Examples

### Transformer Model

```bash
python -m evaluation.validation.validate --model transformer --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt
```

### IV3-GRU Model

```bash
python -m evaluation.validation.validate --model iv3_gru --checkpoint trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt
```

### With Options

```bash
python -m evaluation.validation.validate --model transformer --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt --batch-size 16 --save-predictions --device cpu
```

### Custom Dataset

```bash
python -m evaluation.validation.validate --model transformer --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt --data-dir data/processed/fsl_train --labels-csv data/processed/fsl_train.csv
```

## Output Format

### Overall Results (`overall_results.json`)

```json
{
  "gloss_accuracy": 0.8547,
  "category_accuracy": 0.9123,
  "gloss_precision": 0.8456,
  "gloss_recall": 0.8547,
  "gloss_f1_score": 0.8491,
  "category_precision": 0.9087,
  "category_recall": 0.9123,
  "category_f1_score": 0.9105,
  "num_samples": 428
}
```

### Occlusion Analysis

- `occluded_results.json`: Metrics for occluded samples
- `non_occluded_results.json`: Metrics for non-occluded samples

### Per-Class Results (`per_class_results.json`)

```json
{
  "gloss_per_class": {
    "0": {
      "class": "GOOD MORNING (0)",
      "precision": 0.9231,
      "recall": 0.8571,
      "f1-score": 0.8889,
      "occurrences": 28
    }
  },
  "category_per_class": {
    "0": {
      "class": "GREETING (0)",
      "precision": 0.9156,
      "recall": 0.9089,
      "f1-score": 0.9122,
      "occurrences": 98
    }
  }
}
```

- **class**: Label name with ID
- **precision**: TP / (TP + FP)
- **recall**: TP / (TP + FN)
- **f1-score**: Harmonic mean of precision and recall
- **occurrences**: Number of samples in validation set

### Individual Predictions (`individual_predictions/`)

When using `--save-predictions`:

```json
{
  "file": "clip_0138_nice to meet you",
  "ground_truth": {
    "gloss": "NICE TO MEET YOU (6)",
    "category": "GREETING (0)",
    "occluded": false
  },
  "prediction": {
    "gloss": "NICE TO MEET YOU (6)",
    "category": "GREETING (0)",
    "gloss_probability": 0.9234,
    "category_probability": 0.9876
  },
  "correct": {
    "gloss": true,
    "category": true
  }
}
```

## Metrics

- **Accuracy**: Correctly predicted / Total samples
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

Metrics computed for:

- Overall performance (all samples)
- Occluded subset
- Non-occluded subset
- Per-class (each gloss and category)

## Model Comparison

```bash
# Transformer
python -m evaluation.validation.validate --model transformer --checkpoint trained_models/transformer/optimal/SignTransformer_best.pt --output-dir results_transformer

# IV3-GRU
python -m evaluation.validation.validate --model iv3_gru --checkpoint trained_models/iv3_gru/optimal/InceptionV3GRU_best.pt --output-dir results_iv3gru
```

Compare `overall_results.json` from each output directory.

## Troubleshooting

**File Not Found**: Verify NPZ files exist in data directory and CSV paths are correct

**CUDA Out of Memory**: Reduce batch size (`--batch-size 16`) or use CPU (`--device cpu`)

**Model Loading Errors**: Check checkpoint path and model type match architecture

**Empty Results**: Verify data directory contains NPZ files and CSV format is correct

## Dataset

- **Training**: `data/processed/fsl_train` (80%)
- **Validation**: `data/processed/fsl_val` (20%)
- **Glosses**: 105 classes (0-104)
- **Categories**: 10 classes (0-9): GREETING, SURVIVAL, NUMBER, CALENDAR, DAYS, FAMILY, RELATIONSHIPS, COLOR, FOOD, DRINK

Label mappings: [LABEL_MAPPING_TABLE.md](../../data/labels/LABEL_MAPPING_TABLE.md)
