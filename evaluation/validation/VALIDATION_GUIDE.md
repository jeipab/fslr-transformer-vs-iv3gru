# Sign Language Recognition Model Validation Guide

Validate trained Sign Language Recognition models using `validate.py`.

## Usage

### Basic Command

```bash
python validate.py --model <model_type> --checkpoint <checkpoint_path> [options]
```

### Required Arguments

- `--model`: Model type (`transformer` or `iv3_gru`)
- `--checkpoint`: Path to model checkpoint (.pt file)

### Optional Arguments

- `--data-dir`: Directory with validation NPZ files (default: `../data/processed/seq prepro_30 fps_09-13`)
- `--labels-csv`: Path to labels CSV (default: `../data/processed/val_labels.csv`)
- `--output-dir`: Output directory (default: `results-validate`)
- `--device`: Device (`cpu`, `cuda`, `auto`) (default: `auto`)
- `--batch-size`: Batch size (default: `32`)
- `--save-predictions`: Save individual predictions to JSON files
- `--verbose`: Enable detailed output

### Output Files

```
results-validate/
├── overall_results.json          # Overall metrics
├── occluded_results.json         # Occluded samples
├── non_occluded_results.json     # Non-occluded samples
├── per_class_results.json        # Per-class metrics
├── confusion_matrices.json       # Confusion matrices
├── complete_validation_results.json  # All results
└── individual_predictions/       # Individual predictions (if --save-predictions)
    ├── clip_0001_validation.json
    └── ...
```

## Features

- Overall performance metrics (accuracy, precision, recall, F1-score)
- Occlusion analysis (occluded vs non-occluded samples)
- Per-class metrics for each gloss and category
- Confusion matrices
- Individual predictions (optional)

## Examples

### Basic Validation

```bash
# Transformer model
python validate.py --model transformer --checkpoint transformer/model.pt

# IV3-GRU model
python validate.py --model iv3_gru --checkpoint iv3_gru/model.pt
```

### Advanced Usage

```bash
# Custom batch size and save predictions
python validate.py --model transformer --checkpoint transformer/model.pt --batch-size 16 --save-predictions

# Custom data paths
python validate.py --model transformer --checkpoint transformer/model.pt \
    --data-dir ../data/processed/seq\ prepro_30\ fps_09-13 \
    --labels-csv ../data/processed/val_labels.csv

# Force CPU usage
python validate.py --model iv3_gru --checkpoint iv3_gru/model.pt --device cpu
```

## Results

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

Detailed metrics for each class with actual labels:

```json
{
  "gloss_per_class": {
    "0": {
      "class": "GOOD MORNING (0)",
      "precision": 0.9231,
      "recall": 0.8571,
      "f1-score": 0.8889,
      "occurrences": 28
    },
    "1": {
      "class": "THANK YOU (1)",
      "precision": 0.9,
      "recall": 0.8182,
      "f1-score": 0.8571,
      "occurrences": 22
    }
  }
}
```

**Column Definitions:**

- **class**: Display name showing actual label with class ID (format: "LABEL_NAME (ID)")
- **precision**: True positives / (True positives + False positives)
- **recall**: True positives / (True positives + False negatives)
- **f1-score**: Harmonic mean of precision and recall
- **occurrences**: Number of actual occurrences of this class in the validation dataset

### Individual Predictions

When using `--save-predictions`:

```json
{
  "file": "clip_0001_good morning",
  "ground_truth": {
    "gloss": "GOOD MORNING (0)",
    "category": "GREETING (0)",
    "occluded": false
  },
  "prediction": {
    "gloss": "GOOD MORNING (0)",
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

## Performance Metrics

- **Gloss Accuracy**: Percentage of correctly predicted sign words
- **Category Accuracy**: Percentage of correctly predicted semantic categories
- **F1-Score**: Harmonic mean of precision and recall

### Understanding Per-Class Metrics

The per-class analysis provides detailed performance metrics for each individual sign class:

- **Precision**: Of all predictions for this class, how many were correct?
- **Recall**: Of all actual instances of this class, how many were correctly identified?
- **F1-Score**: Balanced measure combining precision and recall
- **Occurrences**: Total number of samples belonging to this class in the validation dataset

**Example**: If a class has occurrences=50, it means there are 50 samples of that sign in the validation set. This helps interpret whether low performance metrics are due to class imbalance (low occurrences) or actual model difficulty with that class.

**Note**: The per-class results now include a "class" field that shows actual sign language labels with class IDs (like "GOOD MORNING (0)", "THANK YOU (1)"), making the results more readable and interpretable, similar to the format used in individual predictions.

### Occlusion Impact

Compare occluded vs non-occluded samples:

```
Occluded Gloss Accuracy: 0.8234
Non-Occluded Gloss Accuracy: 0.8765
Accuracy Difference: +0.0531
```

Positive difference indicates better performance on non-occluded samples.

### Model Comparison

1. Compare overall accuracy metrics
2. Check occlusion impact (smaller differences = better performance)
3. Review per-class F1-scores for balanced performance

## Use Cases

### Model Selection

```bash
# Compare multiple models
python validate.py --model transformer --checkpoint transformer/model_v1.pt --output-dir results_v1
python validate.py --model transformer --checkpoint transformer/model_v2.pt --output-dir results_v2
```

### Error Analysis

```bash
# Generate detailed predictions
python validate.py --model transformer --checkpoint transformer/model.pt --save-predictions
```

### Performance Monitoring

```bash
# Validate after training
python validate.py --model transformer --checkpoint transformer/best_model.pt --batch-size 64
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Check NPZ files exist and CSV file names match
2. **CUDA Out of Memory**: Reduce batch size (`--batch-size 16`) or use CPU (`--device cpu`)
3. **Model Loading Errors**: Verify checkpoint exists and model type matches
4. **Empty Results**: Check data directory contains NPZ files and CSV format

### Performance Tips

- Use `--device cuda` for faster validation
- Increase `--batch-size` for better GPU utilization (if memory allows)
- Use `--verbose` for detailed error information

## Scripts

### Automated Validation

```bash
#!/bin/bash
MODEL_TYPE=$1
CHECKPOINT=$2
OUTPUT_DIR=$3

python validate.py \
    --model $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --batch-size 32 \
    --save-predictions
```

### Model Comparison

```bash
#!/bin/bash
MODELS=(
    "transformer/transformer_30_epochs/SignTransformer_best.pt"
    "transformer/transformer_100_epochs/SignTransformer_best.pt"
)

for model in "${MODELS[@]}"; do
    MODEL_TYPE="transformer"
    OUTPUT_DIR="results_$(basename $(dirname $model))"

    python validate.py \
        --model $MODEL_TYPE \
        --checkpoint $model \
        --output-dir $OUTPUT_DIR \
        --batch-size 32
done
```

## Best Practices

1. Validate models after each training session
2. Compare different epochs to find optimal stopping point
3. Check occlusion impact for real-world deployment
4. Use individual predictions for error analysis
5. Save validation results with model metadata
