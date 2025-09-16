# Sign Language Recognition Model Validation Guide

This guide provides comprehensive instructions for validating trained Sign Language Recognition models using the `validate.py` script.

## Command Line Usage

### Basic Syntax

```bash
python validate.py --model <model_type> --checkpoint <checkpoint_path> [options]
```

### Required Arguments

- `--model`: Model type to validate

  - Choices: `transformer`, `iv3_gru`
  - Example: `--model transformer`

- `--checkpoint`: Path to model checkpoint file
  - Must be a `.pt` file containing trained model weights
  - Example: `--checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt`

### Optional Arguments

- `--data-dir`: Directory containing validation NPZ files

  - Default: `../data/processed/seq prepro_30 fps_09-13`
  - Example: `--data-dir /path/to/validation/data`

- `--labels-csv`: Path to validation labels CSV file

  - Default: `../data/processed/val_labels.csv`
  - Must contain columns: `file`, `gloss`, `cat`, `occluded`
  - Example: `--labels-csv /path/to/labels.csv`

- `--output-dir`: Output directory for validation results

  - Default: `results-validate`
  - Example: `--output-dir my_validation_results`

- `--device`: Device to use for inference

  - Choices: `cpu`, `cuda`, `auto`
  - Default: `auto` (automatically selects best available device)
  - Example: `--device cuda`

- `--batch-size`: Batch size for evaluation

  - Default: `32`
  - Larger batch sizes may improve GPU utilization but require more memory
  - Example: `--batch-size 64`

- `--save-predictions`: Save individual predictions to JSON files

  - Creates detailed prediction files for each sample
  - Useful for error analysis and debugging
  - Example: `--save-predictions`

- `--verbose`: Enable detailed output and error traces
  - Useful for debugging issues
  - Example: `--verbose`

### Output Structure

The validation script generates comprehensive results in the specified output directory:

```
results-validate/
├── overall_results.json          # Complete validation metrics
├── occluded_results.json         # Occluded samples analysis
├── non_occluded_results.json     # Non-occluded samples analysis
├── per_class_results.json        # Per-class performance metrics
├── confusion_matrices.json       # Confusion matrices
├── complete_validation_results.json  # All results combined
└── individual_predictions/       # Individual predictions (if --save-predictions)
    ├── clip_0001_validation.json
    ├── clip_0002_validation.json
    └── ...
```

## Overview

The validation system provides detailed performance analysis of trained models on the validation dataset, including:

- **Overall Performance Metrics**: Accuracy, precision, recall, F1-score
- **Occlusion Analysis**: Performance comparison between occluded and non-occluded samples
- **Per-Class Metrics**: Detailed performance for each gloss and category class
- **Confusion Matrices**: Visual analysis of prediction patterns
- **Individual Predictions**: Sample-level prediction details (optional)

## Quick Start

### Basic Validation

```bash
# Validate a Transformer model
python validate.py --model transformer --checkpoint transformer/transformer_low-acc_09-15/SignTransformer_best.pt

# Validate an IV3-GRU model
python validate.py --model iv3_gru --checkpoint iv3_gru/model.pt
```

### Advanced Usage

```bash
# Custom batch size and save individual predictions
python validate.py --model transformer --checkpoint transformer/model.pt --batch-size 16 --save-predictions

# Use custom data paths
python validate.py --model transformer --checkpoint transformer/model.pt \
    --data-dir ../data/processed/seq\ prepro_30\ fps_09-13 \
    --labels-csv ../data/processed/val_labels.csv

# Force CPU usage
python validate.py --model iv3_gru --checkpoint iv3_gru/model.pt --device cpu
```

## Understanding Results

### Overall Results (`overall_results.json`)

Contains comprehensive metrics for the entire validation dataset:

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

The system automatically analyzes performance differences between occluded and non-occluded samples:

- **`occluded_results.json`**: Metrics for samples with occlusion (occluded=1)
- **`non_occluded_results.json`**: Metrics for samples without occlusion (occluded=0)

This analysis helps understand how occlusion affects model performance in real-world scenarios.

### Per-Class Results (`per_class_results.json`)

Detailed metrics for each individual class:

```json
{
  "gloss_per_class": {
    "0": {
      "precision": 0.9231,
      "recall": 0.8571,
      "f1-score": 0.8889,
      "support": 28
    },
    "1": {
      "precision": 0.8182,
      "recall": 0.9,
      "f1-score": 0.8571,
      "support": 20
    }
  }
}
```

### Confusion Matrices (`confusion_matrices.json`)

Confusion matrices for both gloss and category predictions, useful for identifying common misclassification patterns.

### Individual Predictions

When using `--save-predictions`, detailed prediction files are created for each sample:

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
  "gloss_top5": [
    ["GOOD MORNING (0)", 0.9234],
    ["GOOD AFTERNOON (1)", 0.0456],
    ["HELLO (3)", 0.0123],
    ["HOW ARE YOU (4)", 0.0089],
    ["GOOD EVENING (2)", 0.0056]
  ],
  "category_top3": [
    ["GREETING (0)", 0.9876],
    ["SURVIVAL (1)", 0.0087],
    ["FOOD (8)", 0.0034]
  ],
  "correct": {
    "gloss": true,
    "category": true
  }
}
```

## Performance Interpretation

### Accuracy Metrics

- **Gloss Accuracy**: Percentage of correctly predicted sign words
- **Category Accuracy**: Percentage of correctly predicted semantic categories
- **F1-Score**: Harmonic mean of precision and recall, useful for imbalanced datasets

### Occlusion Impact

Compare performance between occluded and non-occluded samples:

```bash
# Look for accuracy differences
Occluded Gloss Accuracy: 0.8234
Non-Occluded Gloss Accuracy: 0.8765
Accuracy Difference: +0.0531
```

A positive difference indicates the model performs better on non-occluded samples, which is expected for sign language recognition.

### Model Comparison

Use validation results to compare different models:

1. **Overall Performance**: Compare `gloss_accuracy` and `category_accuracy`
2. **Robustness**: Compare occlusion impact (smaller differences indicate better robustness)
3. **Class Balance**: Check per-class F1-scores for balanced performance across all signs

## Common Use Cases

### Model Selection

```bash
# Validate multiple model checkpoints
python validate.py --model transformer --checkpoint transformer/model_v1.pt --output-dir results_v1
python validate.py --model transformer --checkpoint transformer/model_v2.pt --output-dir results_v2

# Compare results to select best model
```

### Error Analysis

```bash
# Generate detailed predictions for analysis
python validate.py --model transformer --checkpoint transformer/model.pt --save-predictions

# Analyze incorrect predictions in individual_predictions/ directory
```

### Performance Monitoring

```bash
# Validate model after training
python validate.py --model transformer --checkpoint transformer/best_model.pt --batch-size 64

# Check if performance meets requirements
```

## Troubleshooting

### Common Issues

1. **File Not Found Errors**

   - Ensure NPZ files exist in the specified data directory
   - Check that file names in CSV match NPZ file names (without extension)

2. **CUDA Out of Memory**

   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

3. **Model Loading Errors**

   - Verify checkpoint file exists and is not corrupted
   - Ensure model type matches checkpoint (transformer vs iv3_gru)

4. **Empty Results**
   - Check that validation data directory contains NPZ files
   - Verify labels CSV has correct format and matching file names

### Performance Tips

1. **GPU Acceleration**: Use `--device cuda` for faster validation
2. **Batch Size**: Increase `--batch-size` for better GPU utilization (if memory allows)
3. **Parallel Processing**: The script automatically uses optimal settings for your hardware

### Debug Mode

Use `--verbose` flag for detailed error information:

```bash
python validate.py --model transformer --checkpoint model.pt --verbose
```

## Integration Examples

### Automated Validation Pipeline

```bash
#!/bin/bash
# validate_model.sh

MODEL_TYPE=$1
CHECKPOINT=$2
OUTPUT_DIR=$3

python validate.py \
    --model $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --batch-size 32 \
    --save-predictions

echo "Validation complete. Results saved to $OUTPUT_DIR"
```

### Model Comparison Script

```bash
#!/bin/bash
# compare_models.sh

MODELS=(
    "transformer/transformer_30_epochs/SignTransformer_best.pt"
    "transformer/transformer_100_epochs/SignTransformer_best.pt"
    "iv3_gru/iv3gru_30_epochs/InceptionV3GRU_best.pt"
    "iv3_gru/iv3gru_100_epochs/InceptionV3GRU_best.pt"
)

for model in "${MODELS[@]}"; do
    if [[ $model == *"transformer"* ]]; then
        MODEL_TYPE="transformer"
    else
        MODEL_TYPE="iv3_gru"
    fi

    OUTPUT_DIR="results_$(basename $(dirname $model))"

    echo "Validating $model..."
    python validate.py \
        --model $MODEL_TYPE \
        --checkpoint $model \
        --output-dir $OUTPUT_DIR \
        --batch-size 32

    echo "Results saved to $OUTPUT_DIR"
    echo "---"
done
```

## Best Practices

1. **Regular Validation**: Validate models after each training session
2. **Multiple Checkpoints**: Compare different epochs to find optimal stopping point
3. **Occlusion Analysis**: Always check occlusion impact for real-world deployment
4. **Error Analysis**: Use individual predictions to identify problematic samples
5. **Documentation**: Save validation results with model metadata for future reference

## Support

For issues or questions:

1. Check this guide for common solutions
2. Use `--verbose` flag for detailed error information
3. Verify data paths and file formats
4. Check model compatibility (transformer vs iv3_gru)
