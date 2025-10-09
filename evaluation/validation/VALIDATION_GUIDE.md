# Sign Language Recognition Model Validation Guide

Validate trained Sign Language Recognition models using `validate.py`.

## Usage

### Basic Command

```powershell
python -m evaluation.validation.validate --model <model_type> --checkpoint <checkpoint_path> [options]
```

### Required Arguments

- `--model`: Model type (`transformer` or `iv3_gru`)
- `--checkpoint`: Path to model checkpoint (.pt file)

### Optional Arguments

- `--data-dir`: Directory with validation NPZ files (default: `data\processed\cmb_val`)
- `--labels-csv`: Path to labels CSV (default: `data\processed\cmb_val.csv`)
- `--output-dir`: Output directory (default: `results-validate`)
- `--device`: Device (`cpu`, `cuda`, `auto`) (default: `auto`)
- `--batch-size`: Batch size (default: `32`)
- `--save-predictions`: Save individual predictions to JSON files
- `--verbose`: Enable detailed output

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

## Features

- Overall performance metrics (accuracy, precision, recall, F1-score)
- Occlusion analysis (occluded vs non-occluded samples)
- Per-class metrics for each gloss and category
- Confusion matrices for error analysis
- Individual predictions (optional)

## Examples

### Basic Validation

**Transformer model:**

```powershell
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt
```

**IV3-GRU model:**

```powershell
python -m evaluation.validation.validate ^
  --model iv3_gru ^
  --checkpoint trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt
```

### Advanced Usage

**Custom batch size and save predictions:**

```powershell
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --batch-size 16 ^
  --save-predictions
```

**Custom data paths:**

```powershell
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --data-dir data\processed\fsl_val ^
  --labels-csv data\processed\fsl_val.csv
```

**Force CPU usage:**

```powershell
python -m evaluation.validation.validate ^
  --model iv3_gru ^
  --checkpoint trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt ^
  --device cpu
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

- `occluded_results.json`: Metrics for samples marked as occluded during preprocessing
- `non_occluded_results.json`: Metrics for clean samples

Compare to assess model robustness to occlusion.

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
    "6": {
      "class": "NICE TO MEET YOU (6)",
      "precision": 0.9,
      "recall": 0.8182,
      "f1-score": 0.8571,
      "occurrences": 22
    }
  }
}
```

**Column Definitions:**

- **class**: Display name showing actual label with ID (format: "LABEL_NAME (ID)")
- **precision**: True positives / (True positives + False positives)
- **recall**: True positives / (True positives + False negatives)
- **f1-score**: Harmonic mean of precision and recall
- **occurrences**: Number of actual occurrences of this class in validation dataset

### Individual Predictions

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

## Performance Metrics

- **Gloss Accuracy**: Percentage of correctly predicted sign words (105 classes)
- **Category Accuracy**: Percentage of correctly predicted semantic categories (10 classes)
- **F1-Score**: Harmonic mean of precision and recall (better for imbalanced classes)

### Understanding Per-Class Metrics

The per-class analysis provides detailed performance metrics for each individual sign class:

- **Precision**: Of all predictions for this class, how many were correct?
- **Recall**: Of all actual instances of this class, how many were correctly identified?
- **F1-Score**: Balanced measure combining precision and recall
- **Occurrences**: Total number of samples belonging to this class in validation dataset

**Example**: If a class has occurrences=50, there are 50 samples of that sign in the validation set. This helps interpret whether low performance metrics are due to class imbalance (low occurrences) or actual model difficulty with that class.

### Occlusion Impact

Compare occluded vs non-occluded samples:

```
Occluded Gloss Accuracy:     0.8234
Non-Occluded Gloss Accuracy: 0.8765
Accuracy Difference:         +0.0531
```

Positive difference indicates better performance on non-occluded samples.

### Model Comparison

1. Compare overall accuracy metrics
2. Check occlusion impact (smaller differences = more robust model)
3. Review per-class F1-scores for balanced performance
4. Analyze confusion matrices for error patterns

## Use Cases

### Model Selection

```powershell
# Compare multiple models
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --output-dir results_best

python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_last.pt ^
  --output-dir results_last
```

### Error Analysis

```powershell
# Generate detailed predictions for analysis
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --save-predictions ^
  --verbose
```

### Performance Monitoring

```powershell
# Validate after training
python -m evaluation.validation.validate ^
  --model transformer ^
  --checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt ^
  --batch-size 64
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Verify NPZ files exist in data directory and CSV file paths are correct
2. **CUDA Out of Memory**: Reduce batch size (`--batch-size 16`) or use CPU (`--device cpu`)
3. **Model Loading Errors**: Verify checkpoint exists and model type matches checkpoint architecture
4. **Empty Results**: Check data directory contains NPZ files and CSV format is correct
5. **Mismatched Labels**: Ensure num_gloss=105 and num_cat=10 in model checkpoint

### Performance Tips

- Use `--device cuda` for faster validation (if GPU available)
- Increase `--batch-size` for better GPU utilization (if memory allows)
- Use `--verbose` for detailed progress and error information
- Skip `--save-predictions` for faster overall validation

## Scripts

### Automated Validation (PowerShell)

```powershell
# validate_model.ps1
param(
    [string]$ModelType,
    [string]$Checkpoint,
    [string]$OutputDir
)

python -m evaluation.validation.validate `
    --model $ModelType `
    --checkpoint $Checkpoint `
    --output-dir $OutputDir `
    --batch-size 32 `
    --save-predictions
```

**Usage:**

```powershell
.\validate_model.ps1 -ModelType transformer -Checkpoint trained_models\transformer\cmb_optimal\SignTransformer_best.pt -OutputDir results_transformer
```

### Model Comparison (PowerShell)

```powershell
# compare_models.ps1
$models = @(
    @{Type="transformer"; Path="trained_models\transformer\cmb_optimal\SignTransformer_best.pt"},
    @{Type="iv3_gru"; Path="trained_models\iv3_gru\cmb_optimal\InceptionV3GRU_best.pt"}
)

foreach ($model in $models) {
    $outputDir = "results_$($model.Type)"

    python -m evaluation.validation.validate `
        --model $model.Type `
        --checkpoint $model.Path `
        --output-dir $outputDir `
        --batch-size 32
}
```

## Best Practices

1. Validate models after each training session
2. Compare best vs last checkpoint to find optimal model
3. Check occlusion impact for real-world deployment readiness
4. Use individual predictions for detailed error analysis
5. Save validation results with timestamps for tracking
6. Monitor per-class metrics to identify difficult signs
7. Compare multiple models on same validation set for fair comparison

## Dataset Information

- **Training set**: `data\processed\cmb_train` (80% of data)
- **Validation set**: `data\processed\cmb_val` (20% of data)
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
- **Dataset**: Combined fsl-105 + sample-105

For label mappings, see [LABEL_MAPPING_TABLE.md](../../data/labels/LABEL_MAPPING_TABLE.md)
