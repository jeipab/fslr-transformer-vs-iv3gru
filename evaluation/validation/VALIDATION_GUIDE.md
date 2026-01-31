# Validation Guide

Guide for validating trained sign language recognition models on validation datasets.

## Quick Start

### Isolated Classification Validation

```powershell
python -m evaluation.validation.validate ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv
```

### Continuous CTC Validation

```powershell
python -m evaluation.validation.validate ^
  --model transformer_continuous ^
  --checkpoint trained_models\transformer\FSL105_ctc\SignTransformerCtc_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv
```

## Arguments

### Required

- `--model`: Model type
  - Isolated: `transformer_isolated`, `iv3_gru_isolated`, `mediapipe_gru_isolated`
  - Continuous: `transformer_continuous`, `iv3_gru_continuous`, `mediapipe_gru_continuous`
- `--checkpoint`: Path to model checkpoint (.pt file)
- `--data-dir`: Directory containing validation NPZ files
- `--labels-csv`: Path to validation labels CSV file

### Optional

- `--output-dir`: Output directory for results (default: `results-validate`)
- `--device`: Device to use (`cpu`, `cuda`, `auto` - default: `auto`)
- `--batch-size`: Batch size for evaluation (default: `32`)
- `--save-predictions`: Save individual predictions to JSON files
- `--signer-filter`: Filter by specific signer(s) (e.g., `--signer-filter S1 S2`)
- `--category-filter`: Filter by specific category(ies) (e.g., `--category-filter 0 1 2`)
- `--verbose`: Enable detailed output

## Output

Results are saved to the output directory:

- `overall_results.json`: Overall accuracy, precision, recall, F1-score
- `occluded_results.json`: Metrics for occluded samples
- `non_occluded_results.json`: Metrics for non-occluded samples
- `per_class_results.json`: Per-class metrics for each gloss and category
- `per_signer_results.json`: Per-signer performance metrics
- `per_category_results.json`: Per-category performance metrics
- `duration_analysis.json`: Performance by video duration bins
- `confusion_matrices.json`: Confusion matrices with TP/FP/TN/FN breakdowns
- `complete_validation_results.json`: Complete results dictionary
- `individual_predictions/`: Individual prediction JSON files (if `--save-predictions`)

## Metrics

### Overall Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Top-K Accuracy

- **Top-1**: Standard accuracy
- **Top-5**: Ground truth in top 5 predictions
- **Top-10**: Ground truth in top 10 predictions (gloss only)

### Per-Class Metrics

For each gloss and category:
- **TP (True Positive)**: Correct predictions
- **FP (False Positive)**: Incorrect predictions
- **TN (True Negative)**: Correct rejections
- **FN (False Negative)**: Missed predictions
- **Precision, Recall, F1-Score**: Derived from TP/FP/FN

### Occlusion Analysis

Separate metrics for:
- **Occluded samples**: Samples with occlusion flags
- **Non-occluded samples**: Clean samples without occlusion

## Examples

### Filter by Signer

```powershell
python -m evaluation.validation.validate ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv ^
  --signer-filter S0 S1
```

### Save Individual Predictions

```powershell
python -m evaluation.validation.validate ^
  --model transformer_isolated ^
  --checkpoint trained_models\transformer\FSL105_classification\SignTransformer_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv ^
  --save-predictions ^
  --output-dir results\validation_detailed
```

### CPU Only

```powershell
python -m evaluation.validation.validate ^
  --model iv3_gru_isolated ^
  --checkpoint trained_models\iv3_gru\FSL105_classification\InceptionV3GRU_best.pt ^
  --data-dir data\processed\FSL105_val ^
  --labels-csv data\processed\FSL105_val.csv ^
  --device cpu
```

## Troubleshooting

**Checkpoint not found**: Verify .pt file path exists

**Missing NPZ key**: Ensure NPZ files contain required keys (`X` for transformer, `X2048` for IV3-GRU)

**CUDA out of memory**: Use `--device cpu` or reduce `--batch-size`

**No valid samples**: Check that NPZ files exist in data-dir and match filenames in labels CSV

**Help**: Run `python -m evaluation.validation.validate --help`
