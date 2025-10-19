# Updated Validation Guide

This guide covers the enhanced validation system that now supports 6-column CSV format with signer and duration information.

## New Features

### 1. Enhanced CSV Format
The validation system now expects CSV files with 6 columns:
- `file`: File identifier (without .npz extension)
- `gloss`: Ground truth gloss label (integer)
- `cat`: Ground truth category label (integer)
- `occluded`: Occlusion flag (0 or 1)
- `signer`: Signer ID (string, e.g., "S1", "S2")
- `duration`: Duration in seconds (float)

### 2. Signer-Aware Validation
- **Per-signer accuracy metrics**: Individual performance analysis for each signer
- **Signer filtering**: Validate specific signer(s) using `--signer-filter`
- **Signer comparison**: Compare performance across different signers

### 3. Duration Analysis
- **Duration statistics**: Mean, std, min, max, median duration
- **Duration-based performance**: Accuracy analysis by duration bins
- **Duration bins**: 0-1s, 1-2s, 2-3s, 3-5s, 5-10s, 10s+

### 4. Enhanced Confusion Matrix
- **Proper TP, FP, TN, FN calculations**: Accurate per-class metrics
- **Class-specific metrics**: Precision, Recall, F1-Score for each class
- **Detailed confusion analysis**: Better understanding of model errors

### 5. Per-Category Metrics
- **Category-specific accuracy**: Performance analysis by sign category
- **Category filtering**: Validate specific categories using `--category-filter`
- **Category comparison**: Compare performance across different categories

## Usage Examples

### Basic Validation
```bash
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --data-dir data/processed/fsl_val \
    --labels-csv data/processed/fsl_val.csv
```

### Signer-Specific Validation
```bash
# Validate only signer S1
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --signer-filter S1

# Validate multiple signers
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --signer-filter S1 S2 S3
```

### Category-Specific Validation
```bash
# Validate only greeting category (category 0)
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --category-filter 0

# Validate multiple categories
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --category-filter 0 1 2
```

### Combined Filtering
```bash
# Validate signer S1 on greeting and survival categories
python evaluation/validation/validate.py \
    --model transformer \
    --checkpoint trained_models/transformer/optimal/model.pt \
    --signer-filter S1 \
    --category-filter 0 1
```

## Output Files

The validation system now generates additional output files:

### New Result Files
- `per_signer_results.json`: Per-signer accuracy metrics
- `per_category_results.json`: Per-category accuracy metrics
- `duration_analysis.json`: Duration-based performance analysis

### Enhanced Confusion Matrix
- `confusion_matrices.json`: Now includes TP, FP, TN, FN calculations
- `gloss_class_metrics`: Per-class metrics for gloss predictions
- `category_class_metrics`: Per-class metrics for category predictions

### Updated Individual Predictions
Individual prediction files now include:
- `signer`: Signer ID
- `duration`: Duration in seconds
- Enhanced metadata for better analysis

## CSV Format Example

```csv
file,gloss,cat,occluded,signer,duration
clip_0001,0,0,0,S1,2.5
clip_0002,1,0,1,S1,3.2
clip_0003,2,1,0,S2,1.8
clip_0004,3,1,0,S2,2.9
clip_0005,4,2,1,S1,4.1
```

## New Metrics Explained

### Per-Signer Metrics
- **Gloss Accuracy**: Sign recognition accuracy for each signer
- **Category Accuracy**: Category classification accuracy for each signer
- **Sample Count**: Number of samples per signer

### Duration Analysis
- **Overall Stats**: Mean, std, min, max, median duration
- **Bin Analysis**: Performance metrics for different duration ranges
- **Duration Impact**: How duration affects model performance

### Enhanced Confusion Matrix
- **TP (True Positive)**: Correctly predicted positive cases
- **FP (False Positive)**: Incorrectly predicted positive cases
- **TN (True Negative)**: Correctly predicted negative cases
- **FN (False Negative)**: Incorrectly predicted negative cases
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)

## Migration from Old Format

If you have CSV files with the old 4-column format, you need to add the `signer` and `duration` columns:

1. **Add signer column**: Assign signer IDs (e.g., "S1", "S2", etc.)
2. **Add duration column**: Calculate or estimate duration in seconds
3. **Update column order**: Ensure columns are in the correct order

## Troubleshooting

### Common Issues
1. **Missing columns**: Ensure your CSV has all 6 required columns
2. **Data type errors**: Check that gloss/cat are integers, duration is float
3. **File not found**: Verify NPZ files exist in the data directory
4. **Filter issues**: Check that signer/category filters match your data

### Validation
Use the test script to verify your setup:
```bash
python test_validation.py
```

## Performance Considerations

- **Memory usage**: Signer and duration analysis may increase memory usage
- **Processing time**: Additional metrics computation may slow down validation
- **Storage**: More output files require more disk space

## Future Enhancements

Potential future improvements:
- Signer-specific confusion matrices
- Duration-based confusion matrices
- Interactive visualization tools
- Advanced statistical analysis
- Cross-signer performance comparison
