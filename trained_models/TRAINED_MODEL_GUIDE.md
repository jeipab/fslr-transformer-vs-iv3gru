# Trained Model Guide

Management and usage of trained model artifacts for the FSLR pipeline.

## Directory Structure

```
trained_models/
├── transformer/           # Transformer model checkpoints
│   ├── [model_folder_1]/
│   │   ├── [ModelName]_best.pt
│   │   ├── [ModelName]_last.pt
│   │   └── training_log.txt
│   ├── [model_folder_2]/
│   └── [additional_models]/
├── iv3_gru/             # InceptionV3-GRU model checkpoints
│   ├── [model_folder_1]/
│   │   ├── [ModelName]_best.pt
│   │   ├── [ModelName]_last.pt
│   │   └── training_log.txt
│   └── [additional_models]/
└── TRAINED_MODEL_GUIDE.md
```

## Model Organization

### Naming Convention

Model folders should follow the pattern: `{model_type}_{description}_{date}`

Examples:

- `transformer_low-acc_09-15` - Transformer model with low accuracy from Sep 15
- `transformer_high-acc_09-20` - Transformer model with high accuracy from Sep 20
- `transformer_baseline_09-25` - Transformer baseline model from Sep 25
- `iv3_gru_baseline_09-15` - IV3-GRU baseline model from Sep 15
- `iv3_gru_improved_09-20` - IV3-GRU improved model from Sep 20

### File Structure per Model

Each model folder should contain:

```
{model_name}/
├── {ModelName}_best.pt      # Best validation performance checkpoint
├── {ModelName}_last.pt      # Final epoch checkpoint
├── training_log.txt         # Training progress log
├── config.json              # Training configuration (optional)
└── README.md                # Model-specific documentation (optional)
```

## Model Checkpoints (.pt)

### Format

PyTorch checkpoint files containing:

- `model`: Model state_dict
- `epoch`: Training epoch number
- `best_metric`: Best validation metric achieved
- `optimizer`: Optimizer state (optional)
- `scheduler`: Learning rate scheduler state (optional)
- `config`: Training configuration (optional)

### Loading Models

```python
import torch
from models.transformer import SignTransformer
from models.iv3_gru import InceptionV3GRU

# Load Transformer model
transformer_checkpoint = torch.load('trained_models/transformer/[model_folder]/[ModelName]_best.pt')
transformer_model = SignTransformer(
    input_dim=156,
    emb_dim=256,
    num_heads=8,
    num_layers=6,
    num_gloss=105,
    num_cat=10
)
transformer_model.load_state_dict(transformer_checkpoint['model'])

# Load IV3-GRU model
iv3_checkpoint = torch.load('trained_models/iv3_gru/[model_folder]/[ModelName]_best.pt')
iv3_model = InceptionV3GRU(
    num_gloss=105,
    num_cat=10,
    gru_hidden=16,
    dropout=0.3
)
iv3_model.load_state_dict(iv3_checkpoint['model'])
```

## Training Logs

### Format

Training logs should be in CSV format with columns:

```csv
epoch,train_loss,val_loss,gloss_acc,cat_acc,lr
1,2.456,2.123,0.234,0.567,0.001
2,2.134,1.987,0.289,0.612,0.001
```

### Key Metrics

- `train_loss`: Training loss
- `val_loss`: Validation loss
- `gloss_acc`: Gloss classification accuracy
- `cat_acc`: Category classification accuracy
- `lr`: Learning rate

## Model Performance

### Evaluation Results

Store evaluation results in model folders:

```
[model_folder]/
├── evaluation_results/
│   ├── summary_metrics.csv
│   ├── confusion_matrix.png
│   ├── per_class_metrics.csv
│   └── predictions.csv
```

### Performance Tracking

Track key metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Per-Class Performance**: Individual class metrics
- **Confusion Matrix**: Classification error patterns

## Model Versioning

### Version Tags

Use semantic versioning for model releases:

- `v1.0.0` - Initial release
- `v1.1.0` - Performance improvements
- `v1.1.1` - Bug fixes
- `v2.0.0` - Major architecture changes

### Changelog

Maintain a changelog for each model:

```markdown
# Changelog

## v1.1.0 - 2024-09-20

- Improved accuracy from 85% to 89%
- Added data augmentation
- Reduced training time by 20%

## v1.0.0 - 2024-09-15

- Initial release
- Baseline transformer model
- 85% accuracy on validation set
```

## Sharing and Collaboration

### Git Integration

- Use `.gitignore` to exclude large model files
- Track configuration and documentation files
- Use Git LFS for essential model files if needed

### Model Sharing

For sharing trained models:

1. **Compress model folders** for easy transfer
2. **Include README** with performance metrics
3. **Document dependencies** and requirements
4. **Provide loading examples** for easy integration

### Team Collaboration

- **Naming convention** ensures consistent organization
- **Performance tracking** helps compare models
- **Documentation** enables easy model adoption
- **Version control** manages model evolution

## Model Lifecycle

### Development Phase

1. Train model with experimental configuration
2. Save checkpoints with descriptive names
3. Document performance and configuration
4. Compare with baseline models

### Production Phase

1. Select best performing model
2. Create production-ready checkpoint
3. Document final performance metrics
4. Archive experimental models

### Maintenance Phase

1. Monitor model performance over time
2. Retrain with new data if needed
3. Update documentation
4. Archive outdated models

## Best Practices

### File Management

- **Organize by model type** (transformer, iv3_gru)
- **Use descriptive folder names** with dates
- **Keep configuration files** for reproducibility
- **Document performance metrics** clearly

### Performance Monitoring

- **Track key metrics** consistently
- **Compare with baselines** regularly
- **Document improvements** and regressions
- **Maintain evaluation results** for reference

### Collaboration

- **Follow naming conventions** strictly
- **Update documentation** when adding models
- **Share performance insights** with team
- **Archive old models** to avoid confusion

## Integration with Training Pipeline

Models integrate with the training system:

```bash
# Train and save to trained_models/
python -m training.train --model transformer --save-dir trained_models/transformer/

# Load for inference
python -m streamlit_app.main --model-path trained_models/transformer/[model_folder]/[ModelName]_best.pt
```

See [Training Guide](../training/TRAINING_GUIDE.md) for detailed training instructions.
