# Sharing Guide

## Purpose

The `shared/` directory contains deployment resources for running the project on Vast.ai instances, enabling remote access to the sign language recognition application.

## Directory Structure

```
shared/
├── for vast ai/
│   ├── VAST.AI_GUIDE.md              # Complete Vast.ai setup and deployment guide
│   ├── PREPROCESS_VAST.md            # Video preprocessing on Vast.ai
│   ├── visualization_vast_ai.py       # Vast.ai-compatible visualization component
│   └── validation_components_vast_ai.py  # Vast.ai-compatible validation component
└── SHARING_GUIDE.md                   # This file
```

## Vast.ai Deployment Resources

### Configuration Files

The `for vast ai/` folder contains modified Streamlit components that are compatible with Vast.ai environments:

- **visualization_vast_ai.py**: Modified visualization component without local-only features
- **validation_components_vast_ai.py**: Modified validation component for remote environments

### Documentation

- **VAST.AI_GUIDE.md**: Complete setup instructions including port configuration, data download, model deployment, and tunnel setup
- **PREPROCESS_VAST.md**: Instructions for preprocessing raw video data on Vast.ai instances

## Using Shared Resources

### For Vast.ai Deployment

1. Clone the repository on your Vast.ai instance
2. Follow steps in `shared/for vast ai/VAST.AI_GUIDE.md`
3. Replace local Streamlit components with Vast.ai versions:

```bash
cp "shared/for vast ai/visualization_vast_ai.py" "streamlit_app/components/visualization.py"
cp "shared/for vast ai/validation_components_vast_ai.py" "streamlit_app/components/validation_components.py"
```

### For Local Development

Local development uses the standard components in `streamlit_app/components/` without modifications. No changes needed.

## Data Structure

Processed data is stored in `data/processed/` with the following structure:

- **Training/validation splits**: `cmb_train/`, `cmb_val/`, `fsl_train/`, `fsl_val/`, `smp_train/`, `smp_val/`
- **Label files**: `cmb_train.csv`, `cmb_val.csv`, `fsl_train.csv`, `fsl_val.csv`, `smp_train.csv`, `smp_val.csv`
- **NPZ format**: Each file contains `X` (keypoints), `X2048` (InceptionV3 features), `mask`, `timestamps_ms`, and `meta`

## Trained Models

Pre-trained models are located in `trained_models/cmb/`:

- **Transformer**: `trained_models/cmb/transformer/`
- **IV3-GRU**: `trained_models/cmb/iv3_gru/`

Both models are trained on the combined dataset (fsl-105 + sample-105).

## Training with Processed Data

### Transformer Training

```powershell
python -m training.train ^
  --model transformer ^
  --keypoints-train data\processed\cmb_train ^
  --keypoints-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv
```

### IV3-GRU Training

```powershell
python -m training.train ^
  --model iv3_gru ^
  --features-train data\processed\cmb_train ^
  --features-val data\processed\cmb_val ^
  --labels-train-csv data\processed\cmb_train.csv ^
  --labels-val-csv data\processed\cmb_val.csv ^
  --feature-key X2048
```

## Validation

Validate NPZ files before training:

```powershell
python -m preprocessing.utils.validate_npz data\processed\cmb_train
python -m preprocessing.utils.validate_npz data\processed\cmb_val --require-x2048
```

## Notes

- NPZ files contain both keypoints (X) and InceptionV3 features (X2048)
- Vast.ai components are optimized for remote/headless environments
- Port 8081 must be exposed for Vast.ai deployment
- Use cloudflared for external tunnel access
