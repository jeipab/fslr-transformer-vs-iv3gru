# PANSINAYAN

### *Where Every Sign Gets Attention*

**Filipino Sign Language Recognition** — a research codebase that compares a Multi-Head Attention Transformer with an InceptionV3-GRU baseline on isolated signs and continuous sequences.

PANSINAYAN covers the full pipeline: video preprocessing, model training, evaluation, and an interactive Streamlit app. Isolated models classify **105 glosses** across **10 categories**. Continuous models use CTC decoding to recognize unsegmented sign sequences.

## Highlights

- **Fair architecture comparison** — Transformer (MediaPipe keypoints) vs InceptionV3-GRU (visual features), trained under matching conditions
- **Isolated + continuous** — classification heads for single signs; CTC heads for variable-length gloss sequences
- **Occlusion-aware preprocessing** — 178-D keypoints, 2048-D InceptionV3 features, and visibility masks
- **PANSINAYAN app** — upload video or NPZ, compare models, inspect attention and CTC alignments
- **Analysis toolkit** — confusion matrices, occlusion breakdowns, and paired statistical tests

## Dataset

| | |
| --- | --- |
| **Glosses** | 105 Filipino sign words (IDs 0–104) |
| **Categories** | 10: Greeting, Survival, Number, Calendar, Days, Family, Relationships, Color, Food, Drink |
| **Training set** | FSL-105 (80/20 train/val) |
| **Labels** | [Label Mapping Table](data/labels/LABEL_MAPPING_TABLE.md) |

Raw videos and full processed splits are not stored in git. Checkpoints are stored with [Git LFS](https://git-lfs.github.com/). After cloning:

```bash
git lfs install
git lfs pull --include="trained_models/**/*.pt"
```

Place checkpoints as described in [Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md).

## Models

### Isolated (classification)

| Model | Input | Role |
| --- | --- | --- |
| **SignTransformer** | Keypoints `[T, 178]` | Attention encoder; lighter and interpretable |
| **InceptionV3-GRU** | Features `[T, 2048]` | CNN + GRU baseline with ImageNet transfer |

Each model predicts a **gloss** (105 classes) and a **category** (10 classes).

### Continuous (CTC)

| Model | Input | Output |
| --- | --- | --- |
| **SignTransformerCtc** | Keypoints `[T, 178]` | Variable-length gloss sequence |
| **InceptionV3GRUCtc** | Features `[T, 2048]` | Variable-length gloss sequence |

CTC training does not require frame-level alignment. See [Model Guide](models/MODEL_GUIDE.md) for architectures.

A lightweight **MediaPipe-GRU** variant exists for prototyping and mobile export (`training/export_mobile.py`). It is not part of the main comparison and is hidden in the Streamlit app.

## Quick Start

**Requirements:** Python 3.9–3.11

```bash
git clone https://github.com/jeipab/fslr-transformer-vs-iv3gru.git
cd fslr-transformer-vs-iv3gru

python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
git lfs pull --include="trained_models/**/*.pt"
```

### PANSINAYAN app

```bash
streamlit run run_app.py
```

Opens at [http://localhost:8501](http://localhost:8501). For access from other devices on the same network, run `python show_network_info.py`.

The app supports NPZ files and raw video (MP4, MOV, AVI), dual-model comparison, keypoint animation, and occlusion-aware validation. See [Tool Guide](streamlit_app/TOOL_GUIDE.md).

### Predict from the CLI

```bash
python -m evaluation.prediction.predict \
  --model transformer_isolated \
  --checkpoint trained_models/transformer/FSL105_classification/SignTransformer_best.pt \
  --input path/to/clip.npz   # or a .mp4
```

```
Gloss: NICE TO MEET YOU (6) (confidence: 0.882)
Category: GREETING (0) (confidence: 0.774)
```

## Project Structure

```
fslr-transformer-vs-iv3gru/
├── data/                 Labels, splitting, demo/sample clips
├── preprocessing/        Keypoints, InceptionV3 features, occlusion, CTC sequences
├── models/               SignTransformer, InceptionV3-GRU, MediaPipe-GRU
├── training/             Isolated + CTC training; mobile export
├── evaluation/           Prediction, validation, CTC metrics
├── metrics/              Confusion matrices, occlusion splits, statistical tests
├── streamlit_app/        PANSINAYAN web UI
├── trained_models/       Checkpoints (Git LFS)
├── run_app.py            App launcher
└── requirements.txt
```

## Pipeline

### 1. Preprocess videos

```bash
python -m preprocessing.core.preprocess data/raw/videos data/processed/output \
  --write-keypoints --write-iv3-features \
  --workers 8 --batch-size 32 --target-fps 30 --disable-parquet
```

Each `.npz` contains `X [T,178]`, `X2048 [T,2048]`, `mask [T,89]`, `timestamps_ms [T]`, and occlusion metadata. Details: [Preprocessing Guide](preprocessing/docs/PREPROCESS_GUIDE.MD).

### 2. Split data

```bash
python data/splitting/assign.py
python data/splitting/data_split.py \
  --processed-root data/processed/FSL-105 \
  --labels data/processed/FSL-105/labels.csv \
  --out-root data/processed --copy --train-ratio 0.8 \
  --train-dir FSL105_train --val-dir FSL105_val \
  --train-csv FSL105_train.csv --val-csv FSL105_val.csv
```

Details: [Data Guide](data/DATA_GUIDE.md).

### 3. Train

Isolated Transformer:

```bash
python -m training.train \
  --model transformer_isolated \
  --keypoints-train data/processed/FSL105_train \
  --keypoints-val data/processed/FSL105_val \
  --labels-train-csv data/processed/FSL105_train.csv \
  --labels-val-csv data/processed/FSL105_val.csv \
  --num-gloss 105 --num-cat 10 --epochs 100 --batch-size 32 \
  --amp --auto-workers --auto-batch-size
```

Isolated InceptionV3-GRU: use `--model iv3_gru_isolated`, `--features-train` / `--features-val`, and `--feature-key X2048`.

Continuous (CTC): `--model transformer_continuous` or `iv3_gru_continuous`, plus `--grad-clip 1.0`.

Details: [Training Guide](training/TRAINING_GUIDE.md).

### 4. Validate

```bash
python -m preprocessing.utils.validate_npz data/processed/FSL105_val --require-x2048

python -m evaluation.validation.validate \
  --model transformer_isolated \
  --checkpoint trained_models/transformer/FSL105_classification/SignTransformer_best.pt \
  --data-dir data/processed/FSL105_val \
  --labels-csv data/processed/FSL105_val.csv
```

Smoke tests:

```bash
python -m training.train --model transformer_isolated --smoke-test --num-gloss 105 --num-cat 10
python -m training.train --model iv3_gru_isolated --smoke-test --num-gloss 105 --num-cat 10
```

### 5. Continuous prediction and evaluation

```bash
python -m evaluation.prediction.predict_ctc \
  --model transformer_continuous --checkpoint model.pt --input clip.npz

python -m evaluation.validation.evaluate_ctc \
  --model transformer_continuous --checkpoint model.pt \
  --test-data data/processed/FSL105_val --test-labels FSL105_val.csv
```

CTC evaluation reports WER and sequence accuracy. See [Prediction Guide](evaluation/prediction/PREDICTION_GUIDE.md) and [Validation Guide](evaluation/validation/VALIDATION_GUIDE.md).

## Documentation

| Guide | Topic |
| --- | --- |
| [Tool Guide](streamlit_app/TOOL_GUIDE.md) | PANSINAYAN app |
| [Prediction Guide](evaluation/prediction/PREDICTION_GUIDE.md) | Isolated and CTC inference |
| [Validation Guide](evaluation/validation/VALIDATION_GUIDE.md) | Metrics and evaluation |
| [Training Guide](training/TRAINING_GUIDE.md) | Isolated and CTC training |
| [Model Guide](models/MODEL_GUIDE.md) | Architectures |
| [Trained Models Guide](trained_models/TRAINED_MODEL_GUIDE.md) | Checkpoints |
| [Data Guide](data/DATA_GUIDE.md) | NPZ format and splits |
| [Label Mapping Table](data/labels/LABEL_MAPPING_TABLE.md) | Gloss and category IDs |
| [Preprocessing Guide](preprocessing/docs/PREPROCESS_GUIDE.MD) | Video → NPZ |
| [Occlusion Guide](preprocessing/docs/OCCLUSION_GUIDE.md) | Hand occlusion detection |
| [Continuous Signs Guide](preprocessing/docs/CONTINUOUS_SIGNS_GUIDE.md) | CTC sequence construction |
| [Statistical Analysis Guide](metrics/stat%20test/STATISTICAL_ANALYSIS_GUIDE.md) | Paired tests and effect sizes |

Remote GPU notes: [Vast.ai Guide](shared/for%20vast%20ai/VAST.AI_GUIDE.md).

## Troubleshooting

| Issue | What to check |
| --- | --- |
| File not found | CSV `file` values must match NPZ basenames (no extension) |
| Wrong shapes | Transformer needs `X [T,178]`; IV3-GRU needs `X2048 [T,2048]` |
| Label ranges | `gloss` in `[0, 104]`; `cat` in `[0, 9]` |
| Tiny `.pt` files | Git LFS pointers — run `git lfs pull` |
| Port in use | `streamlit run run_app.py --server.port 8502` |
| CUDA / OOM | `--device cpu`, smaller `--batch-size`, or `--amp` |

Training tips: `--amp`, `--auto-workers`, `--auto-batch-size`, `--compile-model` (PyTorch 2+). Preprocessing tips: `--workers 8`, `--disable-parquet`, lower `--target-fps` if needed.

## Citation

Undergraduate thesis, Polytechnic University of the Philippines (2025).

```bibtex
@thesis{pansinayan2025,
  title={PANSINAYAN: Multi-Head Attention Transformer for Filipino Sign Language Recognition},
  author={Estrella, Novelle Lyn and Magtibay, Nathaniel L. and Migueh, Rica Joi C. and Pablo, Jeremias G.},
  year={2025},
  school={Polytechnic University of the Philippines}
}
```

## License

Academic research code. No license file is included; contact the authors if you need to reuse it beyond personal or scholarly use.

**Repository:** [github.com/jeipab/fslr-transformer-vs-iv3gru](https://github.com/jeipab/fslr-transformer-vs-iv3gru)
