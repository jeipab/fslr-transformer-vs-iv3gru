# Filipino Sign Language Recognition Tool

This repository contains the implementation of our thesis project:
**Multi-Head Attention Transformer for Filipino Sign Language**.

## Repository Structure

- `preprocessing/` → keypoint extraction and occlusion handling
- `models/` → model architectures (IV3-GRU, Transformer)
- `training/` → training scripts, utilities, evaluation
- `ui/` → Streamlit demo application
- `notebooks/` → Jupyter notebooks for experiments

## Setup

We recommend using Python **3.9–3.11**, as these versions have the most stable support for PyTorch.

Clone the repository and install dependencies:

```bash
git clone https://github.com/jeipab/fslr-transformer-vs-iv3gru.git
cd fslr-transformer-vs-iv3gru
pip install -r requirments.txt
```

## Run the Streamlit demo (UI)

Use PowerShell from the repo root:

```powershell
# 1) (Optional) create & activate a virtual environment
python -m venv .venv
\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r .\requirments.txt

# 3) Run the app
python -m streamlit run ui\app.py
```

- Open the Local URL shown (e.g., http://localhost:8501). Press Ctrl+C in the terminal to stop.
- If the default port is busy, specify another port:

```powershell
python -m streamlit run ui\app.py --server.port 8502
```

- The placeholder UI accepts a preprocessed `.npz` with at least key `X` shaped `[T,156]` and will simulate predictions.

## Guide

- Preprocessing guide: [preprocessing/PREPROCESS_GUIDE.md](preprocessing/PREPROCESS_GUIDE.md)
- Training guide: [training/TRAINING_GUIDE.md](training/TRAINING_GUIDE.md)
- Data layout: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md)
- Sharing guide: [shared/SHARING_GUIDE.md](shared/SHARING_GUIDE.md)
