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
pip install -r requirements.txt
```
