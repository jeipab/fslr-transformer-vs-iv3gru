Transformer Model Directory
============================

This directory should contain trained Transformer model checkpoints for Filipino Sign Language Recognition.

Directory Structure
-------------------
Place your trained Transformer models in the following structure:

transformer/
├── FSL105_classification/
│   └── SignTransformer_best.pt
└── FSL105_ctc/
    └── SignTransformerCtc_best.pt

Required Files
--------------
1. Classification Model (Isolated Sign Recognition):
   - Path: FSL105_classification/SignTransformer_best.pt
   - Used for: Isolated sign classification
   - Input: Keypoints [T, 178] from MediaPipe

2. CTC Model (Continuous Sign Recognition):
   - Path: FSL105_ctc/SignTransformerCtc_best.pt
   - Used for: Continuous sign sequence recognition
   - Input: Keypoints [T, 178] from MediaPipe

Usage
-----
The Streamlit app automatically loads these models from:
- streamlit_app/core/config.py

Model Configuration:
- Input dimension: 178 (MediaPipe keypoints)
- Gloss classes: 105
- Category classes: 10
- CTC classes: 106 (for continuous model)

To use these models:
1. Copy your trained .pt checkpoint files to the appropriate subdirectories
2. Ensure file names match exactly: SignTransformer_best.pt and SignTransformerCtc_best.pt
3. The app will automatically detect and load them

Note: Model checkpoints should be PyTorch .pt files containing the model state_dict.

