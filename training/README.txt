Actual training with NPZ data
=============================

Prerequisites
- Install deps (optionally inside a venv):
  pip install -r requirments.txt
- Prepare label CSVs with columns: file,gloss,cat

Transformer (keypoints [T,156])
- Expected .npz: key 'X' (configurable via --kp-key) shaped [T,156]
- Example dirs:
  keypoints_train\*.npz
  keypoints_val\*.npz
- Run:
  python -m training.train --model transformer --keypoints-train path\to\keypoints_train --keypoints-val path\to\keypoints_val --labels-train-csv path\to\train_labels.csv --labels-val-csv path\to\val_labels.csv --num-gloss 105 --num-cat 10 --epochs 30 --batch-size 32 --output-dir data\processed

IV3-GRU (precomputed features [T,2048])
- Expected .npz: key 'X2048' (or 'X') shaped [T,2048]
- Example dirs:
  features_train\*.npz
  features_val\*.npz
- Run:
  python -m training.train --model iv3_gru --features-train path\to\features_train --features-val path\to\features_val --labels-train-csv path\to\train_labels.csv --labels-val-csv path\to\val_labels.csv --feature-key X2048 --num-gloss 105 --num-cat 10 --epochs 30 --batch-size 32 --output-dir data\processed

Notes
- Use module mode (python -m training.train) to avoid import errors.
- GPU is used automatically if available; otherwise CPU.
- For Transformer, sequences are padded and a mask from lengths is applied.
- Ensure class counts (--num-gloss/--num-cat) match your labels.

Smoke tests
===========

Quick commands to verify the training script and models without real data.

Prerequisites
- Install deps (optionally inside a venv):
  pip install -r requirments.txt

Transformer smoke test
- Runs a tiny forward/backward on random [B,T,156] and saves a checkpoint.
  python -m training.train --model transformer --smoke-test --num-gloss 105 --num-cat 10

IV3-GRU smoke test (no weights download)
- Uses random [B,T,2048] features with lengths; saves a checkpoint.
  python -m training.train --model iv3_gru --smoke-test --num-gloss 105 --num-cat 10 --no-pretrained-backbone

Outputs
- Checkpoints are saved to data/processed (default). Override with:
  --output-dir path\to\out

Notes
- Use module mode (python -m training.train) to avoid import errors.
- GPU is used automatically if available; otherwise CPU.
- The IV3-GRU warning about InceptionV3 init is harmless for smoke tests.
