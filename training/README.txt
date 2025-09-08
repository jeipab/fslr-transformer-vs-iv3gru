Training with NPZ data
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

Advanced training
=================

Options
- Learning: --lr, --weight-decay
- Precision: --amp (mixed precision)
- Stability: --grad-clip N (max-norm)
- Scheduling: --scheduler [plateau|cosine], --scheduler-patience K
- Early stop: --early-stop K (epochs without improvement)
- Checkpoints: --resume path\to\{ModelName}_last.pt
- Logging: --log-csv logs\train.csv (writes epoch, losses, accs, lr)
- DataLoader: --num-workers N, --pin-memory, --prefetch-factor K
- Reproducibility: --seed S, --deterministic

Notes
- Best/last checkpoints saved to --output-dir as {ModelName}_best.pt and {ModelName}_last.pt.
- Early stopping and scheduler use validation gloss accuracy as the metric.

Example (Transformer + keypoints)
  python -m training.train --model transformer --keypoints-train path\to\kp_train --keypoints-val path\to\kp_val --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv --num-gloss 105 --num-cat 10 --epochs 50 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 --amp --grad-clip 1.0 --scheduler cosine --early-stop 10 --log-csv logs\transformer_train.csv --num-workers 4 --pin-memory

Example (IV3-GRU + features)
  python -m training.train --model iv3_gru --features-train path\to\feat_train --features-val path\to\feat_val --labels-train-csv path\to\train.csv --labels-val-csv path\to\val.csv --feature-key X2048 --num-gloss 105 --num-cat 10 --epochs 40 --batch-size 32 --lr 1e-4 --scheduler plateau --scheduler-patience 3 --early-stop 8 --log-csv logs\iv3_gru_train.csv --num-workers 4 --pin-memory
