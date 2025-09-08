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
