# Continuous Sequence Generation

Generate continuous signing sequences from isolated sign videos for CTC-based model evaluation.

---

## Quick Usage

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/fsl_val.csv \
    --val-dir data/processed/fsl_val \
    --strategy 1 \
    --sequences-per-signer 10
```

---

## Overview

This module concatenates isolated sign videos from the validation set to create continuous signing sequences. This simulates real-world continuous signing scenarios for evaluating models with CTC decoders.

**Features:**

- Two concatenation strategies (same category / different categories)
- Signer-specific sequences (no mixing)
- Configurable sequence length
- Proper timestamp offset handling
- JSON metadata generation

---

## Command Examples

### Strategy 1: Same Category

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/fsl_val.csv \
    --val-dir data/processed/fsl_val \
    --output-dir data/processed/continuous_sequences \
    --strategy 1 \
    --sequences-per-signer 10 \
    --min-glosses 3 \
    --max-glosses 6
```

### Strategy 2: Different Categories

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/fsl_val.csv \
    --val-dir data/processed/fsl_val \
    --output-dir data/processed/continuous_sequences \
    --strategy 2 \
    --sequences-per-signer 10 \
    --min-glosses 4 \
    --max-glosses 5
```

### Preview (Dry Run)

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/fsl_val.csv \
    --val-dir data/processed/fsl_val \
    --strategy 1 \
    --dry-run
```

---

## Arguments

| Argument                 | Required | Default                 | Description                                       |
| ------------------------ | -------- | ----------------------- | ------------------------------------------------- |
| `--val-csv`              | Yes      | -                       | Path to validation CSV file                       |
| `--val-dir`              | Yes      | -                       | Directory containing validation NPZ files         |
| `--output-dir`           | No       | `continuous_sequences/` | Output directory                                  |
| `--strategy`             | No       | `1`                     | Strategy: 1=same category, 2=different categories |
| `--sequences-per-signer` | No       | `10`                    | Number of sequences per signer                    |
| `--min-glosses`          | No       | `3`                     | Minimum glosses per sequence                      |
| `--max-glosses`          | No       | `6`                     | Maximum glosses per sequence                      |
| `--seed`                 | No       | `42`                    | Random seed for reproducibility                   |
| `--dry-run`              | No       | `False`                 | Preview without creating files                    |

---

## Strategies

### Strategy 1: Same Category

Creates sequences where all signs belong to the same semantic category.

**Example**: All GREETING signs  
`GOOD MORNING → HELLO → THANK YOU → SEE YOU TOMORROW`

**Use Case**: Test model's ability to distinguish similar signs within one category.

### Strategy 2: Different Categories

Creates sequences where each sign belongs to a different category.

**Example**: Mixed categories  
`HELLO (GREETING) → YES (SURVIVAL) → SIX (NUMBER) → FATHER (FAMILY)`

**Use Case**: Test model's ability to handle diverse signs across categories.

---

## Input Format

### Validation CSV

Required columns: `file`, `gloss`, `cat`  
Optional columns: `occluded`, `signer`, `duration`

```csv
file,gloss,cat,occluded,signer,duration
clip_0001_good morning_S0.npz,0,0,0,S0,4.2
clip_0082_hello_S0.npz,3,0,0,S0,4.1
clip_0234_thank you_S0.npz,7,0,0,S0,4.0
```

### NPZ Files

Each file must contain:

- `X`: Keypoints `[T, 156]`
- `X2048`: Features `[T, 2048]` (optional)
- `mask`: Visibility `[T, 78]`
- `timestamps_ms`: Timestamps `[T]`

---

## Output Format

### File Structure

```
continuous_sequences/
├── continuous_0001_S0_strategy2.npz
├── continuous_0001_S0_strategy2.json
├── continuous_0002_S0_strategy2.npz
├── continuous_0002_S0_strategy2.json
├── ...
└── generation_summary.json
```

### NPZ Contents

Concatenated arrays with cumulative timestamps:

- `X`: `[T_total, 156]` - All keypoints concatenated
- `X2048`: `[T_total, 2048]` - All features concatenated
- `mask`: `[T_total, 78]` - All masks concatenated
- `timestamps_ms`: `[T_total]` - Cumulative timestamps

### JSON Metadata

```json
{
  "file_name": "continuous_0001_S0_strategy2.npz",
  "signer": "S0",
  "strategy": 2,
  "total_duration_sec": 16.8,
  "num_segments": 4,
  "segments": [
    {
      "index": 0,
      "timestamp_start_ms": 0,
      "timestamp_end_ms": 4200,
      "gloss": 0,
      "gloss_label": "GOOD MORNING",
      "category": 0,
      "category_label": "GREETING",
      "signer": "S0",
      "original_file": "clip_0001_good morning_S0.npz"
    }
  ]
}
```

---

## Common Errors

**"Validation CSV not found"**  
Check `--val-csv` path exists.

**"X NPZ files missing"**  
Ensure all files in CSV exist in `--val-dir`.

**"Not enough videos for signer"**  
Reduce `--sequences-per-signer` or `--max-glosses`, or increase validation set size.

**"Signer has max Y glosses in category (need Z)"**  
For strategy 1: Signer needs at least `min_glosses` videos in at least one category.

**"Signer has videos in Y categories (need Z)"**  
For strategy 2: Signer needs videos in at least `min_glosses` different categories.

---

## CTC Setup

```python
from data.labels.label_mapping import get_ctc_config
config = get_ctc_config()  # {'num_gloss': 105, 'num_ctc_classes': 106, 'blank_id': 105}

model = SignTransformerCtc(num_ctc_classes=config['num_ctc_classes'])
criterion = nn.CTCLoss(blank=config['blank_id'], zero_infinity=True)
```

---

## Timestamp Handling

Timestamps are cumulative across segments:

- Segment 1: `[0ms - 4200ms]`
- Segment 2: `[4200ms - 8400ms]`
- Segment 3: `[8400ms - 12600ms]`

No gaps between segments (immediate concatenation).

---

## Example Workflow

```bash
# 1. Split dataset into train/val
python data/splitting/data_split.py \
    --processed-root data/processed/fsl-105_full \
    --labels data/processed/fsl-105_full/labels.csv \
    --out-root data/processed \
    --train-ratio 0.8

# 2. Train model on 80%
python training/train.py --model transformer ...

# 3. Generate continuous sequences from 20% validation
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/fsl_val.csv \
    --val-dir data/processed/fsl_val \
    --strategy 1 \
    --sequences-per-signer 10

# 4. Evaluate with CTC
python evaluation/prediction/predict_ctc.py \
    --model-path trained_models/transformer/best.pt \
    --input-dir data/processed/continuous_sequences
```
