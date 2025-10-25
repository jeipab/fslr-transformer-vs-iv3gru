# Continuous Sequence Generation

Generate continuous signing sequences from isolated sign videos for CTC-based model evaluation.

---

## Quick Usage

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/greetings_val.csv \
    --val-dir data/processed/greetings_val \
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
    --val-csv data/processed/greetings_val.csv \
    --val-dir data/processed/greetings_val \
    --output-dir data/processed/continuous_sequences \
    --strategy 1 \
    --sequences-per-signer 10 \
    --min-glosses 3 \
    --max-glosses 6
```

### Strategy 2: Different Categories

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/greetings_val.csv \
    --val-dir data/processed/greetings_val \
    --output-dir data/processed/continuous_sequences \
    --strategy 2 \
    --sequences-per-signer 10 \
    --min-glosses 4 \
    --max-glosses 5
```

### Preview (Dry Run)

```bash
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/greetings_val.csv \
    --val-dir data/processed/greetings_val \
    --strategy 1 \
    --dry-run
```

---

## PowerShell Usage (Windows)

On Windows PowerShell, use single-line commands or backticks for line continuation:

```powershell
# Single line (recommended)
python preprocessing/continuous/create_continuous_signs.py --val-csv data/processed/greetings_val.csv --val-dir data/processed/greetings_val --strategy 1 --dry-run

# Multi-line with backticks
python preprocessing/continuous/create_continuous_signs.py `
    --val-csv data/processed/greetings_val.csv `
    --val-dir data/processed/greetings_val `
    --strategy 1 `
    --dry-run
```

**Note**: Backslash (`\`) line continuation doesn't work in PowerShell. Use backticks (`` ` ``) instead.

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
clip_0559_how_are_you_S5,4,0,0,S5,5.265789035160807
clip_0412_hello_S4,3,0,1,S4,4.232627895350775
clip_1005_youre_welcome_S4,8,0,1,S4,4.232627895350775
```

### NPZ Files

Each file must contain:

- `X`: Keypoints `[T, 178]`
- `X2048`: Features `[T, 2048]` (optional)
- `mask`: Visibility `[T, 89]`
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

- `X`: `[T_total, 178]` - All keypoints concatenated
- `X2048`: `[T_total, 2048]` - All features concatenated
- `mask`: `[T_total, 89]` - All masks concatenated
- `timestamps_ms`: `[T_total]` - Cumulative timestamps

### JSON Metadata

```json
{
  "file_name": "continuous_0001_S0_strategy1.npz",
  "signer": "S0",
  "strategy": 1,
  "strategy_name": "same_category",
  "total_duration_sec": 12.13,
  "num_segments": 3,
  "segments": [
    {
      "index": 0,
      "timestamp_start_ms": 0,
      "timestamp_end_ms": 4066,
      "gloss": 0,
      "gloss_label": "GOOD MORNING",
      "category": 0,
      "category_label": "GREETING",
      "signer": "S0",
      "original_file": "clip_0001_good_morning_S0.npz"
    }
  ]
}
```

---

## Common Errors

**"Validation CSV not found"**  
Check `--val-csv` path exists.

**"X NPZ files missing"**  
Ensure all files in CSV exist in `--val-dir`. The script automatically adds `.npz` extension if missing from CSV filenames.

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
# 1. Generate continuous sequences from validation data
python preprocessing/continuous/create_continuous_signs.py \
    --val-csv data/processed/greetings_val.csv \
    --val-dir data/processed/greetings_val \
    --output-dir data/processed/continuous_sequences \
    --strategy 1 \
    --sequences-per-signer 10 \
    --min-glosses 3 \
    --max-glosses 6

# 2. Use continuous sequences in Streamlit app
# Upload the generated NPZ files to test CTC models
# Or use for offline evaluation with predict_ctc.py
```
