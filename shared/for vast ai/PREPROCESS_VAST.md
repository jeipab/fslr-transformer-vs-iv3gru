# Preprocessing Guide for Vast.ai

This guide covers preprocessing video data for the fslr-transformer-vs-iv3gru project on Vast.ai instances.

---

## Prerequisites

- Ensure you're in the project directory with virtual environment activated:

```bash
cd fslr-transformer-vs-iv3gru
source venv/bin/activate
```

---

## Download Raw Video Data

**Step 1: Create raw directory and navigate to target directory**

```bash
mkdir -p data/raw/[TARGET_DIRECTORY]
cd data/raw/[TARGET_DIRECTORY]
```

**Step 2: Download video zip file from Google Drive**

Replace `[GOOGLE_DRIVE_ID]` with the actual file ID:

```bash
gdown https://drive.google.com/uc?id=[GOOGLE_DRIVE_ID]
```

**Step 3: Unzip the downloaded file**

```bash
unzip *.zip
```

---

## Preprocess Videos to NPZ Format

**Step 1: Navigate back to project root**

```bash
cd ../../../
```

**Step 2: Create processed directory**

```bash
mkdir -p data/processed/[TARGET_DIRECTORY]
```

**Step 3: Preprocess videos to NPZ format**

Replace `[TARGET_DIRECTORY]` with your target directory name:

```bash
python -m preprocessing.core.preprocess data/raw/[TARGET_DIRECTORY] data/processed/[TARGET_DIRECTORY] --write-keypoints --write-iv3-features --workers 4 --batch-size 16 --disable-parquet
```

**Optimization Notes for RTX 5080 (1 GPU, 12 CPU cores, 15.9 GB VRAM):**

- `--workers 4`: Uses 4 parallel processes (limited by single GPU memory, not CPU cores)
- `--batch-size 16`: Conservative batch size to avoid CUDA OOM with multiple workers sharing 1 GPU
- `--disable-parquet`: Skips parquet files for faster processing

**For different hardware configurations:**

- **More GPU memory available**: Increase `--batch-size` (e.g., 32, 64)
- **More CPU cores**: Increase `--workers` (e.g., 8, 12)
- **Less GPU memory**: Decrease `--batch-size` (e.g., 8, 4)

---

## Create Labels CSV

**Step 1: Generate labels.csv from processed NPZ files**

Replace `[TARGET_DIRECTORY]` with your processed directory name:

```bash
python3 -c "
import os
import numpy as np
import pandas as pd
import json

# Get all npz files in processed directory
processed_dir = 'data/processed/[TARGET_DIRECTORY]'
files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]

# Extract occluded flag from each NPZ file's metadata
data = []
for f in files:
    npz_path = os.path.join(processed_dir, f)
    npz_data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(npz_data['meta']))
    occluded = meta.get('occluded_flag', 0)
    data.append({'file': f, 'occluded': occluded})

df = pd.DataFrame(data)
df.to_csv(f'{processed_dir}/labels.csv', index=False)
print(f'Created labels.csv with {len(files)} files (with occluded column)')
"
```

---

## Assign Labels

**Step 1: Copy labels to main processed directory**

```bash
cp data/processed/[TARGET_DIRECTORY]/labels.csv data/processed/labels.csv
```

**Step 2: Run the assign script to add gloss and category IDs**

```bash
python data/splitting/assign.py
```

**Step 3: Copy updated labels back to target directory**

```bash
cp data/processed/labels.csv data/processed/[TARGET_DIRECTORY]/labels.csv
```

---

## Verify Output

**Check that the labels.csv file was created with proper mappings:**

```bash
head data/processed/[TARGET_DIRECTORY]/labels.csv
```

**Check processed NPZ files:**

```bash
ls -la data/processed/[TARGET_DIRECTORY]/*.npz | head -10
```

---

## Clean Up

**Remove original zip file:**

```bash
cd data/raw/[TARGET_DIRECTORY]
rm -f *.zip
cd ../../../
```

---

## Example Usage

**For sample data:**

```bash
# Replace placeholders
TARGET_DIRECTORY="sample"
GOOGLE_DRIVE_ID="12S2q_RmHKAsl40ZNAsJpNL2p2F65rVeJ"

# Follow the steps above with these values
```

**For full dataset:**

```bash
# Replace placeholders
TARGET_DIRECTORY="full_dataset"
GOOGLE_DRIVE_ID="your_full_dataset_id"

# Follow the steps above with these values
```

---

## Troubleshooting

**Common issues:**

1. **CUDA OOM errors**: Reduce `--batch-size` and/or `--workers`
2. **Slow processing**: Increase `--workers` if you have more CPU cores
3. **Memory issues**: Process smaller batches or use `--disable-parquet`
4. **File not found**: Ensure correct directory paths and file IDs

**Performance monitoring:**

```bash
# Monitor GPU usage
nvidia-smi

# Monitor CPU usage
htop

# Monitor disk space
df -h
```
