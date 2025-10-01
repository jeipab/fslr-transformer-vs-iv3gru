# Vast.ai Deployment Guide for fslr-transformer-vs-iv3gru

---

## 1. Expose Required Port

- Ensure port **8081** is exposed in your vast.ai template.

---

## 2. Quick Start (Pre-Setup Project)

If the project has already been properly set up with all dependencies, data, and models in place, follow these steps:

**Prerequisites:**

- Project cloned and dependencies installed
- Data downloaded and processed
- Models copied to appropriate directories
- Virtual environment created

**Step 1: Navigate to project directory**

```bash
cd fslr-transformer-vs-iv3gru
```

**Step 2: Activate virtual environment**

```bash
source venv/bin/activate
```

**Step 3: Run Streamlit application**

```bash
streamlit run run_app.py --server.port 8081 --server.address 0.0.0.0 --server.headless true
```

**Step 4: Open new terminal and create tunnel**

In a new terminal window:

```bash
cloudflared tunnel --url http://localhost:8081
```

**Step 5: Access your application**

- **Local access:** [http://localhost:8081]
- **External access:** Use the tunnel URL provided by cloudflared

---

## 3. Set Up the Codebase

**Clone the repository:**

```bash
git clone https://github.com/jeipab/fslr-transformer-vs-iv3gru.git
```

**Navigate to the project directory:**

```bash
cd fslr-transformer-vs-iv3gru
```

**Set up a Python virtual environment:**

```bash
python -m venv venv
```

**Activate the virtual environment:**

```bash
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 4. Upload and Prepare Data

**Navigate to the processed data directory:**

```bash
cd data
mkdir processed
cd processed
```

**Install gdown (if not already installed):**

```bash
pip install gdown
```

**Download the zipped NPZ data:**

```bash
gdown https://drive.google.com/uc?id=1C_td2Jqb07Z19t-uaoE3m3y7cAnHOZwI
```

**Extract the zip file:**

```bash
unzip *.zip
```

---

## 5. Assign Labels

**Return to the project root:**

```bash
cd ../../
```

**Assign gloss and category labels:**

```bash
python data/splitting/assign.py
```

---

## 6. Split Data

**For greetings only:**

```bash
python data/splitting/data_split.py --processed-root data/processed --labels data/processed/labels.csv --out-root data/processed --copy --cats greeting --label-ref data/splitting/labels_reference.csv
```

**For full dataset:**

```bash
python data/splitting/data_split.py --processed-root data/processed --labels data/processed/labels.csv --out-root data/processed --copy --label-ref data/splitting/labels_reference.csv
```

---

## 7. Add Required Files

1. Data files

2. Trained model files
   **For the Transformer model, copy the checkpoint file to the optimal folder:**

```bash
cp shared/current/transformer/anti_overfitting/SignTransformer_best.pt trained_models/transformer/anti_overfitting/
```

**For the IV3-GRU model, copy the checkpoint file to the 70-gloss_acc folder:**

```bash
cp shared/current/iv3_gru/70-gloss_acc/InceptionV3GRU_best.pt trained_models/iv3_gru/70-gloss_acc/
```

3. Replace Streamlit components with vast.ai versions:

```bash
cp "shared/for vast ai/visualization_vast_ai.py" "streamlit_app/components/visualization.py"
```

```bash
cp "shared/for vast ai/validation_components_vast_ai.py" "streamlit_app/components/validation_components.py"
```

---

## 8. Run the Streamlit Application

**Start the app with proper settings:**

```bash
streamlit run run_app.py --server.port 8081 --server.address 0.0.0.0 --server.headless true
```

**In another terminal, create a tunnel for external access:**

```bash
cloudflared tunnel --url http://localhost:8081
```

**Test locally first:**

```bash
curl -I http://localhost:8081
```

---

## 9. Access Your Application

- **Local access:** [http://localhost:8081]
- **External access:** Use the tunnel URL provided by cloudflared (e.g., [https://something-random.trycloudflare.com])

---

## 10. Video Generation Dependencies (if needed)

If you require video generation, install the following:

```bash
apt-get update && apt-get install -y ffmpeg libx264-dev
```

---

## 11. Optional: Test IV3-GRU Model Loading

To verify IV3-GRU loads correctly on CPU:

```bash
python -c "
try:
    from evaluation.prediction.predict import ModelPredictor
    predictor = ModelPredictor('iv3_gru', 'trained_models/iv3_gru/70-gloss_acc/InceptionV3GRU_best.pt', device='cpu')
    print('IV3-GRU loaded successfully on CPU')
except Exception as e:
    print('IV3-GRU loading failed:', str(e))
"
```

---

## 12. Download and Process Sample Data

This section covers downloading sample video data and processing it for testing purposes.

**Prerequisites:**

- Ensure you're in the project directory with virtual environment activated:

```bash
cd fslr-transformer-vs-iv3gru
source venv/bin/activate
```

**Step 1: Create raw directory and navigate to sample directory**

```bash
mkdir -p data/raw/sample
cd data/raw/sample
```

**Step 2: Download the sample zip file from Google Drive**

```bash
gdown https://drive.google.com/uc?id=12S2q_RmHKAsl40ZNAsJpNL2p2F65rVeJ
```

**Step 3: Unzip the downloaded file**

```bash
unzip *.zip
```

**Step 4: Preprocess videos to NPZ format**

Navigate back to project root and preprocess the MP4 videos:

```bash
cd ../../../
mkdir -p data/processed/sample
python -m preprocessing.core.preprocess data/raw/sample data/processed/sample --write-keypoints --write-iv3-features --workers 4 --batch-size 16 --disable-parquet
```

Note: Optimized for RTX 5080 (1 GPU, 12 CPU cores, 15.9 GB VRAM):

- `--workers 4`: Uses 4 parallel processes (limited by single GPU memory, not CPU cores)
- `--batch-size 16`: Conservative batch size to avoid CUDA OOM with multiple workers sharing 1 GPU
- `--disable-parquet`: Skips parquet files for faster processing

**Step 6: Create labels.csv for the processed sample data**

Create labels.csv from the processed NPZ files (including occluded flag):

```bash
python3 -c "
import os
import numpy as np
import pandas as pd
import json

# Get all npz files in processed sample directory
processed_dir = 'data/processed/sample'
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
df.to_csv('data/processed/sample/labels.csv', index=False)
print(f'Created labels.csv with {len(files)} files (with occluded column)')
"
```

**Step 7: Run the assign script on sample data**

Run assign script to add gloss and category IDs:

```bash
cp data/processed/sample/labels.csv data/processed/labels.csv
python data/splitting/assign.py
cp data/processed/labels.csv data/processed/sample/labels.csv
```

**Step 8: Verify the output**

Check that the labels.csv file was created with proper mappings:

```bash
head data/processed/sample/labels.csv
```

---

For further details, refer to the project documentation and guides included in the repository.
