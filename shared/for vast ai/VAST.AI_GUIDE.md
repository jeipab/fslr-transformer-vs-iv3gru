# Vast.ai Deployment Guide for fslr-transformer-vs-iv3gru

---

## 1. Expose Required Port
- Ensure port **8501** is exposed in your vast.ai template.

---

## 2. Set Up the Codebase
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

## 3. Upload and Prepare Data
**Navigate to the processed data directory:**
   ```bash
   cd data/processed
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

## 4. Assign Labels
**Return to the project root:**
   ```bash
   cd ../../
   ```
**Assign gloss and category labels:**
   ```bash
   python data/splitting/assign.py
   ```

---

## 5. Split Data
**For greetings only:**
  ```bash
  python data/splitting/data_split.py --processed-root data/processed --labels data/processed/labels.csv --out-root data/processed --copy --cats greeting --label-ref data/splitting/labels_reference.csv
  ```
**For full dataset:**
  ```bash
  python data/splitting/data_split.py --processed-root data/processed --labels data/processed/labels.csv --out-root data/processed --copy --label-ref data/splitting/labels_reference.csv
  ```

---

## 6. Add Required Files
Ensure the following are present:
1. Data files
2. Trained model files
3. `streamlit_app/components/visualization.py` (replace if needed)
4. `streamlit_app/components/validation_components.py` (replace if needed)

---

## 7. Run the Streamlit Application
Start the app with:
```bash
streamlit run run_app.py --server.port 8081 --server.address 0.0.0.0
```

---

## 8. Create a Tunnel (for External Access)
- Access locally: [http://localhost:8081](http://localhost:8081)
- Example Cloudflare tunnel: [https://capable-oasis-aaa-load.trycloudflare.com](https://capable-oasis-aaa-load.trycloudflare.com)

---

## 9. Optional: Test IV3-GRU Model Loading
To verify IV3-GRU loads correctly on CPU:
```bash
python -c "
try:
    from evaluation.prediction.predict import ModelPredictor
    predictor = ModelPredictor('iv3_gru', 'trained_models/iv3_gru/iv3gru_100_epochs_09-16/InceptionV3GRU_best.pt', device='cpu')
    print('IV3-GRU loaded successfully on CPU')
except Exception as e:
    print('IV3-GRU loading failed:', str(e))
"
```

---

## 10. Video Generation Dependencies (if needed)
If you require video generation, install the following:
```bash
apt-get update && apt-get install -y ffmpeg libx264-dev
```

---

For further details, refer to the project documentation and guides included in the repository.