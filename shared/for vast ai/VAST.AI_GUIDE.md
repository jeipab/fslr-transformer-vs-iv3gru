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
# Recommended: Use HTTP/2 protocol (more reliable on Vast.AI)
cloudflared tunnel --url http://localhost:8081 --protocol http2
```

**Note**: If you encounter connection timeouts, the `--protocol http2` flag forces TCP instead of UDP/QUIC, which works better on most Vast.AI instances.

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

**Download the preprocessed fsl-105 clips:**

```bash
gdown https://drive.google.com/uc?id=1c6SP6Hecv0p5c7c4ntEy-VxZaGeg7bMD
```

**Download the preprocessed smp-105 clips:**

```bash
gdown https://drive.google.com/uc?id=1wKc5lu1pmonnkqDOG6p0vCjrHxQDvjqA
```

**Unzip all zip files:**

```bash
unzip '*.zip'
```

**Clean up zip files:**

```bash
rm -f *.zip
```

---

## 5. Split Data

**Return to project root:**

```bash
cd ../../
```

**For fsl-105 data split:**

```bash
python data/splitting/data_split.py --processed-root data/processed/fsl-105_10-08 --labels data/processed/fsl-105_10-08/labels.csv --out-root data/processed --copy --train-ratio 0.8 --train-dir fsl_train --val-dir fsl_val --train-csv fsl_train.csv --val-csv fsl_val.csv
```

**For sample-105 data split:**

```bash
python data/splitting/data_split.py --processed-root data/processed/smp-105_10-08 --labels data/processed/smp-105_10-08/labels.csv --out-root data/processed --copy --train-ratio 0.8 --train-dir smp_train --val-dir smp_val --train-csv smp_train.csv --val-csv smp_val.csv
```

**For combined data split:**

```bash
python data/splitting/data_split.py --processed-root data/processed/fsl-105_10-08 data/processed/smp-105_10-08 --labels data/processed/fsl-105_10-08/labels.csv data/processed/smp-105_10-08/labels.csv --out-root data/processed --copy --train-ratio 0.8 --train-dir cmb_train --val-dir cmb_val --train-csv cmb_train.csv --val-csv cmb_val.csv
```

---

## 6. Clean Up Source Folders

After data splitting is complete, navigate back to the processed folder and remove the source directories:

**Navigate to processed directory:**

```bash
cd data/processed
```

**Remove source folders:**

```bash
rm -rf fsl-105_10-08 smp-105_10-08
```

**Return to project root:**

```bash
cd ../../
```

---

## 7. Replace Streamlit Components

Replace Streamlit components with vast.ai versions:

```bash
cp "shared/for vast ai/visualization_vast_ai.py" "streamlit_app/components/visualization.py"
```

```bash
cp "shared/for vast ai/validation_components_vast_ai.py" "streamlit_app/components/validation_components.py"
```

---

## 8. Configure Streamlit for Vast.AI

Use the optimized configuration file for Vast.AI + Cloudflare tunnel deployment:

```bash
cp .streamlit/config.toml.optimized .streamlit/config.toml
```

**What this does:**

- ✅ Sets correct port (8081) for Vast.AI
- ✅ Enables CORS for Cloudflare tunnel
- ✅ Configures extended timeouts for mobile uploads
- ✅ Sets maxMessageSize to 700 MB (supports Base64 encoding)
- ✅ Optimizes WebSocket compression

**What the Vast.AI config.py does:**

- ✅ Enables Base64 encoding for mobile camera uploads (`use_base64_preview: True`)
- ✅ Eliminates `MediaFileStorageError` on mobile devices
- ✅ Improves mobile upload consistency through Cloudflare tunnel

---

## 9. Run the Streamlit Application

**Start the app with proper settings:**

```bash
streamlit run run_app.py --server.port 8081 --server.address 0.0.0.0 --server.headless true
```

**In another terminal, create a tunnel for external access:**

```bash
# Recommended: Use HTTP/2 protocol for better reliability
cloudflared tunnel --url http://localhost:8081 --protocol http2
```

**Alternative (if HTTP/2 has issues):**

```bash
# Standard command (may fail with QUIC timeout errors)
cloudflared tunnel --url http://localhost:8081
```

**Troubleshooting Tunnel Connection:**

If you see errors like `"failed to dial to edge with quic: timeout"`, use the HTTP/2 protocol:

```bash
cloudflared tunnel --url http://localhost:8081 --protocol http2
```

This works better on Vast.AI because:

- ✅ Uses TCP instead of UDP/QUIC
- ✅ More compatible with Vast.AI network setup
- ✅ Bypasses UDP firewall restrictions

**Test locally first:**

```bash
curl -I http://localhost:8081
```

---

## 10. Access Your Application

- **Local access:** [http://localhost:8081]
- **External access:** Use the tunnel URL provided by cloudflared (e.g., [https://something-random.trycloudflare.com])

---

## 11. Video Generation Dependencies (if needed)

If you require video generation, install the following:

```bash
apt-get update && apt-get install -y ffmpeg libx264-dev
```

---

## 12. Optional: Test IV3-GRU Model Loading

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

## 13. Troubleshooting

### Cloudflare Tunnel Connection Issues

**Problem**: Tunnel fails with errors like:

```
ERR Failed to dial a quic connection error="failed to dial to edge with quic: timeout"
```

**Solution**: Use HTTP/2 protocol instead of QUIC:

```bash
cloudflared tunnel --url http://localhost:8081 --protocol http2
```

**Why this works:**

- Vast.AI instances may have UDP/QUIC blocked for security
- HTTP/2 uses TCP, which is always allowed
- Proven to work on most Vast.AI configurations

**Additional options if needed:**

```bash
# More aggressive settings for difficult networks
cloudflared tunnel --url http://localhost:8081 \
  --protocol http2 \
  --edge-ip-version 4 \
  --no-autoupdate
```

### Verify Streamlit is Running

Before troubleshooting the tunnel, verify Streamlit is accessible locally:

```bash
curl -I http://localhost:8081
```

You should see HTTP headers with status `200 OK`.

### Check Network Connectivity

```bash
# Test basic connectivity to Cloudflare
ping -c 3 cloudflare.com
curl https://cloudflare.com

# Both should work without errors
```

### Alternative: Direct IP Access

If Cloudflare tunnel continues to fail, use Vast.AI's direct port access:

1. Find your instance's public IP:

   ```bash
   curl ifconfig.me
   ```

2. Configure Vast.AI to expose port 8081 in your instance settings

3. Access directly: `http://YOUR_PUBLIC_IP:8081`

---

For further details, refer to the project documentation and guides included in the repository.
