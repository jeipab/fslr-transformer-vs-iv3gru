# Which Config Should I Use?

## Quick Decision Guide

### Running Locally (Just Your Computer)

**Use**: `.streamlit/config.toml.local`

```bash
cp .streamlit/config.toml.local .streamlit/config.toml
```

**Best for**:

- ✅ Development and testing
- ✅ Running on your laptop/desktop
- ✅ Quick local demos
- ✅ Single-user testing

**Port**: `http://localhost:8501`

---

### Running on Home/Office Network

**Use**: `.streamlit/config.toml.local` (same as above)

```bash
cp .streamlit/config.toml.local .streamlit/config.toml
```

**Best for**:

- ✅ Sharing with devices on same WiFi
- ✅ Testing on mobile devices at home
- ✅ Small team demos
- ✅ Local network access

**Access from other devices**:

1. Find your IP: `python show_network_info.py`
2. Access: `http://YOUR_IP:8501`

---

### Running on Vast.AI + Cloudflare Tunnel

**Use**: `.streamlit/config.toml.optimized` (or keep your current)

```bash
cp .streamlit/config.toml.optimized .streamlit/config.toml
```

**Best for**:

- ✅ Public deployment
- ✅ Remote access from anywhere
- ✅ Mobile users on internet
- ✅ Production use

**Access**: Via Cloudflare tunnel URL

---

## Comparison Table

| Feature              | Local Config       | Vast.AI Config                  |
| -------------------- | ------------------ | ------------------------------- |
| **Port**             | 8501 (standard)    | 8081 (Vast.AI)                  |
| **CORS**             | Disabled           | Enabled (for tunnel)            |
| **XSRF Protection**  | Enabled            | Disabled (tunnel handles it)    |
| **Timeouts**         | Standard           | Extended (slow mobile networks) |
| **runOnSave**        | Enabled (dev mode) | Disabled (production)           |
| **showErrorDetails** | True (debugging)   | False (security)                |
| **Best Use**         | Development/Local  | Production/Internet             |

## Your Current Config

Your current `.streamlit/config.toml` is **Vast.AI optimized**.

### If Running Locally Now:

It will still work! But it's over-configured. You have:

- ⚠️ Port 8081 instead of standard 8501
- ⚠️ Extended timeouts you don't need
- ⚠️ XSRF disabled (less secure locally)
- ⚠️ CORS enabled (not needed locally)

**Quick Fix**: Just use the local config

```bash
cp .streamlit/config.toml.local .streamlit/config.toml
```

### If Deploying to Vast.AI Later:

You can switch back:

```bash
cp .streamlit/config.toml.optimized .streamlit/config.toml
```

## Easy Switching Strategy

### Keep All Versions:

```
.streamlit/
├── config.toml                # Active config (what Streamlit uses)
├── config.toml.local         # For local development
├── config.toml.optimized     # For Vast.AI deployment
├── config.toml.backup        # Your original
└── config.toml.example       # Reference template
```

### Switch Anytime:

```bash
# Going to develop locally?
cp .streamlit/config.toml.local .streamlit/config.toml

# Deploying to Vast.AI?
cp .streamlit/config.toml.optimized .streamlit/config.toml

# Restart Streamlit after switching
```

## The ONE Critical Setting for Both

**Regardless of which config you use**, make sure you have:

```toml
maxMessageSize = 700  # MB
```

This is needed if you enable Base64 encoding for mobile uploads.

## When to Enable Base64 Encoding?

### Local/Network Use:

```python
# In streamlit_app/core/config.py
UPLOAD_CONFIG['use_base64_preview'] = False  # Not needed locally
```

- ❌ Not necessary on local network
- ✅ Everything works fine without it

### Vast.AI + Cloudflare:

```python
# In streamlit_app/core/config.py
UPLOAD_CONFIG['use_base64_preview'] = True  # Recommended for tunnel
```

- ✅ More reliable mobile uploads
- ✅ Better with tunnel routing
- ✅ Works better on slow networks

## Commands Summary

### For Local Development (Now):

```bash
# Backup current
cp .streamlit/config.toml .streamlit/config.toml.backup

# Use local config
cp .streamlit/config.toml.local .streamlit/config.toml

# Run app
streamlit run run_app.py

# Access at: http://localhost:8501
```

### For Vast.AI Deployment (Later):

```bash
# Use optimized config
cp .streamlit/config.toml.optimized .streamlit/config.toml

# Run app
streamlit run run_app.py --server.port 8081 --server.address 0.0.0.0 --server.headless true

# Start tunnel in another terminal
cloudflared tunnel --url http://localhost:8081
```

## Still Confused?

**Simple rule**:

- 🏠 **Local/Network**: Use `config.toml.local`
- ☁️ **Vast.AI/Internet**: Use `config.toml.optimized`
- 📝 **Just testing**: Your current config works fine, just update `maxMessageSize = 700`

---

**Bottom Line**: If you're running locally right now, use `config.toml.local` for a cleaner setup. Your current config will work but has unnecessary settings for Cloudflare tunnel.
