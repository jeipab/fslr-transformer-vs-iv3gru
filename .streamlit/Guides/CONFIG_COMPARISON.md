# Config Comparison: Current vs Optimized vs Example

## Your Deployment Setup

- **Platform**: Vast.AI (powerful GPU instance)
- **External Access**: Cloudflare tunnel (`cloudflared tunnel --url http://localhost:8081`)
- **Architecture**: Single instance (no load balancing)
- **Port**: 8081
- **Binding**: 0.0.0.0 (all interfaces)

## Key Differences

### ✅ What You're Already Doing Right

Your current config is **excellent** for your setup! Here's what's working:

1. **Port & Address**

   ```toml
   port = 8081
   address = "0.0.0.0"
   ```

   ✅ Perfect for Vast.AI + Cloudflare tunnel

2. **CORS Enabled**

   ```toml
   enableCORS = true
   ```

   ✅ Required for Cloudflare tunnel to work

3. **Large Upload Limits**

   ```toml
   maxUploadSize = 500
   ```

   ✅ Good for mobile video uploads

4. **XSRF Disabled**

   ```toml
   enableXsrfProtection = false
   ```

   ✅ Helps with mobile uploads (Cloudflare provides security)

5. **Extended Timeouts**
   ```toml
   browserWebsocketDelay = 30000
   maxUploadTime = 300
   ```
   ✅ Perfect for slow mobile connections

### 📊 Comparison Table

| Setting                   | Your Current | Example     | Optimized   | Recommendation         |
| ------------------------- | ------------ | ----------- | ----------- | ---------------------- |
| **maxUploadSize**         | 500 MB       | 200 MB      | 500 MB      | ✅ **Keep yours**      |
| **maxMessageSize**        | 500 MB       | 300 MB      | 700 MB      | ⚠️ **Increase to 700** |
| **enableCORS**            | true         | false       | true        | ✅ **Keep yours**      |
| **enableXsrfProtection**  | false        | true        | false       | ✅ **Keep yours**      |
| **port**                  | 8081         | 8501        | 8081        | ✅ **Keep yours**      |
| **address**               | 0.0.0.0      | not set     | 0.0.0.0     | ✅ **Keep yours**      |
| **browserWebsocketDelay** | 30000        | not set     | 30000       | ✅ **Keep yours**      |
| **[runner] section**      | ❌ missing   | ✅ included | ✅ included | ⚠️ **Add**             |
| **[client] section**      | ❌ missing   | ✅ included | ✅ included | ⚠️ **Add**             |
| **[logger] section**      | ❌ missing   | ✅ included | ✅ included | 📝 Optional            |

## 🎯 Recommendations

### 1. CRITICAL: Increase `maxMessageSize` ⚠️

**Current**: 500 MB  
**Recommended**: 700 MB

**Why?**

- Base64 encoding increases file size by ~33%
- A 500 MB video becomes ~665 MB when Base64-encoded
- Without this, mobile uploads with Base64 will fail

**Change:**

```toml
[server]
maxUploadSize = 500  # MB
maxMessageSize = 700  # MB - ← INCREASE THIS
```

### 2. ADD: Runner & Client Sections

Add these sections for better performance and security:

```toml
[runner]
magicEnabled = false      # Cleaner production code
fastReruns = true         # Better performance
fixMatplotlib = true      # Fixes plotting issues

[client]
showErrorDetails = false  # Hide errors from users
showToolbarMode = "viewer"  # Production mode
```

### 3. OPTIONAL: Logger Section

For better debugging (optional):

```toml
[logger]
level = "info"  # Use "debug" when troubleshooting
messageFormat = "%(asctime)s %(levelname)s: %(message)s"
```

## 🚀 Why Base64 Encoding Matters for Your Setup

### Your Architecture

```
Mobile User → Internet → Cloudflare Tunnel → Your Vast.AI Instance
                                          ↓
                                    Streamlit (port 8081)
```

### Without Base64 (Current Risk)

1. User uploads video via WebSocket ✅
2. Streamlit saves to temp storage ✅
3. Preview requests file via HTTP ⚠️
4. HTTP request goes through Cloudflare tunnel 🤔
5. **Potential issue**: Tunnel routing + temp file timing

### With Base64 (Recommended)

1. User uploads video via WebSocket ✅
2. Convert to Base64 string ✅
3. Send Base64 through same WebSocket ✅
4. Display directly in browser ✅
5. **No HTTP requests** = More reliable 🎯

## 📱 Mobile Upload Strategy for Cloudflare Tunnel

Since you're using a **single instance** (no load balancing), you have two options:

### Option 1: Current Setup (Good)

- ✅ Works most of the time
- ⚠️ Occasional issues with mobile camera uploads
- ⚠️ May have timing issues with Cloudflare tunnel

### Option 2: Enable Base64 (Better for Mobile)

- ✅ Much more reliable mobile uploads
- ✅ Data passes through WebSocket only
- ✅ No HTTP/tunnel routing issues
- ✅ Works great with slow mobile networks
- ⚠️ Requires `maxMessageSize = 700 MB`

**To enable**: In `streamlit_app/core/config.py`:

```python
UPLOAD_CONFIG = {
    'use_base64_preview': True,  # ← Change to True
    'base64_size_threshold_mb': 50,
    'enable_mobile_camera': True,
    'show_upload_feedback': True,
    'enable_enhanced_sync': True,
}
```

## 🔧 Recommended Action

### Quick Fix (2 minutes)

Edit `.streamlit/config.toml` and make these changes:

```toml
[server]
maxUploadSize = 500
maxMessageSize = 700  # ← CHANGE FROM 500 TO 700

# ... rest stays the same ...

# ADD THESE SECTIONS:
[runner]
magicEnabled = false
fastReruns = true
fixMatplotlib = true

[client]
showErrorDetails = false
showToolbarMode = "viewer"
```

### For Mobile Uploads (5 minutes)

1. **Update config.toml** (as above)
2. **Enable Base64** in `streamlit_app/core/config.py`:
   ```python
   UPLOAD_CONFIG['use_base64_preview'] = True
   ```
3. **Restart your Streamlit app**

## 📋 Files Available

1. **`.streamlit/config.toml`** - Your current config (great base!)
2. **`.streamlit/config.toml.example`** - General example (for any deployment)
3. **`.streamlit/config.toml.optimized`** - Optimized for your Cloudflare+Vast.AI setup
4. **This file** - Detailed comparison

## 🎯 Quick Decision Matrix

### Should I use Base64 encoding?

| Scenario               | Recommendation             |
| ---------------------- | -------------------------- |
| Mostly desktop users   | ❌ Not needed              |
| Some mobile users      | 📝 Optional (good to have) |
| Many mobile users      | ✅ **Highly recommended**  |
| Mobile-first app       | ✅ **Essential**           |
| Users on slow networks | ✅ **Essential**           |
| Large files (>50MB)    | ⚠️ Use standard mode       |

### Your situation?

- Filipino Sign Language app
- Likely mobile users
- Video captures via camera
- Through Cloudflare tunnel

**Verdict**: ✅ **Enable Base64 for best mobile experience**

## 🔍 Testing Plan

After making changes:

1. **Desktop Browser**

   ```
   Visit: http://localhost:8081
   Upload: Test video file
   ✅ Should work perfectly
   ```

2. **Mobile Browser (via tunnel)**

   ```
   Visit: https://your-tunnel.trycloudflare.com
   Tap: "Browse files"
   Select: "Camera"
   Record: Short video
   ✅ Should upload reliably with Base64
   ```

3. **Monitor Logs**
   ```bash
   # Watch for any errors
   streamlit run run_app.py --server.port 8081 \
     --server.address 0.0.0.0 --server.headless true
   ```

## 📞 Support

If you encounter issues:

1. Check browser console (F12) for errors
2. Check Streamlit logs for upload errors
3. Try with smaller video files first
4. Test both with and without Base64
5. See `streamlit_app/DEPLOYMENT_GUIDE.md` for troubleshooting

---

**TL;DR**: Your config is already great! Just increase `maxMessageSize` to 700 MB and optionally enable Base64 encoding for better mobile support.
