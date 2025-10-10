# Quick Start: Mobile Camera Uploads

## 🚀 Enable Robust Mobile Uploads in 3 Steps

### Step 1: Enable Base64 Encoding (Recommended for Mobile)

In your `streamlit_app/core/config.py`:

```python
UPLOAD_CONFIG = {
    'use_base64_preview': True,  # ← Change this to True
    'base64_size_threshold_mb': 50,
    'enable_mobile_camera': True,
    'show_upload_feedback': True,
    'enable_enhanced_sync': True,
}
```

### Step 2: Use Enhanced Upload Components

Your upload components already use the enhanced system:

```python
from streamlit_app.components.components import render_file_upload

# This automatically includes:
# ✓ Mobile camera support
# ✓ Enhanced event synchronization
# ✓ Visual upload feedback
uploaded_files = render_file_upload()
```

### Step 3: Deploy with Session Affinity (for Production)

Choose one option:

#### Option A: AWS Load Balancer

```yaml
TargetGroup:
  Stickiness:
    Enabled: true
    Type: lb_cookie
```

#### Option B: Nginx

```nginx
upstream streamlit {
    ip_hash;  # Session stickiness
    server backend1:8501;
    server backend2:8501;
}
```

#### Option C: Kubernetes

```yaml
spec:
  sessionAffinity: ClientIP
```

## ⚡ Quick Settings

### For Mobile-First Apps

```python
from streamlit_app.core.config import update_upload_config

# Enable all mobile optimizations
update_upload_config('use_base64_preview', True)
update_upload_config('show_upload_feedback', True)
update_upload_config('enable_mobile_camera', True)
```

### For Large Video Files

```python
# In .streamlit/config.toml
[server]
maxUploadSize = 500  # MB
maxMessageSize = 700  # For Base64
```

### For Slow Networks

```python
# In .streamlit/config.toml
[server]
enableWebsocketCompression = true
```

## 📱 Testing Mobile Uploads

### Quick Test Checklist

1. Open app on mobile device
2. Tap "Browse files"
3. Select "Camera" or "Take Photo/Video"
4. Record a short video (< 10 seconds)
5. Verify upload indicator appears
6. Check if file uploads successfully

### Browser Testing

- ✅ iOS Safari 14+
- ✅ Android Chrome 90+
- ✅ Samsung Internet 14+

## 🔧 Troubleshooting

### Upload doesn't work on mobile?

```python
# Enable Base64 encoding
update_upload_config('use_base64_preview', True)
```

### Upload works sometimes but not always?

```
Enable session affinity in your load balancer
OR use Base64 encoding (already configured)
```

### Large videos fail to upload?

```toml
# In .streamlit/config.toml
[server]
maxUploadSize = 500
maxMessageSize = 700
```

## 📚 More Information

- **Full Guide**: [CAMERA_UPLOAD_IMPROVEMENTS.md](./CAMERA_UPLOAD_IMPROVEMENTS.md)
- **Deployment**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Streamlit Docs**: [Server-Client Design](https://docs.streamlit.io/library/advanced-features/app-design)

## 💡 Pro Tips

1. **Enable Base64 for mobile**: Set `use_base64_preview = True`
2. **Use session affinity**: Configure your load balancer
3. **Test on real devices**: Don't rely on browser emulation
4. **Monitor uploads**: Track success rates in production
5. **Set size limits**: Prevent large file issues

## 🎯 One-Line Summary

**For reliable mobile camera uploads, enable Base64 encoding or configure session affinity on your load balancer.**

---

_Need help? Check the full documentation or Streamlit's official docs on session management._
