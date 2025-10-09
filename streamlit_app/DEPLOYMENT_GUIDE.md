# Streamlit App Deployment Guide

## Overview

This guide provides best practices for deploying the FSLR Streamlit app, with special focus on handling camera uploads and ensuring consistency across mobile and desktop platforms.

## Camera Upload & Session Management

### Problem: Inconsistent Mobile Camera Uploads

When deploying Streamlit apps with file uploads (especially camera captures from mobile devices), you may encounter inconsistencies due to how Streamlit handles WebSocket sessions and HTTP media requests.

### Root Cause

- Streamlit uses **WebSockets** for maintaining session state
- Media files are served via **HTTP requests** (separate from WebSocket)
- In load-balanced deployments, HTTP requests may route to different server instances
- If the media file doesn't exist on all replicas, uploads fail with `MediaFileStorageError` or HTTP 400

### Solutions Implemented

#### 1. Enhanced JavaScript Event Handling

Our `render_file_upload()` function includes:

- Multiple event triggers (`input`, `change`) to ensure Streamlit catches file selections
- Focus/blur cycles to trigger Streamlit reactivity
- Visual feedback for mobile users with upload indicators
- Retry logic with delayed triggers (100ms, 300ms)

#### 2. Base64 Data URI Encoding (Recommended)

For critical uploads or mobile camera captures, we provide Base64 encoding:

```python
from streamlit_app.components.utils import encode_file_to_base64, get_mime_type_from_extension

# Encode uploaded file to Base64 data URI
file_data = uploaded_file.getvalue()
mime_type = get_mime_type_from_extension(uploaded_file.name)
data_uri = encode_file_to_base64(file_data, mime_type)

# Display video using Base64 (passes through WebSocket)
st.markdown(f'<video src="{data_uri}" controls></video>', unsafe_allow_html=True)
```

**Benefits:**

- Data passes through WebSocket (not HTTP)
- No session affinity required
- Consistent across all server replicas
- Works reliably on mobile devices

**Trade-offs:**

- ~33% size increase due to Base64 encoding
- Best for files < 50 MB
- For larger files, consider external storage (S3, etc.)

#### 3. Key File Upload Parameter

We use `key="main_file_uploader"` to ensure Streamlit properly tracks the uploader state:

```python
st.file_uploader(
    "Choose files",
    type=["npz", "mp4", "mov", "webm"],
    accept_multiple_files=True,
    key="main_file_uploader"  # Important for state management
)
```

## Deployment Configurations

### Option A: Session Affinity (Stickiness)

**Best for:** Production deployments with multiple replicas

Configure your load balancer to enable session affinity:

#### AWS Application Load Balancer (ALB)

```yaml
TargetGroup:
  Stickiness:
    Enabled: true
    Type: lb_cookie
    Duration: 86400 # 24 hours in seconds
```

#### Nginx Load Balancer

```nginx
upstream streamlit_backends {
    ip_hash;  # Enables session stickiness
    server backend1:8501;
    server backend2:8501;
    server backend3:8501;
}

server {
    location / {
        proxy_pass http://streamlit_backends;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### Docker Swarm

```yaml
services:
  streamlit:
    deploy:
      mode: replicated
      replicas: 3
      endpoint_mode: dnsrr # Enables session affinity
```

#### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: streamlit-service
spec:
  sessionAffinity: ClientIP # Enables session stickiness
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800 # 3 hours
  selector:
    app: streamlit
  ports:
    - protocol: TCP
      port: 8501
      targetPort: 8501
```

### Option B: External File Storage

**Best for:** Large files, long-term storage, or complex deployments

Store media files externally (S3, Azure Blob, etc.):

```python
import boto3
import streamlit as st

# Upload to S3
s3_client = boto3.client('s3')
s3_client.upload_fileobj(
    uploaded_file,
    'your-bucket-name',
    uploaded_file.name
)

# Generate presigned URL
url = s3_client.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'your-bucket-name', 'Key': uploaded_file.name},
    ExpiresIn=3600
)

# Display video from S3
st.video(url)
```

### Option C: Base64 Encoding (Current Implementation)

**Best for:** Small to medium files (< 50 MB), mobile camera uploads

Already implemented in our app:

- Automatic for video preview (optional)
- Configurable via `use_base64` parameter
- Falls back to standard display for large files

```python
# Enable Base64 encoding for video preview
render_video_preview(uploaded_file, use_base64=True)
```

## Mobile-Specific Considerations

### Camera Capture Attributes

Our implementation includes:

```javascript
fileInput.setAttribute("capture", "environment"); // Use rear camera
fileInput.setAttribute("accept", "video/*,.mp4,.mov,.webm,.npz");
```

### Testing Mobile Uploads

1. **Chrome DevTools Mobile Emulation**: Limited - can't test actual camera
2. **Real Device Testing**: Recommended approach
   - Test on iOS Safari
   - Test on Android Chrome
   - Test on different network conditions (WiFi, 4G, 5G)
3. **Browser Compatibility**:
   - iOS Safari 14+
   - Android Chrome 90+
   - Samsung Internet 14+

### Mobile Upload Best Practices

1. **File Size Limits**: Set reasonable limits (e.g., 100 MB)
2. **User Feedback**: Show upload progress and status
3. **Error Handling**: Graceful fallbacks if upload fails
4. **Retry Logic**: Automatic retries with exponential backoff
5. **Compression**: Consider client-side compression for large videos

## Monitoring & Debugging

### Key Metrics to Monitor

```python
import streamlit as st
import time

# Track upload performance
if 'upload_metrics' not in st.session_state:
    st.session_state.upload_metrics = []

def track_upload(filename, size_bytes, duration_seconds):
    st.session_state.upload_metrics.append({
        'filename': filename,
        'size_mb': size_bytes / (1024 * 1024),
        'duration': duration_seconds,
        'timestamp': time.time()
    })
```

### Debug Mode

Enable Streamlit debug mode to see detailed WebSocket logs:

```bash
streamlit run app.py --server.enableWebsocketCompression=true --logger.level=debug
```

### Common Issues & Solutions

#### Issue: Upload works on desktop but fails on mobile

**Solution:**

1. Enable Base64 encoding for mobile uploads
2. Check browser console for JavaScript errors
3. Verify file size is within limits
4. Test with different mobile browsers

#### Issue: Upload fails randomly in production

**Solution:**

1. Enable session affinity on load balancer
2. Use Base64 encoding for critical uploads
3. Check server logs for `MediaFileStorageError`
4. Ensure all replicas have shared storage

#### Issue: Large video uploads timeout

**Solution:**

1. Increase Streamlit's max upload size:
   ```toml
   # .streamlit/config.toml
   [server]
   maxUploadSize = 500  # MB
   ```
2. Use external storage (S3) for large files
3. Implement chunked uploads

## Configuration Files

### `.streamlit/config.toml`

Recommended production configuration:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200  # MB
enableWebsocketCompression = true

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false
fastReruns = true

[client]
showErrorDetails = false
```

### Environment Variables

```bash
# Production deployment
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

## Performance Optimization

### 1. Caching

Use Streamlit's caching for expensive operations:

```python
@st.cache_data
def process_video(video_bytes):
    # Expensive processing
    return processed_data
```

### 2. Session State Management

Clean up old uploads:

```python
# Clear old files after processing
def cleanup_session_files():
    if 'uploaded_files' in st.session_state:
        # Keep only recent uploads
        st.session_state.uploaded_files = st.session_state.uploaded_files[-5:]
```

### 3. Compression

Enable WebSocket compression:

```python
# In config.toml
[server]
enableWebsocketCompression = true
```

## Security Considerations

### 1. File Validation

```python
def validate_upload(file):
    # Check file type
    allowed_types = ['video/mp4', 'video/webm', 'application/x-npz']
    if file.type not in allowed_types:
        raise ValueError(f"Invalid file type: {file.type}")

    # Check file size
    max_size = 200 * 1024 * 1024  # 200 MB
    if len(file.getvalue()) > max_size:
        raise ValueError(f"File too large: {len(file.getvalue())} bytes")

    return True
```

### 2. Input Sanitization

```python
import re

def sanitize_filename(filename):
    # Remove potentially dangerous characters
    return re.sub(r'[^\w\-\.]', '_', filename)
```

### 3. Rate Limiting

Implement rate limiting to prevent abuse:

```python
import time

if 'last_upload_time' not in st.session_state:
    st.session_state.last_upload_time = 0

current_time = time.time()
if current_time - st.session_state.last_upload_time < 5:  # 5 seconds
    st.error("Please wait before uploading again")
    st.stop()

st.session_state.last_upload_time = current_time
```

## Testing Checklist

Before deploying to production:

- [ ] Test uploads on desktop (Chrome, Firefox, Safari)
- [ ] Test camera capture on iOS Safari
- [ ] Test camera capture on Android Chrome
- [ ] Test with various file sizes (small, medium, large)
- [ ] Test with slow network conditions
- [ ] Verify session affinity is enabled (if using multiple replicas)
- [ ] Test upload failures and error handling
- [ ] Monitor upload metrics and success rates
- [ ] Test concurrent users
- [ ] Verify security measures (file validation, sanitization)

## Additional Resources

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [WebSocket Session Management](https://docs.streamlit.io/library/advanced-features/session-state)
- [AWS ALB Sticky Sessions](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/sticky-sessions.html)
- [Nginx Load Balancing](https://nginx.org/en/docs/http/load_balancing.html)

## Support

For issues or questions:

1. Check browser console for JavaScript errors
2. Check Streamlit server logs
3. Enable debug mode for detailed diagnostics
4. Review GitHub issue #4173 for known upload issues

---

**Last Updated:** October 2025
**Version:** 1.0
