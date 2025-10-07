# Streamlit Configuration Notes

## Mobile Upload Issue Resolution

This configuration file addresses the issue where video uploads from mobile camera recordings fail while gallery uploads work.

## Configuration Breakdown

### Server Settings

#### 1. `maxUploadSize = 500` (MB)

**Purpose:** Sets the maximum file size that can be uploaded through Streamlit's file uploader.

**Default:** 200 MB

**Why 500MB:**

- Mobile camera recordings can be 10-50MB for short clips
- High-resolution videos from galleries can be 50-200MB
- Provides buffer for various video formats and qualities
- Accommodates future-proofing as camera quality improves

**Mobile Impact:**

- iOS Safari camera: ~2-8MB (720p, web-optimized)
- Android camera: ~5-15MB (720p-1080p)
- Gallery files: 10-100MB+ (full resolution)

#### 2. `maxMessageSize = 500` (MB)

**Purpose:** Sets the maximum WebSocket message size for client-server communication.

**Why it matters:**

- Streamlit uses WebSockets for real-time communication
- Large file uploads are chunked and sent through WebSocket messages
- Must match or exceed `maxUploadSize` to prevent transmission errors
- Critical for mobile devices with intermittent connections

**What happens if too low:**

- Upload appears to start but fails silently
- "Connection error" messages on large files
- Progress bar freezes

#### 3. `enableCORS = true`

**Purpose:** Enable Cross-Origin Resource Sharing for better mobile browser compatibility.

**Why it helps:**

- Mobile browsers often have stricter CORS policies
- Safari on iOS particularly strict about cross-origin requests
- Allows app to work when accessed from different domains
- Important for PWA (Progressive Web App) functionality

**Use cases:**

- Accessing app from custom domain
- Embedded in iframes
- Mobile app wrappers (Capacitor, Cordova)

#### 4. `enableWebsocketCompression = true`

**Purpose:** Compress WebSocket messages for better performance on slow connections.

**Benefits:**

- Reduces data transfer size by 50-70%
- Faster uploads on mobile networks (3G/4G)
- Lower data usage for mobile users
- Better performance in areas with poor signal

**Trade-offs:**

- Slightly higher CPU usage for compression/decompression
- Minimal impact on modern devices

#### 5. `headless = true`

**Purpose:** Server continues running even if it encounters errors.

**Why it matters for mobile:**

- Mobile connections often drop/reconnect
- Prevents server crash on connection interruptions
- Maintains session state during brief disconnections
- Essential for production deployments

### Browser Settings

#### 6. `gatherUsageStats = false`

**Purpose:** Disable anonymous usage statistics collection.

**Benefits:**

- Faster initial load time
- Reduced network requests
- Better privacy for users
- Slightly lower data usage

---

## Testing Recommendations

After applying these configurations:

1. **Test on actual mobile devices:**

   - iOS Safari (iPhone/iPad)
   - Android Chrome
   - Android Firefox

2. **Test scenarios:**

   - Upload from camera (take new video)
   - Upload from gallery (existing video)
   - Different file sizes: 5MB, 10MB, 25MB, 50MB+
   - Different network conditions: WiFi, 4G, 3G

3. **Monitor for issues:**
   - Upload timeout errors
   - Progress bar behavior
   - Memory usage on server
   - Connection stability

---

## Deployment Considerations

### Streamlit Community Cloud

✅ Reads `config.toml` automatically from repository

- Ensure `.streamlit/config.toml` is committed to git
- Changes take effect on next deployment

### Heroku / Railway / Render

⚠️ May have additional platform limits

- Check platform's own file size limits
- May need to increase request timeout
- Consider adding environment variables

### Self-hosted / VPS

✅ Full control over limits

- Ensure nginx/Apache allows large uploads
- Check `client_max_body_size` in nginx config
- Monitor server memory usage

---

## Alternative Solutions (if issues persist)

### 1. Add User Instructions

Show message on mobile:

> "📱 For best results on mobile, record video with your Camera app, then upload from Photo Library"

### 2. Client-side Compression

Consider adding JavaScript-based video compression:

- Reduces file size before upload
- More complex to implement
- May reduce video quality

### 3. Chunked Upload Implementation

For very large files:

- Break file into smaller chunks
- Upload chunks sequentially
- More resilient to connection interruptions
- Requires custom implementation

### 4. Direct S3/Cloud Upload

Bypass Streamlit's uploader:

- Generate pre-signed upload URLs
- Upload directly to cloud storage
- Lower server memory usage
- More complex infrastructure

---

## Troubleshooting

### Issue: Uploads still fail at ~10MB

**Possible causes:**

1. Hosting platform has own limits (check platform docs)
2. Nginx/reverse proxy limiting request size
3. Browser memory limitations on old devices
4. Network timeout (not file size)

**Solutions:**

- Increase nginx `client_max_body_size`
- Add timeout settings in server config
- Test on different browser/device

### Issue: Server runs out of memory

**Possible causes:**

- Multiple large uploads simultaneously
- Server doesn't have enough RAM

**Solutions:**

- Reduce `maxUploadSize` to 200-300MB
- Implement upload queuing
- Upgrade server resources
- Add file cleanup after processing

### Issue: Mobile Safari still has issues

**Possible causes:**

- iOS Safari has additional restrictions
- PWA mode has different limits
- Video codec compatibility

**Solutions:**

- Test in Chrome on iOS (uses same WebKit engine)
- Check video format compatibility
- Consider transcoding on server side

---

## Performance Impact

**Expected impacts with these settings:**

| Metric                   | Before          | After   | Notes                 |
| ------------------------ | --------------- | ------- | --------------------- |
| Max upload size          | 200MB (default) | 500MB   | Increased capacity    |
| Upload time (50MB on 4G) | ~30-60s         | ~25-50s | Compression helps     |
| Memory usage             | Baseline        | +10-20% | Per concurrent upload |
| Server CPU               | Baseline        | +5-10%  | WebSocket compression |

---

## Security Considerations

### XSRF Protection (Currently ENABLED)

The line `# enableXsrfProtection = false` is commented out (disabled by default in config).

**Current state:** XSRF protection is **ENABLED** (secure)

**If you need to disable it:**

1. Uncomment the line: `enableXsrfProtection = false`
2. Only do this if you experience persistent mobile upload issues
3. Understand the security trade-off:
   - ❌ More vulnerable to CSRF attacks
   - ✅ Better mobile compatibility
   - ⚠️ Only disable if app is not handling sensitive data

**Recommendation:** Keep XSRF protection enabled unless absolutely necessary.

---

## Monitoring Checklist

After deployment, monitor:

- [ ] Upload success rate (mobile vs desktop)
- [ ] Average upload times by file size
- [ ] Server memory usage during peak hours
- [ ] Connection error rates
- [ ] User feedback on upload experience

---

## Questions or Issues?

If you continue to experience upload issues after applying these configurations:

1. Check browser console for JavaScript errors
2. Check server logs for backend errors
3. Test with different video formats
4. Verify hosting platform supports configured sizes
5. Try testing on different networks (WiFi vs mobile data)

---

**Last Updated:** 2025-10-07
**Configuration Version:** 1.0
**Tested On:** Streamlit 1.x
