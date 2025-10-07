# 🚀 Quick Test Guide - Mobile Upload Fix

## Step 1: Restart Your App

The config changes **won't take effect** until you restart Streamlit.

### On Windows (PowerShell):

```powershell
# Find and kill Streamlit process
Get-Process | Where-Object {$_.ProcessName -like "*python*" -and $_.CommandLine -like "*streamlit*"} | Stop-Process -Force

# Or simply:
taskkill /F /IM python.exe

# Start fresh
streamlit run run_app.py
```

### On Linux/Mac:

```bash
# Kill Streamlit
pkill -f streamlit

# Start fresh
streamlit run run_app.py
```

---

## Step 2: Test Configuration

Run the test script to verify config is active:

```bash
streamlit run .streamlit/test_upload_config.py
```

**What to check:**

- ✅ Max Upload Size shows **500 MB** (not default)
- ✅ WebSocket Compression: **Enabled**
- ✅ CORS: **Enabled**

---

## Step 3: Test Actual Uploads

### Test A: Desktop (WiFi) Test

1. Open your main app
2. Upload a **10-15MB video** from your computer
3. **Expected**: Upload completes in 5-10 seconds

### Test B: Mobile (WiFi) Test

1. Open app on mobile device (connected to WiFi)
2. Try **camera recording** (10 seconds)
3. Try **gallery upload** (10-20MB file)
4. **Expected**: Both should work

### Test C: Mobile (4G) Test

1. Switch mobile to cellular data
2. Upload **8MB video** first (baseline - should work)
3. Upload **10MB video** (this was failing before)
4. **Expected**: 10MB upload now works (may take 20-40 seconds)

---

## Step 4: Monitor Performance

While testing, check:

### Upload Time Expectations:

| File Size | WiFi      | 4G Mobile  | 3G Mobile   |
| --------- | --------- | ---------- | ----------- |
| 5 MB      | 2-5 sec   | 8-15 sec   | 20-40 sec   |
| 8 MB      | 3-7 sec   | 12-25 sec  | 30-60 sec   |
| 10 MB     | 4-8 sec   | 15-30 sec  | 40-80 sec   |
| 20 MB     | 7-15 sec  | 30-60 sec  | 80-180 sec  |
| 50 MB     | 15-35 sec | 80-150 sec | 200-400 sec |

### Signs of Success:

- ✅ Progress bar moves smoothly
- ✅ No timeout errors
- ✅ 10MB+ files upload completely
- ✅ Upload completes even on slower connections

### Signs of Failure:

- ❌ Upload freezes at certain percentage
- ❌ "Connection error" or "WebSocket error"
- ❌ Progress bar disappears
- ❌ App reloads unexpectedly

---

## Troubleshooting

### Problem: Still fails at 10MB

**Check:**

1. Did you restart the app?

   ```bash
   # Verify no old process is running
   ps aux | grep streamlit  # Linux/Mac
   Get-Process python       # Windows
   ```

2. Is config being read?

   ```bash
   # Run test script to verify
   streamlit run .streamlit/test_upload_config.py
   ```

3. Are you testing on the same deployment?
   - Local test: Config applied immediately after restart
   - Cloud deployment: Need to commit & push changes

### Problem: Works on WiFi but not on 4G

**This is normal!** Try:

1. Enable compression in your preprocessing settings
2. Ask users to switch to WiFi for large uploads
3. Add user guidance:
   ```python
   st.info("📱 For best results on mobile data, keep videos under 8MB")
   ```

### Problem: Works sometimes but not always

**Possible causes:**

1. **Network instability**: Mobile networks vary in speed
2. **Server load**: Multiple users uploading simultaneously
3. **Browser cache**: Clear browser cache and retry

**Solutions:**

1. Implement client-side compression
2. Add retry logic
3. Show upload progress more clearly

---

## Optional: Enable Debug Mode

If you still have issues, uncomment this in `.streamlit/config.toml`:

```toml
[server]
# ... existing config ...

# Uncomment if issues persist
enableXsrfProtection = false
```

**⚠️ Warning**: This reduces security. Only use for testing.

---

## Expected Results After Fix

### Before Fix:

- ❌ Camera: Fails at 8-10MB
- ⚠️ Gallery: Works for small files, fails for large
- ❌ Mobile data: Very unreliable

### After Fix:

- ✅ Camera: Works up to 15-20MB
- ✅ Gallery: Works up to 500MB
- ✅ Mobile data: Works (just slower)

---

## Success Checklist

- [ ] Config shows 500MB limit (not default 200MB)
- [ ] App restarted successfully
- [ ] 10MB file uploads on desktop
- [ ] 10MB file uploads on mobile WiFi
- [ ] Camera recording uploads successfully
- [ ] Gallery upload works for large files
- [ ] No timeout errors

---

## Next Steps

Once testing confirms it works:

1. **Commit changes:**

   ```bash
   git add .streamlit/config.toml
   git commit -m "Fix mobile upload timeout issues (increase limits & WebSocket settings)"
   git push origin main
   ```

2. **Deploy to production** (if using cloud platform)

3. **Monitor user feedback** for any remaining issues

4. **Consider implementing** chunked uploads for very large files (optional)

---

**Need Help?**

- Check `.streamlit/CONFIG_NOTES.md` for detailed explanations
- Check `.streamlit/IMPLEMENTATION_SUMMARY.md` for technical details
- Run test script: `streamlit run .streamlit/test_upload_config.py`

---

**Last Updated**: October 7, 2025  
**Configuration Version**: 1.1 (with timeout fixes)
