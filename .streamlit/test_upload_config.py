"""
Test script to verify Streamlit upload configuration and monitor performance.
Run this to check if your config changes are active.
"""

import streamlit as st
import time
from pathlib import Path
import toml

def main():
    st.title("🔍 Upload Configuration Test")
    
    # Check configuration
    st.header("Current Configuration")
    
    config_path = Path(".streamlit/config.toml")
    if config_path.exists():
        try:
            config = toml.load(config_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Max Upload Size",
                    f"{config.get('server', {}).get('maxUploadSize', 'default (200)')} MB"
                )
                st.metric(
                    "Max Message Size",
                    f"{config.get('server', {}).get('maxMessageSize', 'default (200)')} MB"
                )
            
            with col2:
                st.metric(
                    "WebSocket Compression",
                    "✅ Enabled" if config.get('server', {}).get('enableWebsocketCompression', False) else "❌ Disabled"
                )
                st.metric(
                    "CORS",
                    "✅ Enabled" if config.get('server', {}).get('enableCORS', False) else "❌ Disabled"
                )
            
            with st.expander("📄 Full Configuration"):
                st.json(config)
                
        except Exception as e:
            st.error(f"Error reading config: {e}")
    else:
        st.warning("⚠️ No config.toml found!")
    
    st.markdown("---")
    
    # Upload test
    st.header("Upload Performance Test")
    
    st.info("""
    **Test Instructions:**
    1. Upload a video file
    2. Check the upload time and speed
    3. Files over 10MB should now work
    """)
    
    uploaded_file = st.file_uploader(
        "Upload test video",
        type=["mp4", "mov", "avi"],
        help="Test your upload configuration"
    )
    
    if uploaded_file:
        start_time = time.time()
        
        # Read file data
        file_data = uploaded_file.getvalue()
        file_size_mb = len(file_data) / (1024 * 1024)
        
        duration = time.time() - start_time
        upload_speed = file_size_mb / duration if duration > 0 else 0
        
        st.success("✅ Upload Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        
        with col2:
            st.metric("Upload Time", f"{duration:.2f} seconds")
        
        with col3:
            st.metric("Upload Speed", f"{upload_speed:.2f} MB/s")
        
        # Analysis
        st.markdown("---")
        st.subheader("📊 Analysis")
        
        if file_size_mb > 10:
            st.success("✅ Successfully uploaded file > 10MB! Configuration is working.")
        elif file_size_mb > 8:
            st.info("ℹ️ File is in the 8-10MB range. Try uploading a larger file to fully test.")
        else:
            st.info("ℹ️ File is under 8MB. Upload a larger file (>10MB) to test the fix.")
        
        if duration > 30:
            st.warning("⚠️ Upload took >30 seconds. May still timeout on slower connections.")
        
        # Network speed estimation
        st.markdown("**Estimated Network Speed:**")
        if upload_speed > 5:
            st.success(f"🚀 Fast connection: {upload_speed:.1f} MB/s")
        elif upload_speed > 1:
            st.info(f"📶 Medium connection: {upload_speed:.1f} MB/s")
        else:
            st.warning(f"🐌 Slow connection: {upload_speed:.1f} MB/s - May have issues with large files")
    
    # Recommendations
    st.markdown("---")
    st.header("💡 Recommendations")
    
    st.markdown("""
    **For best results:**
    
    1. **Restart your Streamlit app** after changing config:
       ```bash
       pkill -f streamlit
       streamlit run run_app.py
       ```
    
    2. **Test on WiFi first** before testing on mobile networks
    
    3. **Expected upload times:**
       - 8MB: 10-20 seconds (mobile) / 2-5 seconds (WiFi)
       - 10MB: 15-30 seconds (mobile) / 3-7 seconds (WiFi)
       - 50MB: 60-150 seconds (mobile) / 10-30 seconds (WiFi)
    
    4. **If uploads still fail at 10MB:**
       - Check if app was restarted
       - Try on different network
       - Enable debug mode to see error messages
       - Consider uncommenting `enableXsrfProtection = false` in config
    """)

if __name__ == "__main__":
    main()

