"""Demo video component using Streamlit's native dialog functionality."""

import streamlit as st


@st.dialog("FSLR Demo Video", width="medium")
def show_demo_video():
    """Show the demo video in a native Streamlit dialog."""
    
    # Google Drive video ID
    video_id = "1Y-c9S74wCY6wxCDcmHtnBUrFnAk2OShm"
    
    # Compact Google Drive video
    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-bottom: 15px;">
        <iframe 
            src="https://drive.google.com/file/d/{video_id}/preview" 
            width="640" 
            height="480" 
            style="border: none; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 100%;"
            allow="autoplay; fullscreen" 
            allowfullscreen>
        </iframe>
    </div>
    """, unsafe_allow_html=True)

def render_demo_button():
    """Render the demo button that opens the video dialog."""
    
    # Create the demo button in the sidebar
    if st.sidebar.button(
        "Play Demo Video", 
        help="Watch the FSLR demo video",
        use_container_width=True,
        type="secondary"
    ):
        # Open the dialog
        show_demo_video()
