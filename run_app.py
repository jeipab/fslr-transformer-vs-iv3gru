#!/usr/bin/env python3
"""Launcher script for the FSLR Streamlit app."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app."""
    project_root = Path(__file__).parent
    app_path = project_root / "streamlit_app" / "main.py"
    
    # Run streamlit on the main.py file
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

if __name__ == "__main__":
    main()
