#!/usr/bin/env python3
"""
Simple runner script for Vast.ai deployment
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("VIDEO COLOR CORRECTION - VAST.AI DEPLOYMENT")
    print("=" * 60)
    
    # Check if input folder exists
    input_folder = Path("demo")
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' not found!")
        print("Please upload your video files to the 'demo' folder")
        sys.exit(1)
    
    # Create output folder
    output_folder = Path("corrected_output")
    output_folder.mkdir(exist_ok=True)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    # Run color correction
    cmd = [
        sys.executable, "color_correction.py",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "--strength", "0.5",
        "--no-cuda",
        "--batch-size", "128"  # Very aggressive batch size - requires high RAM
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ Color correction completed successfully!")
        print(f"Check the '{output_folder}' folder for corrected videos")
        
        # List output files
        output_files = list(output_folder.glob("*.mp4"))
        if output_files:
            print(f"\nGenerated {len(output_files)} corrected videos:")
            for file in output_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.1f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running color correction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
