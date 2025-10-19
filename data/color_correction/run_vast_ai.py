#!/usr/bin/env python3
"""
Comprehensive Vast.ai runner script with debugging
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("VAST.AI VIDEO COLOR CORRECTION - COMPREHENSIVE PACKAGE")
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
    
    # Step 1: Run debug script
    print("Step 1: Running environment debug...")
    try:
        result = subprocess.run([sys.executable, "vast_ai_debug.py"], check=True)
        print("✅ Debug completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Debug script had issues: {e}")
        print("Continuing anyway...")
    
    print("=" * 60)
    
    # Step 2: Run color correction with conservative settings
    print("Step 2: Running color correction...")
    cmd = [
        sys.executable, "color_correction_vast_ai.py",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "--strength", "0.5",
        "--no-cuda",
        "--batch-size", "128"  # High performance batch size
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
        else:
            print("\n⚠ No output files found!")
            print("This might indicate the videos are still black.")
            print("Check the debug output above for clues.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running color correction: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check the debug output above")
        print("2. Try reducing batch size: --batch-size 1")
        print("3. Try different correction strength: --strength 0.3")
        sys.exit(1)

if __name__ == "__main__":
    main()
