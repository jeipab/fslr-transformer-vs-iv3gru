#!/usr/bin/env python3
"""
Example script for running multi-process preprocessing.

This script demonstrates how to use the multi_preprocess.py script
with optimal settings for your hardware configuration.
"""

import os
import sys
import subprocess
import argparse

def run_preprocessing(video_dir, output_dir, workers=10, batch_size=64, target_fps=15, 
                     disable_parquet=True, write_keypoints=True, write_iv3_features=True,
                     gloss_id=None, cat_id=None, labels_csv=None):
    """
    Run multi-process preprocessing with optimal settings.
    
    Args:
        video_dir: Directory containing input videos
        output_dir: Directory for processed output
        workers: Number of parallel workers (default: 10)
        batch_size: Batch size for InceptionV3 (default: 64)
        target_fps: Target frames per second (default: 15)
        disable_parquet: Disable parquet output for speed (default: True)
        write_keypoints: Extract keypoints (default: True)
        write_iv3_features: Extract InceptionV3 features (default: True)
        gloss_id: Gloss ID for labeling (optional)
        cat_id: Category ID for labeling (optional)
        labels_csv: Path to labels CSV (optional)
    """
    
    # Build command
    cmd = [
        sys.executable, "preprocessing/multi_preprocess.py",
        video_dir,
        output_dir,
        "--workers", str(workers),
        "--batch-size", str(batch_size),
        "--target-fps", str(target_fps)
    ]
    
    # Add optional flags
    if write_keypoints:
        cmd.append("--write-keypoints")
    if write_iv3_features:
        cmd.append("--write-iv3-features")
    if disable_parquet:
        cmd.append("--disable-parquet")
    
    # Add labeling options
    if gloss_id is not None:
        cmd.extend(["--gloss-id", str(gloss_id)])
    if cat_id is not None:
        cmd.extend(["--cat-id", str(cat_id)])
    if labels_csv is not None:
        cmd.extend(["--labels-csv", labels_csv])
    
    # Add occlusion detection
    cmd.append("--occ-enable")
    
    print("Running multi-process preprocessing with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nPreprocessing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nPreprocessing failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run multi-process preprocessing with optimal settings")
    parser.add_argument("video_dir", help="Directory containing input videos")
    parser.add_argument("output_dir", help="Directory for processed output")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for InceptionV3 (default: 64)")
    parser.add_argument("--target-fps", type=int, default=15, help="Target frames per second (default: 15)")
    parser.add_argument("--enable-parquet", action="store_true", help="Enable parquet output (default: disabled)")
    parser.add_argument("--no-keypoints", action="store_true", help="Disable keypoint extraction")
    parser.add_argument("--no-iv3", action="store_true", help="Disable InceptionV3 feature extraction")
    parser.add_argument("--gloss-id", type=int, help="Gloss ID for labeling")
    parser.add_argument("--cat-id", type=int, help="Category ID for labeling")
    parser.add_argument("--labels-csv", help="Path to labels CSV file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory '{args.video_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run preprocessing
    success = run_preprocessing(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        target_fps=args.target_fps,
        disable_parquet=not args.enable_parquet,
        write_keypoints=not args.no_keypoints,
        write_iv3_features=not args.no_iv3,
        gloss_id=args.gloss_id,
        cat_id=args.cat_id,
        labels_csv=args.labels_csv
    )
    
    if success:
        print(f"\nProcessed videos saved to: {args.output_dir}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
