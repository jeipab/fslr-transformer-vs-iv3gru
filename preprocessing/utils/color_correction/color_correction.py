#!/usr/bin/env python3
"""
Vast.ai Optimized Video Color Correction

This version is specifically optimized for Vast.ai deployment with better error handling
and compatibility checks.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Third-party imports
import torch
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VastAIColorCorrector:
    """
    Vast.ai optimized video color corrector with robust error handling.
    """
    
    def __init__(self, 
                 use_cuda: bool = False,
                 batch_size: int = 4,
                 correction_strength: float = 0.5):
        """
        Initialize the color corrector.
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.batch_size = batch_size
        self.correction_strength = correction_strength
        
        # Video formats to process
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Device setup
        if self.use_cuda:
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for processing")
        
        # Test codec compatibility
        self.fourcc = self._get_working_codec()
        
        logger.info(f"Initialized VastAIColorCorrector: "
                   f"CUDA: {self.use_cuda}, Batch size: {batch_size}, "
                   f"Correction strength: {correction_strength}")
    
    def _get_working_codec(self):
        """Find a working video codec"""
        codecs_to_try = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('DIVX', cv2.VideoWriter_fourcc(*'DIVX'))
        ]
        
        for codec_name, fourcc in codecs_to_try:
            try:
                # Test creating a video writer
                out = cv2.VideoWriter('test_codec.mp4', fourcc, 30.0, (640, 360))
                if out.isOpened():
                    out.release()
                    os.remove('test_codec.mp4')  # Clean up test file
                    logger.info(f"Using codec: {codec_name}")
                    return fourcc
            except Exception as e:
                logger.warning(f"Codec {codec_name} failed: {e}")
                continue
        
        logger.warning("No codec found, using default mp4v")
        return cv2.VideoWriter_fourcc(*'mp4v')
    
    def get_video_files(self, input_folder: Path) -> List[Path]:
        """Get all video files from the input folder."""
        video_files = []
        for file_path in input_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                video_files.append(file_path)
        
        logger.info(f"Found {len(video_files)} video files in {input_folder}")
        return video_files
    
    def apply_color_correction_cpu(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply color correction using CPU-only methods (more reliable on Vast.ai).
        """
        try:
            # Convert to LAB color space for better color manipulation
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply conservative color correction
            # Reduce yellow tint by adjusting a and b channels
            a_corrected = cv2.add(a, -int(5 * self.correction_strength))
            b_corrected = cv2.add(b, -int(8 * self.correction_strength))
            
            # Enhance contrast conservatively
            l_corrected = cv2.convertScaleAbs(l, alpha=1.0 + 0.05 * self.correction_strength, beta=int(2 * self.correction_strength))
            
            # Merge channels and convert back to BGR
            lab_corrected = cv2.merge([l_corrected, a_corrected, b_corrected])
            corrected_frame = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
            
            # Additional simple color temperature adjustment
            # Reduce yellow tint by adjusting channels
            corrected_frame[:, :, 0] = cv2.add(corrected_frame[:, :, 0], int(5 * self.correction_strength))  # Increase blue
            corrected_frame[:, :, 1] = cv2.add(corrected_frame[:, :, 1], -int(3 * self.correction_strength))  # Slightly reduce green
            corrected_frame[:, :, 2] = cv2.add(corrected_frame[:, :, 2], -int(5 * self.correction_strength))  # Slightly reduce red
            
            # Ensure values are in valid range
            corrected_frame = np.clip(corrected_frame, 0, 255).astype(np.uint8)
            
            return corrected_frame
            
        except Exception as e:
            logger.warning(f"Color correction failed, returning original frame: {e}")
            return frame
    
    def correct_video_color(self, video_path: Path, output_path: Path) -> bool:
        """
        Correct color of a single video file with robust error handling.
        """
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open input video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open input video: {video_path}")
                return False
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing {video_path.name}: {width}x{height}, {fps:.1f} fps, {frame_count} frames")
            
            # Setup output video writer
            out = cv2.VideoWriter(str(output_path), self.fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Cannot create output video: {output_path}")
                cap.release()
                return False
            
            # Process frames
            processed_frames = 0
            failed_frames = 0
            
            with tqdm(total=frame_count, desc=f"Processing {video_path.name}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    try:
                        # Apply color correction
                        corrected_frame = self.apply_color_correction_cpu(frame)
                        
                        # Write frame
                        out.write(corrected_frame)
                        processed_frames += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process frame {processed_frames} in {video_path.name}: {e}")
                        failed_frames += 1
                        # Write original frame if correction fails
                        out.write(frame)
                        processed_frames += 1
                    
                    pbar.update(1)
            
            # Cleanup
            cap.release()
            out.release()
            
            if processed_frames > 0:
                logger.info(f"Successfully processed {video_path.name}: "
                           f"{processed_frames} frames, {failed_frames} failed")
                return True
            else:
                logger.error(f"Failed to process any frames in {video_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            return False
    
    def correct_video_batch(self, video_batch: List[Path], output_folder: Path) -> List[bool]:
        """Process a batch of videos in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for I/O operations
        with ThreadPoolExecutor(max_workers=min(self.batch_size, 16)) as executor:  # Allow up to 16 threads for high batch sizes
            futures = []
            
            for video_path in video_batch:
                # Calculate output path
                relative_path = video_path.relative_to(video_path.parents[len(video_path.parents) - 2])
                output_path = output_folder / relative_path.with_suffix('.mp4')
                
                # Submit correction task
                future = executor.submit(self.correct_video_color, video_path, output_path)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per video
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    results.append(False)
        
        return results
    
    def correct_videos(self, input_folder: Path, output_folder: Path) -> Tuple[int, int]:
        """Correct all videos in the input folder."""
        # Get all video files
        video_files = self.get_video_files(input_folder)
        if not video_files:
            logger.warning(f"No video files found in {input_folder}")
            return 0, 0
        
        logger.info(f"Starting batch processing of {len(video_files)} videos")
        logger.info(f"Correction strength: {self.correction_strength}")
        
        # Process videos in batches
        successful_count = 0
        total_count = len(video_files)
        
        # Create batches
        batches = [video_files[i:i + self.batch_size] 
                  for i in range(0, len(video_files), self.batch_size)]
        
        with tqdm(total=len(batches), desc="Processing batches") as batch_pbar:
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                           f"({len(batch)} videos)")
                
                # Process batch
                batch_results = self.correct_video_batch(batch, output_folder)
                successful_count += sum(batch_results)
                
                batch_pbar.update(1)
                
                # Memory cleanup
                if self.use_cuda:
                    torch.cuda.empty_cache()
        
        logger.info(f"Processing complete: {successful_count}/{total_count} videos successful")
        return successful_count, total_count


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Vast.ai optimized video color correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python color_correction_vast_ai.py -i demo -o output
  python color_correction_vast_ai.py -i input_folder -o output_folder --batch-size 4 --strength 0.5
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input folder containing video files')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output folder for color-corrected videos')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of videos to process in parallel (default: 4)')
    parser.add_argument('--strength', type=float, default=0.5,
                       help='Color correction strength (0.0-2.0, default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA acceleration')
    
    args = parser.parse_args()
    
    # Validate arguments
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        logger.error(f"Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Validate batch size
    if args.batch_size < 1:
        logger.error("Batch size must be at least 1")
        sys.exit(1)
    
    # Validate correction strength
    if not (0.0 <= args.strength <= 2.0):
        logger.error("Correction strength must be between 0.0 and 2.0")
        sys.exit(1)
    
    # Print system information
    logger.info("=" * 60)
    logger.info("VAST.AI VIDEO COLOR CORRECTION")
    logger.info("=" * 60)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Correction strength: {args.strength}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CUDA enabled: {not args.no_cuda}")
    logger.info(f"System CPU cores: {psutil.cpu_count()}")
    logger.info(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
    else:
        logger.info("CUDA not available")
    logger.info("=" * 60)
    
    # Initialize color corrector
    corrector = VastAIColorCorrector(
        use_cuda=not args.no_cuda,
        batch_size=args.batch_size,
        correction_strength=args.strength
    )
    
    # Process videos
    start_time = time.time()
    successful, total = corrector.correct_videos(input_folder, output_folder)
    end_time = time.time()
    
    # Print summary
    duration = end_time - start_time
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {total - successful}")
    logger.info(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "Success rate: N/A")
    logger.info(f"Total time: {duration:.1f} seconds")
    logger.info(f"Average time per video: {duration/total:.1f} seconds" if total > 0 else "Average time per video: N/A")
    logger.info("=" * 60)
    
    if successful == total:
        logger.info("All videos processed successfully!")
        sys.exit(0)
    else:
        logger.warning(f"{total - successful} videos failed to process")
        sys.exit(1)


if __name__ == "__main__":
    main()
