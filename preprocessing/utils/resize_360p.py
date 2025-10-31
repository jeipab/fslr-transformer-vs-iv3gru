"""
Video Resizer - Convert videos to 640x360 resolution with CUDA acceleration

Features:
- CUDA-accelerated video processing using PyTorch/torchvision
- Batch processing for multiple videos
- Progress tracking with detailed statistics
- Error handling and logging
- Support for common video formats (mp4, avi, mov, mkv, etc.)
- Memory-efficient processing for large video collections

Usage:
    cd data
    python resize_360p.py --input /path/to/input/folder --output /path/to/output/folder
    
    cd data
    python resize_360p.py -i input_folder -o output_folder --batch-size 4 --cuda
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
import torchvision.transforms as transforms
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_resize.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VideoResizer:
    """
    CUDA-accelerated video resizer with batch processing capabilities.
    
    This class handles video resizing operations with GPU acceleration,
    batch processing, and comprehensive error handling.
    """
    
    def __init__(self, 
                 target_width: int = 640, 
                 target_height: int = 360,
                 use_cuda: bool = True,
                 batch_size: int = 4,
                 quality: int = 23):
        """
        Initialize the video resizer.
        
        Args:
            target_width (int): Target video width in pixels
            target_height (int): Target video height in pixels
            use_cuda (bool): Whether to use CUDA acceleration
            batch_size (int): Number of videos to process in parallel
            quality (int): FFmpeg quality parameter (18-28, lower = better quality)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.batch_size = batch_size
        self.quality = quality
        
        # Video formats to process
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Initialize CUDA device if available
        if self.use_cuda:
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for processing")
            
        # Initialize video codec parameters
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        logger.info(f"Initialized VideoResizer: {target_width}x{target_height}, "
                   f"CUDA: {self.use_cuda}, Batch size: {batch_size}")
    
    def get_video_files(self, input_folder: Path) -> List[Path]:
        """
        Get all video files from the input folder.
        
        Args:
            input_folder (Path): Path to input folder
            
        Returns:
            List[Path]: List of video file paths
        """
        video_files = []
        for file_path in input_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                video_files.append(file_path)
        
        logger.info(f"Found {len(video_files)} video files in {input_folder}")
        return video_files
    
    def get_video_info(self, video_path: Path) -> Tuple[int, int, float, int]:
        """
        Get video metadata.
        
        Args:
            video_path (Path): Path to video file
            
        Returns:
            Tuple[int, int, float, int]: (width, height, fps, frame_count)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        return width, height, fps, frame_count
    
    def resize_video_cuda(self, video_path: Path, output_path: Path) -> bool:
        """
        Resize video using CUDA acceleration.
        
        Args:
            video_path (Path): Input video path
            output_path (Path): Output video path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get video metadata
            width, height, fps, frame_count = self.get_video_info(video_path)
            
            # Skip if already correct resolution
            if width == self.target_width and height == self.target_height:
                logger.info(f"Skipping {video_path.name} - already {self.target_width}x{self.target_height}")
                return True
            
            # Open input video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open input video: {video_path}")
                return False
            
            # Setup output video writer
            out = cv2.VideoWriter(
                str(output_path), 
                self.fourcc, 
                fps, 
                (self.target_width, self.target_height)
            )
            
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
                        # Convert to tensor for CUDA processing
                        if self.use_cuda:
                            # Convert BGR to RGB and normalize
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                            
                            # Resize using torchvision (CUDA-accelerated)
                            resized_tensor = transforms.Resize(
                                (self.target_height, self.target_width),
                                interpolation=transforms.InterpolationMode.BILINEAR
                            )(frame_tensor)
                            
                            # Convert back to numpy array
                            resized_frame = resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            resized_frame = (resized_frame * 255).astype(np.uint8)
                            resized_frame_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                        else:
                            # CPU fallback
                            resized_frame_bgr = cv2.resize(
                                frame, 
                                (self.target_width, self.target_height),
                                interpolation=cv2.INTER_AREA
                            )
                        
                        out.write(resized_frame_bgr)
                        processed_frames += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process frame {processed_frames} in {video_path.name}: {e}")
                        failed_frames += 1
                        continue
                    
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
    
    def resize_video_batch(self, video_batch: List[Path], input_folder: Path, output_folder: Path) -> List[bool]:
        results = []
        
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            
            for video_path in video_batch:
                relative_path = video_path.relative_to(input_folder)
                output_path = output_folder / relative_path.with_suffix('.mp4')
                
                future = executor.submit(self.resize_video_cuda, video_path, output_path)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    results.append(False)
        
        return results
    
    def resize_videos(self, input_folder: Path, output_folder: Path) -> Tuple[int, int]:
        video_files = self.get_video_files(input_folder)
        if not video_files:
            logger.warning(f"No video files found in {input_folder}")
            return 0, 0
        
        logger.info(f"Starting batch processing of {len(video_files)} videos")
        logger.info(f"Target resolution: {self.target_width}x{self.target_height}")
        
        successful_count = 0
        total_count = len(video_files)
        
        batches = [video_files[i:i + self.batch_size] 
                  for i in range(0, len(video_files), self.batch_size)]
        
        with tqdm(total=len(batches), desc="Processing batches") as batch_pbar:
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                           f"({len(batch)} videos)")
                
                batch_results = self.resize_video_batch(batch, input_folder, output_folder)
                successful_count += sum(batch_results)
                
                batch_pbar.update(1)
                
                if self.use_cuda:
                    torch.cuda.empty_cache()
        
        logger.info(f"Processing complete: {successful_count}/{total_count} videos successful")
        return successful_count, total_count


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Resize videos to 640x360 with CUDA acceleration and batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resize_360p.py -i /path/to/input -o /path/to/output
  python resize_360p.py -i input_folder -o output_folder --batch-size 8 --no-cuda
  python resize_360p.py -i videos -o resized --quality 20 --batch-size 4
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input folder containing video files')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output folder for resized videos')
    parser.add_argument('--width', type=int, default=640,
                       help='Target width (default: 640)')
    parser.add_argument('--height', type=int, default=360,
                       help='Target height (default: 360)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of videos to process in parallel (default: 4)')
    parser.add_argument('--quality', type=int, default=23,
                       help='Video quality parameter (18-28, lower=better, default: 23)')
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
    
    # Validate quality parameter
    if not (18 <= args.quality <= 28):
        logger.warning(f"Quality parameter {args.quality} is outside recommended range (18-28)")
    
    # Print system information
    logger.info("=" * 60)
    logger.info("VIDEO RESIZER - CUDA ACCELERATED BATCH PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Target resolution: {args.width}x{args.height}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CUDA enabled: {not args.no_cuda}")
    logger.info(f"System CPU cores: {psutil.cpu_count()}")
    logger.info(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
    else:
        logger.info("CUDA not available")
    logger.info("=" * 60)
    
    # Initialize resizer
    resizer = VideoResizer(
        target_width=args.width,
        target_height=args.height,
        use_cuda=not args.no_cuda,
        batch_size=args.batch_size,
        quality=args.quality
    )
    
    # Process videos
    start_time = time.time()
    successful, total = resizer.resize_videos(input_folder, output_folder)
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
