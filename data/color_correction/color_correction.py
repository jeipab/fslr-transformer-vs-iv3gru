"""
Video Color Correction - Remove yellowish tints and improve video clarity

Features:
- CUDA-accelerated color correction using PyTorch
- Batch processing for multiple videos
- Advanced color correction algorithms (white balance, color temperature, saturation)
- Progress tracking with detailed statistics
- Error handling and logging
- Support for common video formats (mp4, avi, mov, mkv, etc.)
- Memory-efficient processing for large video collections

Usage:
    cd data
    python color_correction.py --input /path/to/input/folder --output /path/to/output/folder
    
    cd data
    python color_correction.py -i input_folder -o output_folder --batch-size 4 --cuda
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
        logging.FileHandler('video_color_correction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VideoColorCorrector:
    """
    CUDA-accelerated video color corrector with batch processing capabilities.
    
    This class handles color correction operations to remove yellowish tints
    and improve video clarity using GPU acceleration and batch processing.
    """
    
    def __init__(self, 
                 use_cuda: bool = True,
                 batch_size: int = 4,
                 quality: int = 23,
                 correction_strength: float = 1.0):
        """
        Initialize the video color corrector.
        
        Args:
            use_cuda (bool): Whether to use CUDA acceleration
            batch_size (int): Number of videos to process in parallel
            quality (int): FFmpeg quality parameter (18-28, lower = better quality)
            correction_strength (float): Strength of color correction (0.0-2.0)
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.batch_size = batch_size
        self.quality = quality
        self.correction_strength = max(0.0, min(2.0, correction_strength))
        
        # Video formats to process
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Initialize device
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for processing")
            
        # Initialize video codec parameters - use H264 for better compatibility
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        logger.info(f"Initialized VideoColorCorrector: "
                   f"CUDA: {self.use_cuda}, Batch size: {batch_size}, "
                   f"Correction strength: {correction_strength}")
    
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
    
    def correct_color_temperature(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Correct color temperature to remove yellowish tint.
        
        Args:
            frame_tensor (torch.Tensor): Input frame tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Color-corrected frame tensor
        """
        # Convert to float and normalize
        frame_float = frame_tensor.float() / 255.0
        
        # Split channels
        r, g, b = frame_float[:, 0:1], frame_float[:, 1:2], frame_float[:, 2:3]
        
        # Color temperature correction matrix (removes yellow tint)
        # Increase blue channel, slightly decrease red and green
        correction_matrix = torch.tensor([
            [0.95, 0.0, 0.05],    # Red channel: reduce red, add blue
            [0.0, 0.95, 0.05],    # Green channel: reduce green, add blue  
            [0.0, 0.0, 1.1]       # Blue channel: increase blue
        ], device=frame_tensor.device, dtype=frame_tensor.dtype)
        
        # Apply correction
        corrected_r = correction_matrix[0, 0] * r + correction_matrix[0, 1] * g + correction_matrix[0, 2] * b
        corrected_g = correction_matrix[1, 0] * r + correction_matrix[1, 1] * g + correction_matrix[1, 2] * b
        corrected_b = correction_matrix[2, 0] * r + correction_matrix[2, 1] * g + correction_matrix[2, 2] * b
        
        # Combine channels
        corrected_frame = torch.cat([corrected_r, corrected_g, corrected_b], dim=1)
        
        # Apply correction strength
        corrected_frame = frame_float + self.correction_strength * (corrected_frame - frame_float)
        
        # Clamp values to valid range
        corrected_frame = torch.clamp(corrected_frame, 0.0, 1.0)
        
        return corrected_frame
    
    def rgb_to_hsv(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to HSV tensor.
        
        Args:
            rgb_tensor (torch.Tensor): RGB tensor (B, C, H, W) with values in [0, 1]
            
        Returns:
            torch.Tensor: HSV tensor (B, C, H, W) with values in [0, 1]
        """
        r, g, b = rgb_tensor[:, 0:1], rgb_tensor[:, 1:2], rgb_tensor[:, 2:3]
        
        # Get max and min values
        max_val = torch.max(torch.max(r, g), b)
        min_val = torch.min(torch.min(r, g), b)
        diff = max_val - min_val
        
        # Calculate V (value)
        v = max_val
        
        # Calculate S (saturation)
        s = torch.where(max_val > 0, diff / max_val, torch.zeros_like(max_val))
        
        # Calculate H (hue)
        h = torch.zeros_like(max_val)
        
        # Red is max
        mask_r = (max_val == r) & (diff > 0)
        h = torch.where(mask_r, (60 * ((g - b) / diff) + 360) % 360, h)
        
        # Green is max
        mask_g = (max_val == g) & (diff > 0)
        h = torch.where(mask_g, 60 * ((b - r) / diff) + 120, h)
        
        # Blue is max
        mask_b = (max_val == b) & (diff > 0)
        h = torch.where(mask_b, 60 * ((r - g) / diff) + 240, h)
        
        # Normalize hue to [0, 1]
        h = h / 360.0
        
        return torch.cat([h, s, v], dim=1)
    
    def hsv_to_rgb(self, hsv_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert HSV tensor to RGB tensor.
        
        Args:
            hsv_tensor (torch.Tensor): HSV tensor (B, C, H, W) with values in [0, 1]
            
        Returns:
            torch.Tensor: RGB tensor (B, C, H, W) with values in [0, 1]
        """
        h, s, v = hsv_tensor[:, 0:1], hsv_tensor[:, 1:2], hsv_tensor[:, 2:3]
        
        # Convert hue to degrees
        h = h * 360.0
        
        # Calculate chroma
        c = v * s
        
        # Calculate x
        x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
        
        # Calculate m
        m = v - c
        
        # Initialize RGB
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        # Calculate RGB based on hue range
        mask1 = (h >= 0) & (h < 60)
        r = torch.where(mask1, c, r)
        g = torch.where(mask1, x, g)
        b = torch.where(mask1, 0, b)
        
        mask2 = (h >= 60) & (h < 120)
        r = torch.where(mask2, x, r)
        g = torch.where(mask2, c, g)
        b = torch.where(mask2, 0, b)
        
        mask3 = (h >= 120) & (h < 180)
        r = torch.where(mask3, 0, r)
        g = torch.where(mask3, c, g)
        b = torch.where(mask3, x, b)
        
        mask4 = (h >= 180) & (h < 240)
        r = torch.where(mask4, 0, r)
        g = torch.where(mask4, x, g)
        b = torch.where(mask4, c, b)
        
        mask5 = (h >= 240) & (h < 300)
        r = torch.where(mask5, x, r)
        g = torch.where(mask5, 0, g)
        b = torch.where(mask5, c, b)
        
        mask6 = (h >= 300) & (h < 360)
        r = torch.where(mask6, c, r)
        g = torch.where(mask6, 0, g)
        b = torch.where(mask6, x, b)
        
        # Add m to all channels
        r = r + m
        g = g + m
        b = b + m
        
        return torch.cat([r, g, b], dim=1)

    def enhance_contrast_and_saturation(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhance contrast and saturation for clearer appearance.
        
        Args:
            frame_tensor (torch.Tensor): Input frame tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Enhanced frame tensor
        """
        # Convert to HSV for better color manipulation
        frame_hsv = self.rgb_to_hsv(frame_tensor)
        h, s, v = frame_hsv[:, 0:1], frame_hsv[:, 1:2], frame_hsv[:, 2:3]
        
        # Enhance saturation
        s_enhanced = torch.clamp(s * 1.15, 0.0, 1.0)
        
        # Enhance value (brightness/contrast) using histogram equalization
        v_enhanced = torch.clamp(v * 1.1, 0.0, 1.0)
        
        # Recombine HSV
        enhanced_hsv = torch.cat([h, s_enhanced, v_enhanced], dim=1)
        
        # Convert back to RGB
        enhanced_rgb = self.hsv_to_rgb(enhanced_hsv)
        
        return enhanced_rgb
    
    def apply_white_balance(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply white balance correction to remove color casts.
        
        Args:
            frame_tensor (torch.Tensor): Input frame tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: White balance corrected frame tensor
        """
        # Calculate mean values for each channel
        r_mean = frame_tensor[:, 0:1].mean()
        g_mean = frame_tensor[:, 1:2].mean()
        b_mean = frame_tensor[:, 2:3].mean()
        
        # Calculate white balance gains (more conservative)
        gray_value = (r_mean + g_mean + b_mean) / 3.0
        r_gain = gray_value / (r_mean + 1e-8)
        g_gain = gray_value / (g_mean + 1e-8)
        b_gain = gray_value / (b_mean + 1e-8)
        
        # Limit gains to prevent extreme values
        r_gain = torch.clamp(r_gain, 0.5, 2.0)
        g_gain = torch.clamp(g_gain, 0.5, 2.0)
        b_gain = torch.clamp(b_gain, 0.5, 2.0)
        
        # Apply gains with correction strength (more conservative)
        r_corrected = frame_tensor[:, 0:1] * (1.0 + self.correction_strength * 0.3 * (r_gain - 1.0))
        g_corrected = frame_tensor[:, 1:2] * (1.0 + self.correction_strength * 0.3 * (g_gain - 1.0))
        b_corrected = frame_tensor[:, 2:3] * (1.0 + self.correction_strength * 0.3 * (b_gain - 1.0))
        
        # Combine channels
        corrected_frame = torch.cat([r_corrected, g_corrected, b_corrected], dim=1)
        
        # Clamp values
        corrected_frame = torch.clamp(corrected_frame, 0.0, 1.0)
        
        return corrected_frame
    
    def correct_video_color(self, video_path: Path, output_path: Path) -> bool:
        """
        Correct color of a single video file.
        
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
            
            # Open input video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open input video: {video_path}")
                return False
            
            # Setup output video writer with fallback codecs
            codecs_to_try = [cv2.VideoWriter_fourcc(*'H264'), cv2.VideoWriter_fourcc(*'mp4v'), cv2.VideoWriter_fourcc(*'XVID')]
            out = None
            
            for codec in codecs_to_try:
                out = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
                if out.isOpened():
                    logger.info(f"Using codec: {codec}")
                    break
                else:
                    out.release()
            
            if not out or not out.isOpened():
                logger.error(f"Cannot create output video with any codec: {output_path}")
                cap.release()
                return False
            
            # Process frames
            processed_frames = 0
            failed_frames = 0
            
            with tqdm(total=frame_count, desc=f"Color correcting {video_path.name}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    try:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        if self.use_cuda:
                            # Convert to tensor for CUDA processing
                            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                            
                            # Apply color corrections
                            # 1. White balance correction
                            corrected_frame = self.apply_white_balance(frame_tensor)
                            
                            # 2. Color temperature correction
                            corrected_frame = self.correct_color_temperature(corrected_frame)
                            
                            # 3. Enhance contrast and saturation
                            corrected_frame = self.enhance_contrast_and_saturation(corrected_frame)
                            
                            # Convert back to numpy array
                            corrected_frame_np = corrected_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            corrected_frame_np = (corrected_frame_np * 255).astype(np.uint8)
                            corrected_frame_bgr = cv2.cvtColor(corrected_frame_np, cv2.COLOR_RGB2BGR)
                        else:
                            # CPU fallback - simplified color correction
                            # Convert to LAB color space for better color manipulation
                            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            
                            # Apply color correction (more conservative)
                            # Reduce yellow tint by adjusting a and b channels
                            a = cv2.add(a, -5)  # Reduce green-magenta (less aggressive)
                            b = cv2.add(b, -8)  # Reduce yellow-blue (less aggressive)
                            
                            # Enhance contrast (more conservative)
                            l = cv2.convertScaleAbs(l, alpha=1.05, beta=2)
                            
                            # Merge channels and convert back to BGR
                            lab_corrected = cv2.merge([l, a, b])
                            corrected_frame_bgr = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
                            
                            # Additional simple color temperature adjustment
                            # Reduce yellow tint by adjusting blue channel
                            corrected_frame_bgr[:, :, 0] = cv2.add(corrected_frame_bgr[:, :, 0], 5)  # Increase blue
                            corrected_frame_bgr[:, :, 1] = cv2.add(corrected_frame_bgr[:, :, 1], -3)  # Slightly reduce green
                            corrected_frame_bgr[:, :, 2] = cv2.add(corrected_frame_bgr[:, :, 2], -5)  # Slightly reduce red
                        
                        out.write(corrected_frame_bgr)
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
    
    def correct_video_batch(self, video_batch: List[Path], output_folder: Path) -> List[bool]:
        """
        Process a batch of videos in parallel.
        
        Args:
            video_batch (List[Path]): List of video files to process
            output_folder (Path): Output folder path
            
        Returns:
            List[bool]: List of success status for each video
        """
        results = []
        
        # Create thread pool for I/O operations
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            
            for video_path in video_batch:
                # Calculate relative path and create output path
                relative_path = video_path.relative_to(video_path.parents[len(video_path.parents) - 2])
                output_path = output_folder / relative_path.with_suffix('.mp4')
                
                # Submit color correction task
                future = executor.submit(self.correct_video_color, video_path, output_path)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per video
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    results.append(False)
        
        return results
    
    def correct_videos(self, input_folder: Path, output_folder: Path) -> Tuple[int, int]:
        """
        Correct colors of all videos in the input folder.
        
        Args:
            input_folder (Path): Input folder containing videos
            output_folder (Path): Output folder for corrected videos
            
        Returns:
            Tuple[int, int]: (successful_count, total_count)
        """
        # Get all video files
        video_files = self.get_video_files(input_folder)
        if not video_files:
            logger.warning(f"No video files found in {input_folder}")
            return 0, 0
        
        logger.info(f"Starting batch color correction of {len(video_files)} videos")
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
        
        logger.info(f"Color correction complete: {successful_count}/{total_count} videos successful")
        return successful_count, total_count


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Correct video colors to remove yellowish tints and improve clarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python color_correction.py -i /path/to/input -o /path/to/output
  python color_correction.py -i input_folder -o output_folder --batch-size 8 --no-cuda
  python color_correction.py -i videos -o corrected --strength 1.5 --batch-size 4
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input folder containing video files')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output folder for color-corrected videos')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of videos to process in parallel (default: 4)')
    parser.add_argument('--quality', type=int, default=23,
                       help='Video quality parameter (18-28, lower=better, default: 23)')
    parser.add_argument('--strength', type=float, default=1.0,
                       help='Color correction strength (0.0-2.0, default: 1.0)')
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
    
    # Validate quality parameter
    if not (18 <= args.quality <= 28):
        logger.warning(f"Quality parameter {args.quality} is outside recommended range (18-28)")
    
    # Print system information
    logger.info("=" * 60)
    logger.info("VIDEO COLOR CORRECTION - CUDA ACCELERATED BATCH PROCESSING")
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
    corrector = VideoColorCorrector(
        use_cuda=not args.no_cuda,
        batch_size=args.batch_size,
        quality=args.quality,
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
