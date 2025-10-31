#!/usr/bin/env python3
"""
Vast.ai Debug Script for Video Color Correction

This script helps debug why videos are black on Vast.ai but work locally.
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def debug_environment():
    """Debug environment differences"""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT DEBUG")
    logger.info("=" * 60)
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    
    # OpenCV version
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    # PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # NumPy version
    logger.info(f"NumPy version: {np.__version__}")

def debug_video_codecs():
    """Debug video codec support"""
    logger.info("=" * 60)
    logger.info("VIDEO CODEC DEBUG")
    logger.info("=" * 60)
    
    # Test different codecs
    codecs_to_test = [
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('MP4V', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('DIVX', cv2.VideoWriter_fourcc(*'DIVX'))
    ]
    
    for codec_name, fourcc in codecs_to_test:
        try:
            # Test creating a video writer
            out = cv2.VideoWriter('test_codec.mp4', fourcc, 30.0, (640, 360))
            if out.isOpened():
                logger.info(f"✅ {codec_name} codec: SUPPORTED")
                out.release()
            else:
                logger.info(f"❌ {codec_name} codec: NOT SUPPORTED")
        except Exception as e:
            logger.info(f"❌ {codec_name} codec: ERROR - {e}")

def debug_color_correction():
    """Debug color correction on a test image"""
    logger.info("=" * 60)
    logger.info("COLOR CORRECTION DEBUG")
    logger.info("=" * 60)
    
    # Create a test image (yellowish)
    test_image = np.full((100, 100, 3), [120, 140, 150], dtype=np.uint8)  # BGR format
    logger.info(f"Original test image shape: {test_image.shape}")
    logger.info(f"Original test image range: {test_image.min()} - {test_image.max()}")
    logger.info(f"Original test image mean: {test_image.mean(axis=(0,1))}")
    
    # Save original
    cv2.imwrite('debug_original.jpg', test_image)
    logger.info("✅ Saved debug_original.jpg")
    
    # Test LAB color space correction
    try:
        lab = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        logger.info(f"LAB conversion successful")
        logger.info(f"L channel range: {l.min()} - {l.max()}")
        logger.info(f"A channel range: {a.min()} - {a.max()}")
        logger.info(f"B channel range: {b.min()} - {b.max()}")
        
        # Apply conservative correction
        a_corrected = cv2.add(a, -5)
        b_corrected = cv2.add(b, -8)
        l_corrected = cv2.convertScaleAbs(l, alpha=1.05, beta=2)
        
        # Merge and convert back
        lab_corrected = cv2.merge([l_corrected, a_corrected, b_corrected])
        bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        logger.info(f"Corrected image range: {bgr_corrected.min()} - {bgr_corrected.max()}")
        logger.info(f"Corrected image mean: {bgr_corrected.mean(axis=(0,1))}")
        
        # Save corrected
        cv2.imwrite('debug_corrected.jpg', bgr_corrected)
        logger.info("✅ Saved debug_corrected.jpg")
        
    except Exception as e:
        logger.error(f"❌ LAB color correction failed: {e}")

def debug_video_processing():
    """Debug video processing with a simple test video"""
    logger.info("=" * 60)
    logger.info("VIDEO PROCESSING DEBUG")
    logger.info("=" * 60)
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_input.mp4', fourcc, 30.0, (640, 360))
    
    if not out.isOpened():
        logger.error("❌ Cannot create test video writer")
        return
    
    # Create 30 frames of yellowish video
    for i in range(30):
        # Create frame with slight variation
        frame = np.full((360, 640, 3), [120 + i, 140 + i, 150 + i], dtype=np.uint8)
        out.write(frame)
    
    out.release()
    logger.info("✅ Created test_input.mp4")
    
    # Now try to read and process it
    cap = cv2.VideoCapture('test_input.mp4')
    if not cap.isOpened():
        logger.error("❌ Cannot open test video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    
    # Try to process first frame
    ret, frame = cap.read()
    if ret:
        logger.info(f"First frame shape: {frame.shape}")
        logger.info(f"First frame range: {frame.min()} - {frame.max()}")
        logger.info(f"First frame mean: {frame.mean(axis=(0,1))}")
        
        # Apply color correction
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a_corrected = cv2.add(a, -5)
        b_corrected = cv2.add(b, -8)
        l_corrected = cv2.convertScaleAbs(l, alpha=1.05, beta=2)
        lab_corrected = cv2.merge([l_corrected, a_corrected, b_corrected])
        corrected_frame = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        logger.info(f"Corrected frame range: {corrected_frame.min()} - {corrected_frame.max()}")
        logger.info(f"Corrected frame mean: {corrected_frame.mean(axis=(0,1))}")
        
        # Try to write corrected video
        out_corrected = cv2.VideoWriter('test_output.mp4', fourcc, fps, (width, height))
        if out_corrected.isOpened():
            out_corrected.write(corrected_frame)
            out_corrected.release()
            logger.info("✅ Created test_output.mp4")
        else:
            logger.error("❌ Cannot create corrected video writer")
    else:
        logger.error("❌ Cannot read first frame")
    
    cap.release()

def main():
    """Main debug function"""
    logger.info("Starting Vast.ai debug session...")
    
    # Debug environment
    debug_environment()
    
    # Debug video codecs
    debug_video_codecs()
    
    # Debug color correction
    debug_color_correction()
    
    # Debug video processing
    debug_video_processing()
    
    logger.info("=" * 60)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 60)
    logger.info("Check the generated files:")
    logger.info("- debug_original.jpg (original test image)")
    logger.info("- debug_corrected.jpg (corrected test image)")
    logger.info("- test_input.mp4 (test input video)")
    logger.info("- test_output.mp4 (test output video)")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
