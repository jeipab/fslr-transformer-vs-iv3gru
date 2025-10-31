# Vast.ai Troubleshooting Guide

## If Videos Are Still Black

### 1. Check Debug Output
Run the debug script first:
```bash
python vast_ai_debug.py
```

Look for:
- ✅ Codec support
- ✅ Color correction test images
- ❌ Any error messages

### 2. Try Different Settings

#### Conservative Settings (Most Reliable):
```bash
python color_correction_vast_ai.py -i demo -o output --strength 0.3 --batch-size 32 --no-cuda
```

#### Moderate Settings:
```bash
python color_correction_vast_ai.py -i demo -o output --strength 0.5 --batch-size 64 --no-cuda
```

#### High Performance Settings (Default):
```bash
python color_correction_vast_ai.py -i demo -o output --strength 0.5 --batch-size 128 --no-cuda
```

### 3. Check Generated Files

After running, check these files:
- `debug_original.jpg` - Should show yellowish test image
- `debug_corrected.jpg` - Should show corrected test image
- `test_input.mp4` - Test input video
- `test_output.mp4` - Test output video

### 4. Common Issues

#### Issue: "Cannot create output video"
**Solution**: Try different codec
```bash
# Edit color_correction_vast_ai.py and change the codec order
```

#### Issue: "Color correction failed"
**Solution**: Reduce correction strength
```bash
python color_correction_vast_ai.py -i demo -o output --strength 0.2 --no-cuda
```

#### Issue: "Out of memory"
**Solution**: Reduce batch size
```bash
python color_correction_vast_ai.py -i demo -o output --batch-size 32 --no-cuda
```

### 5. Manual Testing

Test with a single video:
```bash
python color_correction_vast_ai.py -i demo -o test_output --strength 0.5 --batch-size 128 --no-cuda
```

### 6. Check Video Properties

Use ffmpeg to check video properties:
```bash
ffmpeg -i input_video.mp4 -f null -
ffmpeg -i output_video.mp4 -f null -
```

## Expected Results

- Input videos should have yellowish tint
- Output videos should have improved color balance
- File sizes should be similar (not much smaller)
- Videos should not be completely black

## Still Having Issues?

1. Check Vast.ai instance specs (RAM, CPU)
2. Try a different Vast.ai instance
3. Check if input videos are corrupted
4. Verify all dependencies are installed correctly
