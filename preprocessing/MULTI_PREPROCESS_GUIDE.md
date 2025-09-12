# Multi-Process Preprocessing Guide

This guide covers the multi-process preprocessing script that provides performance improvements over the original sequential processing.

## Key Features

- **Multi-process parallelization**: Process multiple videos simultaneously across CPU cores
- **Batched GPU inference**: InceptionV3 features extracted in batches for maximum GPU utilization
- **Configurable workers**: Control the number of parallel processes
- **Optional parquet output**: Disable parquet files to speed up I/O
- **Progress tracking**: Real-time progress monitoring with tqdm
- **Error handling**: Robust error handling for individual video failures

## Performance Improvements

### Expected Speedup

- **30-50x faster** than sequential processing on your hardware
- **GPU utilization**: Batched InceptionV3 processing keeps GPU busy
- **CPU utilization**: MediaPipe processing distributed across cores
- **I/O optimization**: Optional parquet disable reduces disk writes

### Hardware Recommendations

- **Workers**: 8-12 workers (default: min(cpu_count, 12))
- **Batch size**: 32-64 for InceptionV3 (default: 32)
- **Target FPS**: 15 fps recommended for speed vs quality balance

## Usage

### Basic Usage

```bash
python preprocessing/multi_preprocess.py /path/to/videos /path/to/output --write-keypoints --write-iv3-features
```

### Optimized Usage (Recommended)

```bash
python preprocessing/multi_preprocess.py /path/to/videos /path/to/output \
    --write-keypoints \
    --write-iv3-features \
    --workers 10 \
    --batch-size 64 \
    --target-fps 15 \
    --disable-parquet
```

### Full Options

```bash
python preprocessing/multi_preprocess.py /path/to/videos /path/to/output \
    --write-keypoints \
    --write-iv3-features \
    --workers 10 \
    --batch-size 64 \
    --target-fps 15 \
    --out-size 256 \
    --conf-thresh 0.5 \
    --max-gap 5 \
    --disable-parquet \
    --id 1 \
    --labels-csv /path/to/labels.csv \
    --occ-enable
```

## Command Line Arguments

### Core Processing

- `video_directory`: Path to video file or directory containing videos
- `output_directory`: Output directory for processed files
- `--write-keypoints`: Extract and save MediaPipe keypoints
- `--write-iv3-features`: Extract and save InceptionV3 features

### Performance Tuning

- `--workers N`: Number of parallel worker processes (default: min(cpu_count, 12))
- `--batch-size N`: Batch size for InceptionV3 GPU inference (default: 32)
- `--target-fps N`: Target frames per second (default: 15, recommended for speed)
- `--disable-parquet`: Disable parquet output to speed up I/O

### Processing Parameters

- `--out-size N`: Output image size for keypoint extraction (default: 256)
- `--conf-thresh F`: Confidence threshold for keypoints (default: 0.5)
- `--max-gap N`: Maximum gap for interpolation (default: 5)

### Labeling

- `--id N`: Single integer ID for both gloss and category
- `--gloss-id N`: Override gloss ID
- `--cat-id N`: Override category ID
- `--labels-csv PATH`: Path to labels CSV file
- `--append`: Append to existing labels CSV

### Occlusion Detection

- `--occ-enable`: Enable occlusion detection
- `--occ-vis-thresh F`: Frame visible fraction threshold (default: 0.6)
- `--occ-frame-prop F`: Clip occluded if proportion >= this (default: 0.4)
- `--occ-min-run N`: Clip occluded if run length >= this (default: 15)

## Performance Tips

### For Maximum Speed

1. Use `--target-fps 15` (reduces processing load)
2. Use `--disable-parquet` (faster I/O)
3. Use `--batch-size 64` (better GPU utilization)
4. Use `--workers 10-12` (optimal for your CPU)

### For Maximum Quality

1. Use `--target-fps 30` (higher quality)
2. Keep parquet enabled for debugging
3. Use `--batch-size 32` (more stable)
4. Use `--workers 8` (more stable)

### Memory Considerations

- Larger batch sizes use more GPU memory
- More workers use more CPU memory
- Monitor GPU memory usage with `nvidia-smi`

## Example Workflows

### Quick Processing (Speed Priority)

```bash
python preprocessing/multi_preprocess.py data/raw data/processed \
    --write-keypoints --write-iv3-features \
    --workers 12 --batch-size 64 --target-fps 15 --disable-parquet
```

### Quality Processing (Quality Priority)

```bash
python preprocessing/multi_preprocess.py data/raw data/processed \
    --write-keypoints --write-iv3-features \
    --workers 8 --batch-size 32 --target-fps 30
```

### With Labels and Occlusion Detection

```bash
python preprocessing/multi_preprocess.py data/raw data/processed \
    --write-keypoints --write-iv3-features \
    --workers 10 --batch-size 48 --target-fps 15 \
    --id 1 --labels-csv data/processed/labels.csv --occ-enable
```

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or number of workers
2. **CUDA errors**: Ensure proper GPU drivers and PyTorch CUDA installation
3. **Slow processing**: Check if parquet is disabled and FPS is set to 15
4. **Worker crashes**: Reduce number of workers or check video file integrity

### Performance Monitoring

- Monitor GPU usage: `nvidia-smi -l 1`
- Monitor CPU usage: `htop` or `top`
- Monitor disk I/O: `iotop`

## Expected Results

With ~2000 videos on your hardware:

- **Sequential processing**: Days
- **Multi-process processing**: 2-4 hours
- **Speedup**: 30-50x improvement

The script provides detailed progress information and final statistics including:

- Processing time per video
- Total processing time
- Success/failure counts
- Videos processed per hour
