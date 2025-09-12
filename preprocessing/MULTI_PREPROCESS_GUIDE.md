# Multi-Process Preprocessing Guide

Multi-process preprocessing script with 30-50x performance improvement over sequential processing.

## Key Features

- Multi-process parallelization across CPU cores
- Batched GPU inference for InceptionV3 features
- Configurable workers and batch sizes
- Optional parquet output disable
- Real-time progress tracking

## Hardware Recommendations

- **Workers**: 8-12 (default: min(cpu_count, 12))
- **Batch size**: 32-64 (default: 32)
- **Target FPS**: 15 fps for speed vs quality balance

## Usage

### Basic

```bash
python preprocessing/multi_preprocess.py /path/to/videos /path/to/output --write-keypoints --write-iv3-features
```

### Optimized (Recommended)

```bash
python preprocessing/multi_preprocess.py /path/to/videos /path/to/output \
    --write-keypoints --write-iv3-features \
    --workers 10 --batch-size 64 --target-fps 15 --disable-parquet
```

## Arguments

### Core

- `video_directory`: Path to video file or directory
- `output_directory`: Output directory for processed files
- `--write-keypoints`: Extract MediaPipe keypoints
- `--write-iv3-features`: Extract InceptionV3 features

### Performance

- `--workers N`: Parallel worker processes (default: min(cpu_count, 12))
- `--batch-size N`: InceptionV3 batch size (default: 32)
- `--target-fps N`: Target FPS (default: 15)
- `--disable-parquet`: Disable parquet output

### Processing

- `--out-size N`: Image size for keypoints (default: 256)
- `--conf-thresh F`: Keypoint confidence threshold (default: 0.5)
- `--max-gap N`: Max interpolation gap (default: 5)

### Labels

- `--id N`: Single ID for gloss and category
- `--labels-csv PATH`: Labels CSV file path
- `--append`: Append to existing CSV

### Occlusion

- `--occ-enable`: Enable occlusion detection
- `--occ-vis-thresh F`: Visibility threshold (default: 0.6)
- `--occ-frame-prop F`: Occluded frame proportion (default: 0.4)
- `--occ-min-run N`: Min consecutive occluded frames (default: 15)

## Performance Tips

### Speed Priority

- `--target-fps 15` --disable-parquet --batch-size 64 --workers 10-12`

### Quality Priority

- `--target-fps 30` --batch-size 32 --workers 8`

### Memory

- Larger batch sizes = more GPU memory
- More workers = more CPU memory
- Monitor with `nvidia-smi`

## Examples

### Speed Priority

```bash
python preprocessing/multi_preprocess.py data/raw data/processed \
    --write-keypoints --write-iv3-features \
    --workers 12 --batch-size 64 --target-fps 15 --disable-parquet
```

### Quality Priority

```bash
python preprocessing/multi_preprocess.py data/raw data/processed \
    --write-keypoints --write-iv3-features \
    --workers 8 --batch-size 32 --target-fps 30
```

### With Labels

```bash
python preprocessing/multi_preprocess.py data/raw data/processed \
    --write-keypoints --write-iv3-features \
    --workers 10 --batch-size 48 --target-fps 15 \
    --id 1 --labels-csv data/processed/labels.csv --occ-enable
```

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or workers
2. **CUDA errors**: Check GPU drivers and PyTorch CUDA installation
3. **Slow processing**: Use `--disable-parquet --target-fps 15`
4. **Worker crashes**: Reduce workers or check video integrity
5. **Conflicting dependencies**: Install dedicated kernel:
   ```bash
   python -m ipykernel install --user --name=workspace-venv --display-name="Python (workspace-venv)"
   ```

### Monitoring

- GPU: `nvidia-smi -l 1`
- CPU: `htop` or `top`
- Disk I/O: `iotop`

## Expected Results

- **Sequential**: Days
- **Multi-process**: 2-4 hours
- **Speedup**: 30-50x improvement
