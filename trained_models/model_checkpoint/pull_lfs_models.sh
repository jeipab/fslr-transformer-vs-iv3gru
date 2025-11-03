#!/bin/bash
# Script to pull Git LFS model files

echo "Checking Git LFS installation..."
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not installed."
    echo "Install it with: sudo apt-get install git-lfs (or brew install git-lfs on macOS)"
    exit 1
fi

echo "Initializing Git LFS..."
git lfs install

echo "Pulling model files from Git LFS..."
git lfs pull --include="trained_models/**/*.pt"

echo "Verifying files..."
for file in trained_models/*/FSL105_*/*.pt; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    if [ "$size" -gt 1000 ]; then
        echo "✓ $(basename "$file"): $size bytes"
    else
        echo "✗ $(basename "$file"): Still a pointer file ($size bytes)"
    fi
done
