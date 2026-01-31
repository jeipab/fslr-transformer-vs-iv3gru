"""Label assignment script for Filipino sign language recognition.

Creates labels.csv from NPZ files and maps gloss text labels to numeric IDs.
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

def extract_metadata_from_npz(npz_path):
    """Extract metadata from NPZ file."""
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
        
        meta = {}
        occluded = 0
        signer = 'N/A'
        duration = 0.0

        if 'meta' in npz_data:
            meta_content = npz_data['meta'].item()
            if isinstance(meta_content, str):
                meta = json.loads(meta_content)
            elif isinstance(meta_content, dict):
                meta = meta_content
            
            occluded = meta.get('occluded_flag', 0)
            signer = meta.get('signer', 'N/A')
            duration = meta.get('duration_sec', meta.get('duration', 0.0))

        if signer == 'N/A' or signer is None:
            match = re.search(r'_(S[0-7])\.npz$', npz_path.name)
            if match:
                signer = match.group(1)

        if signer is None or not re.match(r'^S[0-7]$', str(signer)):
            print(f"[WARN] Invalid signer format for {npz_path.name}: {signer}")
            signer = 'N/A'

        if duration == 0.0 and 'timestamps_ms' in npz_data:
            timestamps = npz_data['timestamps_ms']
            if len(timestamps) > 1:
                duration = (timestamps[-1] - timestamps[0]) / 1000.0

        return {'occluded': occluded, 'signer': signer, 'duration': duration}
    except Exception as e:
        print(f"[WARN] Could not read metadata from {npz_path.name}: {e}")
        return {'occluded': 0, 'signer': 'N/A', 'duration': 0.0}


def create_labels_from_directory(directory, output_file=None):
    """Create labels.csv by scanning NPZ files in directory."""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return None
    
    if output_file is None:
        output_file = directory / "labels.csv"
    else:
        output_file = Path(output_file)
    
    npz_files = sorted(directory.rglob("*.npz"))
    
    if not npz_files:
        print(f"[ERROR] No NPZ files found in {directory}")
        return None
    
    print(f"Scanning directory (recursive): {directory}")
    print(f"Found {len(npz_files)} NPZ files")
    
    data = []
    for i, npz_path in enumerate(npz_files, 1):
        if i % 100 == 0 or i == 1:
            print(f"  Progress: {i}/{len(npz_files)} files...")
        
        meta = extract_metadata_from_npz(npz_path)
        rel_path = npz_path.relative_to(directory)
        rel_path_str = str(rel_path).replace('\\', '/')
        data.append({
            'file': rel_path_str,
            'occluded': meta['occluded'],
            'signer': meta['signer'],
            'duration': meta['duration']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"\nCreated labels.csv with {len(data)} files")
    
    print(f"\nOcclusion Statistics:")
    print(f"  Clear (0): {(df['occluded'] == 0).sum()} files")
    print(f"  Occluded (1): {(df['occluded'] == 1).sum()} files")
    
    print(f"\nSigner Distribution:")
    signer_counts = df['signer'].value_counts().sort_index()
    for signer, count in signer_counts.items():
        print(f"  {signer}: {count} samples")

    print(f"\nDuration Statistics:")
    print(f"  Min: {df['duration'].min():.2f}s")
    print(f"  Max: {df['duration'].max():.2f}s")
    print(f"  Average: {df['duration'].mean():.2f}s")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Create labels.csv from NPZ files and assign gloss/category IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode - scan directory and create/update labels.csv
  python data/splitting/assign.py --directory data/processed/fsl-105_10-08
  
  # Labels mode - update existing labels.csv
  python data/splitting/assign.py --labels data/processed/labels.csv
        """
    )
    parser.add_argument("--directory", type=str, default=None,
                       help="Directory containing NPZ files (creates labels.csv automatically)")
    parser.add_argument("--labels", type=str, default=None,
                       help="Path to existing labels CSV file")
    parser.add_argument("--reference", type=str, default="data/labels_reference.csv",
                       help="Path to reference CSV file (default: data/labels_reference.csv)")
    
    args = parser.parse_args()
    
    reference_path = Path(args.reference)
    
    if not reference_path.exists():
        print(f"[ERROR] Reference file not found: {reference_path}")
        return
    
    if args.directory:
        directory = Path(args.directory)
        labels_path = directory / "labels.csv"
        
        print("Auto mode: Scanning NPZ files...")
        result = create_labels_from_directory(directory, labels_path)
        if result is None:
            return
    elif args.labels:
        labels_path = Path(args.labels)
        if not labels_path.exists():
            print(f"[ERROR] Labels file not found: {labels_path}")
            return
    else:
        labels_path = Path("data/processed/labels.csv")
        if not labels_path.exists():
            print(f"[ERROR] Labels file not found: {labels_path}")
            print("Use --directory to scan NPZ files or --labels to specify a file")
            return
    
    print(f"\nLoading reference: {reference_path}")
    print(f"Processing labels: {labels_path}")
    
    gloss_cat = pd.read_csv(reference_path)
    labels = pd.read_csv(labels_path)
    
    print(f"\nAssigning gloss and category IDs...")
    normalized_labels = (gloss_cat["label"]
                        .str.lower()
                        .str.replace(' ', '_', regex=False)
                        .str.replace('-', '_', regex=False)
                        .str.replace(r'[^a-z0-9_]', '', regex=True)
                        .str.replace(r'_+', '_', regex=True)
                        .str.strip('_'))
    gloss_map = dict(zip(normalized_labels, gloss_cat["gloss_id"]))
    cat_map = dict(zip(normalized_labels, gloss_cat["cat_id"]))
    
    def get_gloss_from_filename(filename):
        """Extract gloss text from filename."""
        filename = Path(filename).name
        match = re.match(r'clip_\d+_(.*?)_S[0-7]\.npz', filename)
        if match:
            return match.group(1).lower()
        else:
            return filename.split("_", 2)[-1].replace(".npz", "").strip().lower()

    labels["gloss_text"] = labels["file"].apply(get_gloss_from_filename)
    labels["gloss"] = labels["gloss_text"].map(gloss_map)
    labels["cat"] = labels["gloss_text"].map(cat_map)
    
    unmapped = labels[labels["gloss"].isna()]
    if len(unmapped) > 0:
        print(f"\nWARNING: {len(unmapped)} files have unmapped labels:")
        for _, row in unmapped.head(10).iterrows():
            print(f"  - {row['file']}: '{row['gloss_text']}' not found in reference")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")
        print("\nPlease check labels_reference.csv for missing entries")

    labels = labels.drop(columns=["gloss_text"])
    
    column_order = ["file", "gloss", "cat", "occluded", "signer", "duration"]
    
    for col in column_order:
        if col not in labels.columns:
            labels[col] = 'N/A'
            
    labels = labels[column_order]
    
    labels.to_csv(labels_path, index=False)
    
    print(f"\nlabels.csv has been updated with gloss_id and cat_id mappings.")
    print(f"Output: {labels_path}")
    print(f"Total files: {len(labels)}")
    print(f"Unique glosses: {labels['gloss'].nunique()}")
    print(f"Unique categories: {labels['cat'].nunique()}")

if __name__ == "__main__":
    main()