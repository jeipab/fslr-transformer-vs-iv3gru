"""
Label assignment script for Filipino sign language recognition.

This script creates labels.csv from NPZ files and maps gloss text labels to numeric IDs.

Two modes:
1. Auto mode: Scans NPZ directory, creates labels.csv, and assigns IDs
2. Labels mode: Updates existing labels.csv with IDs

Usage:
    # Auto mode - scans directory and creates/updates labels.csv
    python data/splitting/assign.py --directory data/processed/fsl-105_10-08
    
    # Labels mode - updates existing labels.csv
    python data/splitting/assign.py --labels data/processed/labels.csv

Output: labels.csv with columns: file, gloss, cat, occluded
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

def extract_metadata_from_npz(npz_path):
    """Extract metadata from NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        dict: Metadata with occluded flag
    """
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # Try to load metadata
        if 'meta' in npz_data:
            meta = json.loads(str(npz_data['meta']))
            occluded = meta.get('occluded_flag', 0)
        else:
            # No metadata, default to not occluded
            occluded = 0
            
        return {'occluded': occluded}
    except Exception as e:
        print(f"[WARN] Could not read metadata from {npz_path.name}: {e}")
        return {'occluded': 0}


def create_labels_from_directory(directory, output_file=None):
    """Create labels.csv by scanning NPZ files in directory.
    
    Args:
        directory: Directory containing NPZ files
        output_file: Output CSV path (default: directory/labels.csv)
        
    Returns:
        Path: Path to created labels.csv
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return None
    
    # Set default output path
    if output_file is None:
        output_file = directory / "labels.csv"
    else:
        output_file = Path(output_file)
    
    # Get all NPZ files
    npz_files = sorted(directory.glob("*.npz"))
    
    if not npz_files:
        print(f"[ERROR] No NPZ files found in {directory}")
        return None
    
    print(f"📂 Scanning directory: {directory}")
    print(f"📊 Found {len(npz_files)} NPZ files")
    
    # Extract metadata from each NPZ file
    data = []
    for i, npz_path in enumerate(npz_files, 1):
        if i % 100 == 0 or i == 1:
            print(f"  Progress: {i}/{len(npz_files)} files...")
        
        meta = extract_metadata_from_npz(npz_path)
        data.append({
            'file': npz_path.name,
            'occluded': meta['occluded']
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Created labels.csv with {len(data)} files")
    print(f"📊 Occlusion Statistics:")
    print(f"  Clear (0): {(df['occluded'] == 0).sum()} files")
    print(f"  Occluded (1): {(df['occluded'] == 1).sum()} files")
    
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
    parser.add_argument("--reference", type=str, default="data/splitting/labels_reference.csv",
                       help="Path to reference CSV file (default: data/splitting/labels_reference.csv)")
    
    args = parser.parse_args()
    
    reference_path = Path(args.reference)
    
    if not reference_path.exists():
        print(f"[ERROR] Reference file not found: {reference_path}")
        return
    
    # Determine mode and labels path
    if args.directory:
        # Auto mode: create labels.csv from directory
        directory = Path(args.directory)
        labels_path = directory / "labels.csv"
        
        # Create or recreate labels.csv from NPZ files
        print("🔍 Auto mode: Scanning NPZ files...")
        result = create_labels_from_directory(directory, labels_path)
        if result is None:
            return
    elif args.labels:
        # Labels mode: use existing labels.csv
        labels_path = Path(args.labels)
        if not labels_path.exists():
            print(f"[ERROR] Labels file not found: {labels_path}")
            return
    else:
        # Default: use data/processed/labels.csv
        labels_path = Path("data/processed/labels.csv")
        if not labels_path.exists():
            print(f"[ERROR] Labels file not found: {labels_path}")
            print("💡 Use --directory to scan NPZ files or --labels to specify a file")
            return
    
    print(f"\n📂 Loading reference: {reference_path}")
    print(f"📂 Processing labels: {labels_path}")
    
    # Load reference and labels
    gloss_cat = pd.read_csv(reference_path)
    labels = pd.read_csv(labels_path)
    
    print(f"\n🏷️  Assigning gloss and category IDs...")

    # Create mapping dictionaries
    gloss_map = dict(zip(gloss_cat["label"].str.lower(), gloss_cat["gloss_id"]))
    cat_map = dict(zip(gloss_cat["label"].str.lower(), gloss_cat["cat_id"]))
    
    def get_gloss_from_filename(filename):
        """Extract gloss text from filename.
        
        Args:
            filename: Video filename
            
        Returns:
            Extracted gloss text in lowercase
        """
        name = filename.split("_", 2)[-1].replace(".npz", "").strip().lower()
        return name
    
    # Extract gloss text from filenames
    labels["gloss_text"] = labels["file"].apply(get_gloss_from_filename)
    
    # Map text labels to numeric IDs
    labels["gloss"] = labels["gloss_text"].map(gloss_map)
    labels["cat"] = labels["gloss_text"].map(cat_map)
    
    # Check for unmapped labels
    unmapped = labels[labels["gloss"].isna()]
    if len(unmapped) > 0:
        print(f"\n⚠️  WARNING: {len(unmapped)} files have unmapped labels:")
        for _, row in unmapped.head(10).iterrows():
            print(f"  - {row['file']}: '{row['gloss_text']}' not found in reference")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")
        print("\n💡 Please check labels_reference.csv for missing entries")

    # Remove helper column
    labels = labels.drop(columns=["gloss_text"])
    
    # Reorder columns to match expected format: file, gloss, cat, occluded
    column_order = ["file", "gloss", "cat"]
    if "occluded" in labels.columns:
        column_order.append("occluded")
    labels = labels[column_order]
    
    # Save updated labels.csv
    labels.to_csv(labels_path, index=False)
    
    print(f"\n✅ labels.csv has been updated with gloss_id and cat_id mappings.")
    print(f"📁 Output: {labels_path}")
    print(f"📊 Total files: {len(labels)}")
    print(f"📊 Unique glosses: {labels['gloss'].nunique()}")
    print(f"📊 Unique categories: {labels['cat'].nunique()}")

if __name__ == "__main__":
    main()
