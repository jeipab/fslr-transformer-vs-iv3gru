#!/usr/bin/env python3
"""
Rename and reorganize processed .npz files from sign language video clips.

Input structure (hierarchical with category folders):
FSL-105/
   C0/
      clip_00001_good_morning_S0.npz
      clip_00002_good_morning_S1.npz
   C1/
      clip_00004_good_evening_S2.npz
      ...
   
OR flat structure:
FSL-105/
   clip_00001_good_morning_S0.npz
   clip_00002_good_morning_S1.npz
   ...

Output options:
1. Renumber files (fix gaps): Renumber sequentially without gaps
2. Reorganize into hierarchical structure: Move to C/G/S folders
3. Rename based on labels_reference.csv: Validate/update labels

Usage:
    # For hierarchical input (C0/, C1/, ... folders):
    python rename_npz.py --input FSL-105 --hierarchical --summary
    
    # Renumber files to fix gaps
    python rename_npz.py --input FSL-105 --hierarchical --renumber --dry-run
    python rename_npz.py --input FSL-105 --hierarchical --renumber
    
    # Reorganize into C/G/S hierarchical structure
    python rename_npz.py --input FSL-105 --hierarchical --reorganize --output Renamed --dry-run
    python rename_npz.py --input FSL-105 --hierarchical --reorganize --output Renamed
    
    # Validate and fix labels based on labels_reference.csv
    python rename_npz.py --input FSL-105 --hierarchical --validate-labels --dry-run
    python rename_npz.py --input FSL-105 --hierarchical --validate-labels
    
    # Summary statistics
    python rename_npz.py --input FSL-105 --hierarchical --summary
    
    # For flat structure, omit --hierarchical flag
    python rename_npz.py --input FSL-105 --summary
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


def slugify_label(s: str) -> str:
    """Convert label to filename-friendly format."""
    s = s.strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'uncategorized'


def find_labels_file(root: Path, explicit_path: Path = None) -> Path:
    """Finds labels_reference.csv in common locations."""
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    possible_locations = [
        root / "labels_reference.csv",
        root / "data" / "labels_reference.csv",
        root / "data" / "raw" / "labels_reference.csv",
    ]
    
    for path in possible_locations:
        if path.exists():
            return path
    
    return explicit_path if explicit_path else (root / "data" / "labels_reference.csv")


def read_labels(labels_csv: Path) -> Dict[int, Tuple[str, int]]:
    """Reads the labels_reference.csv file and returns a mapping from gloss_id to (label, cat_id)."""
    print(f"🔍 Looking for labels_reference.csv at: {labels_csv}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels_reference.csv not found at {labels_csv}")
    
    mapping: Dict[int, Tuple[str, int]] = {}
    with labels_csv.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'gloss_id', 'label', 'cat_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"labels_reference.csv must contain columns: {required}. Found: {reader.fieldnames}")
        for row in reader:
            try:
                gloss_id = int(row['gloss_id'])
                cat_id = int(row['cat_id'])
                label = slugify_label(row['label'])
                mapping[gloss_id] = (label, cat_id)
            except (ValueError, KeyError) as e:
                print(f"[WARN] ❗ Skipping row with invalid data: {row}. Error: {e}", file=sys.stderr)
    return mapping


def parse_npz_filename(filename: str) -> Optional[Dict]:
    """Parse npz filename: clip_XXXXX_label_SY.npz"""
    match = re.match(r'^clip_(\d+)_(.+?)_(S\d+)(\.npz)$', filename)
    if not match:
        return None
    
    return {
        'number': int(match.group(1)),
        'label': match.group(2),
        'signer': match.group(3),
        'extension': match.group(4)
    }


def collect_npz_files(input_dir: Path, recursive: bool = False, hierarchical: bool = False) -> List[Tuple[Path, Dict]]:
    """Collect all .npz files and parse their filenames.
    
    Args:
        input_dir: Directory containing .npz files
        recursive: If True, search recursively in all subdirectories
        hierarchical: If True, expect C0/, C1/, ... structure and search within category folders
    """
    files = []
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if hierarchical:
        # Search for category directories (C0, C1, etc.)
        cat_dirs = sorted([p for p in input_dir.iterdir() 
                          if p.is_dir() and re.match(r'^C\d+$', p.name)])
        
        if not cat_dirs:
            print(f"[WARN] No category directories (C0, C1, ...) found in {input_dir}", file=sys.stderr)
            print(f"[INFO] Searching recursively for .npz files...", file=sys.stderr)
            # Fall back to recursive search
            for npz_file in input_dir.rglob('*.npz'):
                parsed = parse_npz_filename(npz_file.name)
                if parsed:
                    files.append((npz_file, parsed))
        else:
            print(f"📁 Found {len(cat_dirs)} category directories")
            # Search within each category directory
            for cat_dir in cat_dirs:
                for npz_file in cat_dir.rglob('*.npz'):
                    parsed = parse_npz_filename(npz_file.name)
                    if parsed:
                        files.append((npz_file, parsed))
                    else:
                        print(f"[WARN] ❗ Skipping file with invalid name: {npz_file.name}", file=sys.stderr)
    else:
        # Flat structure or recursive search
        pattern = '**/*.npz' if recursive else '*.npz'
        for npz_file in input_dir.glob(pattern):
            parsed = parse_npz_filename(npz_file.name)
            if parsed:
                files.append((npz_file, parsed))
            else:
                print(f"[WARN] ❗ Skipping file with invalid name: {npz_file.name}", file=sys.stderr)
    
    return sorted(files, key=lambda x: x[1]['number'])


def get_label_to_gloss_id(labels_map: Dict[int, Tuple[str, int]]) -> Dict[str, int]:
    """Reverse mapping: label -> gloss_id"""
    return {label: gloss_id for gloss_id, (label, _) in labels_map.items()}


def validate_labels(files: List[Tuple[Path, Dict]], labels_map: Dict[int, Tuple[str, int]]) -> List[Tuple[Path, Path, str]]:
    """Validate labels in filenames against labels_reference.csv and generate rename operations."""
    label_to_gloss = get_label_to_gloss_id(labels_map)
    operations = []
    errors = []
    
    for npz_file, parsed in files:
        label = parsed['label']
        current_path = npz_file
        
        # Check if label exists in reference
        if label not in label_to_gloss:
            errors.append(f"Unknown label '{label}' in {npz_file.name}")
            continue
        
        gloss_id = label_to_gloss[label]
        expected_label, cat_id = labels_map[gloss_id]
        
        # If label matches expected, no change needed
        if label == expected_label:
            continue
        
        # Generate new filename with corrected label
        new_name = f"clip_{parsed['number']:05d}_{expected_label}_{parsed['signer']}.npz"
        new_path = npz_file.parent / new_name
        
        if new_path != current_path:
            operations.append((current_path, new_path, f"Label correction: '{label}' -> '{expected_label}'"))
    
    if errors:
        print("\n⚠️  Validation errors found:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
    
    return operations


def renumber_files(files: List[Tuple[Path, Dict]], start_index: int = 1, digits: int = 5) -> List[Tuple[Path, Path]]:
    """Renumber files sequentially to fix gaps."""
    operations = []
    counter = start_index
    
    for npz_file, parsed in files:
        new_name = f"clip_{counter:0{digits}d}_{parsed['label']}_{parsed['signer']}.npz"
        new_path = npz_file.parent / new_name
        
        if new_path != npz_file:
            operations.append((npz_file, new_path))
        
        counter += 1
    
    return operations


def reorganize_into_hierarchical(
    files: List[Tuple[Path, Dict]], 
    labels_map: Dict[int, Tuple[str, int]], 
    output_dir: Path,
    start_index: int = 1,
    digits: int = 5
) -> List[Tuple[Path, Path]]:
    """Reorganize flat structure into C/G/S hierarchical structure."""
    label_to_gloss = get_label_to_gloss_id(labels_map)
    operations = []
    counter = start_index
    
    for npz_file, parsed in files:
        label = parsed['label']
        
        if label not in label_to_gloss:
            print(f"[WARN] ❗ Unknown label '{label}' in {npz_file.name}, skipping", file=sys.stderr)
            continue
        
        gloss_id = label_to_gloss[label]
        _, cat_id = labels_map[gloss_id]
        signer = parsed['signer']
        
        # Create hierarchical path: C{cat_id}/G{gloss_id}/{signer}/
        dest_dir = output_dir / f"C{cat_id}" / f"G{gloss_id}" / signer
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # New filename with sequential numbering
        new_name = f"clip_{counter:0{digits}d}_{label}_{signer}.npz"
        new_path = dest_dir / new_name
        
        operations.append((npz_file, new_path))
        counter += 1
    
    return operations


def generate_summary(files: List[Tuple[Path, Dict]], labels_map: Dict[int, Tuple[str, int]] = None):
    """Generate summary statistics."""
    print("📊 Generating summary...")
    print(f"Total .npz files found: {len(files)}")
    
    # Statistics by signer
    signer_counts = {}
    for _, parsed in files:
        signer = parsed['signer']
        signer_counts[signer] = signer_counts.get(signer, 0) + 1
    
    print("\nFiles per signer:")
    for signer in sorted(signer_counts.keys()):
        print(f"  {signer}: {signer_counts[signer]} files")
    
    # Statistics by label
    label_counts = {}
    for _, parsed in files:
        label = parsed['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nFiles per label:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"  {label}: {count} files")
    
    # Check for gaps in numbering
    numbers = [parsed['number'] for _, parsed in files]
    if numbers:
        min_num = min(numbers)
        max_num = max(numbers)
        expected_count = max_num - min_num + 1
        actual_count = len(numbers)
        gaps = expected_count - actual_count
        
        print(f"\nNumbering:")
        print(f"  Range: {min_num} to {max_num}")
        print(f"  Expected files: {expected_count}")
        print(f"  Actual files: {actual_count}")
        if gaps > 0:
            print(f"  ⚠️  Gaps detected: {gaps} missing numbers")
        else:
            print(f"  ✅ No gaps detected")
    
    # Validate against labels_reference.csv if provided
    if labels_map:
        label_to_gloss = get_label_to_gloss_id(labels_map)
        unknown_labels = set()
        for _, parsed in files:
            if parsed['label'] not in label_to_gloss:
                unknown_labels.add(parsed['label'])
        
        if unknown_labels:
            print(f"\n⚠️  Unknown labels (not in labels_reference.csv):")
            for label in sorted(unknown_labels):
                print(f"  {label}")
        else:
            print(f"\n✅ All labels match labels_reference.csv")
    
    print("✅ Summary complete.")


def main():
    ap = argparse.ArgumentParser(description="Rename and reorganize processed .npz files.")
    ap.add_argument("--input", type=Path, required=True, help="Input directory containing .npz files")
    ap.add_argument("--output", type=Path, default=None, help="Output directory (for --reorganize)")
    ap.add_argument("--root", type=Path, default=Path("."), help="Project root for finding labels_reference.csv")
    ap.add_argument("--labels", type=Path, default=None, help="Path to labels_reference.csv")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index for numbering (default: 1)")
    ap.add_argument("--digits", type=int, default=5, help="Zero-pad width for numbers (default: 5)")
    ap.add_argument("--recursive", action="store_true", help="Search for .npz files recursively")
    ap.add_argument("--hierarchical", action="store_true", help="Input is hierarchical with C0/, C1/, ... category folders")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would happen")
    ap.add_argument("--renumber", action="store_true", help="Renumber files sequentially to fix gaps")
    ap.add_argument("--reorganize", action="store_true", help="Reorganize into hierarchical C/G/S structure")
    ap.add_argument("--validate-labels", action="store_true", help="Validate and fix labels based on labels_reference.csv")
    ap.add_argument("--summary", action="store_true", help="Generate summary statistics only")
    args = ap.parse_args()

    input_dir = args.input.resolve()
    root = args.root.resolve()
    
    print(f"📂 Input directory: {input_dir}")
    
    # Collect all .npz files
    files = collect_npz_files(input_dir, recursive=args.recursive, hierarchical=args.hierarchical)
    
    if not files:
        print("[ERROR] ❌ No valid .npz files found", file=sys.stderr)
        return
    
    print(f"📦 Found {len(files)} .npz files")
    
    # Load labels if needed
    labels_map = None
    if args.validate_labels or args.reorganize or args.summary:
        explicit_labels_path = args.labels.resolve() if args.labels is not None else None
        labels_csv = find_labels_file(root, explicit_labels_path)
        labels_map = read_labels(labels_csv)
        print(f"✅ Loaded {len(labels_map)} labels from reference")
    
    # Summary mode
    if args.summary:
        generate_summary(files, labels_map)
        return
    
    # Validate labels mode
    if args.validate_labels:
        operations = validate_labels(files, labels_map)
        if not operations:
            print("✅ All labels are correct, no changes needed")
            return
        
        print(f"\n📝 Found {len(operations)} files to rename:")
        if args.dry_run:
            for src, dest, reason in operations[:10]:
                print(f"  {src.name} -> {dest.name} ({reason})")
            if len(operations) > 10:
                print(f"  ... and {len(operations) - 10} more")
            print("\n💡 Run without --dry-run to apply changes")
        else:
            for src, dest, reason in operations:
                try:
                    src.rename(dest)
                    print(f"✅ {src.name} -> {dest.name}")
                except Exception as e:
                    print(f"[ERROR] ❗ Could not rename {src}: {e}", file=sys.stderr)
            print(f"\n✅ Renamed {len(operations)} files")
        return
    
    # Renumber mode
    if args.renumber:
        operations = renumber_files(files, args.start_index, args.digits)
        if not operations:
            print("✅ All files already numbered correctly")
            return
        
        print(f"\n📝 Will renumber {len(operations)} files:")
        if args.dry_run:
            for src, dest in operations[:10]:
                print(f"  {src.name} -> {dest.name}")
            if len(operations) > 10:
                print(f"  ... and {len(operations) - 10} more")
            print("\n💡 Run without --dry-run to apply changes")
        else:
            # Use temporary names to avoid conflicts
            temp_renames = []
            for i, (src, dest) in enumerate(operations):
                temp_name = src.parent / f"__temp_{i}.npz"
                src.rename(temp_name)
                temp_renames.append((temp_name, dest))
            
            for temp, dest in temp_renames:
                temp.rename(dest)
            
            print(f"✅ Renumbered {len(operations)} files")
        return
    
    # Reorganize mode
    if args.reorganize:
        if not labels_map:
            print("[ERROR] ❌ --reorganize requires labels_reference.csv", file=sys.stderr)
            return
        
        output_dir = args.output.resolve() if args.output else (input_dir.parent / "Renamed").resolve()
        print(f"📂 Output directory: {output_dir}")
        
        operations = reorganize_into_hierarchical(files, labels_map, output_dir, args.start_index, args.digits)
        
        print(f"\n📝 Will reorganize {len(operations)} files into hierarchical structure:")
        if args.dry_run:
            for src, dest in operations[:10]:
                rel_src = src.relative_to(input_dir)
                rel_dest = dest.relative_to(output_dir)
                print(f"  {rel_src} -> {rel_dest}")
            if len(operations) > 10:
                print(f"  ... and {len(operations) - 10} more")
            print("\n💡 Run without --dry-run to apply changes")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            for src, dest in operations:
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src.rename(dest)
                except Exception as e:
                    print(f"[ERROR] ❗ Could not move {src} to {dest}: {e}", file=sys.stderr)
            print(f"✅ Reorganized {len(operations)} files")
        return
    
    # No action specified
    print("[ERROR] ❌ Please specify one of: --renumber, --reorganize, --validate-labels, or --summary", file=sys.stderr)


if __name__ == "__main__":
    main()

