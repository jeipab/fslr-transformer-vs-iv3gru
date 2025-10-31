#!/usr/bin/env python3
"""
Rename and renumber video files in hierarchical C/G/S structure.

Input structure:
clips/
   C0/
      G0/
         S0/
            0.MOV
            1.MOV
            3.MOV  ← Gap here (missing 2.MOV)
         S1/
         ...
      G1/
      ...
   C1/
   ...

This script can:
1. Renumber files to fix gaps (0.MOV, 1.MOV, 2.MOV instead of 0.MOV, 1.MOV, 3.MOV)
2. Rename to clip_XXXX_label_SY format while keeping structure
3. Add leading zeros for consistent formatting

Usage:
    # Renumber files to fix gaps (0.MOV, 1.MOV, 2.MOV, etc.)
    python rename_hierarchical_clips.py --input clips --renumber --dry-run
    python rename_hierarchical_clips.py --input clips --renumber --start-index 0 --digits 4
    
    # Rename to clip_XXXX_label_SY format (requires labels_reference.csv)
    python rename_hierarchical_clips.py --input clips --rename-to-clip-format --root . --dry-run
    python rename_hierarchical_clips.py --input clips --rename-to-clip-format --root .
    
    # Summary statistics
    python rename_hierarchical_clips.py --input clips --summary
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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


def collect_hierarchical_clips(clips_dir: Path, video_extensions=None) -> List[Dict]:
    """Collects clips from a 3-level hierarchical structure (C/G/S)."""
    if video_extensions is None:
        video_extensions = {'.MOV', '.mov', '.mp4', '.MP4', '.avi', '.AVI'}
    
    items = []
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips directory not found at {clips_dir}")
    
    cat_dirs = sorted([p for p in clips_dir.iterdir() if p.is_dir() and re.match(r'^C\d+$', p.name)])
    for cat_dir in cat_dirs:
        cat_id = int(cat_dir.name[1:])
        gloss_dirs = sorted([p for p in cat_dir.iterdir() if p.is_dir() and re.match(r'^G\d+$', p.name)])
        for gloss_dir in gloss_dirs:
            gloss_id = int(gloss_dir.name[1:])
            signer_dirs = sorted([p for p in gloss_dir.iterdir() if p.is_dir() and re.match(r'^S\d+$', p.name)])
            for signer_dir in signer_dirs:
                signer = signer_dir.name
                # Find all video files (use set to avoid duplicates)
                video_files = set()
                for ext in video_extensions:
                    video_files.update(signer_dir.glob(f"*{ext}"))
                
                for video_file in sorted(video_files):
                    # Check if filename is just a number
                    base_name = video_file.stem
                    file_num = None
                    if base_name.isdigit():
                        file_num = int(base_name)
                    
                    items.append({
                        "path": video_file,
                        "cat_id": cat_id,
                        "gloss_id": gloss_id,
                        "signer": signer,
                        "file_num": file_num,  # None if not a number
                        "directory": str(signer_dir)
                    })
    return items


def renumber_hierarchical_clips(items: List[Dict], start_index: int = 0, digits: int = 4) -> List[Tuple[Path, Path]]:
    """Renumber files in hierarchical structure to fix gaps.
    
    Groups files by directory (C/G/S path) and renumbers them sequentially.
    """
    operations = []
    
    # Group files by directory
    by_directory = {}
    for item in items:
        directory = item['directory']
        if directory not in by_directory:
            by_directory[directory] = []
        by_directory[directory].append(item)
    
    # Process each directory separately
    for directory in sorted(by_directory.keys()):
        files = by_directory[directory]
        
        # Filter to only numbered files and sort by number
        numbered_files = [(f['file_num'], f) for f in files if f['file_num'] is not None]
        numbered_files.sort(key=lambda x: x[0])
        
        if not numbered_files:
            continue
        
        # Renumber sequentially starting from start_index
        counter = start_index
        for original_num, item in numbered_files:
            video_file = item['path']
            
            # Format new name with leading zeros
            new_name = f"{counter:0{digits}d}{video_file.suffix}"
            dest = video_file.parent / new_name
            
            # Only add if the name needs to change
            if video_file.name != new_name:
                operations.append((video_file, dest))
            
            counter += 1
    
    return operations


def rename_to_clip_format(items: List[Dict], labels_map: Dict[int, Tuple[str, int]], 
                          start_index: int = 1, digits: int = 4, keep_structure: bool = True) -> List[Tuple[Path, Path]]:
    """Rename files from 0.MOV format to clip_XXXX_label_SY.MOV format.
    
    If keep_structure is True, files stay in their C/G/S directories.
    Otherwise, they're moved to a flat structure.
    """
    operations = []
    counter = start_index
    
    for item in items:
        video_file = item['path']
        gloss_id = item['gloss_id']
        signer = item['signer']
        file_num = item['file_num']
        
        # Get label from mapping
        if gloss_id not in labels_map:
            print(f"[WARN] ❗ No label found for gloss_id={gloss_id}; skipping {video_file}", file=sys.stderr)
            continue
        
        label, cat_id = labels_map[gloss_id]
        
        # Generate new filename
        new_name = f"clip_{counter:0{digits}d}_{label}_{signer}{video_file.suffix}"
        
        if keep_structure:
            # Keep in same directory
            dest = video_file.parent / new_name
        else:
            # Flatten to output directory
            # This would need an output_dir parameter
            dest = video_file.parent / new_name  # For now, keep in place
        
        # Only add if the name needs to change
        if video_file.name != new_name:
            operations.append((video_file, dest))
        
        counter += 1
    
    return operations


def generate_summary(items: List[Dict], labels_map: Dict[int, Tuple[str, int]] = None):
    """Generate summary statistics."""
    print("📊 Generating summary...")
    print(f"Total video files found: {len(items)}")
    
    # Files per category
    cat_counts = {}
    for item in items:
        cat_id = item['cat_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    print("\nFiles per category:")
    for cat_id in sorted(cat_counts.keys()):
        print(f"  C{cat_id}: {cat_counts[cat_id]} files")
    
    # Files per signer
    signer_counts = {}
    for item in items:
        signer = item['signer']
        signer_counts[signer] = signer_counts.get(signer, 0) + 1
    
    print("\nFiles per signer:")
    for signer in sorted(signer_counts.keys()):
        print(f"  {signer}: {signer_counts[signer]} files")
    
    # Check for gaps in numbering per directory
    print("\nGaps detected:")
    by_directory = {}
    for item in items:
        directory = item['directory']
        if directory not in by_directory:
            by_directory[directory] = []
        by_directory[directory].append(item)
    
    total_gaps = 0
    for directory in sorted(by_directory.keys()):
        files = by_directory[directory]
        numbered_files = [f['file_num'] for f in files if f['file_num'] is not None]
        if numbered_files:
            numbered_files.sort()
            min_num = min(numbered_files)
            max_num = max(numbered_files)
            expected_count = max_num - min_num + 1
            actual_count = len(numbered_files)
            gaps = expected_count - actual_count
            if gaps > 0:
                rel_dir = Path(directory).relative_to(Path(directory).parents[3])
                print(f"  {rel_dir}: {gaps} gaps (range {min_num}-{max_num}, found {actual_count}/{expected_count})")
                total_gaps += gaps
    
    if total_gaps == 0:
        print("  ✅ No gaps detected")
    else:
        print(f"\n  Total gaps across all directories: {total_gaps}")
    
    if labels_map:
        # Files per gloss
        gloss_counts = {}
        for item in items:
            gloss_id = item['gloss_id']
            if gloss_id in labels_map:
                gloss_counts[gloss_id] = gloss_counts.get(gloss_id, 0) + 1
        
        print("\nFiles per gloss (top 10):")
        sorted_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for gloss_id, count in sorted_glosses:
            label, _ = labels_map.get(gloss_id, ("unknown", -1))
            print(f"  G{gloss_id} ({label}): {count} files")
    
    print("✅ Summary complete.")


def main():
    ap = argparse.ArgumentParser(description="Rename and renumber video files in hierarchical C/G/S structure.")
    ap.add_argument("--input", type=Path, required=True, help="Input directory (clips/) containing C/G/S structure")
    ap.add_argument("--root", type=Path, default=Path("."), help="Project root for finding labels_reference.csv")
    ap.add_argument("--labels", type=Path, default=None, help="Path to labels_reference.csv")
    ap.add_argument("--start-index", type=int, default=0, help="Starting index for numbering (default: 0)")
    ap.add_argument("--digits", type=int, default=4, help="Zero-pad width for numbers (default: 4)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would happen")
    ap.add_argument("--renumber", action="store_true", help="Renumber files to fix gaps (0.MOV, 1.MOV, 2.MOV format)")
    ap.add_argument("--rename-to-clip-format", action="store_true", help="Rename to clip_XXXX_label_SY format")
    ap.add_argument("--keep-structure", action="store_true", default=True, help="Keep files in C/G/S structure (default: True)")
    ap.add_argument("--summary", action="store_true", help="Generate summary statistics only")
    args = ap.parse_args()
    
    input_dir = args.input.resolve()
    root = args.root.resolve()
    
    print(f"📂 Input directory: {input_dir}")
    
    # Collect all clips from hierarchical structure
    try:
        items = collect_hierarchical_clips(input_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] ❌ {e}", file=sys.stderr)
        return
    
    if not items:
        print("[ERROR] ❌ No video files found", file=sys.stderr)
        return
    
    print(f"📦 Found {len(items)} video files")
    
    # Load labels if needed
    labels_map = None
    if args.rename_to_clip_format or args.summary:
        explicit_labels_path = args.labels.resolve() if args.labels is not None else None
        labels_csv = find_labels_file(root, explicit_labels_path)
        labels_map = read_labels(labels_csv)
        print(f"✅ Loaded {len(labels_map)} labels from reference")
    
    # Summary mode
    if args.summary:
        generate_summary(items, labels_map)
        return
    
    # Renumber mode
    if args.renumber:
        operations = renumber_hierarchical_clips(items, args.start_index, args.digits)
        
        if not operations:
            print("✅ All files already numbered correctly")
            return
        
        print(f"\n📝 Found {len(operations)} files to renumber:")
        if args.dry_run:
            for src, dest in operations[:20]:
                rel_src = src.relative_to(input_dir)
                rel_dest = dest.relative_to(input_dir)
                print(f"  {rel_src} -> {rel_dest}")
            if len(operations) > 20:
                print(f"  ... and {len(operations) - 20} more")
            print("\n💡 Run without --dry-run to apply changes")
        else:
            # Use temporary names to avoid conflicts
            temp_renames = []
            for i, (src, dest) in enumerate(operations):
                temp_name = src.parent / f"__temp_{i}{src.suffix}"
                src.rename(temp_name)
                temp_renames.append((temp_name, dest))
            
            for temp, dest in temp_renames:
                temp.rename(dest)
            
            print(f"✅ Renumbered {len(operations)} files")
        return
    
    # Rename to clip format mode
    if args.rename_to_clip_format:
        if not labels_map:
            print("[ERROR] ❌ --rename-to-clip-format requires labels_reference.csv", file=sys.stderr)
            return
        
        operations = rename_to_clip_format(items, labels_map, args.start_index, args.digits, args.keep_structure)
        
        if not operations:
            print("✅ All files already in correct format")
            return
        
        print(f"\n📝 Found {len(operations)} files to rename:")
        if args.dry_run:
            for src, dest in operations[:20]:
                rel_src = src.relative_to(input_dir)
                rel_dest = dest.relative_to(input_dir)
                print(f"  {rel_src} -> {rel_dest}")
            if len(operations) > 20:
                print(f"  ... and {len(operations) - 20} more")
            print("\n💡 Run without --dry-run to apply changes")
        else:
            # Use temporary names to avoid conflicts
            temp_renames = []
            for i, (src, dest) in enumerate(operations):
                temp_name = src.parent / f"__temp_{i}{src.suffix}"
                src.rename(temp_name)
                temp_renames.append((temp_name, dest))
            
            for temp, dest in temp_renames:
                temp.rename(dest)
            
            print(f"✅ Renamed {len(operations)} files")
        return
    
    # No action specified
    print("[ERROR] ❌ Please specify one of: --renumber, --rename-to-clip-format, or --summary", file=sys.stderr)


if __name__ == "__main__":
    main()

