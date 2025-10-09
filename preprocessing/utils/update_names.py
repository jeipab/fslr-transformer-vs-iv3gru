#!/usr/bin/env python3
"""
Replace and renumber video files from output directory into videos directory.

This script:
1. Identifies labels in source directory that need to replace files in target directory
2. Removes old files with those labels from target directory
3. Integrates new files from source directory
4. Renumbers ALL files sequentially to maintain proper order

Usage:
    # For MOV files (videos):
    python update_names.py --dry-run      # Preview changes
    python update_names.py --backup       # Create backup before executing
    python update_names.py                # Execute replacement
    
    # For NPZ files (processed):
    python update_names.py --target data/processed/fsl-105_10-08 --source data/processed/orig-repeat --ext npz --dry-run
    python update_names.py --target data/processed/fsl-105_10-08 --source data/processed/orig-repeat --ext npz --backup
    python update_names.py --target data/processed/fsl-105_10-08 --source data/processed/orig-repeat --ext npz
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


def parse_filename(filename: str, ext: str = "MOV") -> Tuple[int, str, str]:
    """
    Parse filename to extract clip number, label, and extension.
    
    Args:
        filename: e.g., "clip_0123_good morning.MOV" or "clip_0123_good morning.npz"
        ext: Expected extension (default: "MOV")
    
    Returns:
        Tuple of (clip_number, label, extension)
        e.g., (123, "good morning", ".MOV")
    """
    match = re.match(r'clip_(\d+)_(.+)\.([^.]+)$', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), match.group(2), f".{match.group(3)}"


def collect_files_by_label(directory: Path, ext: str = "MOV") -> Dict[str, List[Path]]:
    """
    Collect all files grouped by their label.
    
    Args:
        directory: Directory to scan for files
        ext: File extension to search for (e.g., "MOV", "npz")
    
    Returns:
        Dictionary mapping label -> list of file paths
    """
    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return {}
    
    label_to_files = defaultdict(list)
    
    for file_path in sorted(directory.glob(f"*.{ext}")):
        try:
            clip_num, label, file_ext = parse_filename(file_path.name, ext)
            label_to_files[label].append(file_path)
        except ValueError as e:
            print(f"[WARN] Skipping file: {e}")
            continue
    
    # Sort files within each label by clip number
    for label in label_to_files:
        label_to_files[label].sort(key=lambda p: parse_filename(p.name, ext)[0])
    
    return label_to_files


def get_label_order(target_dir: Path, ext: str = "MOV") -> List[str]:
    """
    Get the order of labels as they appear in the target directory.
    This maintains the original sequence.
    
    Args:
        target_dir: Path to target directory
        ext: File extension to search for
    
    Returns:
        List of labels in order of first appearance
    """
    seen_labels = []
    
    for file_path in sorted(target_dir.glob(f"*.{ext}"), 
                           key=lambda p: parse_filename(p.name, ext)[0]):
        try:
            _, label, _ = parse_filename(file_path.name, ext)
            if label not in seen_labels:
                seen_labels.append(label)
        except ValueError:
            continue
    
    return seen_labels


def create_renaming_plan(target_dir: Path, source_dir: Path, ext: str = "MOV") -> List[Tuple[Path, str, str]]:
    """
    Create a plan for renaming and replacing files.
    
    Args:
        target_dir: Target directory (files to keep/replace)
        source_dir: Source directory (new files to integrate)
        ext: File extension
    
    Returns:
        List of tuples (source_path, new_filename, source_type)
        where source_type is either 'target' or 'source'
    """
    target_files = collect_files_by_label(target_dir, ext)
    source_files = collect_files_by_label(source_dir, ext)
    
    # Labels that will be replaced
    replacement_labels = set(source_files.keys())
    
    # Get label order from target directory
    label_order = get_label_order(target_dir, ext)
    
    # Build the new file list
    plan = []
    counter = 1
    
    for label in label_order:
        # Determine which source to use
        if label in replacement_labels:
            # Use files from source directory
            files_to_use = source_files[label]
            source_type = 'source'
        else:
            # Use files from target directory
            files_to_use = target_files.get(label, [])
            source_type = 'target'
        
        # Add all files for this label with sequential numbering
        for file_path in files_to_use:
            new_filename = f"clip_{counter:04d}_{label}.{ext}"
            plan.append((file_path, new_filename, source_type))
            counter += 1
    
    return plan


def print_summary(plan: List[Tuple[Path, str, str]], target_dir: Path, source_dir: Path, ext: str = "MOV"):
    """Print a summary of changes."""
    target_files = collect_files_by_label(target_dir, ext)
    source_files = collect_files_by_label(source_dir, ext)
    
    replacement_labels = set(source_files.keys())
    
    print("\n" + "="*80)
    print("REPLACEMENT SUMMARY")
    print("="*80)
    
    print(f"\n📂 Target Directory: {target_dir}")
    print(f"📂 Source Directory: {source_dir}")
    print(f"📄 File Extension: .{ext}")
    
    print(f"\n🔄 Labels to be replaced: {len(replacement_labels)}")
    
    total_old = 0
    total_new = 0
    
    for label in sorted(replacement_labels):
        old_count = len(target_files.get(label, []))
        new_count = len(source_files.get(label, []))
        diff = new_count - old_count
        diff_str = f"({diff:+d})" if diff != 0 else ""
        
        print(f"  • {label:20s}: {old_count:2d} → {new_count:2d} files {diff_str}")
        total_old += old_count
        total_new += new_count
    
    print(f"\n📊 Total files being replaced: {total_old} → {total_new} ({total_new - total_old:+d})")
    
    # Count files kept from target
    kept_labels = set(target_files.keys()) - replacement_labels
    kept_count = sum(len(target_files[label]) for label in kept_labels)
    
    print(f"📌 Files kept from target: {kept_count}")
    print(f"🎯 Total files after operation: {len(plan)}")
    
    print("\n" + "="*80)


def create_backup(target_dir: Path) -> Path:
    """Create a backup of the target directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = target_dir.name
    backup_dir = target_dir.parent / f"{dir_name}_backup_{timestamp}"
    
    print(f"\n💾 Creating backup: {backup_dir}")
    shutil.copytree(target_dir, backup_dir)
    print(f"✅ Backup created successfully")
    
    return backup_dir


def execute_plan(plan: List[Tuple[Path, str, str]], target_dir: Path, ext: str = "MOV", dry_run: bool = False):
    """
    Execute the renaming and replacement plan.
    
    Args:
        plan: List of (source_path, new_filename, source_type) tuples
        target_dir: Target directory
        ext: File extension
        dry_run: If True, only print what would happen
    """
    if dry_run:
        print("\n" + "="*80)
        print("DRY RUN - No changes will be made")
        print("="*80)
    
    # Create a temporary directory for staging
    temp_dir = target_dir.parent / "temp_staging"
    
    if not dry_run:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Processing {len(plan)} files...")
    
    # Stage 1: Copy all files to temp with new names
    for i, (source_path, new_filename, source_type) in enumerate(plan, 1):
        if i % 100 == 0 or i == 1:
            print(f"  Progress: {i}/{len(plan)} files...")
        
        if dry_run:
            if i <= 20 or i > len(plan) - 5:  # Show first 20 and last 5
                print(f"  [{source_type:6s}] {source_path.name:40s} → {new_filename}")
            elif i == 21:
                print(f"  ... ({len(plan) - 25} more files) ...")
        else:
            dest_path = temp_dir / new_filename
            shutil.copy2(source_path, dest_path)
    
    if dry_run:
        print("\n💡 Run without --dry-run to apply changes")
        return
    
    # Stage 2: Clear target directory and move files from temp
    print(f"\n🗑️  Clearing target directory...")
    for file_path in target_dir.glob(f"*.{ext}"):
        file_path.unlink()
    
    print(f"📦 Moving files to target directory...")
    for file_path in temp_dir.glob(f"*.{ext}"):
        dest_path = target_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
    
    # Clean up temp directory
    temp_dir.rmdir()
    
    print(f"\n✅ Operation completed successfully!")
    print(f"📁 {len(plan)} files now in {target_dir}")


def verify_result(target_dir: Path, expected_count: int, ext: str = "MOV"):
    """Verify the operation was successful."""
    actual_files = list(target_dir.glob(f"*.{ext}"))
    actual_count = len(actual_files)
    
    print(f"\n🔍 Verification:")
    print(f"  Expected files: {expected_count}")
    print(f"  Actual files: {actual_count}")
    
    if actual_count == expected_count:
        print(f"  ✅ File count matches!")
    else:
        print(f"  ⚠️  File count mismatch!")
        return False
    
    # Check sequential numbering
    file_numbers = []
    for file_path in sorted(actual_files, key=lambda p: p.name):
        try:
            clip_num, _, _ = parse_filename(file_path.name, ext)
            file_numbers.append(clip_num)
        except ValueError:
            print(f"  ⚠️  Invalid filename: {file_path.name}")
            return False
    
    expected_sequence = list(range(1, expected_count + 1))
    if file_numbers == expected_sequence:
        print(f"  ✅ Sequential numbering verified (1 to {expected_count})")
        return True
    else:
        print(f"  ⚠️  Numbering issues detected!")
        missing = set(expected_sequence) - set(file_numbers)
        if missing:
            print(f"     Missing numbers: {sorted(missing)[:10]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Replace and renumber files from source to target directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For MOV files (videos):
  python update_names.py --dry-run
  python update_names.py --backup
  python update_names.py
  
  # For NPZ files (processed):
  python update_names.py --target data/processed/fsl-105_10-08 --source data/processed/orig-repeat --ext npz --dry-run
  python update_names.py --target data/processed/fsl-105_10-08 --source data/processed/orig-repeat --ext npz --backup
  python update_names.py --target data/processed/fsl-105_10-08 --source data/processed/orig-repeat --ext npz
        """
    )
    parser.add_argument("--dry-run", action="store_true", 
                       help="Preview changes without executing")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup before executing")
    parser.add_argument("--target", type=Path, default=None,
                       help="Path to target directory (default: data/raw/videos)")
    parser.add_argument("--source", type=Path, default=None,
                       help="Path to source directory (default: data/raw/output)")
    parser.add_argument("--ext", type=str, default="MOV",
                       help="File extension to process (default: MOV)")
    
    args = parser.parse_args()
    
    # Determine paths
    root = Path(__file__).parent
    target_dir = args.target if args.target else root / "data" / "raw" / "videos"
    source_dir = args.source if args.source else root / "data" / "raw" / "output"
    
    target_dir = target_dir.resolve()
    source_dir = source_dir.resolve()
    
    # Validate directories exist
    if not target_dir.exists():
        print(f"[ERROR] Target directory not found: {target_dir}")
        sys.exit(1)
    
    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Create the plan
    print("📋 Creating replacement plan...")
    plan = create_renaming_plan(target_dir, source_dir, args.ext)
    
    if not plan:
        print("[ERROR] No files found to process!")
        sys.exit(1)
    
    # Print summary
    print_summary(plan, target_dir, source_dir, args.ext)
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_dir = create_backup(target_dir)
        print(f"\n📝 Note: Backup created at {backup_dir}")
    
    # Execute the plan
    execute_plan(plan, target_dir, args.ext, dry_run=args.dry_run)
    
    # Verify if not dry run
    if not args.dry_run:
        verify_result(target_dir, len(plan), args.ext)
        print("\n🎉 All done!")


if __name__ == "__main__":
    main()
