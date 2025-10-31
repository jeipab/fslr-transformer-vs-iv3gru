#!/usr/bin/env python3
"""
Rename and flatten sign language video clips based on a hierarchical structure
and a labels_reference.csv file.

Input structure (hierarchical):
clips/
   C0/
      G0/
         S0/
            0.MOV
            1.MOV
         S1/
         ...
      G1/
      ...
   C1/
   ...

labels_reference.csv columns: gloss_id,label,cat_id,category
- gloss_id: integer folder name (0..104)
- label: e.g., "GOOD MORNING" (used in filename, lowercased)
- cat_id: category ID (0..9)
- category: e.g., "GREETING"

Output:
videos/
   clip_0001_good_morning_S0.MOV
   clip_0002_good_morning_S1.MOV
   ...
- A CSV file with metadata: file,gloss,cat,occluded,signer,duration

Usage:
    python rename_clips.py --root .
    python rename_clips.py --root . --keep-structure  # Rename in-place
Optional:
    python rename_clips.py --root . --dry-run
    python rename_clips.py --root . --validate
    python rename_clips.py --root . --summary
Renumber hierarchical structure (C/G/S with numbered files like 0.MOV, 1.MOV):
    python rename_clips.py --renumber-hierarchical --clips data/raw/clips --dry-run
    python rename_clips.py --renumber-hierarchical --clips data/raw/clips --start-index 0 --digits 4
Renumber (fix gaps in already-renamed clip_XXXX_label_SX files):
    python rename_clips.py --renumber --clips data/raw/Greetings-Only --dry-run
    python rename_clips.py --renumber --clips data/raw/Greetings-Only --recursive --dry-run
    python rename_clips.py --renumber --clips data/raw/Greetings-Only --recursive
"""

import argparse
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import cv2

def slugify_label(s: str) -> str:
    s = s.strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'uncategorized'

def find_labels_file(root: Path, explicit_path: Path = None) -> Path:
    """Finds labels_reference.csv in common locations."""
    # If explicit path is provided, use it
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    # Check common locations
    possible_locations = [
        root / "labels_reference.csv",
        root / "data" / "labels_reference.csv",
        root / "data" / "raw" / "labels_reference.csv",
    ]
    
    for path in possible_locations:
        if path.exists():
            return path
    
    # If not found, return the first default location for error message
    return explicit_path if explicit_path else (root / "data" / "labels_reference.csv")

def find_clips_directory(root: Path, explicit_path: Path = None) -> Path:
    """Finds clips directory in common locations."""
    # If explicit path is provided, use it
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    # Check common locations
    possible_locations = [
        root / "FSL-105",
        root / "clips",
        root / "data" / "raw" / "clips",
        root / "data" / "clips",
    ]
    
    for path in possible_locations:
        if path.exists() and path.is_dir():
            return path
    
    # If not found, return the first default location for error message
    return explicit_path if explicit_path else (root / "data" / "raw" / "clips")

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

def get_video_duration(video_path: Path) -> float:
    """Extracts the duration of a video in seconds."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    except Exception as e:
        print(f"[WARN] ❗ Could not get duration for {video_path}: {e}", file=sys.stderr)
        return 0.0

def collect_clips_hierarchical(clips_dir: Path, video_extensions=None) -> List[Dict]:
    """Collects clips from a 3-level hierarchical structure (C/G/S)."""
    if video_extensions is None:
        video_extensions = {'.MOV', '.mov', '.mp4', '.MP4', '.avi', '.AVI', '.npz'}
    
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
                    items.append({
                        "path": video_file,
                        "cat_id": cat_id,
                        "gloss_id": gloss_id,
                        "signer": signer
                    })
    return items

def collect_all_video_files(target_dir: Path, recursive: bool = False) -> List[Path]:
    """Simply collect all video files and npz files in directory"""
    extensions = ['*.MOV', '*.mov', '*.mp4', '*.MP4', '*.avi', '*.AVI', '*.npz']
    
    files = set()
    for ext in extensions:
        if recursive:
            files.update(target_dir.rglob(ext))
        else:
            files.update(target_dir.glob(ext))
    
    return sorted(files)

def renumber_hierarchical_structure(clips_dir: Path, video_extensions=None, digits: int = 4, start_index: int = 0) -> List[Tuple[Path, Path]]:
    """Renumber files in hierarchical C/G/S structure.
    
    Files are expected to be named like: 0.MOV, 1.MOV, etc. in S0/, S1/, etc. directories.
    They will be renumbered sequentially within each signer directory to fix gaps.
    
    Returns list of (source_path, destination_path) tuples for rename operations.
    """
    if video_extensions is None:
        video_extensions = {'.MOV', '.mov', '.mp4', '.MP4', '.avi', '.AVI'}
    
    operations = []
    
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips directory not found at {clips_dir}")
    
    cat_dirs = sorted([p for p in clips_dir.iterdir() if p.is_dir() and re.match(r'^C\d+$', p.name)])
    for cat_dir in cat_dirs:
        gloss_dirs = sorted([p for p in cat_dir.iterdir() if p.is_dir() and re.match(r'^G\d+$', p.name)])
        for gloss_dir in gloss_dirs:
            signer_dirs = sorted([p for p in gloss_dir.iterdir() if p.is_dir() and re.match(r'^S\d+$', p.name)])
            for signer_dir in signer_dirs:
                # Find all video files in this signer directory
                video_files = []
                for ext in video_extensions:
                    video_files.extend(signer_dir.glob(f"*{ext}"))
                
                # Filter to files that are just numbers (0.MOV, 1.MOV, etc.)
                numbered_files = []
                for video_file in video_files:
                    # Check if filename is just a number followed by extension
                    base_name = video_file.stem  # filename without extension
                    if base_name.isdigit():
                        numbered_files.append((int(base_name), video_file))
                
                # Sort by the number in the filename
                numbered_files.sort(key=lambda x: x[0])
                
                # Renumber sequentially starting from start_index
                counter = start_index
                for original_num, video_file in numbered_files:
                    new_name = f"{counter:0{digits}d}{video_file.suffix}"
                    dest = signer_dir / new_name
                    
                    # Only add to operations if the number needs to change
                    if video_file.name != new_name:
                        operations.append((video_file, dest))
                    
                    counter += 1
    
    return operations

def validate_structure(clips_dir: Path, labels_map: Dict[int, Tuple[str, int]]):
    """Validates the hierarchical folder structure."""
    print("🔍 Validating folder structure...")
    expected_cats = set(range(10))
    found_cats = {int(p.name[1:]) for p in clips_dir.iterdir() if p.is_dir() and re.match(r'^C\d+$', p.name)}
    missing_cats = expected_cats - found_cats
    if missing_cats:
        print(f"❌ Missing category folders: {[f'C{c}' for c in missing_cats]}")

    glosses_per_cat = {i: [] for i in range(10)}
    for gloss_id, (_, cat_id) in labels_map.items():
        if cat_id in glosses_per_cat:
            glosses_per_cat[cat_id].append(gloss_id)

    for cat_id in sorted(glosses_per_cat.keys()):
        cat_dir = clips_dir / f"C{cat_id}"
        if not cat_dir.is_dir():
            continue
        
        expected_glosses = set(glosses_per_cat[cat_id])
        found_glosses = {int(p.name[1:]) for p in cat_dir.iterdir() if p.is_dir() and re.match(r'^G\d+$', p.name)}
        missing_glosses = expected_glosses - found_glosses
        if missing_glosses:
            print(f"❌ In C{cat_id}, missing gloss folders: {[f'G{g}' for g in missing_glosses]}")

        for gloss_id in sorted(found_glosses):
            gloss_dir = cat_dir / f"G{gloss_id}"
            expected_signers = {f"S{i}" for i in range(8)}
            found_signers = {p.name for p in gloss_dir.iterdir() if p.is_dir() and re.match(r'^S\d+$', p.name)}
            missing_signers = expected_signers - found_signers
            if missing_signers:
                print(f"❌ In C{cat_id}/G{gloss_id}, missing signer folders: {sorted(list(missing_signers))}")
    print("✅ Validation complete.")

def generate_summary(clips: List[Dict], labels_map: Dict[int, Tuple[str, int]]):
    """Generates and prints a summary of the dataset."""
    print("📊 Generating summary...")
    total_videos = len(clips)
    print(f"Total videos found: {total_videos}")

    # Videos per category
    print("\nVideos per category:")
    cat_counts = {i: 0 for i in range(10)}
    for clip in clips:
        cat_counts[clip['cat_id']] += 1
    for cat_id, count in sorted(cat_counts.items()):
        print(f"  C{cat_id}: {count} videos")

    # Videos per gloss
    print("\nVideos per gloss:")
    gloss_counts = {g: 0 for g in labels_map.keys()}
    for clip in clips:
        if clip['gloss_id'] in gloss_counts:
            gloss_counts[clip['gloss_id']] += 1
    for gloss_id, count in sorted(gloss_counts.items()):
        label, _ = labels_map.get(gloss_id, ("unknown", -1))
        print(f"  G{gloss_id} ({label}): {count} videos")
        
    # Videos per signer
    print("\nVideos per signer:")
    signer_counts = {f"S{i}": 0 for i in range(8)}
    for clip in clips:
        if clip['signer'] in signer_counts:
            signer_counts[clip['signer']] += 1
    for signer, count in sorted(signer_counts.items()):
        print(f"  {signer}: {count} videos")

    # Missing combinations
    print("\nMissing signer/gloss combinations:")
    all_gloss_ids = set(labels_map.keys())
    all_signers = {f"S{i}" for i in range(8)}
    found_combinations = set()
    for clip in clips:
        found_combinations.add((clip['gloss_id'], clip['signer']))

    missing_count = 0
    for gloss_id in sorted(all_gloss_ids):
        for signer in sorted(all_signers):
            if (gloss_id, signer) not in found_combinations:
                label, _ = labels_map.get(gloss_id, ("unknown", -1))
                print(f"  Missing: G{gloss_id} ({label}) - {signer}")
                missing_count += 1
    
    if missing_count == 0:
        print("  None missing.")
    print("✅ Summary generation complete.")

def main():
    ap = argparse.ArgumentParser(description="Rename and flatten video clips based on a hierarchical structure.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Project root containing labels_reference.csv and clips/")
    ap.add_argument("--clips", type=Path, default=None, help="Path to clips/ (default: <root>/data/raw/clips)")
    ap.add_argument("--labels", type=Path, default=None, help="Path to labels_reference.csv (default: <root>/data/labels_reference.csv)")
    ap.add_argument("--out", type=Path, default=None, help="Output folder (default: <root>/data/raw)")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index for clip numbering (default: 1)")
    ap.add_argument("--digits", type=int, default=4, help="Zero-pad width for numbers (default: 4)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would happen")
    ap.add_argument("--validate", action="store_true", help="Validate folder structure completeness before renaming")
    ap.add_argument("--summary", action="store_true", help="Generate statistics without renaming")
    ap.add_argument("--keep-structure", action="store_true", help="Rename files in-place, keeping the original folder structure (C/G/S)")
    ap.add_argument("--input-structure", type=str, default="hierarchical", choices=["hierarchical"], help="Input folder structure type")
    ap.add_argument("--renumber", action="store_true", help="Renumber existing clip_XXXX_label_SX files to fix gaps in numbering")
    ap.add_argument("--renumber-hierarchical", action="store_true", help="Renumber files in C/G/S structure (0.MOV, 1.MOV format)")
    ap.add_argument("--recursive", action="store_true", help="When renumbering, search subdirectories recursively")
    args = ap.parse_args()

    root = args.root.resolve()

    # Use find_clips_directory to search multiple locations
    explicit_clips_path = args.clips.resolve() if args.clips is not None else None
    clips_dir = find_clips_directory(root, explicit_clips_path)
    
    # Use find_labels_file to search multiple locations
    explicit_labels_path = args.labels.resolve() if args.labels is not None else None
    labels_csv = find_labels_file(root, explicit_labels_path)
    
    # Default output directory to Renamed if not specified
    out_dir = args.out.resolve() if args.out is not None else (root / "Renamed").resolve()
    
    print(f"📂 Root directory: {root}")
    print(f"📂 clips_dir: {clips_dir}")
    print(f"📂 labels_csv: {labels_csv}")
    print(f"📂 out_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle hierarchical renumbering mode (for C/G/S structure with numbered files)
    if args.renumber_hierarchical:
        print("Renumbering files in hierarchical C/G/S structure...")
        
        try:
            operations = renumber_hierarchical_structure(
                clips_dir, 
                video_extensions={'.MOV', '.mov', '.mp4', '.MP4', '.avi', '.AVI'},
                digits=args.digits,
                start_index=args.start_index
            )
            
            if not operations:
                print("✅ All files already numbered correctly")
                return
            
            print(f"Found {len(operations)} files to renumber")
            
            if args.dry_run:
                print("\n📝 Files that would be renamed:")
                for src, dest in operations[:20]:
                    rel_src = src.relative_to(clips_dir)
                    rel_dest = dest.relative_to(clips_dir)
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
        except Exception as e:
            print(f"[ERROR] ❗ Error during renumbering: {e}", file=sys.stderr)
            return
    
    # Handle renumbering mode (for already renamed clip_XXXX_label_SX files)
    if args.renumber:
        print("Renumbering all video files...")
        target = args.clips if args.clips else out_dir
        
        files = collect_all_video_files(target, recursive=args.recursive)
        
        if not files:
            print("No video files found")
            return
        
        print(f"Found {len(files)} video files")
        
        # Build rename operations
        operations = []
        counter = args.start_index
        
        for filepath in files:
            # Extract current number, label and signer from filename
            match = re.match(r'^clip_(\d+)_(.+?)_(S\d+)\.[^.]+$', filepath.name)
            if match:
                current_num = int(match.group(1))
                label = match.group(2)
                signer = match.group(3)
            else:
                # Fallback: use generic naming
                current_num = -1
                label = "clip"
                signer = "S0"
            
            new_name = f"clip_{counter:0{args.digits}d}_{label}_{signer}{filepath.suffix}"
            dest = filepath.parent / new_name
            
            # Only add to operations if number needs to change
            if current_num != counter:
                operations.append((filepath, dest))
            
            counter += 1
        
        if not operations:
            print("All files already numbered correctly")
            return
        
        print(f"Will renumber {len(operations)} files")
        
        if args.dry_run:
            for src, dest in operations[:10]:
                print(f"  {src.name} -> {dest.name}")
            if len(operations) > 10:
                print(f"  ... and {len(operations) - 10} more")
            print("Run without --dry-run to apply changes")
            return
        
        # Execute with temp names to avoid conflicts
        print("Renaming...")
        temp_renames = []
        
        for i, (src, dest) in enumerate(operations):
            temp_name = src.parent / f"__temp_{i}{src.suffix}"
            src.rename(temp_name)
            temp_renames.append((temp_name, dest))
        
        for temp, dest in temp_renames:
            temp.rename(dest)
        
        print(f"Done! Renumbered {len(operations)} files")
        return

    id_to_label_cat = read_labels(labels_csv)

    if args.validate:
        validate_structure(clips_dir, id_to_label_cat)
        return

    items = collect_clips_hierarchical(clips_dir)
    if not items:
        print("[INFO] ❌ No video files found under", clips_dir)
        return
        
    if args.summary:
        generate_summary(items, id_to_label_cat)
        return

    counter = args.start_index
    width = max(args.digits, len(str(args.start_index + len(items) - 1)))
    
    csv_data = []
    operations = []  # Changed from 'moves' to 'operations' for clarity

    for item in items:
        src = item['path']
        gloss_id = item['gloss_id']
        signer = item['signer']
        
        if gloss_id not in id_to_label_cat:
            print(f"[WARN] ❗ No label found for gloss_id={gloss_id}; skipping {src}", file=sys.stderr)
            continue
        
        label, cat_id = id_to_label_cat[gloss_id]
        
        # Preserve original file extension
        file_ext = src.suffix
        
        if args.keep_structure:
            # Rename in-place: keep file in same directory
            new_name = f"clip_{counter:0{width}d}_{label}_{signer}{file_ext}"
            dest = src.parent / new_name
            
            # Skip if already properly named
            if re.match(r'^clip_\d{4}_.*', src.name):
                match = re.match(r'^clip_(\d+)_(.+?)_S\d+\.', src.name)
                if match and match.group(2) == label:
                    print(f"[SKIP] Already properly named: {src}")
                    counter += 1
                    continue
        else:
            # Flatten: move to output directory
            new_name = f"clip_{counter:0{width}d}_{label}_{signer}{file_ext}"
            dest = out_dir / new_name
        
        # Check if destination already exists
        while dest.exists() and dest != src:
            counter += 1
            new_name = f"clip_{counter:0{width}d}_{label}_{signer}{file_ext}"
            if args.keep_structure:
                dest = src.parent / new_name
            else:
                dest = out_dir / new_name

        duration = get_video_duration(src)
        csv_data.append({
            "file": new_name if args.keep_structure else new_name,
            "gloss": label,
            "cat": cat_id,
            "occluded": 0,
            "signer": signer,
            "duration": f"{duration:.2f}"
        })
        operations.append((src, dest, args.keep_structure))
        counter += 1

    # Perform the rename/move operations
    for src, dest, is_rename in operations:
        if args.dry_run:
            action = "RENAME" if is_rename else "MOVE"
            print(f"[DRY] {action}: {src} -> {dest}")
        else:
            try:
                if is_rename:
                    # Rename in-place
                    src.rename(dest)
                else:
                    # Copy to new location
                    dest.write_bytes(src.read_bytes())
                    # src.unlink() # Uncomment to delete original files
            except Exception as e:
                print(f"[ERROR] ❗ Could not process {src} to {dest}: {e}", file=sys.stderr)

    # Write CSV output
    csv_output_path = out_dir / "metadata.csv"
    if not args.dry_run:
        with csv_output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["file", "gloss", "cat", "occluded", "signer", "duration"])
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"📄 Metadata saved to {csv_output_path}")

    action_verb = "Renamed" if args.keep_structure else "Moved"
    location = "in original structure" if args.keep_structure else str(out_dir)
    print(f"[DONE] ✅🎉 {'Planned' if args.dry_run else action_verb} {len(operations)} files {location}")
    if args.dry_run:
        print("💡 Run again without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
