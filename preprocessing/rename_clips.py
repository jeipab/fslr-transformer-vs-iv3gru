#!/usr/bin/env python3
"""
Rename and flatten sign language video clips based on labels.csv.

Input structure:
clips/
   0/
      0.MOV
      1.MOV
      ...
   104/
      ...

labels.csv columns: id,label,category
- id: integer folder name (0..104)
- label: e.g., "Good Morning" (not used for filename)
- category: e.g., "Greetings" (used in filename, slugified & lowercased)

Output:
videos/
   clip_0001_greetings.MOV
   clip_0002_greetings.MOV
   ...
   clip_2235_drink.MOV

Usage:
    python rename_clips.py --root .
Optional:
    python rename_clips.py --root . --dry-run

Notes:
- Counts clips sequentially in increasing id and increasing clip filename order.
- Only processes files with extension .mov/.MOV in two-level structure clips/<id>/<n>.MOV
- Slugifies category: lowercase, spaces and dashes -> underscore, other non [a-z0-9_] removed.
"""

import argparse
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

def slugify_category(s: str) -> str:
    s = s.strip().lower()
    s = s.replace('-', ' ').replace('/', ' ')
    s = re.sub(r'\s+', '_', s)                 # spaces -> underscore
    s = re.sub(r'[^a-z0-9_]', '', s)           # keep safe chars
    s = re.sub(r'_+', '_', s).strip('_')       # squeeze underscores
    return s or 'uncategorized'

def read_labels(labels_csv: Path):
    print(f"🔍 Looking for labels.csv at: {labels_csv}")  # Debugging path
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")
    mapping: Dict[int, Tuple[str, str]] = {}  # Storing both label and category
    with labels_csv.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'id', 'label', 'category'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"labels.csv must contain columns: {required}. Found: {reader.fieldnames}")
        for row in reader:
            try:
                i = int(row['id'])
            except Exception as e:
                print(f"[WARN] ❗ Skipping row with non-integer id={row.get('id')!r}: {e}", file=sys.stderr)
                continue
            label = row.get('label', '').strip().lower()  # Use label directly
            cat = slugify_category(row.get('category', ''))
            mapping[i] = (label, cat)  # Store both label and category
    return mapping


def collect_clips(clips_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips directory not found at {clips_dir}")
    for id_dir in sorted((p for p in clips_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        try:
            id_num = int(id_dir.name)
        except ValueError:
            print(f"[WARN] ❗ Skipping non-numeric folder {id_dir}", file=sys.stderr)
            continue
        # gather .mov files inside
        for mov in sorted(id_dir.glob("*.MOV"), key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem)):
            items.append((id_num, mov))
        # case-insensitive: also pick *.mov if any (deduplicate)
        for mov in sorted(id_dir.glob("*.mov"), key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem)):
            if (id_num, mov) not in items:
                items.append((id_num, mov))
    # sort globally by id then by filename stem numeric/text
    def sort_key(t):
        id_num, path = t
        stem = path.stem
        if stem.isdigit():
            return (id_num, 0, int(stem))
        return (id_num, 1, stem)
    items.sort(key=sort_key)
    return items

def main():
    ap = argparse.ArgumentParser(description="Rename and flatten video clips based on labels.csv")
    ap.add_argument("--root", type=Path, default=Path("."), help="Project root containing labels.csv and clips/")
    ap.add_argument("--clips", type=Path, default=None, help="Path to clips/ (default: <root>/data/raw/clips)")
    ap.add_argument("--labels", type=Path, default=None, help="Path to labels.csv (default: <root>/data/raw/labels.csv)")
    ap.add_argument("--out", type=Path, default=None, help="Output folder (default: <root>/videos)")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index for clip numbering (default: 1)")
    ap.add_argument("--digits", type=int, default=4, help="Zero-pad width for numbers (default: 4)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would happen")
    args = ap.parse_args()

    root = args.root.resolve() if args.root else Path(__file__).resolve().parent
    clips_dir = (root / "data/raw/clips").resolve()
    labels_csv = (root / "data/raw/labels.csv").resolve()
    out_dir = (root / "data/raw/videos").resolve()

    print(f"📂 Root directory: {root}")
    print(f"📂 clips_dir: {clips_dir}")
    print(f"📂 labels_csv: {labels_csv}")
    print(f"📂 out_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # read labels
    id_to_cat = read_labels(labels_csv)

    # collect clips
    items = collect_clips(clips_dir)
    if not items:
        print("[INFO] ❌ No video files found under", clips_dir)
        return

    # do the renaming + moving
    counter = args.start_index
    width = max(args.digits, len(str(args.start_index + len(items) - 1)))
    moves: List[Tuple[Path, Path]] = []

    for id_num, src in items:
        cat = id_to_cat.get(id_num)
        if not cat:
            print(f"[WARN] ❗ No category found in labels.csv for id={id_num}; skipping {src}", file=sys.stderr)
            continue
        label, cat = id_to_cat.get(id_num)
        new_name = f"clip_{counter:0{width}d}_{label}.MOV" # Option: Change {label} to {cat} to rename with a category instead
        dest = out_dir / new_name
        # If destination exists, increment counter until free to avoid accidental overwrite.
        while dest.exists():
            counter += 1
            new_name = f"clip_{counter:0{width}d}_{label}.MOV" # Option: Change {label} to {cat} to rename with a category instead
            dest = out_dir / new_name
        moves.append((src, dest))
        counter += 1

    # Execute
    for src, dest in moves:
        if args.dry_run:
            print(f"[DRY] 🏃‍♀️ {src} -> {dest}")
        else:
            dest.write_bytes(src.read_bytes())
            # remove original after successful copy
            try:
                src.unlink()
            except Exception as e:
                print(f"[WARN] ❗ Copied but could not delete {src}: {e}", file=sys.stderr)

    print(f"[DONE] ✅🎉 {'Planned' if args.dry_run else 'Moved'} {len(moves)} files to {out_dir}")
    if args.dry_run:
        print("💡 Run again without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
