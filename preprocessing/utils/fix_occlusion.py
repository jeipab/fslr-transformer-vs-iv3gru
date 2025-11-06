import json
import os
from pathlib import Path
import csv


def normalize_clip_stem(path_or_name: str) -> str:
    """Return lowercase base filename without extension for matching.

    Examples:
    - "C0/clip_00001_good_morning_S0.npz" -> "clip_00001_good_morning_S0"
    - "clip_01642_dont_know_S4.mp4" -> "clip_01642_dont_know_S4"
    """
    base = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(base)
    return stem.lower()


def load_occlusion_mapping(csv_path: Path) -> dict:
    """Load mapping from clip stem to occluded flag (0/1) from labels.csv."""
    mapping: dict[str, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"file", "occluded"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Expected columns {sorted(required_cols)}, missing {sorted(missing)} in {csv_path}"
            )
        for row in reader:
            file_field = row.get("file", "").strip()
            occ_field = row.get("occluded", "").strip()
            if not file_field:
                continue
            key = normalize_clip_stem(file_field)
            if occ_field == "":
                # Skip rows without occluded value
                continue
            try:
                # Be tolerant to values like "0", "1", "0.0", etc.
                occluded_value = int(float(occ_field))
                occluded_value = 1 if occluded_value != 0 else 0
            except ValueError:
                # Invalid numeric value; skip
                continue
            mapping[key] = occluded_value
    return mapping


def update_json_file(json_path: Path, occlusion_map: dict) -> tuple[int, int]:
    """Update 'occluded' for segments in a single JSON file.

    Returns (num_segments, num_updated).
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        return 0, 0

    updated = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        original_file = segment.get("original_file")
        if not original_file:
            continue
        key = normalize_clip_stem(original_file)
        if key not in occlusion_map:
            continue
        new_value = int(1 if occlusion_map[key] else 0)
        old_value = segment.get("occluded")
        if old_value != new_value:
            segment["occluded"] = new_value
            updated += 1

    if updated > 0:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return len(segments), updated


def main() -> None:
    script_dir = Path(__file__).resolve().parent  # data/raw
    data_dir = script_dir.parent  # data
    csv_path = data_dir / "processed" / "labels.csv"
    json_dirs = [
        script_dir / "same_cat_raw-seq-400",
        script_dir / "diff_cat_raw-seq-400",
    ]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    existing_json_dirs = [d for d in json_dirs if d.exists()]
    if not existing_json_dirs:
        raise FileNotFoundError(
            f"No JSON directories found among: {', '.join(str(p) for p in json_dirs)}"
        )

    occlusion_map = load_occlusion_mapping(csv_path)

    total_files = 0
    total_segments = 0
    total_updated = 0

    for dir_path in existing_json_dirs:
        for json_file in sorted(dir_path.glob("*.json")):
            total_files += 1
            seg_count, upd_count = update_json_file(json_file, occlusion_map)
            total_segments += seg_count
            total_updated += upd_count

    print(
        (
            f"Processed {total_files} JSON files | "
            f"Segments: {total_segments} | Updated: {total_updated}"
        )
    )


if __name__ == "__main__":
    main()


