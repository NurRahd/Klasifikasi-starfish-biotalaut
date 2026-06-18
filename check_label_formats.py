"""Check YOLO label formats across the dataset.
 +

Dependencies:
    pandas
    pathlib
    tqdm

Usage:
    python check_label_formats.py
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


DATASET_ROOT = Path(".")
SPLITS = ["train", "valid", "test"]
DATASETS = ["biota_laut/Underwater Marine Species.v2i.yolov8", "starfish_only", "combined_dataset", "combined_dataset_preprocessed"]
REPORT_CSV = Path("label_format_report.csv")


def get_label_directories() -> List[Path]:
    """Scan and return all label directories across datasets."""
    label_dirs = []
    for dataset in DATASETS:
        dataset_path = DATASET_ROOT / dataset
        if not dataset_path.exists():
            continue
        for split in SPLITS:
            label_dir = dataset_path / split / "labels"
            if label_dir.exists():
                label_dirs.append(label_dir)
    return label_dirs


def parse_label_line(line: str) -> Tuple[int, int]:
    """Parse a label line and return (class_id, column_count)."""
    parts = line.strip().split()
    if not parts:
        return -1, 0
    try:
        class_id = int(parts[0])
        col_count = len(parts)
        return class_id, col_count
    except ValueError:
        return -1, len(parts)


def classify_label_file(file_path: Path) -> Tuple[str, List[int], bool]:
    """
    Classify a label file as Detection or Segmentation.

    Returns:
        (format_type, column_counts, has_inconsistency)
        - format_type: 'Detection', 'Segmentation', or 'Unknown'
        - column_counts: list of column counts for each line
        - has_inconsistency: True if file mixes Detection and Segmentation
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
    except Exception:
        return "Error", [], False

    if not lines:
        return "Empty", [], False

    column_counts = []
    for line in lines:
        _, col_count = parse_label_line(line)
        if col_count > 0:
            column_counts.append(col_count)

    if not column_counts:
        return "Invalid", [], False

    # Classify based on column counts
    # Detection: exactly 5 columns (class_id, x, y, w, h)
    # Segmentation: > 5 columns (class_id + polygon points)
    detection_count = sum(1 for c in column_counts if c == 5)
    segmentation_count = sum(1 for c in column_counts if c > 5)
    invalid_count = sum(1 for c in column_counts if c != 5 and c <= 5)

    has_inconsistency = (detection_count > 0 and segmentation_count > 0) or invalid_count > 0

    if segmentation_count > 0 and detection_count == 0:
        return "Segmentation", column_counts, has_inconsistency
    elif detection_count > 0 and segmentation_count == 0:
        return "Detection", column_counts, has_inconsistency
    else:
        return "Mixed", column_counts, True


def scan_all_labels() -> Tuple[Dict[str, int], List[Dict]]:
    """Scan all label files and return statistics and detailed records."""
    label_dirs = get_label_directories()
    statistics = defaultdict(int)
    records = []

    total_files = sum(len(list(d.glob("*.txt"))) for d in label_dirs)

    with tqdm(total=total_files, desc="Scanning label files", unit="file") as pbar:
        for label_dir in label_dirs:
            for label_file in sorted(label_dir.glob("*.txt")):
                format_type, col_counts, has_inconsistency = classify_label_file(label_file)
                statistics[format_type] += 1

                relative_path = label_file.relative_to(DATASET_ROOT)
                records.append({
                    "file_path": str(relative_path),
                    "format": format_type,
                    "object_count": len(col_counts),
                    "column_counts": ",".join(map(str, col_counts)) if col_counts else "",
                    "inconsistency": "Yes" if has_inconsistency else "No",
                })

                pbar.update(1)

    return dict(statistics), records


def save_report(statistics: Dict[str, int], records: List[Dict]) -> None:
    """Save the report to a CSV file."""
    df = pd.DataFrame(records)

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        # Write statistics summary
        handle.write("LABEL FORMAT STATISTICS\n")
        handle.write("format,count\n")
        for fmt, count in sorted(statistics.items()):
            handle.write(f"{fmt},{count}\n")

        handle.write("\n")
        handle.write("DETAILED FILE REPORT\n")
        df.to_csv(handle, index=False)

    print(f"✓ Laporan disimpan ke: {REPORT_CSV}")


def display_summary(statistics: Dict[str, int], records: List[Dict]) -> None:
    """Display a summary of findings."""
    print("\n" + "=" * 70)
    print("LABEL FORMAT ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nFormat Statistics:")
    for fmt in ["Detection", "Segmentation", "Mixed", "Empty", "Invalid", "Error"]:
        count = statistics.get(fmt, 0)
        if count > 0:
            print(f"  - {fmt}: {count}")

    total = sum(statistics.values())
    print(f"\n  Total files scanned: {total}")

    # Check for inconsistencies
    inconsistent = [r for r in records if r["inconsistency"] == "Yes"]
    if inconsistent:
        print(f"\n⚠ Inconsistent files detected: {len(inconsistent)}")
        print("\nFirst 10 inconsistent files:")
        for record in inconsistent[:10]:
            print(f"  - {record['file_path']}")
            print(f"    Format: {record['format']}, Objects: {record['object_count']}")

    # Detection vs Segmentation summary
    detection_count = statistics.get("Detection", 0)
    segmentation_count = statistics.get("Segmentation", 0)
    print(f"\nFormat Distribution:")
    print(f"  - Detection files: {detection_count}")
    print(f"  - Segmentation files: {segmentation_count}")

    print("\n" + "=" * 70)


def main() -> None:
    """Run the label format checker."""
    print("Scanning label files...")
    statistics, records = scan_all_labels()

    display_summary(statistics, records)
    save_report(statistics, records)


if __name__ == "__main__":
    main()
