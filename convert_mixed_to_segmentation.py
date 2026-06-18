"""Convert mixed YOLO label format files to pure Segmentation format.

This script processes YOLO label files that contain mixed Detection and Segmentation
formats, converting all Detection format lines to Segmentation format.

Detection to Segmentation Conversion:
- Detection: class_id center_x center_y width height (5 columns, bbox format)
- Segmentation: class_id x1 y1 x2 y2 ... xn yn (polygon format)

The conversion creates a rectangular polygon from the bbox:
  - Calculate bbox corners from center and dimensions
  - Create 4 polygon points representing the rectangle corners

Conversion Strategy:
  1. Scan all label files
  2. Identify mixed-format files
  3. Convert Detection format lines to Segmentation (rectangle polygon)
  4. Preserve existing Segmentation lines
  5. Backup original files and save converted labels
  6. Generate conversion report

Dependencies:
    pathlib
    tqdm
    pandas (for CSV reporting)

Usage:
    python convert_mixed_to_segmentation.py
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import shutil

import pandas as pd
from tqdm import tqdm


DATASET_ROOT = Path(".")
SPLITS = ["train", "valid", "test"]
DATASETS = ["biota_laut/Underwater Marine Species.v2i.yolov8", "starfish_only", "combined_dataset", "combined_dataset_preprocessed"]
BACKUP_DIR = Path("label_backups")
CONVERSION_REPORT_CSV = Path("conversion_report.csv")


def get_all_label_files() -> List[Tuple[Path, Path]]:
    """Return list of (label_dir, label_file) tuples for all label files."""
    label_files = []
    for dataset in DATASETS:
        dataset_path = DATASET_ROOT / dataset
        if not dataset_path.exists():
            continue
        for split in SPLITS:
            label_dir = dataset_path / split / "labels"
            if label_dir.exists():
                for label_file in sorted(label_dir.glob("*.txt")):
                    label_files.append((label_dir, label_file))
    return label_files


def parse_label_line(line: str) -> Tuple[int, int, List[float]]:
    """Parse a label line and return (class_id, col_count, values)."""
    parts = line.strip().split()
    if not parts:
        return -1, 0, []
    try:
        class_id = int(parts[0])
        values = [float(p) for p in parts[1:]]
        col_count = len(parts)
        return class_id, col_count, values
    except ValueError:
        return -1, len(parts), []


def classify_line_format(class_id: int, col_count: int) -> str:
    """Classify a line as Detection, Segmentation, or Invalid."""
    if col_count == 5:
        return "Detection"
    elif col_count > 5 and col_count % 2 == 1:  # odd number: class_id + pairs of coordinates
        return "Segmentation"
    else:
        return "Invalid"


def bbox_to_polygon(center_x: float, center_y: float, width: float, height: float) -> List[float]:
    """Convert bbox (center + dimensions) to polygon (4 rectangle corners).

    Args:
        center_x, center_y: normalized center coordinates (0-1)
        width, height: normalized dimensions (0-1)

    Returns:
        List of 8 floats: [x1, y1, x2, y2, x3, y3, x4, y4] representing rectangle corners.
    """
    x_min = center_x - width / 2.0
    x_max = center_x + width / 2.0
    y_min = center_y - height / 2.0
    y_max = center_y + height / 2.0

    # Rectangle corners: top-left, top-right, bottom-right, bottom-left
    polygon = [
        x_min, y_min,  # top-left
        x_max, y_min,  # top-right
        x_max, y_max,  # bottom-right
        x_min, y_max,  # bottom-left
    ]
    return polygon


def convert_label_file(label_path: Path, backup_dir: Path) -> Tuple[bool, int, int, str]:
    """Convert a mixed format label file to pure Segmentation format.

    Returns:
        (is_mixed, detection_count, segmentation_count, status_message)
    """
    try:
        with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = [line.strip() for line in handle.readlines()]
    except Exception as e:
        return False, 0, 0, f"Error reading: {e}"

    if not lines:
        return False, 0, 0, "Empty file"

    # Parse all lines
    parsed_lines = []
    detection_count = 0
    segmentation_count = 0
    is_mixed = False

    for line in lines:
        if not line:
            continue

        class_id, col_count, values = parse_label_line(line)
        if class_id < 0:
            continue

        line_format = classify_line_format(class_id, col_count)

        if line_format == "Detection":
            detection_count += 1
            parsed_lines.append((class_id, line_format, values))
        elif line_format == "Segmentation":
            segmentation_count += 1
            parsed_lines.append((class_id, line_format, values))
        else:
            parsed_lines.append((class_id, line_format, values))

    is_mixed = (detection_count > 0 and segmentation_count > 0) or detection_count > 0

    if not is_mixed and segmentation_count == 0:
        return False, detection_count, segmentation_count, "No conversion needed"

    # Backup original file
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / label_path.name
        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)
    except Exception as e:
        return False, detection_count, segmentation_count, f"Backup failed: {e}"

    # Convert and write
    try:
        converted_lines = []
        for class_id, line_format, values in parsed_lines:
            if line_format == "Detection":
                # Convert bbox to polygon
                center_x, center_y, width, height = values
                polygon = bbox_to_polygon(center_x, center_y, width, height)
                converted_line = f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon)
                converted_lines.append(converted_line)
            elif line_format == "Segmentation":
                # Keep as-is
                converted_line = f"{class_id} " + " ".join(f"{v:.6f}" for v in values)
                converted_lines.append(converted_line)

        with label_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(converted_lines))
            if converted_lines:
                handle.write("\n")

        return is_mixed, detection_count, segmentation_count, "Converted"
    except Exception as e:
        return False, detection_count, segmentation_count, f"Conversion failed: {e}"


def main() -> None:
    """Run the conversion process."""
    print("=" * 70)
    print("CONVERT MIXED YOLO LABELS TO SEGMENTATION FORMAT")
    print("=" * 70)

    label_files = get_all_label_files()
    print(f"\nFound {len(label_files)} label files\n")

    records = []
    mixed_count = 0
    total_detection_converted = 0
    total_segmentation_preserved = 0

    for label_dir, label_file in tqdm(label_files, desc="Converting label files", unit="file"):
        is_mixed, detection_count, segmentation_count, status = convert_label_file(label_file, BACKUP_DIR)

        if is_mixed:
            mixed_count += 1
            total_detection_converted += detection_count
            total_segmentation_preserved += segmentation_count

            relative_path = label_file.relative_to(DATASET_ROOT)
            records.append({
                "file_path": str(relative_path),
                "detection_lines": detection_count,
                "segmentation_lines": segmentation_count,
                "status": status,
            })

    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Mixed format files found: {mixed_count}")
    print(f"Total Detection lines converted to Segmentation: {total_detection_converted}")
    print(f"Total Segmentation lines preserved: {total_segmentation_preserved}")
    print(f"Backup directory: {BACKUP_DIR}")

    # Save report
    if records:
        df = pd.DataFrame(records)
        df.to_csv(CONVERSION_REPORT_CSV, index=False)
        print(f"\n✓ Conversion report saved to: {CONVERSION_REPORT_CSV}")
        print(f"\nFirst 10 converted files:")
        for i, record in enumerate(records[:10], 1):
            print(f"  {i}. {record['file_path']}")
            print(f"     Detection → Segmentation: {record['detection_lines']} lines")
    else:
        print("\n✓ No mixed format files needed conversion!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
