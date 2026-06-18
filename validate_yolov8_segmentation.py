"""Validate a YOLOv8 segmentation dataset.

This script inspects the dataset folder structure and checks image/label consistency,
class distributions, and invalid label patterns for a YOLOv8 segmentation dataset.

Expected dataset layout:
    combined_dataset_preprocessed/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/

The script performs the following checks:
1. Count images in train, valid, and test splits.
2. Count objects per class across all splits.
3. Present a class distribution table.
4. Plot class distribution as a matplotlib bar chart.
5. Detect image files without matching labels.
6. Detect label files without matching images.
7. Detect class IDs outside the valid range 0-5.
8. Detect empty label files.
9. Save a summary report to dataset_report.csv.

Dependencies:
    pandas
    matplotlib
    tqdm

Usage:
    python validate_yolov8_segmentation.py
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


DATASET_ROOT = Path("combined_dataset_preprocessed")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VALID_CLASS_RANGE = range(0, 6)
REPORT_CSV = Path("dataset_report.csv")


def list_files(folder: Path, extensions: Set[str]) -> List[Path]:
    """Return a sorted list of files in a folder matching any of the given extensions."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions])


def load_label_lines(label_path: Path) -> List[str]:
    """Load non-empty stripped lines from a YOLO label file."""
    with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle.readlines()]
    return [line for line in lines if line]


def parse_label_line(line: str) -> Tuple[int, List[str]]:
    """Parse a YOLO label line into class ID and the remaining tokens."""
    parts = line.split()
    class_id = int(parts[0])
    return class_id, parts[1:]


def gather_dataset_statistics(root: Path) -> Dict[str, object]:
    """Collect counts and consistency information for the YOLOv8 dataset."""
    split_image_counts: Dict[str, int] = {}
    class_counts: Dict[int, int] = {cls: 0 for cls in VALID_CLASS_RANGE}
    invalid_class_ids: Set[int] = set()
    images_without_labels: List[Path] = []
    labels_without_images: List[Path] = []
    empty_label_files: List[Path] = []

    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"

        images = list_files(image_dir, IMAGE_EXTENSIONS)
        labels = list_files(label_dir, {".txt"})

        split_image_counts[split] = len(images)

        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}

        for image in images:
            if image.stem not in label_stems:
                images_without_labels.append(image)

        for label in labels:
            if label.stem not in image_stems:
                labels_without_images.append(label)

        for label_path in tqdm(labels, desc=f"Checking label files ({split})", unit="file"):
            lines = load_label_lines(label_path)
            if not lines:
                empty_label_files.append(label_path)
                continue

            for line in lines:
                try:
                    class_id, _ = parse_label_line(line)
                except (ValueError, IndexError):
                    invalid_class_ids.add(-1)
                    continue

                if class_id not in VALID_CLASS_RANGE:
                    invalid_class_ids.add(class_id)
                else:
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

    return {
        "split_image_counts": split_image_counts,
        "class_counts": class_counts,
        "invalid_class_ids": sorted(invalid_class_ids),
        "images_without_labels": images_without_labels,
        "labels_without_images": labels_without_images,
        "empty_label_files": empty_label_files,
    }


def build_report_table(stats: Dict[str, object]) -> pd.DataFrame:
    """Create a pandas DataFrame summarizing class distribution and checks."""
    class_counts = stats["class_counts"]
    rows = [
        {"class_id": cls, "object_count": class_counts.get(cls, 0)}
        for cls in sorted(class_counts.keys())
    ]
    return pd.DataFrame(rows)


def save_report_csv(report_df: pd.DataFrame, stats: Dict[str, object], output_path: Path) -> None:
    """Persist the report summary and class distribution to a CSV file."""
    summary_rows = [
        {"metric": "train_images", "value": stats["split_image_counts"]["train"]},
        {"metric": "valid_images", "value": stats["split_image_counts"]["valid"]},
        {"metric": "test_images", "value": stats["split_image_counts"]["test"]},
        {"metric": "total_objects", "value": sum(stats["class_counts"].values())},
        {"metric": "invalid_class_ids", "value": ", ".join(map(str, stats["invalid_class_ids"])) or "none"},
        {"metric": "images_without_labels", "value": len(stats["images_without_labels"])},
        {"metric": "labels_without_images", "value": len(stats["labels_without_images"])},
        {"metric": "empty_label_files", "value": len(stats["empty_label_files"])},
    ]

    summary_df = pd.DataFrame(summary_rows)
    counts_df = report_df.copy()
    counts_df.columns = ["metric", "value"]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        summary_df.to_csv(handle, index=False)
        handle.write("\n")
        handle.write("class_id,object_count\n")
        counts_df.to_csv(handle, index=False, header=False)

    print(f"✓ Laporan ringkas disimpan ke: {output_path}")


def plot_class_distribution(report_df: pd.DataFrame) -> None:
    """Plot class distribution as a bar chart using matplotlib."""
    plt.figure(figsize=(10, 6))
    plt.bar(report_df["class_id"].astype(str), report_df["object_count"], color="#4c72b0")
    plt.xlabel("Class ID")
    plt.ylabel("Object Count")
    plt.title("Distribusi Kelas YOLOv8 Segmentation")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the dataset validation and print the results."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root tidak ditemukan: {DATASET_ROOT}")

    stats = gather_dataset_statistics(DATASET_ROOT)
    report_df = build_report_table(stats)

    print("\n=== Ringkasan Dataset YOLOv8 Segmentation ===")
    print(f"Dataset root: {DATASET_ROOT}")
    print("")
    print("Jumlah gambar per split:")
    for split, count in stats["split_image_counts"].items():
        print(f"  - {split}: {count}")

    print("")
    print("Distribusi kelas:")
    print(report_df.to_string(index=False))

    print("")
    print(f"Jumlah file gambar tanpa label: {len(stats['images_without_labels'])}")
    print(f"Jumlah file label tanpa gambar: {len(stats['labels_without_images'])}")
    print(f"Jumlah file label kosong: {len(stats['empty_label_files'])}")
    print(f"Class ID tidak valid: {stats['invalid_class_ids'] or 'tidak ada'}")

    save_report_csv(report_df, stats, REPORT_CSV)
    plot_class_distribution(report_df)


if __name__ == "__main__":
    main()
