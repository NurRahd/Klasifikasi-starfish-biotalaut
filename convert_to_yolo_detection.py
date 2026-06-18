"""Convert multiple underwater datasets into a unified YOLOv8 Detection dataset.

This script reads a collection of datasets under `datasets/`, normalizes their
labels into YOLO Detection format, validates label/image consistency, and writes
an output dataset at `combined_detection_dataset/` ready for YOLOv8 Detection.

The script supports these label formats:
- YOLO Detection: class_id x_center y_center width height
- YOLO Segmentation: class_id x1 y1 x2 y2 x3 y3 ...

Conversion rules:
- Detection labels are copied as-is after class mapping and normalization.
- Segmentation labels are converted by computing the minimum bounding box from
  polygon vertices and saving the result as YOLO Detection.

The output dataset includes:
- train/images
- train/labels
- valid/images
- valid/labels
- test/images
- test/labels
- data.yaml

A comprehensive conversion report is saved to `conversion_report.csv`.

Dependencies:
    os
    shutil
    pathlib
    cv2
    numpy
    pandas
    matplotlib
    tqdm
    yaml

Usage:
    python convert_to_yolo_detection.py
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# === CONFIGURATION ============================================================
INPUT_ROOT = Path("datasets")
OUTPUT_ROOT = Path("combined_detection_dataset")
REPORT_CSV = Path("conversion_report.csv")
DATA_YAML = OUTPUT_ROOT / "data.yaml"
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VISUAL_SAMPLE_COUNT = 20
FINAL_CLASS_NAMES = ["fish", "jellyfish", "starfish"]

# Mapping from original class labels or numeric IDs to final YOLO classes.
# This mapping can be extended as new datasets appear.
CLASS_NAME_MAPPING: Dict[str, int] = {
    "fish": 0,
    "eel": 0,
    "lionfish": 0,
    "lobster": 0,
    "shark": 0,
    "jellyfish": 1,
    "jelly-fish": 1,
    "starfish": 2,
    "star-fish": 2,
    "sea_star": 2,
    "sea-star": 2,
    "star": 2,
}

# Default numeric label mapping for older YOLO datasets with more class IDs.
NUMERIC_CLASS_MAPPING: Dict[int, int] = {
    0: 0,  # fish / eel / other fish-like classes
    1: 0,
    2: 1,
    3: 0,
    4: 0,
    5: 2,
}

VALID_FINAL_CLASS_IDS = set(range(len(FINAL_CLASS_NAMES)))


@dataclass
class ConversionResult:
    filename: str
    original_format: str
    converted: bool
    final_class_id: Optional[int]
    status: str
    error: str


# === HELPERS ==================================================================

def load_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_dataset_dir(path: Path) -> bool:
    for split in SPLITS:
        images_dir = path / split / "images"
        labels_dir = path / split / "labels"
        if images_dir.exists() and labels_dir.exists():
            return True
    return False


def list_dataset_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and is_dataset_dir(p)]


def list_files(folder: Path, extensions: Iterable[str]) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions])


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_class_token(token: str) -> Optional[int]:
    token_clean = token.strip().lower()
    if token_clean in CLASS_NAME_MAPPING:
        return CLASS_NAME_MAPPING[token_clean]

    try:
        numeric = int(token_clean)
        if numeric in NUMERIC_CLASS_MAPPING:
            return NUMERIC_CLASS_MAPPING[numeric]
        if numeric in VALID_FINAL_CLASS_IDS:
            return numeric
    except ValueError:
        pass
    return None


def read_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, "corrupt image or cannot decode"
    h, w = image.shape[:2]
    return w, h, None


def parse_floats(tokens: List[str]) -> Tuple[List[float], Optional[str]]:
    values: List[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            return [], f"invalid numeric token '{token}'"
    return values, None


def normalize_coordinates(coords: List[float], width: int, height: int) -> List[float]:
    xs = coords[0::2]
    ys = coords[1::2]
    if max(xs, default=0) > 1.0 or max(ys, default=0) > 1.0:
        normalized: List[float] = []
        for x, y in zip(xs, ys):
            normalized.append(x / width)
            normalized.append(y / height)
        return normalized
    return coords


def normalize_detection_values(values: List[float], width: int, height: int) -> List[float]:
    if max(values, default=0) > 1.0:
        cx, cy, w, h = values
        return [cx / width, cy / height, w / width, h / height]
    return values


def polygon_to_bbox(coords: List[float]) -> Tuple[float, float, float, float]:
    xs = np.array(coords[0::2], dtype=np.float64)
    ys = np.array(coords[1::2], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return x_center, y_center, width, height


def is_normalized(coords: List[float]) -> bool:
    return all(0.0 <= v <= 1.0 for v in coords)


def format_type_from_token_count(token_count: int) -> str:
    if token_count == 5:
        return "detection"
    if token_count > 5:
        return "segmentation"
    return "invalid"


def validate_detection_box(values: List[float]) -> List[str]:
    errors: List[str] = []
    if len(values) != 4:
        errors.append("detection label must have 4 coordinates")
    if not is_normalized(values):
        errors.append("detection coordinates must be normalized between 0 and 1")
    return errors


def validate_polygon_coords(coords: List[float]) -> List[str]:
    errors: List[str] = []
    if len(coords) < 6:
        errors.append("segmentation polygon must have at least 3 points")
    if len(coords) % 2 != 0:
        errors.append("segmentation polygon coordinate count must be even")
    if not is_normalized(coords):
        errors.append("segmentation coordinates must be normalized between 0 and 1")
    return errors


def parse_label_file(
    label_path: Path, image_size: Optional[Tuple[int, int]], dataset_name: str
) -> Tuple[List[str], str, bool, Optional[int], str, int, List[str]]:
    """Parse a single label file and return converted lines with status."""
    if not label_path.exists():
        return [], "missing", False, None, "missing label file", 0, []

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if text == "":
        return [], "empty", False, None, "empty label file", 0, []

    lines = text.splitlines()
    output_lines: List[str] = []
    status = "ok"
    errors: List[str] = []
    original_formats = set()
    converted = False
    final_class_id: Optional[int] = None
    object_count = 0

    image_width, image_height = image_size if image_size is not None else (None, None)

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {line_no} is empty")
            status = "error"
            continue

        tokens = line.strip().split()
        fmt = format_type_from_token_count(len(tokens))
        original_formats.add(fmt)

        class_token = tokens[0]
        class_id = normalize_class_token(class_token)
        if class_id is None:
            errors.append(f"line {line_no}: cannot map class '{class_token}'")
            status = "error"
            continue

        if fmt == "invalid":
            errors.append(f"line {line_no}: invalid label format")
            status = "error"
            continue

        values, parse_error = parse_floats(tokens[1:])
        if parse_error:
            errors.append(f"line {line_no}: {parse_error}")
            status = "error"
            continue

        if fmt == "detection":
            if image_width is not None and image_height is not None:
                values = normalize_detection_values(values, image_width, image_height)
            line_errors = validate_detection_box(values)
            if line_errors:
                errors.extend([f"line {line_no}: {err}" for err in line_errors])
                status = "error"
                continue
            output_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in values)}")
            final_class_id = class_id
            object_count += 1

        elif fmt == "segmentation":
            if image_width is not None and image_height is not None:
                if not is_normalized(values):
                    polygon = normalize_coordinates(values, image_width, image_height)
                else:
                    polygon = values
            else:
                polygon = values

            line_errors = validate_polygon_coords(polygon)
            if line_errors:
                errors.extend([f"line {line_no}: {err}" for err in line_errors])
                status = "error"
                continue

            if image_width is not None and image_height is not None:
                polygon = normalize_coordinates(polygon, image_width, image_height)
            bbox = polygon_to_bbox(polygon)
            if not is_normalized(list(bbox)):
                errors.append(f"line {line_no}: converted bbox values out of range")
                status = "error"
                continue

            output_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}")
            final_class_id = class_id
            converted = True
            object_count += 1

    original_format = ",".join(sorted(original_formats)) if original_formats else "none"
    if status == "ok" and object_count == 0:
        status = "warning"
        errors.append("label file contains no valid objects")

    return output_lines, original_format, converted, final_class_id, status, object_count, errors


def build_data_yaml(output_root: Path) -> None:
    yaml_path = output_root / "data.yaml"
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(FINAL_CLASS_NAMES),
        "names": FINAL_CLASS_NAMES,
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file)


def convert_dataset(input_root: Path, output_root: Path) -> Tuple[pd.DataFrame, Counter, int, int, int]:
    ensure_folder(output_root)
    for split in SPLITS:
        ensure_folder(output_root / split / "images")
        ensure_folder(output_root / split / "labels")

    report_rows: List[ConversionResult] = []
    class_counter = Counter()
    total_images = 0
    total_labels = 0
    total_objects = 0

    dataset_dirs = list_dataset_dirs(input_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"Input dataset folder not found or empty: {input_root}")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        for split in SPLITS:
            image_folder = dataset_dir / split / "images"
            label_folder = dataset_dir / split / "labels"
            image_files = list_files(image_folder, IMAGE_EXTENSIONS)
            label_files = {p.stem: p for p in list_files(label_folder, {".txt"})}

            file_index = 0
            for image_path in tqdm(image_files, desc=f"Processing {dataset_name}/{split}", unit="image"):
                file_index += 1
                output_base = f"{dataset_name}_{split}_{file_index:05d}"
                output_image = OUTPUT_ROOT / split / "images" / f"{output_base}{image_path.suffix.lower()}"
                shutil.copy2(image_path, output_image)
                total_images += 1

                image_width, image_height, img_error = read_image_size(image_path)
                label_path = label_files.get(image_path.stem)
                if label_path is None:
                    output_label_path = OUTPUT_ROOT / split / "labels" / f"{output_base}.txt"
                    output_label_path.write_text("", encoding="utf-8")
                    report_rows.append(
                        ConversionResult(
                            filename=str(output_label_path.relative_to(OUTPUT_ROOT)),
                            original_format="none",
                            converted=False,
                            final_class_id=None,
                            status="missing_label",
                            error="no label file found",
                        )
                    )
                    continue

                label_lines, original_format, converted, final_class_id, status, object_count, errors = parse_label_file(
                    label_path,
                    (image_width, image_height) if img_error is None else None,
                    dataset_name,
                )
                output_label_path = OUTPUT_ROOT / split / "labels" / f"{output_base}.txt"

                if img_error is not None:
                    status = "corrupt_image"
                    errors.append(img_error)

                if status != "missing_label":
                    if label_lines:
                        output_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    else:
                        output_label_path.write_text("", encoding="utf-8")

                class_name = FINAL_CLASS_NAMES[final_class_id] if final_class_id is not None else "unknown"
                if status in {"ok", "warning"}:
                    total_labels += 1
                    total_objects += object_count
                    if final_class_id is not None:
                        class_counter[final_class_id] += object_count

                report_rows.append(
                    ConversionResult(
                        filename=str(output_label_path.relative_to(OUTPUT_ROOT)),
                        original_format=original_format,
                        converted=converted,
                        final_class_id=final_class_id,
                        status=status,
                        error="; ".join(errors),
                    )
                )

            for label_stem, label_path in label_files.items():
                if label_stem not in {p.stem for p in image_files}:
                    output_base = f"{dataset_name}_{split}_label_{label_stem}"
                    output_label_path = OUTPUT_ROOT / split / "labels" / f"{output_base}.txt"
                    output_label_path.write_text(label_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                    report_rows.append(
                        ConversionResult(
                            filename=str(output_label_path.relative_to(OUTPUT_ROOT)),
                            original_format="orphan_label",
                            converted=False,
                            final_class_id=None,
                            status="label_without_image",
                            error="label file exists without matching image",
                        )
                    )

    df = pd.DataFrame([row.__dict__ for row in report_rows])
    df.to_csv(REPORT_CSV, index=False)
    total_labels = int((df["filename"].str.endswith(".txt")).sum())
    return df, class_counter, total_images, total_labels, total_objects


def visualize_samples(output_root: Path, report_df: pd.DataFrame) -> None:
    valid_rows = report_df[report_df["status"].isin({"ok", "warning"})]
    sample = valid_rows.sample(min(VISUAL_SAMPLE_COUNT, len(valid_rows)), random_state=42)
    if sample.empty:
        print("No valid samples available for visualization.")
        return

    images = []
    labels = []
    classes = []
    for _, row in sample.iterrows():
        label_path = output_root / row["filename"]
        image_path = output_root / label_path.parent.parent / "images" / f"{label_path.stem}{next((ext for ext in IMAGE_EXTENSIONS if (output_root / label_path.parent.parent / 'images' / f'{label_path.stem}{ext}').exists()), '.jpg')}"
        if not image_path.exists():
            continue
        text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        images.append(image_path)
        labels.append(text)
        classes.append(row["final_class_id"])

    if not images:
        print("No valid visualization samples found.")
        return

    cols = 5
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for idx, (ax, image_path, label_text) in enumerate(zip(axes, images, labels)):
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            ax.set_title("corrupt image")
            ax.axis("off")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        ax.imshow(image)
        ax.set_title(image_path.name)
        ax.axis("off")

        if label_text:
            for line in label_text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="lime", facecolor="none", linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, FINAL_CLASS_NAMES[class_id], color="yellow", fontsize=8, weight="bold")

    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    
    # Save visualization instead of showing (non-blocking)
    output_viz = output_root / "conversion_visualization.png"
    plt.savefig(str(output_viz), dpi=100, bbox_inches="tight")
    print(f"Visualization saved to: {output_viz}")
    plt.close()


def summarize_report(df: pd.DataFrame, class_counter: Counter, total_images: int, total_labels: int, total_objects: int, input_root: Path) -> None:
    print("\n=== CONVERSION SUMMARY ===")
    print(f"Total input datasets: {len(list_dataset_dirs(input_root))}")
    print(f"Total images copied: {total_images}")
    print(f"Total label files written: {total_labels}")
    print(f"Total objects: {total_objects}")
    print("\nClass distribution:")
    for class_id, name in enumerate(FINAL_CLASS_NAMES):
        print(f"  - {name}: {class_counter[class_id]}")

    counts = df["status"].value_counts().to_dict()
    print("\nStatus counts:")
    for status, count in counts.items():
        print(f"  - {status}: {count}")

    if counts.get("label_without_image", 0) > 0:
        print("Warning: there are label files without corresponding images.")
    if counts.get("missing_label", 0) > 0:
        print("Warning: there are images without label files; empty labels were generated.")

    print(f"\nConversion report saved to: {REPORT_CSV}")
    print(f"Dataset ready at: {OUTPUT_ROOT}")


def find_available_root(default_root: Path) -> Path:
    if default_root.exists():
        return default_root

    candidate_root = Path(".")
    candidate_datasets = list_dataset_dirs(candidate_root)
    if candidate_datasets:
        print(f"Warning: default input folder '{default_root}' not found.")
        print(f"Using current working directory as input root with {len(candidate_datasets)} dataset(s): {[d.name for d in candidate_datasets]}")
        return candidate_root

    raise FileNotFoundError(f"Input datasets folder not found: {default_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert multiple underwater datasets into a unified YOLO Detection dataset.")
    parser.add_argument("--input", default=str(INPUT_ROOT), help="Input datasets root folder")
    parser.add_argument("--output", default=str(OUTPUT_ROOT), help="Output folder for combined detection dataset")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    try:
        input_root = find_available_root(input_root)
    except FileNotFoundError as exc:
        raise

    print("=== YOLO Detection Conversion Pipeline ===")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")

    df, class_counter, total_images, total_labels, total_objects = convert_dataset(input_root, output_root)
    build_data_yaml(output_root)
    summarize_report(df, class_counter, total_images, total_labels, total_objects, input_root)
    visualize_samples(output_root, df)


if __name__ == "__main__":
    main()
