"""Dataset Readiness Check for YOLOv8 Segmentation.

This script validates a YOLOv8 Segmentation dataset located in `combined_dataset/`.
It inspects dataset structure, labels, polygons, class distribution, image size,
visualizes samples, and generates a readiness report.

Usage:
    python dataset_readiness_check.py

Dependencies:
    pathlib
    pandas
    numpy
    opencv-python
    matplotlib
    tqdm
    yaml
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


DATASET_ROOT = Path("combined_dataset")
REPORT_CSV = Path("dataset_readiness_report.csv")
VISUAL_SAMPLE_SIZE = 30
CLASS_NAMES = {
    0: "eel",
    1: "fish",
    2: "jellyfish",
    3: "lionfish",
    4: "lobster",
    5: "star-fish",
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_POLYGON_AREA = 0.0005  # normalized area threshold for too-small polygon
MAX_POLYGON_AREA = 1.0


@dataclass
class LabelCheckResult:
    class_id: int
    coords: List[float]
    polygon_valid: bool
    errors: List[str]


@dataclass
class FileReport:
    filename: str
    split: str
    status: str
    jumlah_objek: int
    kelas: str
    error: str


def load_yaml(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def list_image_files(split: str) -> List[Path]:
    folder = DATASET_ROOT / split / "images"
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def list_label_files(split: str) -> List[Path]:
    folder = DATASET_ROOT / split / "labels"
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"] )


def parse_label_line(line: str, line_index: int) -> Tuple[Optional[LabelCheckResult], Optional[str]]:
    tokens = line.strip().split()
    if not tokens:
        return None, f"empty line {line_index}"

    if len(tokens) < 2:
        return None, f"line {line_index} has too few tokens"

    try:
        class_id = int(tokens[0])
    except ValueError:
        return None, f"line {line_index} has invalid class id '{tokens[0]}'"

    coords = []
    errors: List[str] = []
    for idx, token in enumerate(tokens[1:], start=1):
        try:
            coords.append(float(token))
        except ValueError:
            errors.append(f"coord {idx} is not float: '{token}'")

    col_count = len(tokens)
    if class_id not in VALID_CLASS_IDS:
        errors.append(f"class_id {class_id} out of range")
    if col_count <= 5:
        errors.append("label format must be segmentation (>5 values)")
    if len(coords) % 2 != 0:
        errors.append("polygon coordinate count must be even")

    if len(coords) >= 6 and len(coords) % 2 == 0:
        polygon_valid = True
    else:
        polygon_valid = False

    if coords:
        if any(c < 0.0 for c in coords) or any(c > 1.0 for c in coords):
            errors.append("coordinate values must be normalized between 0 and 1")
            polygon_valid = False

    if len(coords) >= 6 and len(coords) % 2 == 0:
        polygon_area = compute_polygon_area(coords)
        if polygon_area <= 0:
            errors.append("polygon area is zero or negative")
            polygon_valid = False
        elif polygon_area < MIN_POLYGON_AREA:
            errors.append("polygon too small")
    else:
        polygon_area = 0.0

    if len(coords) >= 6 and len(coords) % 2 == 0 and polygon_valid:
        if not is_polygon_closed(coords):
            errors.append("polygon is malformed or self-intersecting")
            polygon_valid = False

    result = LabelCheckResult(
        class_id=class_id,
        coords=coords,
        polygon_valid=polygon_valid,
        errors=errors,
    )
    return result, None


def compute_polygon_area(coords: List[float]) -> float:
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def is_polygon_closed(coords: List[float]) -> bool:
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return False
    return not np.any(np.isnan(pts))


def validate_label_file(label_path: Path) -> Tuple[List[LabelCheckResult], List[str]]:
    results: List[LabelCheckResult] = []
    errors: List[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        return [], [f"cannot read label file: {exc}"]

    if not lines:
        errors.append("label file is empty")
        return [], errors

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"empty line {idx}")
            continue
        result, parse_error = parse_label_line(line, idx)
        if parse_error:
            errors.append(parse_error)
            continue
        if result is not None:
            results.append(result)
            errors.extend(result.errors)
    return results, errors


def load_image_shape(image_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, None, "cannot open image"
    h, w = image.shape[:2]
    return w, h, None


def collect_dataset_info() -> Tuple[Dict[str, int], Dict[str, int], List[FileReport], Dict[int, int], List[Tuple[int, int]], List[Path], List[Path], List[Path], List[Path], int, int]:
    image_counts = {}
    label_counts = {}
    file_reports: List[FileReport] = []
    class_obj_counts = Counter()
    image_sizes: List[Tuple[int, int]] = []
    images_without_label = []
    labels_without_image = []
    corrupt_images = []
    empty_files = []
    valid_polygons = 0
    invalid_polygons = 0

    for split in SPLITS:
        images = list_image_files(split)
        labels = list_label_files(split)

        image_counts[split] = len(images)
        label_counts[split] = len(labels)

        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}

        for image in images:
            if image.stem not in label_stems:
                images_without_label.append(image)

        for label in labels:
            if label.stem not in image_stems:
                labels_without_image.append(label)

        for image_path in images:
            width, height, error = load_image_shape(image_path)
            if error:
                corrupt_images.append(image_path)
                continue
            image_sizes.append((width, height))

        for label_path in labels:
            results, errors = validate_label_file(label_path)
            status = "OK"
            if errors:
                status = "ERROR"
            kelas = ",".join(str(r.class_id) for r in results)
            file_reports.append(
                FileReport(
                    filename=str(label_path.relative_to(DATASET_ROOT)),
                    split=split,
                    status=status,
                    jumlah_objek=len(results),
                    kelas=kelas,
                    error="; ".join(errors),
                )
            )
            for result in results:
                if result.polygon_valid:
                    class_obj_counts[result.class_id] += 1
                    valid_polygons += 1
                else:
                    invalid_polygons += 1

            if not results and not errors:
                empty_files.append(label_path)

    return (
        image_counts,
        label_counts,
        file_reports,
        class_obj_counts,
        image_sizes,
        images_without_label,
        labels_without_image,
        corrupt_images,
        empty_files,
        valid_polygons,
        invalid_polygons,
    )


def get_statistics(image_sizes: List[Tuple[int, int]], object_counts: Counter) -> Dict[str, float]:
    if not image_sizes:
        return {
            "min_width": 0,
            "min_height": 0,
            "max_width": 0,
            "max_height": 0,
            "avg_width": 0.0,
            "avg_height": 0.0,
        }
    widths = np.array([w for w, _ in image_sizes], dtype=np.float64)
    heights = np.array([h for _, h in image_sizes], dtype=np.float64)
    return {
        "min_width": float(widths.min()),
        "min_height": float(heights.min()),
        "max_width": float(widths.max()),
        "max_height": float(heights.max()),
        "avg_width": float(widths.mean()),
        "avg_height": float(heights.mean()),
    }


def compute_class_balance(obj_counts: Counter) -> Tuple[Dict[str, int], Dict[str, float], Optional[str]]:
    distribution = {CLASS_NAMES[c]: obj_counts.get(c, 0) for c in VALID_CLASS_IDS}
    total_objects = sum(distribution.values())
    percentage = {k: (v / total_objects * 100 if total_objects > 0 else 0.0) for k, v in distribution.items()}
    min_count = min([v for v in distribution.values() if v > 0], default=0)
    max_count = max(distribution.values(), default=0)
    imbalance_warning = None
    if min_count > 0 and max_count > 5 * min_count:
        imbalance_warning = "class imbalance detected"
    elif min_count == 0 and max_count > 0:
        imbalance_warning = "class(es) missing or zero count"
    return distribution, percentage, imbalance_warning


def generate_visual_samples(file_reports: List[FileReport]) -> None:
    label_paths = [DATASET_ROOT / report.filename for report in file_reports if report.status == "OK" and report.filename.endswith(".txt")]
    if not label_paths:
        print("No valid label files available for visualization.")
        return

    sample_paths = random.sample(label_paths, min(VISUAL_SAMPLE_SIZE, len(label_paths)))
    n = len(sample_paths)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()

    for ax, label_path in zip(axs, sample_paths):
        image_folder = DATASET_ROOT / label_path.parent.parent / "images"
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = image_folder / f"{label_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            ax.set_title("missing image")
            ax.axis("off")
            continue

        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            ax.set_title("cannot open image")
            ax.axis("off")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        line_results, _ = validate_label_file(label_path)
        ax.imshow(image)
        ax.set_title(f"{label_path.name}")
        ax.axis("off")
        for result in line_results:
            if result.polygon_valid:
                pts = np.array(result.coords, dtype=np.float32).reshape(-1, 2)
                h, w = image.shape[:2]
                pts_pixel = np.column_stack((pts[:, 0] * w, pts[:, 1] * h))
                ax.plot(np.append(pts_pixel[:, 0], pts_pixel[0, 0]), np.append(pts_pixel[:, 1], pts_pixel[0, 1]), color="lime", linewidth=1.5)
                ax.text(pts_pixel[0, 0], pts_pixel[0, 1] - 5, CLASS_NAMES.get(result.class_id, str(result.class_id)), color="yellow", fontsize=8, weight="bold")

    for ax in axs[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def recommend_training_params(image_sizes: List[Tuple[int, int]], total_objects: int) -> Dict[str, str]:
    widths = np.array([w for w, _ in image_sizes], dtype=np.float64)
    heights = np.array([h for _, h in image_sizes], dtype=np.float64)
    if len(widths) == 0:
        imgsz = 640
    else:
        mean_short = np.mean(np.minimum(widths, heights))
        if mean_short < 416:
            imgsz = 416
        elif mean_short < 640:
            imgsz = 640
        else:
            imgsz = 800
    if total_objects < 1000:
        batch = 8
    elif total_objects < 5000:
        batch = 16
    else:
        batch = 32
    epochs = 100 if total_objects > 2000 else 150
    return {
        "imgsz": str(int(imgsz)),
        "batch": str(batch),
        "epochs": str(epochs),
    }


def main() -> None:
    print("Dataset Readiness Check for YOLOv8 Segmentation")
    print("Dataset root:", DATASET_ROOT)

    data_yaml = load_yaml(DATASET_ROOT / "data.yaml")
    if data_yaml:
        print("Loaded data.yaml")
    else:
        print("Warning: data.yaml not found or invalid")

    (
        image_counts,
        label_counts,
        file_reports,
        class_obj_counts,
        image_sizes,
        images_without_label,
        labels_without_image,
        corrupt_images,
        empty_files,
        valid_polygons,
        invalid_polygons,
    ) = collect_dataset_info()

    total_images = sum(image_counts.values())
    total_labels = sum(label_counts.values())
    total_objects = sum(class_obj_counts.values())
    invalid_files = [r for r in file_reports if r.status == "ERROR"]

    class_distribution, class_percentage, imbalance_warning = compute_class_balance(class_obj_counts)
    training_params = recommend_training_params(image_sizes, total_objects)

    df_report = pd.DataFrame([report.__dict__ for report in file_reports])
    df_report.to_csv(REPORT_CSV, index=False)

    stats = get_statistics(image_sizes, class_obj_counts)

    print("\n=== SUMMARY ===")
    print(f"Total Images: {total_images}")
    print(f"Total Labels: {total_labels}")
    print(f"Total Objects: {total_objects}")
    print(f"Valid Polygons: {valid_polygons}")
    print(f"Invalid Polygons: {invalid_polygons}")
    print(f"Corrupt Images: {len(corrupt_images)}")
    print(f"Empty Label Files: {len(empty_files)}")
    print(f"Files with Errors: {len(invalid_files)}")
    print(f"Images without Labels: {len(images_without_label)}")
    print(f"Labels without Images: {len(labels_without_image)}")
    print(f"Average Objects per Image: {total_objects / total_images if total_images > 0 else 0:.2f}")
    print(f"Image size min: {stats['min_width']}x{stats['min_height']}")
    print(f"Image size max: {stats['max_width']}x{stats['max_height']}")
    print(f"Image size avg: {stats['avg_width']:.1f}x{stats['avg_height']:.1f}")

    print("\nClass Distribution:")
    for class_name, count in class_distribution.items():
        print(f"  - {class_name}: {count} ({class_percentage[class_name]:.2f}%)")
    if imbalance_warning:
        print("Warning:", imbalance_warning)

    if len(corrupt_images) == 0 and len(invalid_files) == 0 and not imbalance_warning:
        status = "READY FOR TRAINING"
    elif len(corrupt_images) == 0 and len(invalid_files) < 10:
        status = "READY WITH WARNINGS"
    else:
        status = "NOT READY"

    print("\nDataset Status:", status)
    print("Report saved to:", REPORT_CSV)

    print("\nTraining Recommendation:")
    print(f"  - imgsz: {training_params['imgsz']}")
    print(f"  - batch size: {training_params['batch']}")
    print(f"  - epochs: {training_params['epochs']}")

    print("\nGenerating visualization of random samples...")
    generate_visual_samples(file_reports)


if __name__ == "__main__":
    main()
