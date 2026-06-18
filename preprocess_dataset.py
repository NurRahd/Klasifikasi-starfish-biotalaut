"""Preprocess dataset citra underwater untuk YOLOv8 Detection.

Pipeline yang digunakan:
1. Resize ke 640x640
2. Gaussian blur untuk mereduksi noise underwater
3. CLAHE untuk meningkatkan kontras lokal
4. Brightness dan contrast enhancement dengan parameter aman
5. Gamma correction opsional untuk memperbaiki pencahayaan

Output yang dihasilkan:
- preprocessed_dataset/ dengan struktur train/valid/test/images dan labels
- preprocessing_report.csv berisi status setiap file
- visualisasi before-after preprocessing pada 20 sampel acak

Catatan implementasi:
- Label disalin tanpa perubahan untuk pasangan image-label yang sinkron.
- Image yang corrupt, gagal diproses, label kosong, image tanpa label, dan label tanpa image
  semuanya dilaporkan agar dataset tetap bisa diaudit sebelum training.
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# KONFIGURASI UMUM
# ============================================================================

SOURCE_DATASET_PATH = Path("combined_dataset")
OUTPUT_DATASET_PATH = Path("preprocessed_dataset")
REPORT_CSV_PATH = Path("preprocessing_report.csv")

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_SIZE = (640, 640)

# Parameter preprocessing yang aman untuk citra underwater.
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0.8
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BRIGHTNESS_BETA = 6
CONTRAST_ALPHA = 1.10
GAMMA_VALUE = 1.08
ENABLE_GAMMA_CORRECTION = True


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURE
# ============================================================================


@dataclass
class ProcessedRecord:
    """Menyimpan metadata hasil pemrosesan satu image."""

    filename: str
    split: str
    status: str
    original_size: str
    processed_size: str
    error: str
    source_image_path: Optional[Path] = None
    output_image_path: Optional[Path] = None


# ============================================================================
# UTILITAS DASAR
# ============================================================================


def create_output_structure(output_base_path: Path) -> None:
    """Membuat struktur folder output yang kompatibel dengan YOLOv8 Detection."""

    for split in SPLITS:
        (output_base_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_base_path / split / "labels").mkdir(parents=True, exist_ok=True)


def list_image_files(folder: Path) -> List[Path]:
    """Mengambil semua file image yang didukung dari sebuah folder."""

    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def load_image(image_path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[str]]:
    """Memuat image dengan aman, termasuk path Windows yang mengandung spasi."""

    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return None, None, "image corrupt atau tidak bisa dibaca"
        height, width = image.shape[:2]
        return image, (width, height), None
    except Exception as exc:
        return None, None, f"gagal membaca image: {exc}"


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize citra ke ukuran target."""

    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Meningkatkan kontras lokal dengan CLAHE pada kanal lightness."""

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = BRIGHTNESS_BETA,
    contrast: float = CONTRAST_ALPHA,
) -> np.ndarray:
    """Menyesuaikan brightness dan contrast dengan parameter yang aman."""

    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def gamma_correction(image: np.ndarray, gamma: float = GAMMA_VALUE) -> np.ndarray:
    """Menerapkan gamma correction untuk membantu pencahayaan underwater."""

    if gamma <= 0:
        return image

    inv_gamma = 1.0 / gamma
    lookup = np.array([((index / 255.0) ** inv_gamma) * 255 for index in range(256)]).astype("uint8")
    return cv2.LUT(image, lookup)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Pipeline preprocessing utama untuk satu citra underwater."""

    processed = resize_image(image, TARGET_SIZE)
    processed = cv2.GaussianBlur(processed, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    processed = apply_clahe(processed)
    processed = adjust_brightness_contrast(processed)

    if ENABLE_GAMMA_CORRECTION:
        processed = gamma_correction(processed)

    return processed


def save_processed_image(image: np.ndarray, output_path: Path) -> bool:
    """Menyimpan hasil preprocessing dengan cara yang aman di Windows."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(output_path.suffix or ".jpg", image)
        if not success:
            return False
        encoded.tofile(str(output_path))
        return True
    except Exception as exc:
        logger.error(f"Gagal menyimpan image {output_path.name}: {exc}")
        return False


def copy_label_file(source_label_path: Path, output_label_path: Path) -> bool:
    """Menyalin label tanpa perubahan."""

    try:
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_label_path, output_label_path)
        return True
    except Exception as exc:
        logger.error(f"Gagal menyalin label {source_label_path.name}: {exc}")
        return False


def is_label_empty(label_path: Path) -> bool:
    """Memeriksa apakah file label kosong atau hanya berisi spasi/newline."""

    try:
        return not label_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return True


def format_size(size: Optional[Tuple[int, int]]) -> str:
    """Mengubah ukuran image menjadi string yang konsisten untuk report."""

    if size is None:
        return ""
    width, height = size
    return f"{width}x{height}"


# ============================================================================
# PROSES DATASET
# ============================================================================


def process_split(
    split: str,
    source_dataset_path: Path,
    output_dataset_path: Path,
) -> Tuple[List[ProcessedRecord], int, int, List[Tuple[int, int]]]:
    """Memproses satu split dataset dan mengembalikan laporan detailnya."""

    source_split_path = source_dataset_path / split
    output_split_path = output_dataset_path / split
    source_images_path = source_split_path / "images"
    source_labels_path = source_split_path / "labels"
    output_images_path = output_split_path / "images"
    output_labels_path = output_split_path / "labels"

    records: List[ProcessedRecord] = []
    success_count = 0
    failed_count = 0
    loaded_sizes: List[Tuple[int, int]] = []

    image_files = list_image_files(source_images_path)
    label_files = sorted([p for p in source_labels_path.iterdir() if p.is_file()]) if source_labels_path.exists() else []
    label_map = {label_file.stem: label_file for label_file in label_files if label_file.suffix.lower() == ".txt"}
    processed_label_stems = set()

    logger.info(f"Memproses split {split.upper()} - {len(image_files)} image ditemukan")

    for image_path in tqdm(image_files, desc=f"Preprocessing {split}", unit="img"):
        label_path = label_map.get(image_path.stem)
        errors: List[str] = []
        status = "processed"

        if label_path is None:
            status = "processed_missing_label"
            errors.append("label tidak ditemukan")
        else:
            processed_label_stems.add(label_path.stem)
            if is_label_empty(label_path):
                status = "processed_empty_label"
                errors.append("label file kosong")

        image, original_size, load_error = load_image(image_path)
        if load_error is not None or image is None:
            failed_count += 1
            records.append(
                ProcessedRecord(
                    filename=str(image_path.relative_to(source_dataset_path)),
                    split=split,
                    status="failed_corrupt_image",
                    original_size="",
                    processed_size="",
                    error=load_error or "image gagal dibaca",
                    source_image_path=image_path,
                    output_image_path=output_images_path / image_path.name,
                )
            )
            continue

        loaded_sizes.append(original_size or (0, 0))

        try:
            processed_image = preprocess_image(image)
        except Exception as exc:
            failed_count += 1
            errors.append(f"gagal preprocessing: {exc}")
            records.append(
                ProcessedRecord(
                    filename=str(image_path.relative_to(source_dataset_path)),
                    split=split,
                    status="failed_processing",
                    original_size=format_size(original_size),
                    processed_size="",
                    error="; ".join(errors),
                    source_image_path=image_path,
                    output_image_path=output_images_path / image_path.name,
                )
            )
            continue

        output_image_path = output_images_path / image_path.name
        if not save_processed_image(processed_image, output_image_path):
            failed_count += 1
            errors.append("gagal menyimpan hasil preprocessing")
            records.append(
                ProcessedRecord(
                    filename=str(image_path.relative_to(source_dataset_path)),
                    split=split,
                    status="failed_processing",
                    original_size=format_size(original_size),
                    processed_size="640x640",
                    error="; ".join(errors),
                    source_image_path=image_path,
                    output_image_path=output_image_path,
                )
            )
            continue

        if label_path is not None:
            copy_label_file(label_path, output_labels_path / label_path.name)

        success_count += 1
        records.append(
            ProcessedRecord(
                filename=str(image_path.relative_to(source_dataset_path)),
                split=split,
                status=status,
                original_size=format_size(original_size),
                processed_size="640x640",
                error="; ".join(errors),
                source_image_path=image_path,
                output_image_path=output_image_path,
            )
        )

    for label_stem, label_path in label_map.items():
        if label_stem in processed_label_stems:
            continue
        records.append(
            ProcessedRecord(
                filename=str(label_path.relative_to(source_dataset_path)),
                split=split,
                status="label_without_image",
                original_size="",
                processed_size="",
                error="label tidak memiliki pasangan image",
                source_image_path=None,
                output_image_path=None,
            )
        )

    return records, success_count, failed_count, loaded_sizes


def copy_data_yaml(source_dataset_path: Path, output_dataset_path: Path) -> bool:
    """Menyalin data.yaml agar output tetap kompatibel dengan YOLOv8."""

    source_yaml = source_dataset_path / "data.yaml"
    output_yaml = output_dataset_path / "data.yaml"
    if not source_yaml.exists():
        logger.warning("data.yaml tidak ditemukan pada source dataset")
        return False

    try:
        shutil.copy2(source_yaml, output_yaml)
        return True
    except Exception as exc:
        logger.error(f"Gagal menyalin data.yaml: {exc}")
        return False


def write_report_csv(report_path: Path, records: List[ProcessedRecord]) -> None:
    """Menyimpan laporan preprocessing ke CSV."""

    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "split", "status", "original_size", "processed_size", "error"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "filename": record.filename,
                    "split": record.split,
                    "status": record.status,
                    "original_size": record.original_size,
                    "processed_size": record.processed_size,
                    "error": record.error,
                }
            )


def compute_statistics(all_sizes: List[Tuple[int, int]]) -> Dict[str, float]:
    """Menghitung statistik ukuran image dari file yang berhasil dibaca."""

    if not all_sizes:
        return {
            "min_width": 0.0,
            "min_height": 0.0,
            "max_width": 0.0,
            "max_height": 0.0,
            "avg_width": 0.0,
            "avg_height": 0.0,
        }

    widths = np.array([size[0] for size in all_sizes], dtype=np.float64)
    heights = np.array([size[1] for size in all_sizes], dtype=np.float64)
    return {
        "min_width": float(widths.min()),
        "min_height": float(heights.min()),
        "max_width": float(widths.max()),
        "max_height": float(heights.max()),
        "avg_width": float(widths.mean()),
        "avg_height": float(heights.mean()),
    }


def visualize_results(
    records: List[ProcessedRecord],
    sample_count: int = 20,
) -> None:
    """Menampilkan perbandingan before-after preprocessing pada sampel acak."""

    eligible_records = [
        record
        for record in records
        if record.source_image_path is not None
        and record.output_image_path is not None
        and record.output_image_path.exists()
        and record.status not in {"failed_corrupt_image", "failed_processing"}
    ]

    if not eligible_records:
        logger.warning("Tidak ada sampel yang dapat divisualisasikan")
        return

    sampled_records = random.sample(eligible_records, min(sample_count, len(eligible_records)))
    rows = len(sampled_records)
    comparison_path = OUTPUT_DATASET_PATH / "preprocessing_comparison.png"
    pair_cols = 4
    pair_rows = int(np.ceil(rows / 2))
    fig, axes = plt.subplots(
        pair_rows,
        pair_cols,
        figsize=(18, max(6, pair_rows * 3.2)),
        constrained_layout=True,
    )

    if pair_rows == 1:
        axes = np.array([axes])

    if pair_rows == 1 and pair_cols == 1:
        axes = np.array([[axes]])

    fig.suptitle(
        "Perbandingan Gambar Asli vs Hasil Preprocessing",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    fig.text(0.25, 0.985, "Asli", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(
        0.75,
        0.985,
        "Hasil Preprocessing",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    for index, record in enumerate(sampled_records):
        original_image, _, _ = load_image(record.source_image_path)
        processed_image, _, _ = load_image(record.output_image_path)
        row_index = index // 2
        col_index = (index % 2) * 2

        if original_image is not None:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        if processed_image is not None:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        short_name = Path(record.filename).name
        if len(short_name) > 42:
            short_name = short_name[:39] + "..."
        axes[row_index, col_index].imshow(original_image)
        axes[row_index, col_index].set_title(f"Asli\n{index + 1}. {short_name}", fontsize=8)
        axes[row_index, col_index].axis("off")

        axes[row_index, col_index + 1].imshow(processed_image)
        axes[row_index, col_index + 1].set_title(f"Preprocessed\n{index + 1}. {short_name}", fontsize=8)
        axes[row_index, col_index + 1].axis("off")

    total_pair_slots = pair_rows * 2
    for empty_index in range(rows, total_pair_slots):
        row_index = empty_index // 2
        col_index = (empty_index % 2) * 2
        axes[row_index, col_index].axis("off")
        axes[row_index, col_index + 1].axis("off")

    plt.savefig(comparison_path, dpi=120, bbox_inches="tight")
    logger.info(f"Visualisasi disimpan ke: {comparison_path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Menjalankan seluruh pipeline preprocessing dataset."""

    logger.info("=" * 80)
    logger.info("PREPROCESSING DATASET UNDERWATER UNTUK YOLOv8 DETECTION")
    logger.info("=" * 80)

    if not SOURCE_DATASET_PATH.exists():
        logger.error(f"Source dataset tidak ditemukan: {SOURCE_DATASET_PATH}")
        return

    create_output_structure(OUTPUT_DATASET_PATH)

    all_records: List[ProcessedRecord] = []
    all_sizes: List[Tuple[int, int]] = []
    total_success = 0
    total_failed = 0

    for split in SPLITS:
        split_records, success_count, failed_count, loaded_sizes = process_split(
            split,
            SOURCE_DATASET_PATH,
            OUTPUT_DATASET_PATH,
        )
        all_records.extend(split_records)
        all_sizes.extend(loaded_sizes)
        total_success += success_count
        total_failed += failed_count

    copy_data_yaml(SOURCE_DATASET_PATH, OUTPUT_DATASET_PATH)
    write_report_csv(REPORT_CSV_PATH, all_records)

    stats = compute_statistics(all_sizes)

    logger.info("")
    logger.info("RINGKASAN PREPROCESSING")
    logger.info("-" * 80)
    logger.info(f"Total gambar berhasil diproses : {total_success}")
    logger.info(f"Total gambar gagal diproses    : {total_failed}")
    logger.info(f"Ukuran minimum image           : {stats['min_width']:.0f}x{stats['min_height']:.0f}")
    logger.info(f"Ukuran maksimum image          : {stats['max_width']:.0f}x{stats['max_height']:.0f}")
    logger.info(f"Ukuran rata-rata image         : {stats['avg_width']:.2f}x{stats['avg_height']:.2f}")
    logger.info(f"Report CSV                     : {REPORT_CSV_PATH.resolve()}")
    logger.info(f"Output dataset                  : {OUTPUT_DATASET_PATH.resolve()}")

    visualize_results(all_records, sample_count=20)

    logger.info("Preprocessing selesai")


if __name__ == "__main__":
    main()
