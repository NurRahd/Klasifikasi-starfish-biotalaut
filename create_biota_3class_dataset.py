"""
Membuat dataset YOLO khusus biota 3 class:

0 = eel
1 = fish
2 = jellyfish

Script ini membaca label dari preprocessed_dataset, lalu menyalin hanya gambar
yang seluruh objeknya termasuk class 0, 1, atau 2. Gambar yang mengandung
lionfish, lobster, atau star-fish dilewati agar training ulang lebih bersih.
"""

from pathlib import Path
import shutil


SOURCE_ROOT = Path("preprocessed_dataset")
OUTPUT_ROOT = Path("biota_3class_dataset")
KEEP_CLASS_IDS = {0, 1, 2}
CLASS_NAMES = ["eel", "fish", "jellyfish"]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_label_ids(label_path: Path):
    """Baca id class dari file label YOLO."""
    class_ids = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_ids.append(int(float(line.split()[0])))
    return class_ids


def find_image_for_label(images_dir: Path, label_path: Path):
    """Cari file gambar yang pasangannya sama dengan nama file label."""
    for suffix in IMAGE_SUFFIXES:
        image_path = images_dir / f"{label_path.stem}{suffix}"
        if image_path.exists():
            return image_path
    return None


def prepare_output_dirs():
    """Buat struktur folder YOLO train/valid/test."""
    for split in ["train", "valid", "test"]:
        (OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    """Tulis data.yaml untuk training YOLOv8."""
    yaml_text = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: eel\n"
        "  1: fish\n"
        "  2: jellyfish\n"
    )
    (OUTPUT_ROOT / "data.yaml").write_text(yaml_text)


def copy_clean_split(split: str):
    """
    Salin gambar dan label yang bersih untuk satu split.

    Bersih berarti semua class di file label adalah subset dari {0,1,2}.
    """
    source_labels = SOURCE_ROOT / split / "labels"
    source_images = SOURCE_ROOT / split / "images"
    output_labels = OUTPUT_ROOT / split / "labels"
    output_images = OUTPUT_ROOT / split / "images"

    stats = {
        "copied": 0,
        "skipped_other_class": 0,
        "skipped_empty": 0,
        "missing_image": 0,
        "class_objects": {name: 0 for name in CLASS_NAMES},
    }

    for label_path in sorted(source_labels.glob("*.txt")):
        class_ids = read_label_ids(label_path)
        if not class_ids:
            stats["skipped_empty"] += 1
            continue

        if not set(class_ids).issubset(KEEP_CLASS_IDS):
            stats["skipped_other_class"] += 1
            continue

        image_path = find_image_for_label(source_images, label_path)
        if image_path is None:
            stats["missing_image"] += 1
            continue

        shutil.copy2(image_path, output_images / image_path.name)
        shutil.copy2(label_path, output_labels / label_path.name)
        stats["copied"] += 1

        for class_id in class_ids:
            stats["class_objects"][CLASS_NAMES[class_id]] += 1

    return stats


def main():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset tidak ditemukan: {SOURCE_ROOT}")

    prepare_output_dirs()
    write_data_yaml()

    all_stats = {}
    for split in ["train", "valid", "test"]:
        all_stats[split] = copy_clean_split(split)

    print(f"Dataset baru dibuat di: {OUTPUT_ROOT}")
    print("Class final: 0=eel, 1=fish, 2=jellyfish")
    print()

    for split, stats in all_stats.items():
        print(f"[{split}]")
        print(f"  copied images: {stats['copied']}")
        print(f"  skipped other class: {stats['skipped_other_class']}")
        print(f"  skipped empty label: {stats['skipped_empty']}")
        print(f"  missing image: {stats['missing_image']}")
        print(f"  object counts: {stats['class_objects']}")


if __name__ == "__main__":
    main()
