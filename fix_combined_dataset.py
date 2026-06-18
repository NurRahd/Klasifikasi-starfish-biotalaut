from pathlib import Path
import shutil
from typing import List

SPLITS = ["train", "valid", "test"]
BIOTA_CLASS_NAMES = ['eel', 'fish', 'jellyfish', 'lionfish', 'lobster']
STARFISH_CLASS_NAMES = ['star-fish']
COMBINED_CLASS_NAMES = BIOTA_CLASS_NAMES + STARFISH_CLASS_NAMES
STARFISH_CLASS_OFFSET = len(BIOTA_CLASS_NAMES)
STARFISH_PREFIX = "starfish_"


def remap_label_file(source_path: Path, dest_path: Path, offset: int = 0):
    if offset == 0:
        shutil.copy2(source_path, dest_path)
        return

    lines = []
    with source_path.open('r', encoding='utf-8') as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                class_id = int(parts[0])
            except ValueError:
                lines.append(raw_line)
                continue

            if class_id < offset:
                class_id += offset

            lines.append(" ".join([str(class_id)] + parts[1:]) + "\n")

    dest_path.write_text(''.join(lines), encoding='utf-8')


def write_data_yaml(base_path: Path, names: List[str]):
    yaml_path = base_path / 'data.yaml'
    content = (
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(content, encoding='utf-8')
    print(f"✓ File data.yaml ditulis ke: {yaml_path}")


def fix_dataset_labels(dataset_path: Path):
    if not dataset_path.exists():
        print(f"⚠ Dataset tidak ditemukan: {dataset_path}")
        return 0

    fixed_count = 0
    for split in SPLITS:
        labels_path = dataset_path / split / 'labels'
        if not labels_path.exists():
            print(f"⚠ Folder labels tidak ditemukan: {labels_path}")
            continue

        for label_file in labels_path.glob(f"{STARFISH_PREFIX}*.txt"):
            if not label_file.is_file():
                continue

            temp_path = label_file.with_suffix('.tmp')
            remap_label_file(label_file, temp_path, STARFISH_CLASS_OFFSET)
            temp_path.replace(label_file)
            fixed_count += 1

    print(f"✓ Diperbaiki {fixed_count} file label starfish di {dataset_path}")
    return fixed_count


def main():
    combined_path = Path('combined_dataset')
    preprocessed_path = Path('combined_dataset_preprocessed')

    print('\n' + '=' * 70)
    print('FIX COMBINED YOLO DATASET LABELS & CONFIG')
    print('=' * 70)

    # Fix label files and generate data.yaml for combined and preprocessed
    if combined_path.exists():
        fix_dataset_labels(combined_path)
        write_data_yaml(combined_path, COMBINED_CLASS_NAMES)
    else:
        print(f"⚠ Folder combined_dataset tidak ditemukan: {combined_path}")

    if preprocessed_path.exists():
        fix_dataset_labels(preprocessed_path)
        write_data_yaml(preprocessed_path, COMBINED_CLASS_NAMES)
    else:
        print(f"⚠ Folder combined_dataset_preprocessed tidak ditemukan: {preprocessed_path}")

    print('\n✓ Perbaikan selesai. Pastikan training menggunakan data.yaml di folder target.')


if __name__ == '__main__':
    main()
