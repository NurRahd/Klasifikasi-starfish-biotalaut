"""
Script untuk menggabungkan dua dataset YOLOv8 (biota_dataset dan starfish_dataset)
ke dalam satu folder combined_dataset dengan struktur yang terorganisir.

Fitur:
- Menggabungkan train, valid, dan test splits
- Menambahkan prefix 'starfish_' pada file dari starfish_only untuk menghindari konflik nama
- Membuat folder tujuan otomatis jika belum ada
- Menampilkan statistik jumlah file yang berhasil dipindahkan
"""

from pathlib import Path
import shutil
from typing import Dict, Tuple, List


# ============================================================================
# KONFIGURASI DATASET
# ============================================================================

# Path ke kedua dataset sumber
BIOTA_DATASET_PATH = Path("biota_laut/Underwater Marine Species.v2i.yolov8")
STARFISH_DATASET_PATH = Path("starfish_only")

# Path untuk dataset gabungan
COMBINED_DATASET_PATH = Path("combined_dataset")

# Nama-nama split yang akan diproses
SPLITS = ["train", "valid", "test"]

# Kelas untuk dataset biota dan starfish
BIOTA_CLASS_NAMES = ['eel', 'fish', 'jellyfish', 'lionfish', 'lobster']
STARFISH_CLASS_NAMES = ['star-fish']
COMBINED_CLASS_NAMES = BIOTA_CLASS_NAMES + STARFISH_CLASS_NAMES

# Offset class ID untuk starfish dataset agar tidak bentrok dengan kelas biota
STARFISH_CLASS_OFFSET = len(BIOTA_CLASS_NAMES)


# ============================================================================
# FUNGSI UTILITAS
# ============================================================================

def create_directory_structure(base_path: Path) -> Dict[str, Path]:
    """
    Membuat struktur folder untuk combined_dataset.
    
    Args:
        base_path (Path): Path ke folder combined_dataset
        
    Returns:
        Dict[str, Path]: Dictionary berisi path untuk semua folder yang dibuat
    """
    print("\n" + "=" * 70)
    print("MEMBUAT STRUKTUR FOLDER COMBINED_DATASET")
    print("=" * 70)
    
    folder_structure = {}
    
    for split in SPLITS:
        # Membuat folder images dan labels untuk setiap split
        images_path = base_path / split / "images"
        labels_path = base_path / split / "labels"
        
        # Membuat folder otomatis jika belum ada (parents=True membuat parent folders)
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        
        folder_structure[f"{split}_images"] = images_path
        folder_structure[f"{split}_labels"] = labels_path
        
        print(f"✓ Dibuat folder: {images_path}")
        print(f"✓ Dibuat folder: {labels_path}")
    
    return folder_structure


def copy_files_from_dataset(
    source_base_path: Path,
    folder_structure: Dict[str, Path],
    prefix: str = "",
    label_offset: int = 0
) -> Dict[str, int]:
    """
    Menyalin file gambar dan label dari satu dataset ke combined_dataset.
    
    Args:
        source_base_path (Path): Path ke folder dataset sumber
        folder_structure (Dict[str, Path]): Dictionary path folder tujuan
        prefix (str): Prefix untuk nama file (untuk menghindari konflik)
        label_offset (int): Offset class id pada label (misal starfish -> offset 5)
        
    Returns:
        Dict[str, int]: Dictionary berisi jumlah file yang berhasil dipindahkan
    """
    file_counts = {split: 0 for split in SPLITS}
    
    print("\n" + "-" * 70)
    print(f"MEMPROSES DATASET: {source_base_path.name}")
    print("-" * 70)
    
    for split in SPLITS:
        # Path folder images dan labels dari dataset sumber
        source_images_folder = source_base_path / split / "images"
        source_labels_folder = source_base_path / split / "labels"
        
        # Path folder tujuan
        dest_images_folder = folder_structure[f"{split}_images"]
        dest_labels_folder = folder_structure[f"{split}_labels"]
        
        # Cek apakah folder sumber ada
        if not source_images_folder.exists() or not source_labels_folder.exists():
            print(f"⚠ Folder sumber tidak ditemukan: {split}")
            continue
        
        print(f"\n  Memproses split '{split}':")
        
        # Proses file-file gambar
        try:
            # Mengambil semua file gambar dari folder sumber
            image_files = list(source_images_folder.glob("*"))
            
            for image_file in image_files:
                if image_file.is_file():
                    # Tambahkan prefix pada nama file jika diperlukan
                    new_filename = f"{prefix}{image_file.name}" if prefix else image_file.name
                    dest_image_path = dest_images_folder / new_filename
                    
                    # Copy file gambar
                    shutil.copy2(image_file, dest_image_path)
                    file_counts[split] += 1
            
            print(f"    ✓ Berhasil menyalin {len(image_files)} file gambar")
            
        except Exception as e:
            print(f"    ✗ Error saat menyalin file gambar: {e}")
        
        # Proses file-file label
        try:
            # Mengambil semua file label dari folder sumber
            label_files = list(source_labels_folder.glob("*"))
            
            for label_file in label_files:
                if label_file.is_file():
                    # Tambahkan prefix pada nama file jika diperlukan
                    # Pastikan label file sesuai dengan gambar file
                    new_filename = f"{prefix}{label_file.name}" if prefix else label_file.name
                    dest_label_path = dest_labels_folder / new_filename
                    
                    # Copy file label, dengan remapping jika diperlukan
                    if label_offset > 0:
                        remap_label_file(label_file, dest_label_path, label_offset)
                    else:
                        shutil.copy2(label_file, dest_label_path)
            
            print(f"    ✓ Berhasil menyalin {len(label_files)} file label")
            
        except Exception as e:
            print(f"    ✗ Error saat menyalin file label: {e}")
    
    return file_counts


def remap_label_file(source_path: Path, dest_path: Path, offset: int = 0):
    """
    Menyalin dan meremap class id pada file label YOLO.

    Args:
        source_path (Path): Path file label sumber
        dest_path (Path): Path file label tujuan
        offset (int): Offset class id yang akan ditambahkan
    """
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
    """
    Menulis file data.yaml untuk dataset gabungan.

    Args:
        base_path (Path): Path root dataset
        names (List[str]): Daftar nama kelas
    """
    yaml_path = base_path / 'data.yaml'
    content = (
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(content, encoding='utf-8')
    print(f"✓ File data.yaml berhasil dibuat: {yaml_path}")


def print_summary(counts_biota: Dict[str, int], counts_starfish: Dict[str, int]):
    """
    Menampilkan ringkasan jumlah file yang berhasil dipindahkan.
    
    Args:
        counts_biota (Dict[str, int]): Jumlah file dari biota dataset
        counts_starfish (Dict[str, int]): Jumlah file dari starfish dataset
    """
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL PENGGABUNGAN DATASET")
    print("=" * 70)
    
    total_biota = sum(counts_biota.values())
    total_starfish = sum(counts_starfish.values())
    total_combined = total_biota + total_starfish
    
    print("\nJumlah file per split (BIOTA DATASET):")
    for split in SPLITS:
        print(f"  {split.upper():5s}: {counts_biota[split]:4d} file (images + labels)")
    print(f"  TOTAL : {total_biota:4d} file")
    
    print("\nJumlah file per split (STARFISH DATASET dengan prefix 'starfish_'):")
    for split in SPLITS:
        print(f"  {split.upper():5s}: {counts_starfish[split]:4d} file (images + labels)")
    print(f"  TOTAL : {total_starfish:4d} file")
    
    print("\n" + "-" * 70)
    print(f"TOTAL GABUNGAN: {total_combined:4d} file")
    print("-" * 70)
    
    # Breakdown per folder
    print("\nBreakdown per folder split:")
    for split in SPLITS:
        combined_count = counts_biota[split] + counts_starfish[split]
        print(f"  {split.upper():5s}: {combined_count:4d} file")


def main():
    """
    Fungsi utama untuk menjalankan proses penggabungan dataset.
    """
    print("\n" + "=" * 70)
    print("SCRIPT PENGGABUNGAN DATASET YOLOv8")
    print("=" * 70)
    
    # Cek apakah dataset sumber ada
    if not BIOTA_DATASET_PATH.exists():
        print(f"✗ Dataset biota tidak ditemukan: {BIOTA_DATASET_PATH}")
        return
    
    if not STARFISH_DATASET_PATH.exists():
        print(f"✗ Dataset starfish tidak ditemukan: {STARFISH_DATASET_PATH}")
        return
    
    print(f"\n✓ Dataset biota ditemukan: {BIOTA_DATASET_PATH}")
    print(f"✓ Dataset starfish ditemukan: {STARFISH_DATASET_PATH}")
    
    try:
        # 1. Membuat struktur folder untuk combined_dataset
        folder_structure = create_directory_structure(COMBINED_DATASET_PATH)
        
        # 2. Menyalin file dari dataset biota (tanpa prefix)
        counts_biota = copy_files_from_dataset(
            BIOTA_DATASET_PATH,
            folder_structure,
            prefix="",
            label_offset=0
        )
        
        # 3. Menyalin file dari dataset starfish (dengan prefix "starfish_" dan remap class id)
        counts_starfish = copy_files_from_dataset(
            STARFISH_DATASET_PATH,
            folder_structure,
            prefix="starfish_",
            label_offset=STARFISH_CLASS_OFFSET
        )
        
        # 4. Menulis konfigurasi combined data.yaml
        write_data_yaml(COMBINED_DATASET_PATH, COMBINED_CLASS_NAMES)
        
        # 5. Menampilkan ringkasan hasil
        print_summary(counts_biota, counts_starfish)
        
        print("\n" + "=" * 70)
        print("✓ PROSES PENGGABUNGAN DATASET SELESAI!")
        print(f"✓ Dataset gabungan tersimpan di: {COMBINED_DATASET_PATH.absolute()}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
