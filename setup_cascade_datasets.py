"""Setup dataset untuk cascade detection: model fish+jellyfish dan model starfish terpisah."""
from pathlib import Path
import yaml
import shutil

# ============================================================================
# 1. SETUP DATASET FISH + JELLYFISH (tanpa starfish)
# ============================================================================
print("=" * 80)
print("1. Setting up Fish + Jellyfish dataset (class 0 dan 1 saja)")
print("=" * 80)

dataset_fj_dir = Path("cascade_fish_jellyfish_dataset")
dataset_fj_dir.mkdir(exist_ok=True)

# Buat data.yaml untuk fish+jellyfish
data_fj_yaml = {
    "path": str(dataset_fj_dir.absolute()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": ["fish", "jellyfish"]
}

# Copy folder dan filter label
source_dataset = Path("combined_detection_dataset")
for split in ["train", "valid", "test"]:
    src_img = source_dataset / split / "images"
    src_lbl = source_dataset / split / "labels"
    dst_img = dataset_fj_dir / split / "images"
    dst_lbl = dataset_fj_dir / split / "labels"
    
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy images dan filter label (hapus class 2 = starfish)
    kept_count = 0
    removed_count = 0
    for src_lbl_file in src_lbl.glob("*.txt"):
        content = src_lbl_file.read_text().strip()
        if not content:
            continue
        
        # Filter: hapus deteksi dengan class 2
        filtered_lines = []
        for line in content.splitlines():
            parts = line.split()
            if parts and parts[0] != "2":  # Keep class 0 dan 1, hapus class 2
                filtered_lines.append(line)
            elif parts and parts[0] == "2":
                removed_count += 1
        
        if filtered_lines:  # Only copy image jika ada deteksi yang tersisa
            img_name = src_lbl_file.stem.replace(".txt", "")
            # Find corresponding image
            for img_file in src_img.glob(f"{img_name}*"):
                shutil.copy2(img_file, dst_img / img_file.name)
                break
            # Write filtered label
            dst_lbl_file = dst_lbl / src_lbl_file.name
            dst_lbl_file.write_text("\n".join(filtered_lines))
            kept_count += 1
    
    print(f"  {split}: {kept_count} files kept, {removed_count} starfish objects removed")

# Save data.yaml
with open(dataset_fj_dir / "data.yaml", "w") as f:
    yaml.dump(data_fj_yaml, f, default_flow_style=False)
print(f"✓ Saved data.yaml untuk fish+jellyfish model\n")

# ============================================================================
# 2. SETUP DATASET STARFISH ONLY (class 2 diubah jadi class 0)
# ============================================================================
print("=" * 80)
print("2. Setting up Starfish-only dataset (class 2 diubah menjadi class 0)")
print("=" * 80)

dataset_sf_dir = Path("cascade_starfish_dataset")
dataset_sf_dir.mkdir(exist_ok=True)

# Buat data.yaml untuk starfish
data_sf_yaml = {
    "path": str(dataset_sf_dir.absolute()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["starfish"]
}

# Copy folder dan filter label (ubah class 2 menjadi 0)
for split in ["train", "valid", "test"]:
    src_img = source_dataset / split / "images"
    src_lbl = source_dataset / split / "labels"
    dst_img = dataset_sf_dir / split / "images"
    dst_lbl = dataset_sf_dir / split / "labels"
    
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    
    kept_count = 0
    for src_lbl_file in src_lbl.glob("*.txt"):
        content = src_lbl_file.read_text().strip()
        if not content:
            continue
        
        # Filter: ambil hanya class 2 dan ubah menjadi class 0
        filtered_lines = []
        for line in content.splitlines():
            parts = line.split()
            if parts and parts[0] == "2":  # Ambil starfish
                parts[0] = "0"  # Ubah class ID ke 0
                filtered_lines.append(" ".join(parts))
        
        if filtered_lines:  # Only copy image jika ada starfish
            img_name = src_lbl_file.stem.replace(".txt", "")
            # Find corresponding image
            for img_file in src_img.glob(f"{img_name}*"):
                shutil.copy2(img_file, dst_img / img_file.name)
                break
            # Write filtered label
            dst_lbl_file = dst_lbl / src_lbl_file.name
            dst_lbl_file.write_text("\n".join(filtered_lines))
            kept_count += 1
    
    print(f"  {split}: {kept_count} files dengan starfish")

# Save data.yaml
with open(dataset_sf_dir / "data.yaml", "w") as f:
    yaml.dump(data_sf_yaml, f, default_flow_style=False)
print(f"✓ Saved data.yaml untuk starfish model\n")

print("=" * 80)
print("Setup selesai!")
print(f"Folder dataset:\n  - {dataset_fj_dir} (fish + jellyfish)\n  - {dataset_sf_dir} (starfish)")
print("=" * 80)
