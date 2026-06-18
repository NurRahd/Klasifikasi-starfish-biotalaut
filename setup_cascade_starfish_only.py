"""Persiapkan dataset untuk starfish-only model dalam cascade detection."""
from pathlib import Path
import shutil
import yaml

print("=" * 80)
print("Setting Up Cascade Starfish-Only Dataset")
print("=" * 80)

base_dir = Path("combined_detection_dataset")
output_dir = Path("cascade_starfish_dataset")

print(f"\nSource: {base_dir}")
print(f"Output: {output_dir}")

# ============================================================================
# Create directory structure
# ============================================================================
for split in ["train", "valid", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

print("\n✓ Directory structure created")

# ============================================================================
# Filter dan copy data (hanya class 2 = starfish, remap ke class 0)
# ============================================================================
stats = {"train": 0, "valid": 0, "test": 0}

for split in ["train", "valid", "test"]:
    print(f"\nProcessing {split}...")
    
    images_src = base_dir / split / "images"
    images_dst = output_dir / split / "images"
    labels_src = base_dir / split / "labels"
    labels_dst = output_dir / split / "labels"
    
    # Proses setiap label file
    for label_file in sorted(labels_src.glob("*.txt")):
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        # Filter hanya class 2 (starfish)
        starfish_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and int(parts[0]) == 2:  # class 2 = starfish
                # Remap class 2 -> 0 untuk model starfish-only
                parts[0] = "0"
                starfish_lines.append(" ".join(parts) + "\n")
        
        # Jika ada starfish di image ini, copy image dan label
        if starfish_lines:
            # Copy image
            image_file = images_src / label_file.stem  # find matching image
            for ext in [".jpg", ".png", ".jpeg"]:
                possible_image = images_src / (label_file.stem + ext)
                if possible_image.exists():
                    image_file = possible_image
                    shutil.copy2(image_file, images_dst / image_file.name)
                    break
            
            # Write filtered label
            output_label = labels_dst / label_file.name
            with open(output_label, "w") as f:
                f.writelines(starfish_lines)
            
            stats[split] += 1
    
    print(f"  ✓ Kept {stats[split]} images dengan starfish")

# ============================================================================
# Create data.yaml untuk starfish-only model
# ============================================================================
data_yaml = {
    "path": str(output_dir.absolute()),
    "train": str((output_dir / "train" / "images").absolute()),
    "val": str((output_dir / "valid" / "images").absolute()),
    "test": str((output_dir / "test" / "images").absolute()),
    "nc": 1,  # 1 class only
    "names": ["starfish"],
}

yaml_path = output_dir / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\n✓ Created {yaml_path}")
print(f"  - train: {stats['train']} images")
print(f"  - valid: {stats['valid']} images")
print(f"  - test: {stats['test']} images")

print("\n" + "=" * 80)
print("✓ Cascade starfish-only dataset ready!")
print("=" * 80)
