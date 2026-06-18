"""Train 3-class detection model (fish, jellyfish, starfish) untuk cascade system."""
from pathlib import Path
from ultralytics import YOLO
import shutil

print("=" * 80)
print("Training 3-Class Detection Model")
print("=" * 80)

# ============================================================================
# Load pretrained model
# ============================================================================
print("\nLoading pretrained YOLOv8 model...")
model = YOLO("yolov8n.pt")

data_yaml = Path("combined_detection_dataset/data.yaml")

if not data_yaml.exists():
    print(f"\n⚠️  Dataset tidak ditemukan: {data_yaml}")
    exit(1)

print(f"Dataset: {data_yaml}")
print(f"Training dengan 15 epochs...\n")

# ============================================================================
# Train model dengan parameter optimal untuk dataset ini
# ============================================================================
results = model.train(
    data=str(data_yaml),
    epochs=15,  # Lebih banyak dari 5 untuk class imbalance
    imgsz=640,
    batch=8,
    patience=5,
    device="cpu",
    name="train",  # Akan overwrite atau create runs/detect/train
    verbose=True,
)

# ============================================================================
# Verify dan save best model
# ============================================================================
best_model = Path("runs/detect/train/weights/best.pt")
if best_model.exists():
    print(f"\n✓ Model berhasil dilatih!")
    print(f"  Location: {best_model}")
    print(f"  Size: {best_model.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Copy ke root untuk mudah diakses
    shutil.copy2(best_model, Path("trained_3class_model_best.pt"))
    print(f"  Copied to: trained_3class_model_best.pt")
else:
    print(f"\n⚠️  Model tidak ditemukan di {best_model}")

print("\n" + "=" * 80)
print("✓ 3-class model training selesai!")
print("=" * 80)
