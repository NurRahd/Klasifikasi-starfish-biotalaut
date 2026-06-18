"""Latih model starfish-only untuk cascade detection."""
from pathlib import Path
from ultralytics import YOLO
import shutil

print("=" * 80)
print("Training Starfish-Only Model")
print("=" * 80)

# ============================================================================
# Load pretrained model dan latih dengan starfish-only dataset
# ============================================================================
print("\nLoading pretrained model...")
model_sf = YOLO("yolov8n.pt")

data_sf_yaml = Path("cascade_starfish_dataset/data.yaml")

if not data_sf_yaml.exists():
    print(f"\n⚠️  Dataset belum disiapkan!")
    print(f"Silakan jalankan: python setup_cascade_starfish_only.py")
    exit(1)

print(f"Training with {data_sf_yaml}...\n")

results_sf = model_sf.train(
    data=str(data_sf_yaml),
    epochs=20,  # Lebih banyak untuk dataset kecil
    imgsz=640,
    batch=8,
    patience=5,
    device="cpu",
    name="cascade_starfish_only",
    verbose=True,
)

# ============================================================================
# Save best weights
# ============================================================================
best_sf = Path("runs/detect/cascade_starfish_only/weights/best.pt")
if best_sf.exists():
    shutil.copy2(best_sf, Path("cascade_model_starfish_best.pt"))
    print(f"\n✓ Best model saved: cascade_model_starfish_best.pt")
else:
    print(f"\n⚠️  Best model tidak ditemukan di {best_sf}")

print("\n" + "=" * 80)
print("✓ Starfish model training selesai!")
print("=" * 80)
