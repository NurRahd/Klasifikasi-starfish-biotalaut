"""Train cascade models: fish+jellyfish dan starfish terpisah."""
from pathlib import Path
from ultralytics import YOLO
import shutil

print("=" * 80)
print("Training Cascade Detection Models")
print("=" * 80)

# ============================================================================
# MODEL 1: Fish + Jellyfish
# ============================================================================
print("\n1. Training Fish + Jellyfish Model...")
print("-" * 80)

model_fj = YOLO("yolov8n.pt")
data_fj_yaml = Path("cascade_fish_jellyfish_dataset/data.yaml")

results_fj = model_fj.train(
    data=str(data_fj_yaml),
    epochs=10,
    imgsz=640,
    batch=8,
    patience=5,
    device="cpu",
    name="cascade_fish_jellyfish",
)

# Copy best weights
best_fj = Path("runs/detect/cascade_fish_jellyfish/weights/best.pt")
if best_fj.exists():
    shutil.copy2(best_fj, Path("cascade_model_fish_jellyfish_best.pt"))
    print(f"✓ Saved best model: cascade_model_fish_jellyfish_best.pt")

# ============================================================================
# MODEL 2: Starfish
# ============================================================================
print("\n2. Training Starfish Model...")
print("-" * 80)

model_sf = YOLO("yolov8n.pt")
data_sf_yaml = Path("cascade_starfish_dataset/data.yaml")

results_sf = model_sf.train(
    data=str(data_sf_yaml),
    epochs=15,  # Starfish lebih sedikit, perlu lebih banyak epoch
    imgsz=640,
    batch=8,
    patience=5,
    device="cpu",
    name="cascade_starfish",
)

# Copy best weights
best_sf = Path("runs/detect/cascade_starfish/weights/best.pt")
if best_sf.exists():
    shutil.copy2(best_sf, Path("cascade_model_starfish_best.pt"))
    print(f"✓ Saved best model: cascade_model_starfish_best.pt")

print("\n" + "=" * 80)
print("Training selesai!")
print("=" * 80)
