"""Combined inference: existing 3-class model (without starfish) + starfish-only model."""
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import csv

print("=" * 80)
print("Cascade Inference: Fish+Jellyfish (existing) + Starfish (new)")
print("=" * 80)

# ============================================================================
# Load models
# ============================================================================
print("\nLoading models...")

# Model 1: Trained 3-class model (akan filter class 2)
weights_combined = Path("trained_3class_model_best.pt")
if weights_combined.exists():
    model_combined = YOLO(str(weights_combined))
    print(f"✓ Loaded trained 3-class model: {weights_combined}")
else:
    print(f"⚠️  Trained model tidak ditemukan!")
    weights_alt = Path("runs/detect/train-2/weights/best.pt")
    if weights_alt.exists():
        model_combined = YOLO(str(weights_alt))
        print(f"✓ Loaded from: {weights_alt}")
    else:
        print(f"⚠️  Menggunakan pretrained weights...")
        model_combined = YOLO("yolov8n.pt")

# Model 2: Starfish-only model (baru)
model_starfish = None
weights_starfish = Path("cascade_model_starfish_best.pt")
if weights_starfish.exists():
    model_starfish = YOLO(str(weights_starfish))
    print(f"✓ Loaded starfish model: {weights_starfish}")
else:
    print(f"\n⚠️  Starfish model belum tersedia!")
    print(f"Silakan latih dulu: python setup_cascade_starfish_only.py")
    print(f"                  python train_cascade_starfish_only.py")
    exit(1)

# ============================================================================
# Inference settings
# ============================================================================
conf = 0.25  # Confidence threshold
iou = 0.45   # IOU threshold

# Test dataset
test_images_dir = Path("combined_detection_dataset/test/images")
test_images = sorted(list(test_images_dir.glob("*.*")))

print(f"\nInferencing pada {len(test_images)} test images...")
print(f"Confidence threshold: {conf}, IOU threshold: {iou}\n")

# ============================================================================
# Run combined inference
# ============================================================================
results_combined = []
detection_stats = defaultdict(int)

for i, img_path in enumerate(test_images):
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(test_images)}]")
    
    # -----------------------------------------------------------------------
    # Get detections from combined model (filter class 2)
    # -----------------------------------------------------------------------
    results_c = model_combined(img_path, conf=conf, iou=iou, verbose=False)
    detections_combined = []
    
    if results_c[0].boxes is not None:
        for box in results_c[0].boxes:
            class_id = int(box.cls.item())
            
            # FILTER: Ignore class 2 (starfish) dari model ini
            if class_id != 2:
                detections_combined.append({
                    "class": class_id,
                    "conf": float(box.conf.item()),
                    "box": box.xyxy[0].cpu().numpy(),
                    "source": "combined",
                })
                detection_stats[f"class_{class_id}_combined"] += 1
    
    # -----------------------------------------------------------------------
    # Get detections dari starfish model
    # -----------------------------------------------------------------------
    detections_starfish = []
    results_s = model_starfish(img_path, conf=conf, iou=iou, verbose=False)
    
    if results_s[0].boxes is not None:
        for box in results_s[0].boxes:
            # Starfish model output class 0, remap ke class 2
            detections_starfish.append({
                "class": 2,  # Remap 0 -> 2 (starfish)
                "conf": float(box.conf.item()),
                "box": box.xyxy[0].cpu().numpy(),
                "source": "starfish",
            })
            detection_stats["class_2_starfish"] += 1
    
    # -----------------------------------------------------------------------
    # Combine results
    # -----------------------------------------------------------------------
    all_detections = detections_combined + detections_starfish
    
    results_combined.append({
        "image": img_path.name,
        "detections": all_detections,
        "count_class_0": sum(1 for d in all_detections if d["class"] == 0),
        "count_class_1": sum(1 for d in all_detections if d["class"] == 1),
        "count_class_2": sum(1 for d in all_detections if d["class"] == 2),
        "total": len(all_detections),
    })

# ============================================================================
# Statistics
# ============================================================================
print("\n" + "=" * 80)
print("DETECTION STATISTICS")
print("=" * 80)

total_detections = sum(r["total"] for r in results_combined)
images_with_detections = sum(1 for r in results_combined if r["total"] > 0)

print(f"\nTotal detections: {total_detections}")
print(f"Images with detections: {images_with_detections}/{len(test_images)}")

print("\nDetections by class:")
class_0_total = sum(r["count_class_0"] for r in results_combined)
class_1_total = sum(r["count_class_1"] for r in results_combined)
class_2_total = sum(r["count_class_2"] for r in results_combined)

print(f"  Class 0 (fish):      {class_0_total:6d} ({100*class_0_total/total_detections:.1f}%)" if total_detections > 0 else "  Class 0 (fish):      0")
print(f"  Class 1 (jellyfish): {class_1_total:6d} ({100*class_1_total/total_detections:.1f}%)" if total_detections > 0 else "  Class 1 (jellyfish): 0")
print(f"  Class 2 (starfish):  {class_2_total:6d} ({100*class_2_total/total_detections:.1f}%)" if total_detections > 0 else "  Class 2 (starfish):  0")

print("\nDetection sources:")
for key in sorted(detection_stats.keys()):
    print(f"  {key}: {detection_stats[key]}")

# ============================================================================
# Save results to CSV
# ============================================================================
csv_path = Path("cascade_inference_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "class_0_fish", "class_1_jellyfish", "class_2_starfish", "total"])
    writer.writeheader()
    for r in results_combined:
        writer.writerow({
            "image": r["image"],
            "class_0_fish": r["count_class_0"],
            "class_1_jellyfish": r["count_class_1"],
            "class_2_starfish": r["count_class_2"],
            "total": r["total"],
        })

print(f"\n✓ Results saved: {csv_path}")

print("\n" + "=" * 80)
print("✓ Cascade inference complete!")
print("=" * 80)
