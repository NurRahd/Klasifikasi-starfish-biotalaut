"""Run inference-only on test set using pretrained YOLOv8n.

This script is minimal and avoids training to produce quick results.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics not found:", e)
    raise

RESULTS_DIR = Path("results")
RESULTS_DETECTION = RESULTS_DIR / "detection"
RESULTS_PREDICTIONS = RESULTS_DIR / "predictions"
RESULTS_PLOTS = RESULTS_DIR / "plots"

for p in [RESULTS_DIR, RESULTS_DETECTION, RESULTS_PREDICTIONS, RESULTS_PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

WEIGHTS = Path("yolov8n.pt")
if not WEIGHTS.exists():
    print(f"weights {WEIGHTS} not found, yolov8 will download it automatically if needed.")

TEST_IMAGES = Path("preprocessed_dataset") / "test" / "images"
if not TEST_IMAGES.exists():
    raise FileNotFoundError(f"Test images folder not found: {TEST_IMAGES}")

model = YOLO(str(WEIGHTS))

image_paths = sorted([p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
rows = []
for img_path in image_paths:
    print("Predict:", img_path.name)
    res_list = model.predict(source=str(img_path), imgsz=640)
    if not res_list:
        continue
    r = res_list[0]
    # save annotated image
    try:
        annotated = r.plot()
        out_path = RESULTS_PREDICTIONS / img_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    except Exception:
        out_path = None

    detections = 0
    confs = []
    per_class = {}
    try:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
            detections += 1
            confs.append(conf)
            per_class[cls] = per_class.get(cls, 0) + 1
    except Exception:
        pass

    rows.append({
        'image': str(img_path.relative_to(TEST_IMAGES.parent)),
        'detections': detections,
        'avg_confidence': float(np.mean(confs)) if confs else 0.0,
        'per_class': per_class,
        'annotated_path': str(out_path) if out_path else '',
    })

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DETECTION / 'inference_summary.csv', index=False)
    # save simple stats
    total = int(df['detections'].sum())
    avg_conf = float(df['avg_confidence'].mean())
    pd.DataFrame([{'total_objects_detected': total, 'detection_confidence_avg': avg_conf}]).to_csv(RESULTS_DETECTION / 'inference_stats.csv', index=False)
    print('Saved inference CSVs and annotated images to results/')
else:
    print('No predictions made.')
