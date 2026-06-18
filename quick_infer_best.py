"""Run inference using the latest trained best.pt (or last.pt) in runs/detect/*/weights.
Saves annotated images to results/predictions_best and CSV summary to results/detection.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except Exception as e:
    print('ultralytics not available:', e)
    raise

ROOT = Path('.')
RUNS_DETECT = ROOT / 'runs' / 'detect'
RESULTS_PRED = Path('results') / 'predictions_best'
RESULTS_DET = Path('results') / 'detection'
for p in [RESULTS_PRED, RESULTS_DET]:
    p.mkdir(parents=True, exist_ok=True)

# find latest run folder
best_weight = None
if RUNS_DETECT.exists():
    exps = sorted([d for d in RUNS_DETECT.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    if exps:
        latest = exps[-1]
        wdir = latest / 'weights'
        if (wdir / 'best.pt').exists():
            best_weight = wdir / 'best.pt'
        elif (wdir / 'last.pt').exists():
            best_weight = wdir / 'last.pt'

if best_weight is None:
    print('No best.pt/last.pt found in runs/detect. Aborting.')
    raise SystemExit(1)

print('Using weights:', best_weight)

TEST_IMAGES = Path('combined_detection_dataset') / 'test' / 'images'
if not TEST_IMAGES.exists():
    raise FileNotFoundError(f'Test images not found: {TEST_IMAGES}')

model = YOLO(str(best_weight))

rows = []
image_paths = sorted([p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
for img_path in image_paths:
    print('Predict:', img_path.name)
    res_list = model.predict(source=str(img_path), imgsz=640)
    if not res_list:
        continue
    r = res_list[0]
    # annotated
    try:
        annotated = r.plot()
        out = RESULTS_PRED / img_path.name
        cv2.imwrite(str(out), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    except Exception:
        out = None
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
    rows.append({'image': str(img_path.relative_to(TEST_IMAGES.parent)), 'detections': detections, 'avg_confidence': float(np.mean(confs)) if confs else 0.0, 'per_class': per_class, 'annotated_path': str(out) if out else ''})

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DET / 'inference_summary_best.csv', index=False)
    total = int(df['detections'].sum())
    avg_conf = float(df['avg_confidence'].mean())
    pd.DataFrame([{'total_objects_detected': total, 'detection_confidence_avg': avg_conf}]).to_csv(RESULTS_DET / 'inference_stats_best.csv', index=False)
    print('Saved inference outputs to results/')
else:
    print('No predictions made')
