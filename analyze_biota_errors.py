"""
Generate visual error samples for the 3-class biota model.

Outputs:
    results/biota_error_analysis/<split>/false_positive
    results/biota_error_analysis/<split>/false_negative
    results/biota_error_analysis/<split>/low_iou
    results/biota_error_analysis/<split>/summary.json
    results/biota_error_analysis/<split>/cases.csv

Classes:
    0 = eel
    1 = fish
    2 = jellyfish

Example:
    .\\.venv\\Scripts\\python.exe analyze_biota_errors.py --split test --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {
    0: "eel",
    1: "fish",
    2: "jellyfish",
}
COLORS = {
    "gt": (60, 210, 80),
    "tp": (255, 180, 0),
    "fp": (40, 40, 240),
    "fn": (0, 120, 255),
    "low_iou": (220, 80, 220),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path_for_image(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def read_labels(label_path: Path, width: int, height: int):
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_w, box_h = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if class_id not in CLASS_NAMES:
            continue
        x1 = (cx - box_w / 2) * width
        y1 = (cy - box_h / 2) * height
        x2 = (cx + box_w / 2) * width
        y2 = (cy + box_h / 2) * height
        labels.append(
            {
                "class_id": class_id,
                "bbox": [x1, y1, x2, y2],
                "matched": False,
            }
        )
    return labels


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def predict(model, image_path: Path, conf: float, imgsz: int):
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    detections = []
    if not results or results[0].boxes is None:
        return detections
    for box in results[0].boxes:
        class_id = int(box.cls[0].cpu().item())
        if class_id not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detections.append(
            {
                "class_id": class_id,
                "confidence": float(box.conf[0].cpu().item()),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return detections


def match_detections(detections, labels, iou_threshold: float):
    cases = []
    used_labels = set()
    for pred_index, detection in enumerate(
        sorted(detections, key=lambda item: item["confidence"], reverse=True)
    ):
        best_iou = 0.0
        best_label_index = None
        best_same_class_index = None
        best_same_class_iou = 0.0
        for label_index, label in enumerate(labels):
            iou = bbox_iou(detection["bbox"], label["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_label_index = label_index
            if label["class_id"] == detection["class_id"] and iou > best_same_class_iou:
                best_same_class_iou = iou
                best_same_class_index = label_index

        if (
            best_same_class_index is not None and
            best_same_class_index not in used_labels and
            best_same_class_iou >= iou_threshold
        ):
            used_labels.add(best_same_class_index)
            cases.append(("tp", pred_index, best_same_class_index, best_same_class_iou))
        elif best_same_class_index is not None and best_same_class_iou > 0:
            cases.append(("low_iou", pred_index, best_same_class_index, best_same_class_iou))
        else:
            cases.append(("false_positive", pred_index, best_label_index, best_iou))

    for label_index, label in enumerate(labels):
        if label_index not in used_labels:
            cases.append(("false_negative", None, label_index, 0.0))
    return cases


def draw_box(image, bbox, label, color):
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    y_text = max(18, y1 - 6)
    cv2.putText(
        image,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def render_case(image_path: Path, detections, labels, case, output_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return
    status, pred_index, label_index, iou = case

    for label in labels:
        draw_box(
            image,
            label["bbox"],
            f"GT {CLASS_NAMES[label['class_id']]}",
            COLORS["gt"],
        )

    for idx, detection in enumerate(detections):
        color = COLORS["tp"]
        tag = "PRED"
        if pred_index == idx:
            color = COLORS.get(status, COLORS["fp"])
            tag = status.upper().replace("_", " ")
        draw_box(
            image,
            detection["bbox"],
            f"{tag} {CLASS_NAMES[detection['class_id']]} {detection['confidence']:.2f}",
            color,
        )

    if label_index is not None and status == "false_negative":
        label = labels[label_index]
        draw_box(image, label["bbox"], f"FN {CLASS_NAMES[label['class_id']]}", COLORS["fn"])

    cv2.putText(
        image,
        f"case={status} iou={iou:.3f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"case={status} iou={iou:.3f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def collect_images(dataset_root: Path, split: str):
    split_dir = "valid" if split == "val" else split
    images_dir = dataset_root / split_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {images_dir}")
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def analyze(args):
    dataset_root = Path(args.dataset)
    output_root = Path(args.output_dir) / args.split
    for folder in ["false_positive", "false_negative", "low_iou"]:
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    image_paths = collect_images(dataset_root, args.split)
    if args.limit:
        image_paths = image_paths[:args.limit]

    rows = []
    summary = {
        "images": 0,
        "labels": 0,
        "predictions": 0,
        "tp": 0,
        "false_positive": 0,
        "false_negative": 0,
        "low_iou": 0,
    }

    saved_count = {
        "false_positive": 0,
        "false_negative": 0,
        "low_iou": 0,
    }

    for image_number, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(label_path_for_image(image_path), width, height)
        detections = predict(model, image_path, args.conf, args.imgsz)
        cases = match_detections(detections, labels, args.iou)

        summary["images"] += 1
        summary["labels"] += len(labels)
        summary["predictions"] += len(detections)

        for case_number, case in enumerate(cases, 1):
            status, pred_index, label_index, iou = case
            if status == "tp":
                summary["tp"] += 1
                continue

            summary[status] += 1
            pred_class = ""
            pred_conf = ""
            label_class = ""
            if pred_index is not None and pred_index < len(detections):
                pred = sorted(detections, key=lambda item: item["confidence"], reverse=True)[pred_index]
                pred_class = CLASS_NAMES[pred["class_id"]]
                pred_conf = f"{pred['confidence']:.4f}"
            if label_index is not None and label_index < len(labels):
                label_class = CLASS_NAMES[labels[label_index]["class_id"]]

            rows.append(
                {
                    "image": str(image_path),
                    "case": status,
                    "pred_class": pred_class,
                    "pred_confidence": pred_conf,
                    "label_class": label_class,
                    "iou": f"{iou:.4f}",
                    "labels": len(labels),
                    "predictions": len(detections),
                }
            )

            if saved_count[status] < args.max_save:
                output_name = f"{image_path.stem}_{case_number:02d}_{status}.jpg"
                render_case(
                    image_path,
                    sorted(detections, key=lambda item: item["confidence"], reverse=True),
                    labels,
                    case,
                    output_root / status / output_name,
                )
                saved_count[status] += 1

        if not args.quiet:
            print(
                f"[{image_number}/{len(image_paths)}] {image_path.name}: "
                f"labels={len(labels)} preds={len(detections)}"
            )

    csv_path = output_root / "cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "case",
                "pred_class",
                "pred_confidence",
                "label_class",
                "iou",
                "labels",
                "predictions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, summary_path, csv_path, output_root


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze biota model FP/FN/low-IoU samples.")
    parser.add_argument("--dataset", default="biota_3class_dataset")
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--model", default="trained_biota_3class_model_best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-save", type=int, default=80)
    parser.add_argument("--output-dir", default="results/biota_error_analysis")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary, summary_path, csv_path, output_root = analyze(args)
    print("\nBiota error analysis complete")
    print("=" * 60)
    print(f"Images         : {summary['images']}")
    print(f"Labels         : {summary['labels']}")
    print(f"Predictions    : {summary['predictions']}")
    print(f"TP             : {summary['tp']}")
    print(f"False positive : {summary['false_positive']}")
    print(f"False negative : {summary['false_negative']}")
    print(f"Low IoU        : {summary['low_iou']}")
    print(f"Output folder  : {output_root}")
    print(f"Summary        : {summary_path}")
    print(f"Cases CSV      : {csv_path}")


if __name__ == "__main__":
    main()
