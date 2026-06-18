"""
Evaluate dual-model underwater detection mAP on labeled dataset images.

Final class mapping:
    0 = eel
    1 = fish
    2 = jellyfish
    3 = starfish

Examples:
    .\\.venv\\Scripts\\python.exe evaluate_map_dataset.py --dataset biota_3class_dataset --split test
    .\\.venv\\Scripts\\python.exe evaluate_map_dataset.py --dataset starfish_only --split valid
    .\\.venv\\Scripts\\python.exe evaluate_map_dataset.py --dataset all --split test
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np

from dual_model_detection_system import (
    DetectionConfig,
    apply_color_class_corrections,
    load_models,
    merge_detections,
    preprocess_image,
    run_parallel_detection,
)


FINAL_CLASS_NAMES = {
    0: "eel",
    1: "fish",
    2: "jellyfish",
    3: "starfish",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_config(args) -> DetectionConfig:
    config = DetectionConfig()
    config.CLASS_NAMES = FINAL_CLASS_NAMES.copy()
    config.STARFISH_CLASS_ID = 3
    config.CONF_BIOTA = args.conf_biota
    config.CONF_STARFISH = args.conf_starfish
    config.ENABLE_COLOR_CLASS_CORRECTION = not args.disable_color_correction
    return config


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        return image_path.with_suffix(".txt")
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def map_label_class(dataset_root: Path, class_id: int) -> int:
    name = dataset_root.name.lower()
    if name in {"biota_3class_dataset", "combined_dataset"}:
        return class_id
    if name in {"starfish_only", "cascade_starfish_dataset"}:
        return 3
    if name in {"combined_detection_dataset", "preprocessed_detection_dataset"}:
        # Dataset labels: 0 fish, 1 jellyfish, 2 starfish.
        return {0: 1, 1: 2, 2: 3}.get(class_id, class_id)
    return class_id


def read_yolo_labels(label_path: Path, dataset_root: Path, image_size: int):
    labels = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue

        final_class_id = map_label_class(dataset_root, class_id)
        x1 = (cx - width / 2) * image_size
        y1 = (cy - height / 2) * image_size
        x2 = (cx + width / 2) * image_size
        y2 = (cy + height / 2) * image_size
        labels.append(
            {
                "class_id": final_class_id,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return labels


def collect_images(dataset_root: Path, split: str):
    images_dir = dataset_root / split / "images"
    if not images_dir.exists() and split == "val":
        images_dir = dataset_root / "valid" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {images_dir}")

    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def selected_datasets(dataset_arg: str):
    if dataset_arg != "all":
        return [Path(dataset_arg)]
    return [
        Path("biota_3class_dataset"),
        Path("starfish_only"),
        Path("combined_detection_dataset"),
    ]


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


def average_precision(predictions, labels, class_id: int, iou_threshold: float) -> float | None:
    class_labels = [
        label for label in labels
        if label["class_id"] == class_id
    ]
    if not class_labels:
        return None

    class_predictions = sorted(
        [prediction for prediction in predictions if prediction["class_id"] == class_id],
        key=lambda item: item["confidence"],
        reverse=True,
    )
    matched = set()
    tp = []
    fp = []

    for prediction in class_predictions:
        best_iou = 0.0
        best_key = None
        for label_index, label in enumerate(class_labels):
            if label["image_id"] != prediction["image_id"]:
                continue
            key = (label["image_id"], label_index)
            if key in matched:
                continue
            iou = bbox_iou(prediction["bbox"], label["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_key = key

        if best_key is not None and best_iou >= iou_threshold:
            matched.add(best_key)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(class_labels)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    recall_points = np.concatenate(([0.0], recalls, [1.0]))
    precision_points = np.concatenate(([1.0], precisions, [0.0]))
    for idx in range(len(precision_points) - 2, -1, -1):
        precision_points[idx] = max(precision_points[idx], precision_points[idx + 1])

    changed = np.where(recall_points[1:] != recall_points[:-1])[0]
    return float(
        np.sum(
            (recall_points[changed + 1] - recall_points[changed]) *
            precision_points[changed + 1]
        )
    )


def precision_recall_f1(predictions, labels, iou_threshold: float = 0.5):
    labels_by_image_class = {}
    for idx, label in enumerate(labels):
        key = (label["image_id"], label["class_id"])
        labels_by_image_class.setdefault(key, []).append((idx, label))

    matched = set()
    tp = 0
    fp = 0
    ious = []

    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        key = (prediction["image_id"], prediction["class_id"])
        best_iou = 0.0
        best_label_index = None
        for label_index, label in labels_by_image_class.get(key, []):
            if label_index in matched:
                continue
            iou = bbox_iou(prediction["bbox"], label["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_label_index = label_index

        if best_label_index is not None and best_iou >= iou_threshold:
            matched.add(best_label_index)
            tp += 1
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(labels) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return precision, recall, f1, mean_iou, tp, fp, fn


def compute_map(predictions, labels):
    thresholds = [round(value, 2) for value in np.arange(0.5, 1.0, 0.05)]
    per_class = {}
    ap50_values = []
    map_values = []

    present_classes = sorted({label["class_id"] for label in labels})
    for class_id in present_classes:
        ap_by_threshold = []
        for threshold in thresholds:
            ap = average_precision(predictions, labels, class_id, threshold)
            if ap is not None:
                ap_by_threshold.append(ap)
        ap50 = ap_by_threshold[0] if ap_by_threshold else 0.0
        map50_95 = float(np.mean(ap_by_threshold)) if ap_by_threshold else 0.0
        per_class[class_id] = {
            "class_name": FINAL_CLASS_NAMES.get(class_id, str(class_id)),
            "ap50": ap50,
            "map50_95": map50_95,
        }
        ap50_values.append(ap50)
        map_values.append(map50_95)

    return {
        "map50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "map50_95": float(np.mean(map_values)) if map_values else 0.0,
        "per_class": per_class,
    }


def run_prediction(image_path: Path, models, config: DetectionConfig, quiet: bool):
    stream = io.StringIO()
    redirect = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    with redirect:
        _, preprocessed = preprocess_image(str(image_path), config)
        biota_detections, starfish_detections = run_parallel_detection(
            models[0], models[1], preprocessed, config
        )
        biota_detections = apply_color_class_corrections(biota_detections, preprocessed, config)
        starfish_detections = apply_color_class_corrections(starfish_detections, preprocessed, config)
        merged = merge_detections(biota_detections, starfish_detections, config)
    return merged


def evaluate(args):
    config = build_config(args)
    models = load_models(config)

    predictions = []
    labels = []
    per_image_rows = []
    image_counter = 0

    datasets = selected_datasets(args.dataset)
    for dataset_root in datasets:
        image_paths = collect_images(dataset_root, args.split)
        if args.limit:
            image_paths = image_paths[:args.limit]

        for image_path in image_paths:
            image_id = f"{dataset_root.name}/{args.split}/{image_path.name}"
            label_path = label_path_for_image(image_path)
            image_labels = read_yolo_labels(label_path, dataset_root, config.TARGET_SIZE)
            for label in image_labels:
                labels.append({"image_id": image_id, **label})

            detections = run_prediction(image_path, models, config, args.quiet)
            for detection in detections:
                predictions.append(
                    {
                        "image_id": image_id,
                        "class_id": int(detection["class_id"]),
                        "confidence": float(detection["confidence"]),
                        "bbox": [
                            float(detection["x1"]),
                            float(detection["y1"]),
                            float(detection["x2"]),
                            float(detection["y2"]),
                        ],
                    }
                )

            per_image_rows.append(
                {
                    "image_id": image_id,
                    "label_count": len(image_labels),
                    "prediction_count": len(detections),
                    "label_file": str(label_path),
                }
            )
            image_counter += 1
            if not args.quiet:
                print(f"[{image_counter}] {image_id}: labels={len(image_labels)}, preds={len(detections)}")

    precision, recall, f1, mean_iou, tp, fp, fn = precision_recall_f1(predictions, labels)
    map_result = compute_map(predictions, labels)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "images": len(per_image_rows),
        "labels": len(labels),
        "predictions": len(predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "map50": map_result["map50"],
        "map50_95": map_result["map50_95"],
        "per_class": map_result["per_class"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"map_summary_{args.dataset}_{args.split}.json"
    rows_path = output_dir / f"map_per_image_{args.dataset}_{args.split}.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "label_count", "prediction_count", "label_file"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    return summary, summary_path, rows_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-model detection mAP on labeled YOLO dataset images."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset folder to evaluate, or 'all'. Default: all",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "val", "test"],
        help="Dataset split to evaluate. Default: test",
    )
    parser.add_argument("--conf-biota", type=float, default=0.4)
    parser.add_argument("--conf-starfish", type=float, default=0.6)
    parser.add_argument("--disable-color-correction", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit images per dataset for quick testing.")
    parser.add_argument("--output-dir", default="results/map_evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose pipeline logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary, summary_path, rows_path = evaluate(args)

    print("\nEvaluation complete")
    print("=" * 60)
    print(f"Dataset/split : {summary['dataset']} / {summary['split']}")
    print(f"Images        : {summary['images']}")
    print(f"Labels        : {summary['labels']}")
    print(f"Predictions   : {summary['predictions']}")
    print(f"Precision     : {summary['precision']:.4f}")
    print(f"Recall        : {summary['recall']:.4f}")
    print(f"F1-score      : {summary['f1']:.4f}")
    print(f"Mean IoU      : {summary['mean_iou']:.4f}")
    print(f"mAP@0.5       : {summary['map50']:.4f}")
    print(f"mAP@0.5:0.95  : {summary['map50_95']:.4f}")
    print("\nPer class:")
    for class_id, item in summary["per_class"].items():
        print(
            f"  {class_id} {item['class_name']}: "
            f"AP50={item['ap50']:.4f}, mAP50-95={item['map50_95']:.4f}"
        )
    print(f"\nSaved summary : {summary_path}")
    print(f"Saved rows    : {rows_path}")


if __name__ == "__main__":
    main()
