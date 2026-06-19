"""
Local web UI for testing underwater object detection from an uploaded image.

Run:
    .\\.venv\\Scripts\\python.exe detection_ui.py

Then open:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import html
import csv
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from dual_model_detection_system import (
    DetectionConfig,
    apply_color_class_corrections,
    extract_features,
    load_models,
    merge_detections,
    prepare_output_dirs,
    preprocess_image,
    run_parallel_detection,
    segment_object,
    visualize_detections,
)


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7860"))
BASE_DIR = Path(__file__).resolve().parent
UI_RESULTS_DIR = BASE_DIR / "results" / "ui"
UPLOAD_DIR = UI_RESULTS_DIR / "uploads"
OUTPUT_DIR = UI_RESULTS_DIR / "outputs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_LABEL_EXTENSIONS = {".txt"}

MODEL_CACHE = None


@dataclass
class UploadedFile:
    filename: str
    data: bytes


class MultipartForm:
    def __init__(self, fields: dict[str, str], files: dict[str, UploadedFile]):
        self.fields = fields
        self.files = files

    def getfirst(self, name: str, default=None):
        return self.fields.get(name, default)


def parse_multipart_form(handler: SimpleHTTPRequestHandler) -> MultipartForm:
    content_type = handler.headers.get("Content-Type", "")
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    body = handler.rfile.read(content_length)

    if "multipart/form-data" not in content_type:
        raise ValueError("Request harus multipart/form-data.")

    raw_message = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=email_policy).parsebytes(raw_message)

    fields: dict[str, str] = {}
    files: dict[str, UploadedFile] = {}
    for part in message.iter_parts():
        disposition = part.get_content_disposition()
        if disposition != "form-data":
            continue

        name = part.get_param("name", header="content-disposition")
        if not name:
            continue

        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = UploadedFile(Path(filename).name, payload)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")

    return MultipartForm(fields, files)


def get_models(config: DetectionConfig):
    global MODEL_CACHE
    if MODEL_CACHE is None:
        MODEL_CACHE = load_models(config)
    return MODEL_CACHE


def safe_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def json_response(handler: SimpleHTTPRequestHandler, payload, status=HTTPStatus.OK):
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def file_response(handler: SimpleHTTPRequestHandler, path: Path):
    if not path.exists() or not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def make_config(form) -> DetectionConfig:
    config = DetectionConfig()
    config.CLASS_NAMES = {
        0: "eel",
        1: "fish",
        2: "jellyfish",
        3: "starfish",
    }
    config.CLASS_COLORS = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 255, 255),
    }
    config.STARFISH_CLASS_ID = 3
    config.CONF_BIOTA = safe_float(form.getfirst("conf_biota"), config.CONF_BIOTA, 0.05, 0.95)
    config.CONF_STARFISH = safe_float(form.getfirst("conf_starfish"), config.CONF_STARFISH, 0.05, 0.95)
    config.ENABLE_COLOR_CLASS_CORRECTION = form.getfirst("color_correction", "on") == "on"
    config.RESULTS_DIR = UI_RESULTS_DIR
    config.refresh_output_dirs()
    prepare_output_dirs(config)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return config


def build_feature_rows(detections, features, config: DetectionConfig):
    feature_map = {item.get("object_id"): item for item in features}
    rows = []
    for idx, detection in enumerate(detections, 1):
        feature = feature_map.get(idx, {})
        rows.append(
            {
                "object_id": idx,
                "class_name": config.CLASS_NAMES.get(detection["class_id"], "unknown"),
                "confidence": round(float(detection["confidence"]), 4),
                "model": str(detection.get("model", "-")),
                "bbox": [
                    round(float(detection["x1"]), 1),
                    round(float(detection["y1"]), 1),
                    round(float(detection["x2"]), 1),
                    round(float(detection["y2"]), 1),
                ],
                "area": int(feature.get("area", 0) or 0),
                "circularity": round(float(feature.get("circularity", 0) or 0), 4),
                "aspect_ratio": round(float(feature.get("aspect_ratio", 0) or 0), 4),
                "contour_shape": str(feature.get("contour_shape", "-")),
            }
        )
    return rows


def class_counts(rows):
    counts = {}
    for row in rows:
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    return counts


def label_for_dataset_image(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    image_index = parts.index("images")
    label_parts = parts[:]
    label_parts[image_index] = "labels"
    label_path = Path(*label_parts).with_suffix(".txt")
    return label_path if label_path.exists() else None


def find_ground_truth_label(image_name: str, image_path: Path | None = None) -> Path | None:
    stem = Path(image_name).stem

    dataset_roots = [
        BASE_DIR / "combined_detection_dataset",
        BASE_DIR / "preprocessed_detection_dataset",
        BASE_DIR / "biota_3class_dataset",
        BASE_DIR / "starfish_only",
    ]

    for root in dataset_roots:
        if not root.exists():
            continue
        for label_path in root.rglob(f"{stem}.txt"):
            if label_path.is_file() and "labels" in label_path.parts:
                return label_path

    return None


def read_latest_training_metrics(results_path: Path):
    if not results_path.exists():
        return None
    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None

    latest = rows[-1]

    def get_number(column: str) -> float:
        value = latest.get(column, latest.get(f" {column}", "0"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return {
        "precision": get_number("metrics/precision(B)"),
        "recall": get_number("metrics/recall(B)"),
        "ap50": get_number("metrics/mAP50(B)"),
        "map50_95": get_number("metrics/mAP50-95(B)"),
    }


def model_validation_metrics():
    metric_paths = [
        BASE_DIR / "runs" / "detect" / "biota_3class_train-3" / "results.csv",
        BASE_DIR / "runs" / "detect" / "cascade_starfish_only" / "results.csv",
    ]
    values = [read_latest_training_metrics(path) for path in metric_paths]
    values = [value for value in values if value is not None]
    if not values:
        return None

    precision = float(np.mean([value["precision"] for value in values]))
    recall = float(np.mean([value["recall"] for value in values]))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "available": True,
        "reason": "Label per-gambar tidak ditemukan; memakai skor validasi model.",
        "label_file": "runs/detect/*/results.csv",
        "label_mode": "model_validation",
        "ground_truth": 0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(float(f1), 4),
        "mean_iou": None,
        "ap50": round(float(np.mean([value["ap50"] for value in values])), 4),
        "map50_95": round(float(np.mean([value["map50_95"] for value in values])), 4),
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def dataset_overall_metrics():
    summary_path = UI_RESULTS_DIR.parent / "map_evaluation" / "map_summary_all_test.json"
    if not summary_path.exists():
        summary_path = BASE_DIR / "results" / "map_evaluation" / "map_summary_all_test.json"
    if not summary_path.exists():
        return model_validation_metrics()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "reason": "Skor akurasi keseluruhan dataset.",
        "label_file": str(summary_path.relative_to(BASE_DIR)),
        "label_mode": "overall_dataset",
        "dataset": summary.get("dataset", "all"),
        "split": summary.get("split", "test"),
        "images": int(summary.get("images", 0)),
        "ground_truth": int(summary.get("labels", 0)),
        "predictions": int(summary.get("predictions", 0)),
        "precision": round(float(summary.get("precision", 0.0)), 4),
        "recall": round(float(summary.get("recall", 0.0)), 4),
        "f1": round(float(summary.get("f1", 0.0)), 4),
        "mean_iou": round(float(summary.get("mean_iou", 0.0)), 4),
        "ap50": round(float(summary.get("map50", 0.0)), 4),
        "map50_95": round(float(summary.get("map50_95", 0.0)), 4),
        "tp": int(summary.get("tp", 0)),
        "fp": int(summary.get("fp", 0)),
        "fn": int(summary.get("fn", 0)),
    }


def map_label_class_id(class_id: int, label_path: Path, label_mode: str, config: DetectionConfig) -> int:
    normalized_path = str(label_path).replace("\\", "/").lower()
    if label_mode in {"four_class", "biota_3class"}:
        return class_id
    if label_mode == "starfish_only" or "starfish_only" in normalized_path:
        return config.STARFISH_CLASS_ID
    if (
        label_mode == "combined_detection"
        or "combined_detection_dataset" in normalized_path
        or "preprocessed_detection_dataset" in normalized_path
    ):
        combined_map = {
            0: class_name_to_id("fish", config),
            1: class_name_to_id("jellyfish", config),
            2: config.STARFISH_CLASS_ID,
        }
        return combined_map.get(class_id, class_id)
    return class_id


def class_name_to_id(class_name: str, config: DetectionConfig) -> int:
    for class_id, name in config.CLASS_NAMES.items():
        if name == class_name:
            return class_id
    return 0


def read_yolo_labels(
    label_path: Path | None,
    config: DetectionConfig,
    label_mode: str = "auto",
):
    if label_path is None:
        return None

    labels = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue

        class_id = map_label_class_id(class_id, label_path, label_mode, config)

        target = config.TARGET_SIZE
        x1 = (cx - width / 2) * target
        y1 = (cy - height / 2) * target
        x2 = (cx + width / 2) * target
        y2 = (cy + height / 2) * target
        labels.append(
            {
                "class_id": class_id,
                "bbox": [x1, y1, x2, y2],
                "matched": False,
            }
        )
    return labels


def box_iou(box_a, box_b) -> float:
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


def average_precision(predictions, labels, iou_threshold: float):
    if labels is None:
        return None
    if not labels:
        return 1.0 if not predictions else 0.0

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )
    matched_labels = set()
    tp = []
    fp = []

    for prediction in sorted_predictions:
        best_iou = 0.0
        best_index = None
        for idx, label in enumerate(labels):
            if idx in matched_labels or label["class_id"] != prediction["class_id"]:
                continue
            iou = box_iou(prediction["bbox"], label["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = idx

        if best_index is not None and best_iou >= iou_threshold:
            matched_labels.add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(len(labels), 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    recall_points = np.concatenate(([0.0], recalls, [1.0]))
    precision_points = np.concatenate(([1.0], precisions, [0.0]))
    for idx in range(len(precision_points) - 2, -1, -1):
        precision_points[idx] = max(precision_points[idx], precision_points[idx + 1])

    changed = np.where(recall_points[1:] != recall_points[:-1])[0]
    return float(np.sum((recall_points[changed + 1] - recall_points[changed]) * precision_points[changed + 1]))


def evaluate_detections(
    detections,
    image_name: str,
    config: DetectionConfig,
    label_path: Path | None = None,
    label_mode: str = "auto",
    image_path: Path | None = None,
):
    manual_label = label_path is not None
    label_path = label_path or find_ground_truth_label(image_name, image_path=image_path)
    if manual_label and label_mode == "auto":
        label_mode = "four_class"
    labels = read_yolo_labels(label_path, config, label_mode=label_mode)
    if labels is None:
        expected_label = f"{Path(image_name).stem}.txt"
        fallback_metrics = model_validation_metrics()
        if fallback_metrics is not None:
            fallback_metrics["reason"] = (
                f"Label otomatis tidak ditemukan: {expected_label}. "
                "Menampilkan skor validasi model."
            )
            return fallback_metrics
        return {
            "available": False,
            "reason": f"Label otomatis tidak ditemukan: {expected_label}",
            "label_file": "",
            "ground_truth": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "mean_iou": None,
            "ap50": None,
            "map50_95": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    predictions = [
        {
            "class_id": int(det["class_id"]),
            "confidence": float(det["confidence"]),
            "bbox": [float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])],
        }
        for det in detections
    ]

    matched_labels = set()
    matched_ious = []
    tp = 0
    fp = 0

    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        best_iou = 0.0
        best_index = None
        for idx, label in enumerate(labels):
            if idx in matched_labels or label["class_id"] != prediction["class_id"]:
                continue
            iou = box_iou(prediction["bbox"], label["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = idx
        if best_index is not None and best_iou >= 0.5:
            matched_labels.add(best_index)
            matched_ious.append(best_iou)
            tp += 1
        else:
            fp += 1

    fn = max(0, len(labels) - tp)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not labels else 0.0)
    recall = tp / len(labels) if labels else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    ap50 = average_precision(predictions, labels, 0.5)
    ap_values = [
        average_precision(predictions, labels, threshold)
        for threshold in np.arange(0.5, 1.0, 0.05)
    ]

    return {
        "available": True,
        "reason": "",
        "label_file": str(label_path.relative_to(BASE_DIR)),
        "label_mode": label_mode,
        "ground_truth": len(labels),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mean_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "ap50": round(float(ap50 or 0.0), 4),
        "map50_95": round(float(np.mean(ap_values)), 4) if ap_values else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_detection(
    image_path: Path,
    config: DetectionConfig,
    original_name: str | None = None,
    label_path: Path | None = None,
    label_mode: str = "auto",
):
    started = time.perf_counter()
    models = get_models(config)

    original_image, preprocessed_image = preprocess_image(str(image_path), config)
    biota_detections, starfish_detections = run_parallel_detection(
        models[0], models[1], preprocessed_image, config
    )

    biota_detections = apply_color_class_corrections(
        biota_detections, preprocessed_image, config
    )
    starfish_detections = apply_color_class_corrections(
        starfish_detections, preprocessed_image, config
    )
    merged_detections = merge_detections(biota_detections, starfish_detections, config)

    img_biota, img_starfish, img_merged = visualize_detections(
        original_image,
        preprocessed_image,
        merged_detections,
        biota_detections,
        starfish_detections,
        config,
    )

    features = []
    for idx, detection in enumerate(merged_detections, 1):
        seg_result = segment_object(preprocessed_image, detection, config)
        feature = extract_features(seg_result, detection, config)
        if feature:
            feature["filename"] = image_path.stem
            feature["object_id"] = idx
            features.append(feature)

    stamp = f"{image_path.stem}_{int(time.time() * 1000)}"
    merged_path = OUTPUT_DIR / f"{stamp}_merged.jpg"
    biota_path = OUTPUT_DIR / f"{stamp}_biota.jpg"
    starfish_path = OUTPUT_DIR / f"{stamp}_starfish.jpg"
    original_path = OUTPUT_DIR / f"{stamp}_original.jpg"

    cv2.imwrite(str(merged_path), img_merged)
    cv2.imwrite(str(biota_path), img_biota)
    cv2.imwrite(str(starfish_path), img_starfish)
    cv2.imwrite(str(original_path), cv2.resize(original_image, (config.TARGET_SIZE, config.TARGET_SIZE)))

    rows = build_feature_rows(merged_detections, features, config)
    confidences = [row["confidence"] for row in rows]
    metrics = dataset_overall_metrics()

    return {
        "ok": True,
        "summary": {
            "filename": image_path.name,
            "total_objects": len(rows),
            "raw_biota": len(biota_detections),
            "raw_starfish": len(starfish_detections),
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
            "processing_time": round(time.perf_counter() - started, 2),
            "class_counts": class_counts(rows),
        },
        "metrics": metrics,
        "images": {
            "merged": f"/result/{merged_path.name}",
            "biota": f"/result/{biota_path.name}",
            "starfish": f"/result/{starfish_path.name}",
            "original": f"/result/{original_path.name}",
        },
        "detections": rows,
    }


def page_html():
    return """<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Test Deteksi Biota Laut</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #65758b;
      --line: #d8e0ea;
      --brand: #007c89;
      --brand-strong: #005c66;
      --accent: #f2b84b;
      --danger: #b42318;
      --radius: 8px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }
    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(24px, 4vw, 38px);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 760px;
    }
    .status-pill {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
      white-space: nowrap;
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 10px 30px rgba(18, 38, 63, 0.06);
    }
    .controls { padding: 18px; }
    .dropzone {
      border: 1.5px dashed #99a8b8;
      border-radius: var(--radius);
      min-height: 172px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
      cursor: pointer;
      background: #fbfcfe;
      transition: border-color 160ms, background 160ms;
    }
    .dropzone:hover,
    .dropzone.dragover {
      border-color: var(--brand);
      background: #eefafa;
    }
    .upload-mark {
      width: 44px;
      height: 44px;
      margin: 0 auto 10px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: #dff5f6;
      color: var(--brand-strong);
      font-weight: 800;
      font-size: 22px;
    }
    .dropzone strong { display: block; margin-bottom: 4px; }
    .dropzone span { color: var(--muted); font-size: 14px; }
    input[type="file"] { display: none; }
    .file-name {
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .label-tools {
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }
    .label-upload {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px 12px;
      background: #fbfcfe;
      cursor: pointer;
      color: var(--muted);
      font-size: 14px;
    }
    .label-upload strong {
      color: var(--ink);
      font-size: 14px;
    }
    .select-field {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .field { margin-top: 18px; }
    .field-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .value {
      color: var(--ink);
      font-variant-numeric: tabular-nums;
      min-width: 44px;
      text-align: right;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--brand);
    }
    .switch {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-top: 18px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }
    .switch span {
      color: var(--muted);
      font-size: 14px;
    }
    .switch input {
      width: 42px;
      height: 22px;
      accent-color: var(--brand);
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 18px;
    }
    button {
      border: 0;
      border-radius: var(--radius);
      min-height: 42px;
      padding: 0 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .primary {
      flex: 1;
      background: var(--brand);
      color: white;
    }
    .primary:hover { background: var(--brand-strong); }
    .ghost {
      background: #eef2f6;
      color: var(--ink);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .viewer { overflow: hidden; }
    .tabs {
      display: flex;
      gap: 0;
      border-bottom: 1px solid var(--line);
      background: #f9fbfd;
      overflow-x: auto;
    }
    .tab {
      background: transparent;
      color: var(--muted);
      border-radius: 0;
      border-right: 1px solid var(--line);
      min-width: 116px;
    }
    .tab.active {
      background: var(--panel);
      color: var(--brand-strong);
      box-shadow: inset 0 -3px 0 var(--brand);
    }
    .image-stage {
      min-height: 480px;
      display: grid;
      place-items: center;
      background:
        linear-gradient(45deg, #edf2f7 25%, transparent 25%),
        linear-gradient(-45deg, #edf2f7 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #edf2f7 75%),
        linear-gradient(-45deg, transparent 75%, #edf2f7 75%);
      background-size: 22px 22px;
      background-position: 0 0, 0 11px, 11px -11px, -11px 0;
    }
    .placeholder {
      max-width: 360px;
      text-align: center;
      color: var(--muted);
      padding: 28px;
    }
    .result-image {
      display: none;
      width: 100%;
      max-height: 72vh;
      object-fit: contain;
      background: #111820;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 14px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }
    .metric strong {
      font-size: 24px;
      font-variant-numeric: tabular-nums;
    }
    .metric small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .eval-summary {
      grid-template-columns: repeat(6, minmax(0, 1fr));
    }
    .table-panel {
      margin-top: 18px;
      overflow: hidden;
    }
    .table-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .table-head h2 {
      margin: 0;
      font-size: 17px;
    }
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      background: #fbfcfe;
      font-weight: 700;
    }
    .empty-row td {
      color: var(--muted);
      text-align: center;
      padding: 26px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      background: #e8f7f5;
      color: var(--brand-strong);
      font-weight: 700;
    }
    .error {
      margin-top: 12px;
      display: none;
      border: 1px solid #f1a7a1;
      background: #fff5f4;
      color: var(--danger);
      border-radius: var(--radius);
      padding: 10px 12px;
      font-size: 14px;
    }
    .loading {
      display: none;
      color: var(--muted);
      font-size: 14px;
      margin-top: 12px;
    }
    @media (max-width: 920px) {
      header { align-items: start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .eval-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .image-stage { min-height: 340px; }
    }
    @media (max-width: 520px) {
      .shell { width: min(100vw - 20px, 1180px); padding-top: 16px; }
      .summary { grid-template-columns: 1fr; }
      .eval-summary { grid-template-columns: 1fr; }
      .actions { flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Test Deteksi Biota Laut</h1>
        <p class="subtitle">Upload gambar underwater untuk menjalankan deteksi eel, fish, jellyfish, dan starfish memakai pipeline dual-model YOLOv8.</p>
      </div>
      <div class="status-pill" id="statusPill">Model siap dimuat</div>
    </header>

    <section class="grid">
      <form class="panel controls" id="detectForm">
        <label class="dropzone" id="dropzone">
          <input id="imageInput" name="image" type="file" accept="image/*">
          <div>
            <div class="upload-mark">+</div>
            <strong>Pilih atau jatuhkan gambar</strong>
            <span>JPG, PNG, BMP, atau WEBP</span>
            <div class="file-name" id="fileName">Belum ada file</div>
          </div>
        </label>

        <div class="field">
          <div class="field-line">
            <label for="confBiota">Confidence biota</label>
            <span class="value" id="biotaValue">0.40</span>
          </div>
          <input id="confBiota" name="conf_biota" type="range" min="0.05" max="0.95" value="0.40" step="0.05">
        </div>

        <div class="field">
          <div class="field-line">
            <label for="confStarfish">Confidence starfish</label>
            <span class="value" id="starfishValue">0.60</span>
          </div>
          <input id="confStarfish" name="conf_starfish" type="range" min="0.05" max="0.95" value="0.60" step="0.05">
        </div>

        <label class="switch">
          <span>Koreksi warna untuk mengurangi false-positive starfish</span>
          <input id="colorCorrection" name="color_correction" type="checkbox" checked>
        </label>

        <div class="actions">
          <button class="primary" id="runButton" type="submit">Jalankan Deteksi</button>
          <button class="ghost" id="resetButton" type="button">Reset</button>
        </div>
        <div class="loading" id="loadingText">Sedang memproses. Load model pertama bisa agak lama.</div>
        <div class="error" id="errorBox"></div>
      </form>

      <section class="panel viewer">
        <nav class="tabs" aria-label="Tampilan hasil">
          <button class="tab active" data-image="merged" type="button">Merged</button>
          <button class="tab" data-image="original" type="button">Original</button>
          <button class="tab" data-image="biota" type="button">Biota</button>
          <button class="tab" data-image="starfish" type="button">Starfish</button>
        </nav>
        <div class="image-stage">
          <div class="placeholder" id="placeholder">Hasil deteksi akan tampil di sini setelah gambar diproses.</div>
          <img class="result-image" id="resultImage" alt="Hasil deteksi">
        </div>
      </section>
    </section>

    <section class="summary" id="summary">
      <div class="metric"><span>Total objek</span><strong id="mTotal">0</strong></div>
      <div class="metric"><span>Rata-rata confidence</span><strong id="mConf">0.00</strong></div>
      <div class="metric"><span>Raw model biota</span><strong id="mBiota">0</strong></div>
      <div class="metric"><span>Waktu proses</span><strong id="mTime">0s</strong></div>
    </section>

    <section class="summary eval-summary" id="evalSummary">
      <div class="metric"><span>mAP@0.5</span><strong id="mAP50">N/A</strong><small>Skor overall dataset</small></div>
      <div class="metric"><span>mAP@0.5:0.95</span><strong id="mAP5095">N/A</strong><small>Skor overall dataset</small></div>
      <div class="metric"><span>Precision</span><strong id="mPrecision">N/A</strong><small>Overall dataset</small></div>
      <div class="metric"><span>Recall</span><strong id="mRecall">N/A</strong><small>Overall dataset</small></div>
      <div class="metric"><span>F1-score</span><strong id="mF1">N/A</strong><small>Overall dataset</small></div>
      <div class="metric"><span>Mean IoU</span><strong id="mIoU">N/A</strong><small id="mEvalInfo">Skor belum dimuat</small></div>
    </section>

    <section class="panel table-panel">
      <div class="table-head">
        <h2>Detail Deteksi</h2>
        <span class="badge" id="classCounts">Belum ada hasil</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Class</th>
              <th>Confidence</th>
              <th>Model</th>
              <th>Bbox</th>
              <th>Area</th>
              <th>Circularity</th>
              <th>Aspect</th>
              <th>Shape</th>
            </tr>
          </thead>
          <tbody id="resultRows">
            <tr class="empty-row"><td colspan="9">Upload gambar untuk mulai test deteksi.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("detectForm");
    const imageInput = document.getElementById("imageInput");
    const dropzone = document.getElementById("dropzone");
    const fileName = document.getElementById("fileName");
    const runButton = document.getElementById("runButton");
    const resetButton = document.getElementById("resetButton");
    const errorBox = document.getElementById("errorBox");
    const loadingText = document.getElementById("loadingText");
    const resultImage = document.getElementById("resultImage");
    const placeholder = document.getElementById("placeholder");
    const tabs = document.querySelectorAll(".tab");
    const statusPill = document.getElementById("statusPill");
    const rows = document.getElementById("resultRows");
    const classCounts = document.getElementById("classCounts");
    const confBiota = document.getElementById("confBiota");
    const confStarfish = document.getElementById("confStarfish");
    const biotaValue = document.getElementById("biotaValue");
    const starfishValue = document.getElementById("starfishValue");

    let latestImages = {};
    let activeImage = "merged";
    let selectedPreviewUrl = "";

    function syncRangeLabels() {
      biotaValue.textContent = Number(confBiota.value).toFixed(2);
      starfishValue.textContent = Number(confStarfish.value).toFixed(2);
    }

    function setBusy(isBusy) {
      runButton.disabled = isBusy;
      imageInput.disabled = isBusy;
      loadingText.style.display = isBusy ? "block" : "none";
      statusPill.textContent = isBusy ? "Memproses gambar" : "Siap test deteksi";
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.style.display = "block";
    }

    function hideError() {
      errorBox.textContent = "";
      errorBox.style.display = "none";
    }

    function updateImage() {
      const url = latestImages[activeImage];
      if (!url) {
        resultImage.style.display = "none";
        placeholder.style.display = "block";
        return;
      }
      resultImage.src = url.startsWith("blob:") ? url : `${url}?t=${Date.now()}`;
      resultImage.style.display = "block";
      placeholder.style.display = "none";
    }

    function activateTab(name) {
      activeImage = name;
      tabs.forEach((item) => item.classList.toggle("active", item.dataset.image === name));
      updateImage();
    }

    function showSelectedImage(file) {
      if (selectedPreviewUrl) {
        URL.revokeObjectURL(selectedPreviewUrl);
      }
      selectedPreviewUrl = URL.createObjectURL(file);
      latestImages = { original: selectedPreviewUrl };
      activateTab("original");
      statusPill.textContent = "Preview gambar original";
    }

    function formatMetric(value) {
      return value === null || value === undefined ? "N/A" : Number(value).toFixed(4);
    }

    function renderSummary(summary) {
      document.getElementById("mTotal").textContent = summary.total_objects;
      document.getElementById("mConf").textContent = Number(summary.avg_confidence).toFixed(2);
      document.getElementById("mBiota").textContent = summary.raw_biota;
      document.getElementById("mTime").textContent = `${summary.processing_time}s`;
      const counts = Object.entries(summary.class_counts || {})
        .map(([name, count]) => `${name}: ${count}`)
        .join(" | ");
      classCounts.textContent = counts || "Tidak ada objek";
    }

    function renderMetrics(metrics) {
      const available = metrics && metrics.available;
      document.getElementById("mAP50").textContent = available ? formatMetric(metrics.ap50) : "N/A";
      document.getElementById("mAP5095").textContent = available ? formatMetric(metrics.map50_95) : "N/A";
      document.getElementById("mPrecision").textContent = available ? formatMetric(metrics.precision) : "N/A";
      document.getElementById("mRecall").textContent = available ? formatMetric(metrics.recall) : "N/A";
      document.getElementById("mF1").textContent = available ? formatMetric(metrics.f1) : "N/A";
      document.getElementById("mIoU").textContent = available ? formatMetric(metrics.mean_iou) : "N/A";
      document.getElementById("mEvalInfo").textContent = available
        ? (metrics.label_mode === "overall_dataset"
            ? `${metrics.dataset || "all"}/${metrics.split || "test"} | ${metrics.images || 0} gambar | ${metrics.ground_truth || 0} label`
          : metrics.label_mode === "model_validation"
            ? "Skor validasi model"
            : `GT ${metrics.ground_truth} | TP ${metrics.tp} FP ${metrics.fp} FN ${metrics.fn} | ${metrics.label_mode || "auto"}`)
        : (metrics?.reason || "Label ground-truth tidak ditemukan");
    }

    function renderRows(detections) {
      if (!detections.length) {
        rows.innerHTML = '<tr class="empty-row"><td colspan="9">Tidak ada objek terdeteksi pada threshold ini.</td></tr>';
        return;
      }
      rows.innerHTML = detections.map((item) => `
        <tr>
          <td>${item.object_id}</td>
          <td><span class="badge">${item.class_name}</span></td>
          <td>${Number(item.confidence).toFixed(4)}</td>
          <td>${item.model}</td>
          <td>${item.bbox.join(", ")}</td>
          <td>${item.area}</td>
          <td>${Number(item.circularity).toFixed(4)}</td>
          <td>${Number(item.aspect_ratio).toFixed(4)}</td>
          <td>${item.contour_shape}</td>
        </tr>
      `).join("");
    }

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      fileName.textContent = file?.name || "Belum ada file";
      if (file) {
        showSelectedImage(file);
      }
      hideError();
    });

    confBiota.addEventListener("input", syncRangeLabels);
    confStarfish.addEventListener("input", syncRangeLabels);

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (!file) return;
      const transfer = new DataTransfer();
      transfer.items.add(file);
      imageInput.files = transfer.files;
      fileName.textContent = file.name;
      showSelectedImage(file);
      hideError();
    });

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        activateTab(tab.dataset.image);
      });
    });

    resetButton.addEventListener("click", () => {
      form.reset();
      latestImages = {};
      activeImage = "merged";
      if (selectedPreviewUrl) {
        URL.revokeObjectURL(selectedPreviewUrl);
        selectedPreviewUrl = "";
      }
      fileName.textContent = "Belum ada file";
      activateTab("merged");
      document.getElementById("mTotal").textContent = "0";
      document.getElementById("mConf").textContent = "0.00";
      document.getElementById("mBiota").textContent = "0";
      document.getElementById("mTime").textContent = "0s";
      renderMetrics(null);
      classCounts.textContent = "Belum ada hasil";
      rows.innerHTML = '<tr class="empty-row"><td colspan="9">Upload gambar untuk mulai test deteksi.</td></tr>';
      syncRangeLabels();
      hideError();
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      hideError();
      if (!imageInput.files.length) {
        showError("Pilih gambar terlebih dahulu.");
        return;
      }

      const data = new FormData(form);
      setBusy(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      try {
        const response = await fetch("/detect", {
          method: "POST",
          body: data,
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "Deteksi gagal dijalankan.");
        }

        latestImages = payload.images || {};
        renderSummary(payload.summary);
        renderMetrics(payload.metrics);
        renderRows(payload.detections || []);
        activateTab("merged");
      } catch (error) {
        showError(error.name === "AbortError" ? "Deteksi terlalu lama. Coba gambar lebih kecil atau upload label .txt manual." : error.message);
      } finally {
        clearTimeout(timeoutId);
        setBusy(false);
      }
    });

    syncRangeLabels();
    renderMetrics(null);
  </script>
</body>
</html>"""


class DetectionUIHandler(SimpleHTTPRequestHandler):
    server_version = "DetectionUI/1.0"

    def log_message(self, format, *args):
        sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = page_html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/healthz":
            json_response(self, {"ok": True})
            return

        if parsed.path.startswith("/result/"):
            filename = Path(unquote(parsed.path.removeprefix("/result/"))).name
            file_response(self, OUTPUT_DIR / filename)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/detect":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            form = parse_multipart_form(self)
            file_item = form.files.get("image")
            if file_item is None or not file_item.filename:
                json_response(self, {"ok": False, "error": "File gambar belum dipilih."}, HTTPStatus.BAD_REQUEST)
                return

            original_name = Path(file_item.filename).name
            suffix = Path(original_name).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                json_response(
                    self,
                    {"ok": False, "error": "Format file tidak didukung. Gunakan JPG, PNG, BMP, atau WEBP."},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            config = make_config(form)
            safe_name = f"upload_{int(time.time() * 1000)}{suffix}"
            upload_path = UPLOAD_DIR / safe_name
            upload_path.write_bytes(file_item.data)

            payload = run_detection(
                upload_path,
                config,
                original_name=original_name,
                label_mode="auto",
            )
            payload["summary"]["filename"] = html.escape(original_name)
            json_response(self, payload)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), DetectionUIHandler)
    print(f"Detection UI running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
