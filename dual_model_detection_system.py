"""
====================================================
SISTEM DETEKSI MULTI-MODEL YOLOV8 - UNDERWATER
====================================================

JUDUL: Deteksi Bintang Laut dan Biota Laut Dangkal Menggunakan YOLOv8

DESKRIPSI:
Program ini mengimplementasikan sistem deteksi underwater yang menggunakan
dua model YOLOv8 secara paralel:
1. Model deteksi biota laut (eel, fish, jellyfish)
2. Model deteksi starfish

Sistem melakukan:
- Preprocessing citra underwater
- Inferensi paralel dengan confidence threshold berbeda
- Merge detections dari kedua model
- Segmentasi objek menggunakan OpenCV
- Ekstraksi fitur objek
- Visualisasi hasil lengkap
- Penyimpanan hasil ke struktur folder otomatis
- Export ke CSV

AUTHOR: Sistem Deteksi Underwater YOLOv8
DATE: 2026-06-07

====================================================
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


# ====================================================
# KONFIGURASI GLOBAL
# ====================================================

class DetectionConfig:
    """Konfigurasi sistem deteksi"""
    
    # Path model
    MODEL_BIOTA = "trained_biota_3class_model_best.pt"
    MODEL_STARFISH = "cascade_model_starfish_best.pt"
    
    # Class mapping
    CLASS_NAMES = {
        0: 'eel',
        1: 'fish',
        2: 'jellyfish',
        3: 'starfish'
    }
    
    # Class colors BGR untuk OpenCV drawing
    CLASS_COLORS = {
        0: (0, 0, 255),      # eel - red
        1: (0, 255, 0),      # fish - green
        2: (255, 0, 0),      # jellyfish - blue
        3: (0, 255, 255)     # starfish - yellow
    }

    STARFISH_CLASS_ID = 3
    
    # Confidence thresholds
    CONF_BIOTA = 0.4
    CONF_STARFISH = 0.6

    # Post-processing thresholds
    STARFISH_SUPPRESS_IOU = 0.50
    STARFISH_OVERRIDE_MARGIN = 0.20
    DUPLICATE_IOU = 0.70

    # Koreksi warna untuk mengurangi false-positive starfish pada jellyfish.
    ENABLE_COLOR_CLASS_CORRECTION = True
    JELLYFISH_PURPLE_RATIO = 0.25
    JELLYFISH_BLUE_RATIO = 0.25
    JELLYFISH_MAX_ORANGE_RATIO = 0.10
    PALE_JELLYFISH_BLUE_RATIO = 0.45
    PALE_JELLYFISH_MAX_ORANGE_RATIO = 0.35
    PALE_JELLYFISH_MAX_STARFISH_CONF = 0.75
    
    # Image preprocessing
    TARGET_SIZE = 640
    
    # Paths
    RESULTS_DIR = Path("results")
    DETECTION_DIR = RESULTS_DIR / "detection"
    SEGMENTATION_DIR = RESULTS_DIR / "segmentation"
    FEATURES_DIR = RESULTS_DIR / "features"
    VIZ_DIR = RESULTS_DIR / "visualizations"
    CSV_DIR = RESULTS_DIR / "csv"

    def refresh_output_dirs(self):
        """
        Sinkronkan subfolder output dengan RESULTS_DIR.

        Fungsi ini penting saat RESULTS_DIR diubah pada custom config, supaya
        detection/segmentation/features/visualizations/csv tetap dibuat di
        folder hasil yang sama.
        """
        self.RESULTS_DIR = Path(self.RESULTS_DIR)
        self.DETECTION_DIR = self.RESULTS_DIR / "detection"
        self.SEGMENTATION_DIR = self.RESULTS_DIR / "segmentation"
        self.FEATURES_DIR = self.RESULTS_DIR / "features"
        self.VIZ_DIR = self.RESULTS_DIR / "visualizations"
        self.CSV_DIR = self.RESULTS_DIR / "csv"


def prepare_output_dirs(config: DetectionConfig):
    """
    Buat seluruh folder output yang dibutuhkan sistem.
    """
    config.refresh_output_dirs()
    for directory in [
        config.RESULTS_DIR,
        config.DETECTION_DIR,
        config.SEGMENTATION_DIR,
        config.FEATURES_DIR,
        config.VIZ_DIR,
        config.CSV_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def class_name_to_final_id(class_name: str, default_id: int = 0) -> int:
    """
    Ubah nama class dari metadata model menjadi class final sistem.

    Final class:
    0 = eel, 1 = fish, 2 = jellyfish, 3 = starfish
    """
    normalized = str(class_name).strip().lower()
    aliases = {
        "eel": 0,
        "fish": 1,
        "jellyfish": 2,
        "starfish": 3,
    }
    return aliases.get(normalized, default_id)


def get_model_class_name(model, class_id: int) -> str:
    """
    Ambil nama class asli dari metadata model YOLO.
    """
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, list) and class_id < len(names):
        return names[class_id]
    return str(class_id)


def bbox_iou(det_a, det_b) -> float:
    """
    Hitung Intersection over Union untuk dua bounding box deteksi.
    """
    ax1, ay1, ax2, ay2 = det_a['x1'], det_a['y1'], det_a['x2'], det_a['y2']
    bx1, by1, bx2, by2 = det_b['x1'], det_b['y1'], det_b['x2'], det_b['y2']

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def get_detection_color_profile(image, detection):
    """
    Hitung rasio warna sederhana pada crop deteksi.

    HSV OpenCV memakai hue 0-179:
    - oranye/kuning: kandidat kuat starfish
    - biru/ungu: kandidat kuat jellyfish pada dataset underwater ini
    """
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(detection['x1']))
    y1 = max(0, int(detection['y1']))
    x2 = min(w_img, int(detection['x2']))
    y2 = min(h_img, int(detection['y2']))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            'orange_ratio': 0.0,
            'purple_ratio': 0.0,
            'blue_ratio': 0.0,
        }

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    valid = (sat > 35) & (val > 40)
    orange = (hue >= 5) & (hue <= 35) & valid
    blue = (hue >= 90) & (hue <= 130) & valid
    purple = (hue >= 125) & (hue <= 175) & valid

    total_pixels = crop.shape[0] * crop.shape[1]
    return {
        'orange_ratio': float(np.count_nonzero(orange) / total_pixels),
        'purple_ratio': float(np.count_nonzero(purple) / total_pixels),
        'blue_ratio': float(np.count_nonzero(blue) / total_pixels),
    }


def apply_color_class_corrections(detections, preprocessed_image, config: DetectionConfig):
    """
    Koreksi class berbasis warna untuk kasus false-positive starfish.

    Jika model starfish mendeteksi area yang dominan biru/ungu dan hampir tidak
    mengandung oranye, deteksi tersebut lebih cocok dianggap jellyfish.
    """
    if not config.ENABLE_COLOR_CLASS_CORRECTION:
        return detections

    corrected = []
    for detection in detections:
        det = detection.copy()
        profile = get_detection_color_profile(preprocessed_image, det)
        det.update(profile)

        looks_like_jellyfish = (
            profile['purple_ratio'] >= config.JELLYFISH_PURPLE_RATIO and
            profile['blue_ratio'] >= config.JELLYFISH_BLUE_RATIO and
            profile['orange_ratio'] <= config.JELLYFISH_MAX_ORANGE_RATIO
        )
        looks_like_pale_jellyfish = (
            det['confidence'] <= config.PALE_JELLYFISH_MAX_STARFISH_CONF and
            profile['blue_ratio'] >= config.PALE_JELLYFISH_BLUE_RATIO and
            profile['orange_ratio'] <= config.PALE_JELLYFISH_MAX_ORANGE_RATIO
        )

        if det['class_id'] == config.STARFISH_CLASS_ID and (
            looks_like_jellyfish or looks_like_pale_jellyfish
        ):
            det['class_id'] = class_name_to_final_id('jellyfish')
            det['corrected_from'] = 'starfish'
            det['correction_reason'] = (
                'blue_purple_color_profile'
                if looks_like_jellyfish
                else 'pale_jellyfish_color_profile'
            )
            det['model'] = f"{det['model']}+color_correction"

        corrected.append(det)

    return corrected


def resolve_image_path(image_path: str) -> Path:
    """
    Cari path gambar dari input user.

    Fungsi ini membuat demo lebih nyaman: user boleh memberi path lengkap,
    path relatif, atau hanya nama file seperti
    combined_dataset_preprocessed_test_00021.jpg.
    """
    candidate = Path(image_path)
    if candidate.exists():
        return candidate

    search_dirs = [
        Path("combined_detection_dataset/test/images"),
        Path("combined_detection_dataset/valid/images"),
        Path("combined_detection_dataset/train/images"),
        Path("preprocessed_detection_dataset/test/images"),
        Path("preprocessed_detection_dataset/valid/images"),
        Path("preprocessed_detection_dataset/train/images"),
        Path("combined_dataset_preprocessed/test/images"),
        Path("combined_dataset_preprocessed/valid/images"),
        Path("combined_dataset_preprocessed/train/images"),
        Path("combined_dataset/test/images"),
        Path("combined_dataset/valid/images"),
        Path("combined_dataset/train/images"),
        Path("preprocessed_dataset/test/images"),
        Path("preprocessed_dataset/valid/images"),
        Path("preprocessed_dataset/train/images"),
        Path("results/predictions"),
    ]

    for directory in search_dirs:
        found = directory / candidate.name
        if found.exists():
            return found

    if candidate.suffix:
        patterns = [candidate.name]
    else:
        patterns = [f"{candidate.name}{suffix}" for suffix in [".jpg", ".jpeg", ".png", ".bmp"]]

    for directory in search_dirs:
        for pattern in patterns:
            found = directory / pattern
            if found.exists():
                return found

    raise FileNotFoundError(
        f"Image tidak ditemukan: {image_path}. "
        "Gunakan path lengkap atau simpan gambar di folder dataset images."
    )


# ====================================================
# TAHAP 1: LOAD MODEL
# ====================================================

def load_models(config: DetectionConfig):
    """
    Load dua model YOLOv8 dari file .pt
    
    Args:
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        tuple: (model_biota, model_starfish)
    """
    print("[TAHAP 1] Loading Models...")
    print("-" * 50)

    def resolve_model_path(model_path, fallback_names=None):
        """
        Cari file model dari path utama dan beberapa nama fallback.
        """
        candidates = [Path(model_path)]
        for fallback in fallback_names or []:
            candidates.append(Path(fallback))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return Path(model_path)
    
    try:
        biota_path = resolve_model_path(config.MODEL_BIOTA)
        starfish_path = resolve_model_path(
            config.MODEL_STARFISH,
            ["cascade_model_starfish_best.pt", "caascade_model_starfish_best.pt"]
        )

        print(f"Loading model biota: {biota_path}")
        model_biota = YOLO(str(biota_path))
        print("✓ Model biota loaded successfully")
        
        print(f"Loading model starfish: {starfish_path}")
        model_starfish = YOLO(str(starfish_path))
        print("✓ Model starfish loaded successfully")
        
        print("-" * 50)
        return model_biota, model_starfish
    
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found - {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


# ====================================================
# TAHAP 2: PREPROCESSING CITRA UNDERWATER
# ====================================================

def preprocess_image(image_path: str, config: DetectionConfig):
    """
    Lakukan preprocessing citra underwater
    
    Tahapan:
    1. Baca image
    2. Resize ke 640x640
    3. Gaussian Blur
    4. CLAHE contrast enhancement
    5. Brightness & contrast adjustment
    
    Args:
        image_path (str): Path ke file image
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        tuple: (original_image, preprocessed_image)
    """
    print("[TAHAP 2] Preprocessing Image...")
    print("-" * 50)

    image_path = resolve_image_path(image_path)
    
    # 1. Baca image original
    print(f"Loading image: {image_path}")
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Tidak dapat membaca image: {image_path}")
    
    original_h, original_w = original.shape[:2]
    print(f"Original size: {original_w}x{original_h}")
    
    # 2. Resize image ke target size
    print(f"Resizing to {config.TARGET_SIZE}x{config.TARGET_SIZE}")
    resized = cv2.resize(original, (config.TARGET_SIZE, config.TARGET_SIZE))
    
    # 3. Gaussian Blur untuk mengurangi noise
    print("Applying Gaussian Blur")
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    # 4. CLAHE contrast enhancement (untuk underwater imaging)
    print("Applying CLAHE contrast enhancement")
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    clahe_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 5. Brightness & contrast adjustment
    print("Applying brightness & contrast adjustment")
    alpha = 1.1  # contrast
    beta = 10    # brightness
    preprocessed = cv2.convertScaleAbs(clahe_enhanced, alpha=alpha, beta=beta)
    
    print("-" * 50)
    return original, preprocessed


# ====================================================
# TAHAP 3: INFERENSI DUA MODEL
# ====================================================

def run_biota_detection(model_biota, preprocessed_image, config: DetectionConfig):
    """
    Jalankan deteksi biota laut dengan confidence 0.4
    
    Args:
        model_biota: Model YOLOv8 biota
        preprocessed_image: Image yang sudah dipreprocess
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        list: Hasil deteksi biota [x1, y1, x2, y2, confidence, class_id]
    """
    print("[TAHAP 3A] Running Biota Detection...")
    print("-" * 50)
    print(f"Using confidence threshold: {config.CONF_BIOTA}")
    
    results = model_biota(preprocessed_image, conf=config.CONF_BIOTA, verbose=False)
    
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            source_cls = int(box.cls[0].cpu().item())
            source_name = get_model_class_name(model_biota, source_cls)
            cls = class_name_to_final_id(source_name, default_id=source_cls)
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf,
                'class_id': cls,
                'source_class_id': source_cls,
                'source_class_name': source_name,
                'model': 'biota'
            })
    
    print(f"Detected {len(detections)} biota objects")
    print("-" * 50)
    return detections


def run_starfish_detection(model_starfish, preprocessed_image, config: DetectionConfig):
    """
    Jalankan deteksi starfish dengan confidence 0.6
    
    Args:
        model_starfish: Model YOLOv8 starfish
        preprocessed_image: Image yang sudah dipreprocess
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        list: Hasil deteksi starfish [x1, y1, x2, y2, confidence, class_id=3]
    """
    print("[TAHAP 3B] Running Starfish Detection...")
    print("-" * 50)
    print(f"Using confidence threshold: {config.CONF_STARFISH}")
    
    results = model_starfish(preprocessed_image, conf=config.CONF_STARFISH, verbose=False)
    
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            # Class starfish adalah class final khusus dari config.
            cls = config.STARFISH_CLASS_ID
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf,
                'class_id': cls,
                'source_class_id': int(box.cls[0].cpu().item()),
                'source_class_name': 'starfish',
                'model': 'starfish'
            })
    
    print(f"Detected {len(detections)} starfish objects")
    print("-" * 50)
    return detections


def run_parallel_detection(model_biota, model_starfish, preprocessed_image, config: DetectionConfig):
    """
    Jalankan inferensi model biota dan model starfish secara paralel.

    Returns:
        tuple: (biota_detections, starfish_detections)
    """
    print("[TAHAP 3] Running Dual-Model Detection in Parallel...")
    print("-" * 50)

    with ThreadPoolExecutor(max_workers=2) as executor:
        biota_future = executor.submit(
            run_biota_detection, model_biota, preprocessed_image, config
        )
        starfish_future = executor.submit(
            run_starfish_detection, model_starfish, preprocessed_image, config
        )

        biota_detections = biota_future.result()
        starfish_detections = starfish_future.result()

    print("Parallel inference complete")
    print("-" * 50)
    return biota_detections, starfish_detections


# ====================================================
# TAHAP 4: MERGE DETECTION
# ====================================================

def merge_detections(biota_detections, starfish_detections, config: DetectionConfig = None):
    """
    Gabungkan hasil deteksi dari kedua model
    
    Args:
        biota_detections (list): Deteksi dari model biota
        starfish_detections (list): Deteksi dari model starfish
        
    Returns:
        list: Deteksi merged
    """
    if config is None:
        config = DetectionConfig()

    print("[TAHAP 4] Merging Detections...")
    print("-" * 50)
    
    merged = list(biota_detections)
    suppressed_starfish = 0
    starfish_overrides = 0
    duplicate_removed = 0

    for starfish_det in starfish_detections:
        if starfish_det['class_id'] != config.STARFISH_CLASS_ID:
            duplicate_index = None
            for idx, existing in enumerate(merged):
                same_class = existing['class_id'] == starfish_det['class_id']
                strong_overlap = bbox_iou(starfish_det, existing) >= config.STARFISH_SUPPRESS_IOU

                if strong_overlap and same_class:
                    duplicate_index = idx
                    break

                if strong_overlap and 'corrected_from' in starfish_det:
                    duplicate_index = idx
                    break

            if duplicate_index is not None:
                duplicate_removed += 1
                merged[duplicate_index] = starfish_det
            else:
                merged.append(starfish_det)
            continue

        conflicting_index = None
        conflicting_iou = 0.0
        for idx, biota_det in enumerate(merged):
            iou = bbox_iou(starfish_det, biota_det)
            if biota_det['class_id'] != config.STARFISH_CLASS_ID and iou >= config.STARFISH_SUPPRESS_IOU:
                if iou > conflicting_iou:
                    conflicting_index = idx
                    conflicting_iou = iou

        if conflicting_index is not None:
            conflicting_det = merged[conflicting_index]
            starfish_is_stronger = (
                starfish_det['confidence'] >=
                conflicting_det['confidence'] + config.STARFISH_OVERRIDE_MARGIN
            )

            if starfish_is_stronger:
                merged[conflicting_index] = starfish_det
                starfish_overrides += 1
            else:
                suppressed_starfish += 1
            continue

        duplicate_index = None
        for idx, existing in enumerate(merged):
            same_class = existing['class_id'] == starfish_det['class_id']
            if same_class and bbox_iou(starfish_det, existing) >= config.DUPLICATE_IOU:
                duplicate_index = idx
                break

        if duplicate_index is not None:
            duplicate_removed += 1
            if starfish_det['confidence'] > merged[duplicate_index]['confidence']:
                merged[duplicate_index] = starfish_det
            continue

        merged.append(starfish_det)

    print(f"Total detections merged: {len(merged)}")
    print(f"  - Biota: {len(biota_detections)}")
    print(f"  - Starfish: {len(starfish_detections)}")
    print(f"  - Starfish suppressed by overlap: {suppressed_starfish}")
    print(f"  - Starfish selected by confidence: {starfish_overrides}")
    print(f"  - Duplicate removed: {duplicate_removed}")
    print("-" * 50)
    
    return merged


# ====================================================
# TAHAP 5: VISUALISASI DETEKSI
# ====================================================

def visualize_detections(original_image, preprocessed_image, merged_detections, 
                        biota_detections, starfish_detections, config: DetectionConfig):
    """
    Visualisasi semua deteksi dengan bounding box dan label
    
    Args:
        original_image: Image original
        preprocessed_image: Image hasil preprocessing
        merged_detections: Deteksi merged
        biota_detections: Deteksi biota
        starfish_detections: Deteksi starfish
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        tuple: (img_biota, img_starfish, img_merged)
    """
    print("[TAHAP 5] Visualizing Detections...")
    print("-" * 50)
    
    # Prepare images
    img_biota = preprocessed_image.copy()
    img_starfish = preprocessed_image.copy()
    img_merged = preprocessed_image.copy()
    
    # Visualize biota detections
    for det in biota_detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        cls_id = det['class_id']
        conf = det['confidence']
        color = config.CLASS_COLORS[cls_id]
        label = f"{config.CLASS_NAMES[cls_id]} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img_biota, (x1, y1), (x2, y2), color, 2)
        # Draw label
        cv2.putText(img_biota, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Visualize starfish detections
    for det in starfish_detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        cls_id = det['class_id']
        conf = det['confidence']
        color = config.CLASS_COLORS[cls_id]
        label = f"{config.CLASS_NAMES[cls_id]} {conf:.2f}"
        
        cv2.rectangle(img_starfish, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_starfish, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Visualize merged detections
    for det in merged_detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        cls_id = det['class_id']
        conf = det['confidence']
        color = config.CLASS_COLORS[cls_id]
        label = f"{config.CLASS_NAMES[cls_id]} {conf:.2f}"
        
        cv2.rectangle(img_merged, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_merged, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    print(f"Visualisasi created")
    print("-" * 50)
    
    return img_biota, img_starfish, img_merged


# ====================================================
# TAHAP 6: SEGMENTASI OPENCV
# ====================================================

def segment_object(original_image, detection, config: DetectionConfig):
    """
    Segmentasi objek individual menggunakan GrabCut + thresholding
    
    Args:
        original_image: Image original (untuk ukuran original)
        detection (dict): Info deteksi individual
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        dict: Hasil segmentasi (mask, contours, dst)
    """
    
    x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), \
                     int(detection['x2']), int(detection['y2'])
    
    # Tambah padding untuk GrabCut
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(original_image.shape[1], x2 + pad)
    y2 = min(original_image.shape[0], y2 + pad)
    
    # Crop region
    crop_img = original_image[y1:y2, x1:x2].copy()
    
    if crop_img.size == 0:
        return None
    
    # GrabCut segmentation
    mask = np.zeros(crop_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    rect = (10, 10, crop_img.shape[1]-10, crop_img.shape[0]-10)
    
    try:
        cv2.grabCut(crop_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except:
        # Jika GrabCut error, gunakan thresholding
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask.astype(np.uint8)
    
    # Convert mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    
    # Extract contours
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = cv2.bitwise_and(crop_img, crop_img, mask=mask2)
    
    return {
        'mask': mask2,
        'contours': contours,
        'dst': result,
        'crop_img': crop_img,
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
    }


# ====================================================
# TAHAP 7: EKSTRAKSI FITUR
# ====================================================

def extract_features(segmentation_result, detection, config: DetectionConfig):
    """
    Ekstrak fitur objek hasil segmentasi
    
    Fitur yang diekstrak:
    1. Area objek
    2. Perimeter contour
    3. Rata-rata warna RGB
    4. Edge detection (Canny)
    5. Circularity
    6. Aspect ratio
    
    Args:
        segmentation_result (dict): Hasil segmentasi
        detection (dict): Info deteksi
        config (DetectionConfig): Konfigurasi sistem
        
    Returns:
        dict: Fitur yang diekstrak
    """
    
    if segmentation_result is None:
        return None
    
    mask = segmentation_result['mask']
    crop_img = segmentation_result['crop_img']
    contours = segmentation_result['contours']
    
    features = {
        'class': config.CLASS_NAMES[detection['class_id']],
        'confidence': detection['confidence'],
        'model': detection['model'],
        'bbox_x1': detection['x1'],
        'bbox_y1': detection['y1'],
        'bbox_x2': detection['x2'],
        'bbox_y2': detection['y2']
    }
    
    # 1. Area
    area = cv2.countNonZero(mask)
    features['area'] = area
    
    # 2. Perimeter & 5. Circularity
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        features['perimeter'] = perimeter
        
        # Circularity = 4π * Area / Perimeter²
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        features['circularity'] = circularity
        
        # 6. Aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0:
            aspect_ratio = float(w) / h
        else:
            aspect_ratio = 0
        features['aspect_ratio'] = aspect_ratio

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        features['contour_points'] = len(approx)
        if len(approx) <= 4:
            features['contour_shape'] = 'angular'
        elif circularity >= 0.75:
            features['contour_shape'] = 'round'
        else:
            features['contour_shape'] = 'irregular'
    else:
        features['perimeter'] = 0
        features['circularity'] = 0
        features['aspect_ratio'] = 0
        features['contour_points'] = 0
        features['contour_shape'] = 'none'
    
    # 3. Rata-rata warna RGB
    masked_img = crop_img.copy()
    masked_img[mask == 0] = [0, 0, 0]
    
    b_mean = np.mean(crop_img[:, :, 0][mask > 0]) if np.any(mask) else 0
    g_mean = np.mean(crop_img[:, :, 1][mask > 0]) if np.any(mask) else 0
    r_mean = np.mean(crop_img[:, :, 2][mask > 0]) if np.any(mask) else 0
    
    features['mean_b'] = b_mean
    features['mean_g'] = g_mean
    features['mean_r'] = r_mean
    
    # 4. Edge detection (Canny)
    edges = cv2.Canny(mask, 50, 150)
    edge_count = cv2.countNonZero(edges)
    features['edge_count'] = edge_count
    
    return features


# ====================================================
# TAHAP 8: VISUALISASI LENGKAP
# ====================================================

def visualize_complete_analysis(original_image, preprocessed_image, 
                               img_biota, img_starfish, img_merged,
                               segmentation_results, config: DetectionConfig,
                               filename: str):
    """
    Visualisasi lengkap menggunakan matplotlib subplot
    
    Menampilkan:
    1. Gambar asli
    2. Hasil preprocessing
    3. Hasil deteksi biota
    4. Hasil deteksi starfish
    5. Hasil merge detection
    6-8. Segmentasi + contour + edge untuk 3 objek pertama
    
    Args:
        original_image: Image original
        preprocessed_image: Image hasil preprocessing
        img_biota: Visualisasi deteksi biota
        img_starfish: Visualisasi deteksi starfish
        img_merged: Visualisasi merge deteksi
        segmentation_results: Hasil segmentasi semua objek
        config (DetectionConfig): Konfigurasi sistem
        filename (str): Nama file
    """
    
    print("[TAHAP 8] Creating Complete Visualization...")
    print("-" * 50)
    
    # Resize untuk display konsisten
    display_size = 300
    
    def resize_for_display(img):
        return cv2.resize(img, (display_size, display_size))
    
    # Tentukan jumlah subplot berdasarkan jumlah objek
    num_objects = min(3, len(segmentation_results))
    num_rows = 3
    num_cols = 3 + num_objects
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    fig.suptitle(f'Analisis Deteksi Underwater - {filename}', fontsize=16, fontweight='bold')
    
    # Row 1: Original, Preprocessed, Biota
    axes[0, 0].imshow(cv2.cvtColor(resize_for_display(original_image), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(resize_for_display(preprocessed_image), cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('2. Preprocessing')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(resize_for_display(img_biota), cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('3. Biota Detection')
    axes[0, 2].axis('off')
    
    # Row 2: Starfish, Merged, dan segmentasi pertama
    axes[1, 0].imshow(cv2.cvtColor(resize_for_display(img_starfish), cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('4. Starfish Detection')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(resize_for_display(img_merged), cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('5. Merge Detection')
    axes[1, 1].axis('off')
    
    # Row 3: Detail 3 objek
    for idx in range(num_objects):
        if idx < len(segmentation_results):
            seg_result = segmentation_results[idx]
            
            if seg_result is not None:
                # Masking
                axes[0, 3+idx].imshow(cv2.cvtColor(seg_result['dst'], cv2.COLOR_BGR2RGB))
                axes[0, 3+idx].set_title(f'Obj{idx+1}: Masking')
                axes[0, 3+idx].axis('off')
                
                # Contour
                contour_img = seg_result['crop_img'].copy()
                cv2.drawContours(contour_img, seg_result['contours'], -1, (0, 255, 0), 2)
                axes[1, 3+idx].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
                axes[1, 3+idx].set_title(f'Obj{idx+1}: Contour')
                axes[1, 3+idx].axis('off')
                
                # Edge detection
                edges = cv2.Canny(seg_result['mask'], 50, 150)
                axes[2, 3+idx].imshow(edges, cmap='gray')
                axes[2, 3+idx].set_title(f'Obj{idx+1}: Edge Detection')
                axes[2, 3+idx].axis('off')
    
    # Hapus subplot yang tidak digunakan
    for idx in range(num_objects, 3):
        if 3+idx < num_cols:
            for row in range(num_rows):
                fig.delaxes(axes[row, 3+idx])
    
    plt.tight_layout()
    
    # Simpan
    output_path = config.VIZ_DIR / f"{filename}_complete_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()
    
    print("-" * 50)


# ====================================================
# TAHAP 9: PENYIMPANAN HASIL
# ====================================================

def save_detection_results(merged_detections, filename: str, config: DetectionConfig):
    """
    Simpan hasil deteksi ke file txt
    
    Args:
        merged_detections (list): Deteksi merged
        filename (str): Nama file
        config (DetectionConfig): Konfigurasi sistem
    """
    output_file = config.DETECTION_DIR / f"{filename}_detections.txt"
    
    with open(output_file, 'w') as f:
        f.write("Detection Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"File: {filename}\n")
        f.write(f"Total objects: {len(merged_detections)}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        
        for idx, det in enumerate(merged_detections, 1):
            f.write(f"Object {idx}:\n")
            f.write(f"  Class: {config.CLASS_NAMES[det['class_id']]}\n")
            f.write(f"  Confidence: {det['confidence']:.4f}\n")
            f.write(f"  Bbox: ({det['x1']:.1f}, {det['y1']:.1f}, {det['x2']:.1f}, {det['y2']:.1f})\n")
            f.write(f"  Model: {det['model']}\n")
            if 'corrected_from' in det:
                f.write(f"  Corrected from: {det['corrected_from']}\n")
                f.write(f"  Correction reason: {det.get('correction_reason', '-')}\n")
            f.write("\n")
    
    print(f"Detection results saved: {output_file}")


def save_segmentation_results(segmentation_results, filename: str, config: DetectionConfig):
    """
    Simpan hasil mask, contour, dan edge detection tiap objek.
    """
    for idx, seg_result in enumerate(segmentation_results, 1):
        if seg_result is None:
            continue

        mask_path = config.SEGMENTATION_DIR / f"{filename}_object_{idx:03d}_mask.png"
        segmented_path = config.SEGMENTATION_DIR / f"{filename}_object_{idx:03d}_segmented.png"
        contour_path = config.SEGMENTATION_DIR / f"{filename}_object_{idx:03d}_contour.png"
        edge_path = config.SEGMENTATION_DIR / f"{filename}_object_{idx:03d}_edges.png"

        contour_img = seg_result['crop_img'].copy()
        cv2.drawContours(contour_img, seg_result['contours'], -1, (0, 255, 0), 2)
        edges = cv2.Canny(seg_result['mask'], 50, 150)

        cv2.imwrite(str(mask_path), seg_result['mask'])
        cv2.imwrite(str(segmented_path), seg_result['dst'])
        cv2.imwrite(str(contour_path), contour_img)
        cv2.imwrite(str(edge_path), edges)

    print(f"Segmentation artifacts saved: {config.SEGMENTATION_DIR}")


def save_feature_details(all_features, filename: str, config: DetectionConfig):
    """
    Simpan fitur per objek ke file teks untuk kebutuhan laporan/demo.
    """
    output_file = config.FEATURES_DIR / f"{filename}_features.txt"

    with open(output_file, 'w') as f:
        f.write("Feature Extraction Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"File: {filename}\n")
        f.write(f"Total analyzed objects: {len(all_features)}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")

        for feature in all_features:
            f.write(f"Object {feature.get('object_id', '-')}\n")
            f.write(f"  Class: {feature.get('class', '-')}\n")
            f.write(f"  Confidence: {feature.get('confidence', 0):.4f}\n")
            f.write(f"  Area: {feature.get('area', 0)}\n")
            f.write(f"  Perimeter: {feature.get('perimeter', 0):.4f}\n")
            f.write(f"  Mean RGB: ({feature.get('mean_r', 0):.2f}, {feature.get('mean_g', 0):.2f}, {feature.get('mean_b', 0):.2f})\n")
            f.write(f"  Circularity: {feature.get('circularity', 0):.4f}\n")
            f.write(f"  Aspect ratio: {feature.get('aspect_ratio', 0):.4f}\n")
            f.write(f"  Contour shape: {feature.get('contour_shape', '-')}\n")
            f.write(f"  Contour points: {feature.get('contour_points', 0)}\n")
            f.write(f"  Edge count: {feature.get('edge_count', 0)}\n\n")

    print(f"Feature detail saved: {output_file}")


def save_features_csv(all_features, config: DetectionConfig, append: bool = False):
    """
    Simpan hasil feature extraction ke CSV
    
    Args:
        all_features (list): List semua fitur
        config (DetectionConfig): Konfigurasi sistem
    """
    
    csv_columns = [
        'filename',
        'class',
        'confidence',
        'area',
        'perimeter',
        'mean_r',
        'mean_g',
        'mean_b',
        'circularity',
        'aspect_ratio',
        'object_id',
        'model',
        'edge_count',
        'contour_shape',
        'contour_points',
        'bbox_x1',
        'bbox_y1',
        'bbox_x2',
        'bbox_y2',
    ]
    
    df = pd.DataFrame(all_features)
    for column in csv_columns:
        if column not in df.columns:
            df[column] = ''
    df = df[csv_columns]
    
    output_file = config.CSV_DIR / "feature_results.csv"
    write_header = not (append and output_file.exists())
    df.to_csv(output_file, mode='a' if append else 'w', index=False, header=write_header)
    
    print(f"Features CSV saved: {output_file}")
    if not df.empty:
        print(f"\nFeature Statistics:")
        print(df.describe())
    else:
        print("No features to save; CSV header created")


def save_results(merged_detections, segmentation_results, all_features,
                 filename: str, config: DetectionConfig, append_csv: bool = False):
    """
    Simpan seluruh hasil akhir sistem ke struktur folder results/.
    """
    save_detection_results(merged_detections, filename, config)
    save_segmentation_results(segmentation_results, filename, config)
    save_feature_details(all_features, filename, config)
    save_features_csv(all_features, config, append=append_csv)


# ====================================================
# VALIDASI SISTEM
# ====================================================

def validate_system(merged_detections, biota_detections, starfish_detections, 
                   all_features, config: DetectionConfig):
    """
    Validasi sistem dan hitung statistik
    
    Args:
        merged_detections (list): Deteksi merged
        biota_detections (list): Deteksi biota
        starfish_detections (list): Deteksi starfish
        all_features (list): Semua fitur
        config (DetectionConfig): Konfigurasi sistem
    """
    
    print("\n" + "=" * 60)
    print("SISTEM VALIDATION REPORT")
    print("=" * 60)
    
    # Total objects
    print(f"\n[TOTAL OBJECTS]")
    print(f"Total objects detected: {len(merged_detections)}")
    print(f"Raw biota model objects: {len(biota_detections)}")
    print(f"Raw starfish model objects: {len(starfish_detections)}")
    
    # Detection per class
    print(f"\n[DETECTION PER CLASS]")
    class_count = defaultdict(int)
    for det in merged_detections:
        class_name = config.CLASS_NAMES[det['class_id']]
        class_count[class_name] += 1
    
    for class_name, count in sorted(class_count.items()):
        print(f"  {class_name}: {count}")

    final_starfish = class_count.get('starfish', 0)
    final_other_biota = sum(
        count for class_name, count in class_count.items()
        if class_name != 'starfish'
    )
    print(f"\n[FINAL SUMMARY]")
    print(f"Jumlah starfish terdeteksi: {final_starfish}")
    print(f"Jumlah biota laut lain terdeteksi: {final_other_biota}")
    
    # Average confidence
    print(f"\n[AVERAGE CONFIDENCE]")
    if merged_detections:
        avg_conf = np.mean([det['confidence'] for det in merged_detections])
        print(f"Overall average: {avg_conf:.4f}")

        final_biota_detections = [
            det for det in merged_detections
            if det['class_id'] != config.STARFISH_CLASS_ID
        ]
        final_starfish_detections = [
            det for det in merged_detections
            if det['class_id'] == config.STARFISH_CLASS_ID
        ]

        if final_biota_detections:
            final_biota_conf = np.mean([
                det['confidence'] for det in final_biota_detections
            ])
            print(f"Biota average: {final_biota_conf:.4f}")
        else:
            print("Biota average: N/A (no final biota detection)")

        if final_starfish_detections:
            final_starfish_conf = np.mean([
                det['confidence'] for det in final_starfish_detections
            ])
            print(f"Starfish average: {final_starfish_conf:.4f}")
        else:
            print("Starfish average: N/A (no final starfish detection)")

        print(f"\n[RAW MODEL CONFIDENCE]")
        if biota_detections:
            biota_conf = np.mean([det['confidence'] for det in biota_detections])
            print(f"Raw biota model average: {biota_conf:.4f}")
        else:
            print("Raw biota model average: N/A")

        if starfish_detections:
            starfish_conf = np.mean([det['confidence'] for det in starfish_detections])
            print(f"Raw starfish model average: {starfish_conf:.4f}")
        else:
            print("Raw starfish model average: N/A")

        print(f"\n[FINAL CONFIDENCE PER CLASS]")
        for class_id, class_name in config.CLASS_NAMES.items():
            class_detections = [
                det for det in merged_detections
                if det['class_id'] == class_id
            ]
            if class_detections:
                class_conf = np.mean([det['confidence'] for det in class_detections])
                print(f"{class_name} average: {class_conf:.4f}")
            else:
                print(f"{class_name} average: N/A")
    else:
        print("Overall average: N/A (no detections)")
        print("Biota average: N/A (no final biota detection)")
        print("Starfish average: N/A (no final starfish detection)")
    
    # Feature statistics
    if all_features:
        print(f"\n[FEATURE STATISTICS]")
        df_features = pd.DataFrame(all_features)
        print(f"Average area: {df_features['area'].mean():.2f}")
        print(f"Average perimeter: {df_features['perimeter'].mean():.2f}")
        print(f"Average circularity: {df_features['circularity'].mean():.4f}")
        print(f"Average aspect ratio: {df_features['aspect_ratio'].mean():.4f}")
    
    print("\n" + "=" * 60 + "\n")


# ====================================================
# PIPELINE UTAMA
# ====================================================

def process_single_image(image_path: str, config: DetectionConfig = None,
                         models=None, append_csv: bool = False):
    """
    Proses satu image lengkap dari awal sampai akhir
    
    Args:
        image_path (str): Path ke file image
        config (DetectionConfig): Konfigurasi sistem (default: DetectionConfig())
        models (tuple): Model yang sudah diload, agar batch processing lebih cepat
        append_csv (bool): Tambahkan hasil ke CSV yang sudah ada
    """
    
    if config is None:
        config = DetectionConfig()
    
    prepare_output_dirs(config)
    
    print("\n" + "=" * 60)
    print("SISTEM DETEKSI MULTI-MODEL YOLOV8 - UNDERWATER")
    print("=" * 60 + "\n")
    
    try:
        resolved_image_path = resolve_image_path(image_path)
        filename = resolved_image_path.stem

        # TAHAP 1: Load models
        if models is None:
            model_biota, model_starfish = load_models(config)
        else:
            model_biota, model_starfish = models
        
        # TAHAP 2: Preprocessing
        original_image, preprocessed_image = preprocess_image(str(resolved_image_path), config)
        
        # TAHAP 3: Inference paralel dua model
        biota_detections, starfish_detections = run_parallel_detection(
            model_biota, model_starfish, preprocessed_image, config
        )

        # Koreksi post-processing untuk false-positive starfish pada jellyfish.
        biota_detections = apply_color_class_corrections(
            biota_detections, preprocessed_image, config
        )
        starfish_detections = apply_color_class_corrections(
            starfish_detections, preprocessed_image, config
        )
        
        # TAHAP 4: Merge
        merged_detections = merge_detections(biota_detections, starfish_detections, config)
        
        # TAHAP 5: Visualisasi deteksi
        img_biota, img_starfish, img_merged = visualize_detections(
            original_image, preprocessed_image, merged_detections, 
            biota_detections, starfish_detections, config
        )
        
        # TAHAP 6 & 7: Segmentasi dan ekstraksi fitur
        print("[TAHAP 6-7] Segmentation & Feature Extraction...")
        print("-" * 50)
        segmentation_results = []
        all_features = []
        
        for idx, detection in enumerate(merged_detections):
            print(f"Processing object {idx+1}/{len(merged_detections)}")
            
            seg_result = segment_object(preprocessed_image, detection, config)
            segmentation_results.append(seg_result)
            
            if seg_result is not None:
                features = extract_features(seg_result, detection, config)
                if features is not None:
                    features['filename'] = filename
                    features['object_id'] = idx + 1
                    all_features.append(features)
        
        print(f"Segmentasi dan features untuk {len(all_features)} objek")
        print("-" * 50)
        
        # TAHAP 8: Visualisasi lengkap
        visualize_complete_analysis(original_image, preprocessed_image,
                                   img_biota, img_starfish, img_merged,
                                   segmentation_results, config, filename)
        
        # TAHAP 9: Penyimpanan hasil
        print("\n[TAHAP 9] Saving Results...")
        print("-" * 50)
        save_results(
            merged_detections,
            segmentation_results,
            all_features,
            filename,
            config,
            append_csv=append_csv
        )
        
        # Simpan gambar visualisasi individual
        cv2.imwrite(str(config.VIZ_DIR / f"{filename}_biota_detection.jpg"), img_biota)
        cv2.imwrite(str(config.VIZ_DIR / f"{filename}_starfish_detection.jpg"), img_starfish)
        cv2.imwrite(str(config.VIZ_DIR / f"{filename}_merged_detection.jpg"), img_merged)
        
        print(f"Saved detection images to {config.VIZ_DIR}")
        print("-" * 50)
        
        # Validasi
        validate_system(merged_detections, biota_detections, starfish_detections,
                       all_features, config)

        return {
            'filename': filename,
            'detections': merged_detections,
            'features': all_features,
            'biota_detections': biota_detections,
            'starfish_detections': starfish_detections,
        }
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_multiple_images(image_dir: str, config: DetectionConfig = None):
    """
    Proses multiple images dalam satu direktori
    
    Args:
        image_dir (str): Direktori berisi images
        config (DetectionConfig): Konfigurasi sistem
    """
    
    if config is None:
        config = DetectionConfig()

    prepare_output_dirs(config)
    
    image_path = Path(image_dir)
    image_files = (
        list(image_path.glob("*.jpg")) +
        list(image_path.glob("*.jpeg")) +
        list(image_path.glob("*.png")) +
        list(image_path.glob("*.bmp"))
    )
    
    print(f"\nFound {len(image_files)} images to process\n")

    if not image_files:
        save_features_csv([], config, append=False)
        return []

    # Load model sekali untuk seluruh batch agar proses lebih cepat.
    models = load_models(config)
    all_results = []
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {idx}/{len(image_files)}")
        print(f"{'='*60}")
        result = process_single_image(
            str(img_file),
            config,
            models=models,
            append_csv=(idx > 1)
        )
        if result is not None:
            all_results.append(result)

    return all_results


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistem deteksi multi-model YOLOv8 untuk gambar underwater."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help=(
            "Path file gambar, folder gambar, atau nama file yang ada di folder "
            "dataset images."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Folder output hasil deteksi. Default: results",
    )
    args = parser.parse_args()

    if not args.input_path:
        parser.print_help()
        print("\nExamples:")
        print("  python dual_model_detection_system.py starfish_only_test_00018.jpg")
        print("  python dual_model_detection_system.py combined_detection_dataset/test/images")
        raise SystemExit(1)

    input_path = Path(args.input_path)

    # Initialize config
    config = DetectionConfig()

    if args.results_dir:
        config.RESULTS_DIR = Path(args.results_dir)
        config.refresh_output_dirs()

    if input_path.is_dir():
        process_multiple_images(str(input_path), config)
    else:
        process_single_image(args.input_path, config)
