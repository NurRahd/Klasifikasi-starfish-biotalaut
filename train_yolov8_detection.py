"""Evaluasi dan inference YOLOv8 Detection untuk dataset underwater.

Fitur:
- load dataset otomatis dari `combined_detection_dataset/data.yaml`
- pakai `train-2/weights/best.pt` jika sudah tersedia
- evaluasi model pada validation set
- inferensi pada test/images/
- visualisasi dan penyimpanan hasil ke folder `results/`

Jalankan:
    python train_yolov8_detection.py

Catatan: pastikan environment memiliki paket `ultralytics`, `torch`, `opencv-python`, `matplotlib`, `pandas`, `numpy`.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency
    print("ERROR: library 'ultralytics' tidak ditemukan. Install dengan: pip install ultralytics")
    raise


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
SOURCE_DATA_YAML = Path("combined_detection_dataset") / "data.yaml"
PREFERRED_BEST_WEIGHT = Path("runs") / "detect" / "train-2" / "weights" / "best.pt"
WEIGHTS = "yolov8n.pt"
RESULTS_DIR = Path("results")
RESULTS_DETECTION = RESULTS_DIR / "detection"
RESULTS_VALIDATION = RESULTS_DIR / "validation"
RESULTS_PLOTS = RESULTS_DIR / "plots"
RESULTS_PREDICTIONS = RESULTS_DIR / "predictions"

# Training hyperparameters
EPOCHS = 5
IMGSZ = 640
BATCH = 8
PATIENCE = 3
WORKERS = 2
DEVICE = "cpu"
SAVE_PERIOD = 5


def ensure_result_dirs() -> None:
    """Buat folder hasil jika belum ada."""

    for p in [RESULTS_DIR, RESULTS_DETECTION, RESULTS_VALIDATION, RESULTS_PLOTS, RESULTS_PREDICTIONS]:
        p.mkdir(parents=True, exist_ok=True)


def load_dataset_yaml() -> Path:
    """Kembalikan path ke data.yaml (raise jika tidak ada)."""

    if not SOURCE_DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan pada {SOURCE_DATA_YAML}")
    return SOURCE_DATA_YAML


def train_model(
    weights: str = WEIGHTS,
    data: str | Path = SOURCE_DATA_YAML,
    epochs: int = EPOCHS,
    imgsz: int = IMGSZ,
    batch: int = BATCH,
    workers: int = WORKERS,
    device: str = DEVICE,
    patience: int = PATIENCE,
    save_period: int = SAVE_PERIOD,
) -> Tuple[YOLO, Path]:
    """Melatih YOLOv8 dan mengembalikan instance model beserta folder run terbaru.

    Model akan menggunakan weights awal `weights` (pretrained).
    """

    print("Mulai training YOLOv8 detection...")
    model = YOLO(weights)

    # Jalankan training
    result = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        patience=patience,
        save=True,
        save_period=save_period,
    )

    # Ultralyics menyimpan runs di runs/train/expN. Cari run terbaru.
    runs_dir = Path("runs") / "train"
    if runs_dir.exists():
        exps = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
        if exps:
            latest_run = exps[-1]
            print(f"Training selesai; run folder: {latest_run}")
            return model, latest_run

    # fallback
    print("Training selesai; namun folder run tidak ditemukan, gunakan model yang ada di memori")
    return model, Path(".")


def evaluate_model(model: YOLO, data: str | Path = SOURCE_DATA_YAML) -> Dict:
    """Evaluasi model pada validation set dan kembalikan metrik.

    Mengembalikan dictionary metrik utama bila tersedia.
    """

    print("Menjalankan evaluasi pada validation set...")
    try:
        metrics = model.val(data=str(data))
        # `metrics` biasanya berisi summary. Kembalikan apa adanya.
        return {"raw": metrics}
    except Exception as exc:
        print(f"Evaluasi gagal: {exc}")
        return {}


def _save_run_weights(run_folder: Path) -> Tuple[Path, Path]:
    """Salin best.pt dan last.pt ke folder results/detection dan kembalikan pathnya."""

    best = None
    last = None
    for name in ("best.pt", "last.pt"):
        candidate = run_folder / name
        if candidate.exists():
            dest = RESULTS_DETECTION / name
            shutil.copy2(candidate, dest)
            if name == "best.pt":
                best = dest
            else:
                last = dest

    return best, last


def run_inference(model_source: Path | YOLO, test_images_dir: Path) -> List[Dict]:
    """Jalankan inferensi pada folder test dan simpan gambar terannotasi.

    Mengembalikan list hasil per gambar (jumlah deteksi, rata-rata confidence, per-class counts).
    """

    if isinstance(model_source, YOLO):
        model = model_source
        model_desc = "model in-memory"
    else:
        model = YOLO(str(model_source))
        model_desc = str(model_source)

    print(f"Menjalankan inferensi menggunakan {model_desc} pada {test_images_dir}")

    results_summary = []
    image_paths = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    for img_path in image_paths:
        res = model.predict(source=str(img_path), imgsz=IMGSZ)
        # res adalah list of Results; ambil pertama
        if not res:
            continue
        r = res[0]
        # simpan gambar terannotasi
        try:
            annotated = r.plot()
            out_path = RESULTS_PREDICTIONS / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        except Exception:
            out_path = None

        detections = 0
        confs = []
        class_counts: Dict[int, int] = {}
        try:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
                detections += 1
                confs.append(conf)
                class_counts[cls] = class_counts.get(cls, 0) + 1
        except Exception:
            # fallback untuk format lain
            pass

        avg_conf = float(np.mean(confs)) if confs else 0.0
        summary = {
            "image": str(img_path.relative_to(test_images_dir.parent)),
            "detections": detections,
            "avg_confidence": avg_conf,
            "per_class": class_counts,
            "annotated_path": str(out_path) if out_path is not None else "",
        }
        results_summary.append(summary)

    # simpan ringkasan sebagai CSV
    df = pd.DataFrame(results_summary)
    df.to_csv(RESULTS_DETECTION / "inference_summary.csv", index=False)
    return results_summary


def visualize_predictions(samples: List[Dict], n_cols: int = 3) -> None:
    """Visualisasikan beberapa prediksi sampel (original vs annotated).

    `samples` adalah list dict sebagaimana dikembalikan oleh `run_inference`.
    """

    if not samples:
        print("Tidak ada sampel untuk divisualisasikan")
        return

    n = min(len(samples), 12)
    cols = n_cols
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        sample = samples[i]
        annotated_path = Path(sample.get("annotated_path", ""))
        orig_rel = Path(sample.get("image"))
        orig = Path(".") / orig_rel
        ax = axes[i]
        if annotated_path.exists():
            img = cv2.imread(str(annotated_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        elif orig.exists():
            img = cv2.imread(str(orig))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "file not found", ha="center")
        ax.set_title(f"{Path(sample['image']).name}\nDet: {sample['detections']}, conf: {sample['avg_confidence']:.2f}")
        ax.axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    out = RESULTS_PLOTS / "inference_samples.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Visualisasi sampel disimpan ke: {out}")


def aggregate_inference_stats(summaries: List[Dict]) -> Dict:
    """Hitung statistik ringkas dari hasil inferensi."""

    total_detected = sum(s.get('detections', 0) for s in summaries)
    avg_conf = float(np.mean([s.get('avg_confidence', 0.0) for s in summaries])) if summaries else 0.0
    per_class = {}
    for s in summaries:
        for k, v in s.get('per_class', {}).items():
            per_class[k] = per_class.get(k, 0) + v

    stats = {
        'total_objects_detected': int(total_detected),
        'detection_confidence_avg': float(avg_conf),
        'detections_per_class': per_class,
    }
    # simpan
    pd.DataFrame([stats]).to_csv(RESULTS_DETECTION / 'inference_stats.csv', index=False)
    return stats


def main():
    ensure_result_dirs()
    data_yaml = load_dataset_yaml()

    if not PREFERRED_BEST_WEIGHT.exists():
        raise FileNotFoundError(
            f"Weights tidak ditemukan: {PREFERRED_BEST_WEIGHT}. "
            "Letakkan hasil training di lokasi itu atau ubah PREFERRED_BEST_WEIGHT."
        )

    print(f"Menggunakan weights yang sudah ada: {PREFERRED_BEST_WEIGHT}")
    model = YOLO(str(PREFERRED_BEST_WEIGHT))
    best = PREFERRED_BEST_WEIGHT

    # evaluasi
    eval_res = evaluate_model(model, data=data_yaml)
    pd.Series({'eval_raw': str(eval_res)}).to_csv(RESULTS_VALIDATION / 'evaluation_raw.txt', index=False)

    # gunakan best.pt yang sudah ada untuk inferensi
    model_path = best
    print(f'Gunakan best model: {model_path}')

    test_images = Path('combined_detection_dataset') / 'test' / 'images'
    summaries = run_inference(model_path if model_path is not None else model, test_images)
    stats = aggregate_inference_stats(summaries)
    print('Inference summary:', stats)

    visualize_predictions(summaries)

    print('Semua selesai. Lihat folder results/ untuk keluaran dan plot.')


if __name__ == '__main__':
    main()
