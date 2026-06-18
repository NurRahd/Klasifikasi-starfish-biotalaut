"""Train YOLOv8 biota 3-class model: eel, fish, jellyfish."""

from pathlib import Path
import shutil

from ultralytics import YOLO


DATA_YAML = Path("biota_3class_dataset") / "data.yaml"
BASE_WEIGHTS = "yolov8n.pt"
EPOCHS = 15
IMGSZ = 640
BATCH = 2
PATIENCE = 5
DEVICE = 0
WORKERS = 0
RUN_NAME = "biota_3class_train"
OUTPUT_WEIGHT = Path("trained_biota_3class_model_best.pt")


def main():
    print("=" * 80)
    print("Training Biota 3-Class YOLOv8 Model")
    print("Classes: 0=eel, 1=fish, 2=jellyfish")
    print("=" * 80)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML tidak ditemukan: {DATA_YAML}")

    print(f"Dataset: {DATA_YAML}")
    print(f"Base weights: {BASE_WEIGHTS}")
    print(f"Epochs: {EPOCHS}, imgsz: {IMGSZ}, batch: {BATCH}, device: {DEVICE}, workers: {WORKERS}")

    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        name=RUN_NAME,
        verbose=True,
    )

    run_dir = Path("runs") / "detect" / RUN_NAME
    best_model = run_dir / "weights" / "best.pt"
    if not best_model.exists():
        # Ultralytics may auto-increment the run name if it already exists.
        candidates = sorted(
            Path("runs/detect").glob(f"{RUN_NAME}*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            best_model = candidates[-1]

    if not best_model.exists():
        raise FileNotFoundError("best.pt tidak ditemukan setelah training selesai")

    shutil.copy2(best_model, OUTPUT_WEIGHT)
    print("\nTraining selesai.")
    print(f"Best model: {best_model}")
    print(f"Copied to: {OUTPUT_WEIGHT}")


if __name__ == "__main__":
    main()
