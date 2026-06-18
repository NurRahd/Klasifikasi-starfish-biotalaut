"""Lanjutkan training biota 3-class dari checkpoint terakhir."""

from pathlib import Path
import shutil

from ultralytics import YOLO


LAST_WEIGHT = Path("runs/detect/biota_3class_train/weights/last.pt")
BEST_WEIGHT = Path("runs/detect/biota_3class_train/weights/best.pt")
OUTPUT_WEIGHT = Path("trained_biota_3class_model_best.pt")


def main():
    if not LAST_WEIGHT.exists():
        raise FileNotFoundError(f"Checkpoint last.pt tidak ditemukan: {LAST_WEIGHT}")

    model = YOLO(str(LAST_WEIGHT))
    model.train(resume=True)

    if BEST_WEIGHT.exists():
        shutil.copy2(BEST_WEIGHT, OUTPUT_WEIGHT)
        print(f"Best model disalin ke: {OUTPUT_WEIGHT}")


if __name__ == "__main__":
    main()
