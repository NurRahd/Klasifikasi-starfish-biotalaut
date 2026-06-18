# Quick Reference - YOLOv8 Underwater Segmentation Setup

## Checklist

- [x] **Stage 1: Dataset Combination**
  - ✓ `combined_dataset/` created with 7,040 images
  - ✓ Biota + Starfish datasets merged
  - ✓ Prefix added to starfish files

- [ ] **Stage 2: Preprocessing** (In Progress - ~40-50%)
  - [ ] `combined_dataset_preprocessed/` will contain enhanced images
  - [ ] Status: Running (ETA: 5-10 minutes)
  - [ ] Enhancement: 7 techniques applied

- [ ] **Stage 3: Training** (Next)
  - [ ] YOLOv8 model training
  - [ ] Validation & testing
  - [ ] Inference & deployment

## Files Generated

| File | Purpose | Status |
|------|---------|--------|
| `combine_datasets.py` | Dataset combination script | ✓ Complete |
| `combined_dataset/` | Raw combined dataset | ✓ Complete (7,040 images) |
| `preprocess_dataset.py` | Preprocessing script | ✓ Complete |
| `combined_dataset_preprocessed/` | Enhanced dataset | ⏳ In Progress |
| `PREPROCESSING_DOCUMENTATION.md` | Detailed guide | ✓ Complete |
| `WORKFLOW_SUMMARY.md` | Project overview | ✓ Complete |
| `preprocessing.log` | Execution log | ⏳ Being created |

## Quick Start Training

```bash
# After preprocessing completes:
python train_yolov8.py

# Or using YOLOv8 CLI:
yolo detect train data=combined_dataset_preprocessed/data.yaml \
                       model=yolov8s-seg.pt \
                       epochs=100 \
                       imgsz=640 \
                       device=0
```

## Dataset Structure

```
combined_dataset_preprocessed/
├── train/
│   ├── images/      (5,490 images - 640×640 preprocessed)
│   └── labels/      (5,490 YOLO format labels)
├── valid/
│   ├── images/      (1,107 images)
│   └── labels/      (1,107 labels)
├── test/
│   ├── images/      (443 images)
│   └── labels/      (443 labels)
└── data.yaml        (Dataset config)
```

## Preprocessing Techniques Applied

1. ✓ Resize to 640×640
2. ✓ Gaussian Blur (5×5)
3. ✓ Brightness +5, Contrast ×1.2
4. ✓ CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. ✓ Underwater color correction (R×1.3, G×1.1)
6. ✓ Bilateral Filter (edge-preserving denoising)
7. ✓ Histogram normalization

## Performance Metrics

- **Processing Speed:** 17-20 img/s
- **Total Processing Time:** ~7-10 minutes
- **Dataset Size:** ~2-3 GB
- **Label Files:** 7,040 .txt files (YOLO format)

## Next Commands

When preprocessing is complete:

```bash
# View a sample comparison
open preprocessing_comparison.png

# Check log for any issues
cat preprocessing.log

# Verify file count
ls -la combined_dataset_preprocessed/train/images | wc -l
ls -la combined_dataset_preprocessed/valid/images | wc -l
ls -la combined_dataset_preprocessed/test/images | wc -l

# Start training
python -m yolov8 detect train data=combined_dataset_preprocessed/data.yaml
```

## Troubleshooting

**Q: Preprocessing takstalled?**
A: Check `preprocessing.log` for errors. If stuck, can safely restart - it will overwrite.

**Q: Labels not copied?**
A: Verify `combined_dataset_preprocessed/train/labels` exists. Labels should match image count.

**Q: Images look over-processed?**
A: Adjust parameters in `preprocess_dataset.py` and re-run preprocessing.

## Supported Models for Training

```bash
# Segmentation models
yolo detect train model=yolov8n-seg.pt    # Nano (fastest)
yolo detect train model=yolov8s-seg.pt    # Small (recommended)
yolo detect train model=yolov8m-seg.pt    # Medium
yolo detect train model=yolov8l-seg.pt    # Large
yolo detect train model=yolov8x-seg.pt    # Extra Large
```

## Useful Paths

- **Source (Raw):** `combined_dataset/`
- **Processed:** `combined_dataset_preprocessed/`
- **Logs:** `preprocessing.log`, `preprocessing_output.txt`
- **Comparison:** `preprocessing_comparison.png`
- **Scripts:** `combine_datasets.py`, `preprocess_dataset.py`

---

**Status:** Preprocessing in progress (~50% complete)
**ETA Completion:** ~5 minutes
**Next Step:** Wait for completion, then proceed to training
