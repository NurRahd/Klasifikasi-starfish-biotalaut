"""
DUAL MODEL DETECTION SYSTEM - QUICK START GUIDE
================================================

Sistem deteksi underwater multi-model menggunakan YOLOv8 paralel.

REQUIREMENTS
============

✓ Model .pt files:
  - trained_3class_model_best.pt (eel, fish, jellyfish)
  - cascade_model_starfish_best.pt (starfish)

✓ Python packages:
  - ultralytics (YOLO)
  - opencv-python
  - numpy
  - pandas
  - matplotlib


INSTALLATION
=============

1. Ensure .pt models in root directory:
   $ ls *.pt
   trained_3class_model_best.pt
   cascade_model_starfish_best.pt
   yolov8n.pt

2. Install requirements (if not already installed):
   $ pip install ultralytics opencv-python numpy pandas matplotlib


QUICK START
===========

Method 1: Process Single Image
------------------------------

from dual_model_detection_system import process_single_image

process_single_image("path/to/image.jpg")

Hasil tersimpan di: results/


Method 2: Process Multiple Images
---------------------------------

from dual_model_detection_system import process_multiple_images

process_multiple_images("path/to/images/directory")

Semua images di direktori akan diproses.


Method 3: Custom Configuration
------------------------------

from dual_model_detection_system import DetectionConfig, process_single_image

config = DetectionConfig()
config.CONF_BIOTA = 0.3      # Ubah confidence threshold
config.CONF_STARFISH = 0.5

process_single_image("path/to/image.jpg", config)


SYSTEM FLOW
===========

Input: Image underwater
  ↓
[TAHAP 1] Load Models
  - trained_3class_model_best.pt
  - cascade_model_starfish_best.pt
  ↓
[TAHAP 2] Preprocessing
  - Resize to 640x640
  - Gaussian Blur
  - CLAHE enhancement
  - Brightness/Contrast adjustment
  ↓
[TAHAP 3] Parallel Inference
  - Biota model (conf: 0.4)
  - Starfish model (conf: 0.6)
  ↓
[TAHAP 4] Merge Detections
  - Combine results dari 2 model
  ↓
[TAHAP 5] Visualize Detections
  - Draw bounding boxes
  - Add class labels & confidence
  ↓
[TAHAP 6-7] Segmentation & Features
  - GrabCut segmentation
  - Extract 10+ features (area, perimeter, color, etc)
  ↓
[TAHAP 8] Complete Analysis Visualization
  - 8+ subplot dengan semua tahap
  - Save to visualizations/
  ↓
[TAHAP 9] Save Results
  - Detection results (.txt)
  - Feature extraction (.csv)
  - Visualizations (.jpg)
  ↓
Output: results/


OUTPUT STRUCTURE
================

results/
├── detection/
│   ├── image1_detections.txt
│   ├── image2_detections.txt
│   └── ...
├── segmentation/
│   ├── masks/
│   └── contours/
├── features/
│   └── ...
├── visualizations/
│   ├── image1_complete_analysis.png (8+ subplots)
│   ├── image1_biota_detection.jpg
│   ├── image1_starfish_detection.jpg
│   ├── image1_merged_detection.jpg
│   └── ...
└── csv/
    └── feature_results.csv


CSV OUTPUT: feature_results.csv
================================

Columns:
  - filename: Nama file image
  - object_id: ID objek dalam image
  - class: Nama class (eel, fish, jellyfish, starfish)
  - confidence: Confidence score dari model
  - model: Model yang mendeteksi (biota/starfish)
  - area: Area objek dalam pixel
  - perimeter: Perimeter contour objek
  - mean_b, mean_g, mean_r: Rata-rata warna BGR
  - circularity: Bentuk circularity (0-1, higher = lebih circular)
  - aspect_ratio: Ratio tinggi/lebar bounding box
  - edge_count: Jumlah edge pixels


CLASS MAPPING
=============

0 = eel (merah)
1 = fish (hijau)
2 = jellyfish (biru)
3 = starfish (kuning)


CONFIDENCE THRESHOLDS
=====================

Default:
  - Biota model: 0.4
  - Starfish model: 0.6

Tips untuk adjust:
  - Tingkatkan untuk mengurangi false positives (lebih ketat)
  - Turunkan untuk menangkap objek kecil/samar (lebih sensitif)


EXAMPLE: Image Dari Dataset Test
==================================

1. Lihat images di:
   combined_dataset/test/images/

2. Jalankan:
   from dual_model_detection_system import process_single_image
   process_single_image("combined_dataset/test/images/sample.jpg")

3. Lihat hasil di:
   results/visualizations/
   results/csv/feature_results.csv


ADVANCED USAGE
==============

1. Process dengan custom output directory:
   
   config = DetectionConfig()
   config.RESULTS_DIR = Path("my_custom_results")
   config.RESULTS_DIR.mkdir(exist_ok=True)
   process_single_image("image.jpg", config)

2. Extract hanya detections tanpa full analysis:
   
   model_biota, model_starfish = load_models(config)
   original, preprocessed = preprocess_image("image.jpg", config)
   biota_det = run_biota_detection(model_biota, preprocessed, config)
   starfish_det = run_starfish_detection(model_starfish, preprocessed, config)
   merged = merge_detections(biota_det, starfish_det)


TROUBLESHOOTING
===============

Q: Model file not found
A: Pastikan .pt files di root directory atau update path di DetectionConfig

Q: Memory error
A: Turunkan batch size atau gunakan GPU (jika available)

Q: No detections found
A: Turunkan confidence threshold di config

Q: Image size mismatch
A: System otomatis resize ke 640x640, tidak perlu adjust


FITUR UTAMA
===========

✓ Multi-model parallel detection
✓ Advanced preprocessing (CLAHE, denoising)
✓ Automatic result merging
✓ GrabCut segmentation
✓ 10+ feature extraction
✓ Complete visualization (8 subplots)
✓ CSV export
✓ Batch processing
✓ Modular architecture
✓ Comprehensive documentation


UNTUK PRESENTASI/SIDANG
=======================

File yang penting untuk ditampilkan:

1. results/visualizations/*_complete_analysis.png
   - Menunjukkan semua tahapan processing
   - Cocok untuk presentasi teknis

2. results/csv/feature_results.csv
   - Hasil ekstraksi fitur
   - Untuk analisis statistik

3. results/detection/*.txt
   - Detail setiap deteksi
   - Untuk validasi akurasi

4. results/visualizations/*_merged_detection.jpg
   - Hasil deteksi final
   - Menunjukkan efektivitas dual-model


CATATAN PENTING
===============

- Sistem dirancang untuk gambar underwater
- Preprocessing khusus untuk meningkatkan visibility
- Dual-model approach untuk akurasi lebih baik
- Feature extraction untuk analisis mendalam
- Modular design untuk mudah dikustomisasi
- Lengkap dengan dokumentasi kode


KONTRIBUSI NOTES
================

Sistem ini menggabungkan:
- YOLO object detection architecture
- OpenCV computer vision techniques
- Advanced image preprocessing
- Feature extraction & analysis
- Data visualization & reporting

Cocok untuk:
- Penelitian marine biology
- Environmental monitoring
- Oceanographic studies
- Computer vision projects
"""

# QUICK COMMAND REFERENCE
# =======================
# 
# Process 1 image:
#   python -c "from dual_model_detection_system import process_single_image; process_single_image('image.jpg')"
#
# Process directory:
#   python -c "from dual_model_detection_system import process_multiple_images; process_multiple_images('images/')"
#
# View results:
#   - Visualizations: results/visualizations/
#   - CSV data: results/csv/feature_results.csv
#   - Detections: results/detection/
