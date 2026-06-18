# Sistem Deteksi Multi-Model YOLOv8 untuk Underwater Imaging

**Judul Proyek**: Deteksi Bintang Laut dan Biota Laut Dangkal Menggunakan YOLOv8

## 📋 Ringkasan

Sistem ini mengimplementasikan deteksi underwater objects menggunakan **dua model YOLOv8 secara paralel**:

1. **Model Biota Laut**: `trained_3class_model_best.pt`
   - eel (belut)
   - fish (ikan)
   - jellyfish (ubur-ubur)

2. **Model Starfish**: `cascade_model_starfish_best.pt`
   - starfish (bintang laut)

## ✨ Fitur Utama

### 🔄 Multi-Model Architecture
- Dual-model parallel processing untuk akurasi lebih tinggi
- Confidence threshold berbeda untuk setiap model:
  - Biota: 0.4 (lebih sensitif untuk objek kecil)
  - Starfish: 0.6 (lebih ketat untuk presisi)

### 🖼️ Advanced Image Preprocessing
- Resize ke 640x640 (standar YOLO)
- Gaussian Blur untuk noise reduction
- CLAHE contrast enhancement (khusus underwater)
- Brightness & contrast adjustment

### 📊 Komprehensif Feature Extraction
- Area objek
- Perimeter contour
- Rata-rata warna BGR
- Edge detection (Canny)
- Circularity (bentuk objek)
- Aspect ratio

### 🎨 Segmentasi Objek Lanjutan
- GrabCut algorithm untuk segmentasi akurat
- Morphological operations (opening/closing)
- Contour extraction
- Automatic fallback ke thresholding jika diperlukan

### 📈 Visualisasi Lengkap
- 8+ subplot menampilkan setiap tahap processing
- Bounding box dengan color-coding per class
- Confidence score display
- Segmentation masks dan contours

### 💾 Penyimpanan Terstruktur
```
results/
├── detection/          # Detection results (.txt)
├── segmentation/       # Segmentation masks
├── features/           # Feature data
├── visualizations/     # Output images & plots
└── csv/               # feature_results.csv
```

### 📊 Data Export
- Feature results ke CSV untuk analisis lebih lanjut
- Detection text files untuk validasi
- Visualization images untuk presentasi

## 🚀 Quick Start

### 1. Setup

```bash
# Pastikan models ada di directory
ls *.pt
# Output:
# trained_3class_model_best.pt
# cascade_model_starfish_best.pt

# Install dependencies (jika belum)
pip install ultralytics opencv-python numpy pandas matplotlib
```

### 2. Process Single Image

```python
from dual_model_detection_system import process_single_image

process_single_image("path/to/image.jpg")
```

### 3. Process Multiple Images

```python
from dual_model_detection_system import process_multiple_images

process_multiple_images("path/to/images/directory")
```

### 4. Custom Configuration

```python
from dual_model_detection_system import DetectionConfig, process_single_image

config = DetectionConfig()
config.CONF_BIOTA = 0.3      # Lebih sensitif
config.CONF_STARFISH = 0.5   # Lebih ketat

process_single_image("image.jpg", config)
```

## 📁 File Structure

```
dual_model_detection_system.py    # Main system (1000+ lines)
example_usage.py                  # Contoh penggunaan
test_demo.py                      # Testing & diagnostic
DUAL_MODEL_GUIDE.md              # Quick reference guide
README.md                         # Documentation (this file)
```

## 📚 Dokumentasi Detail

### Tahap-Tahap Processing

#### Tahap 1: Load Model
- Load 2 model YOLO dari file .pt
- Validasi model availability

#### Tahap 2: Preprocessing
- Resize image ke 640x640
- Apply Gaussian blur
- CLAHE contrast enhancement
- Brightness/contrast adjustment

#### Tahap 3: Inference Paralel
- Model biota dengan conf=0.4
- Model starfish dengan conf=0.6
- Independent processing

#### Tahap 4: Merge Detections
- Combine results dari kedua model
- Single unified detection list

#### Tahap 5: Visualisasi Deteksi
- Draw bounding boxes dengan warna berbeda
- Tambah label dan confidence score
- Generate 3 visualization images

#### Tahap 6: Segmentasi
- GrabCut algorithm
- Morphological operations
- Contour extraction
- Automatic fallback handling

#### Tahap 7: Feature Extraction
- 10+ fitur per objek
- Normalized values
- Ready untuk ML analysis

#### Tahap 8: Visualization Lengkap
- 8-11 subplot comprehensive analysis
- Menampilkan setiap tahap processing
- High-res output untuk presentasi

#### Tahap 9: Penyimpanan
- Save ke structured directories
- Export ke CSV
- Text reports untuk setiap image

## 🎨 Class Information

| ID | Class | Color (BGR) | Model |
|----|-------|-----------|-------|
| 0 | eel | (255, 0, 0) - Red | Biota |
| 1 | fish | (0, 255, 0) - Green | Biota |
| 2 | jellyfish | (0, 0, 255) - Blue | Biota |
| 3 | starfish | (255, 255, 0) - Yellow | Starfish |

## 📊 Output CSV Format

`results/csv/feature_results.csv` berisi:

```
filename, object_id, class, confidence, model, area, perimeter, 
mean_b, mean_g, mean_r, circularity, aspect_ratio, edge_count
```

## 🔧 Configuration Options

### DetectionConfig

```python
class DetectionConfig:
    # Model paths
    MODEL_BIOTA = "trained_3class_model_best.pt"
    MODEL_STARFISH = "cascade_model_starfish_best.pt"
    
    # Confidence thresholds
    CONF_BIOTA = 0.4
    CONF_STARFISH = 0.6
    
    # Image size
    TARGET_SIZE = 640
    
    # Output paths
    RESULTS_DIR = Path("results")
    # ... other directories
```

## 💡 Advanced Usage

### Batch Processing dengan Different Configs

```python
from dual_model_detection_system import DetectionConfig, process_single_image

configs = {
    'strict': (0.5, 0.7),
    'balanced': (0.4, 0.6),
    'sensitive': (0.3, 0.5)
}

for name, (biota_conf, starfish_conf) in configs.items():
    config = DetectionConfig()
    config.CONF_BIOTA = biota_conf
    config.CONF_STARFISH = starfish_conf
    process_single_image("image.jpg", config)
```

### Custom Preprocessing

```python
from dual_model_detection_system import preprocess_image, DetectionConfig

config = DetectionConfig()
original, preprocessed = preprocess_image("image.jpg", config)

# Modify if needed
import cv2
custom = cv2.bilateralFilter(preprocessed, 9, 75, 75)
```

## 🔍 Testing & Diagnosis

Gunakan `test_demo.py` untuk diagnostic:

```bash
# Run semua tests
python test_demo.py test

# Run specific test
python test_demo.py test1    # Model availability
python test_demo.py test2    # Model loading
python test_demo.py test3    # Directory structure
python test_demo.py test4    # Configuration
python test_demo.py test5    # Image input
python test_demo.py test6    # Pipeline

# Quick demo
python test_demo.py demo
```

## 📈 Performance & Tips

### Untuk Akurasi Lebih Tinggi
- Turunkan confidence threshold
- Tingkatkan jumlah preprocessing steps
- Gunakan model dengan lebih banyak training epochs

### Untuk Processing Lebih Cepat
- Naikkan confidence threshold
- Gunakan smaller image size
- Disable visualizations

### Untuk Objek Kecil
- CONF_BIOTA = 0.3 (lebih sensitif)
- Gunakan preprocessing yang lebih kuat
- Adjust CLAHE clip limit

### Untuk False Positive Reduction
- CONF_STARFISH = 0.7 (lebih ketat)
- Tingkatkan morphological opening strength
- Use NMS (Non-Maximum Suppression)

## 🎓 Cocok Untuk

- ✓ Penelitian computer vision
- ✓ Marine biology applications
- ✓ Environmental monitoring
- ✓ Underwater robotics
- ✓ Scientific publications
- ✓ Proyek akademik & sidang
- ✓ Industry applications

## 📝 Dokumentasi Kode

Setiap fungsi memiliki docstring lengkap:

```python
def process_single_image(image_path: str, config: DetectionConfig = None):
    """
    Proses satu image lengkap dari awal sampai akhir
    
    Args:
        image_path (str): Path ke file image
        config (DetectionConfig): Konfigurasi sistem
    """
```

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Check .pt file location, update path in config |
| Memory error | Reduce image size or use GPU |
| No detections | Lower confidence threshold |
| Poor segmentation | Adjust GrabCut parameters or morphology kernel |
| Image size issue | Automatic resize to 640x640 |
| CSV errors | Ensure write permissions in results/csv/ |

## 📊 System Requirements

- **Python**: 3.8+
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 500MB untuk models + 1GB untuk results
- **GPU**: Optional (CUDA untuk faster processing)

## 🔗 Dependencies

```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
```

## 📖 Reference Guide

- **Main Program**: `dual_model_detection_system.py`
- **Quick Start**: `example_usage.py`
- **Testing**: `test_demo.py`
- **Quick Guide**: `DUAL_MODEL_GUIDE.md`
- **Full Docs**: `README.md`

## 💬 Citation

Jika menggunakan sistem ini untuk penelitian:

```
Deteksi Bintang Laut dan Biota Laut Dangkal Menggunakan YOLOv8
Multi-Model Parallel Detection System for Underwater Imaging
```

## 📜 License & Credits

- YOLOv8 by Ultralytics
- OpenCV by OpenCV team
- Image processing techniques dari literature standar

## 🎯 Next Steps

1. ✅ Verify model files exist
2. ✅ Run tests: `python test_demo.py test`
3. ✅ Process image: `from dual_model_detection_system import process_single_image`
4. ✅ Check results in `results/` directory
5. ✅ Review visualization dan CSV output
6. ✅ Customize configuration untuk use case Anda

## 📞 Support

Untuk issues atau pertanyaan:
- Check `DUAL_MODEL_GUIDE.md`
- Review docstrings di main code
- Run diagnostic tests
- Check results files untuk detail errors

---

**Last Updated**: June 2026  
**Version**: 1.0  
**Status**: Production Ready
