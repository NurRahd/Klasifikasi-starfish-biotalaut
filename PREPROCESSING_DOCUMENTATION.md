
# Dokumentasi Preprocessing Dataset YOLOv8 Underwater Segmentation

## Ringkasan
Script ini melakukan preprocessing lengkap pada dataset YOLOv8 Segmentation underwater dengan 7040 images total (5490 train, 1107 valid, 443 test).

## Pipeline Preprocessing

### 1. **Resize Image (640x640)**
- **Tujuan**: Standardisasi ukuran input untuk model YOLOv8
- **Interpolasi**: Linear interpolation untuk kualitas terbaik
- **Output**: Semua image di-resize ke 640×640 pixel

### 2. **Gaussian Blur (5×5 kernel, σ=0.5)**
- **Tujuan**: Mengurangi noise high-frequency
- **Parameter**: 
  - Kernel size: 5×5
  - Sigma: 0.5
- **Efek**: Smoothing ringan tanpa over-blurring details penting

### 3. **Brightness & Contrast Enhancement**
- **Brightness**: +5 (meningkatkan brightness sedikit)
- **Contrast**: ×1.2 (meningkatkan kontras 20%)
- **Formula**: `new_pixel = contrast × (pixel - 128) + 128 + brightness`
- **Tujuan**: 
  - Gambar underwater cenderung lebih gelap
  - Peningkatan kontras membantu model membedakan objek lebih baik

### 4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- **Alasan**: Better than global histogram equalization untuk underwater images
- **Parameter**:
  - Clip limit: 2.0 (membatasi amplifikasi noise)
  - Tile size: 8×8 (tile grid untuk perhitungan histogram lokal)
- **Proses**:
  1. Konversi BGR → LAB color space
  2. Apply CLAHE pada L channel (Lightness) saja
  3. Merge channels dan konversi kembali ke BGR
- **Efek**: Peningkatan kontras lokal tanpa over-enhancement

### 5. **Underwater Color Correction**
- **Problem**: Cahaya air menyerap spektrum red
- **Solusi**:
  - Tingkatkan Red channel: ×1.3
  - Tingkatkan Green channel: ×1.1
  - Biarkan Blue channel normal
- **Tujuan**: Restore color balance yang hilang karena penyerapan air

### 6. **Bilateral Filter (Denoising)**
- **Parameter**:
  - Diameter: 9 pixels
  - Sigma color: 75.0
  - Sigma space: 75.0
- **Keuntungan**: 
  - Menjaga edge details (penting untuk segmentation)
  - Jauh lebih cepat dari NLM denoising (~15× faster)
  - Noise reduction yang efektif untuk dataset underwater
- **Efek**: Mengurangi noise random sambil mempertahankan object boundaries

### 7. **Normalisasi Intensitas**
- **Proses**:
  1. Konversi BGR → HSV
  2. Apply histogram equalization pada V channel (Value/Brightness)
  3. Konversi kembali ke BGR
- **Tujuan**: Balance intensitas brightness across the image
- **Efek**: Menghilangkan shadow dan hot spots, normalisasi brightness

## Struktur Output

```
combined_dataset_preprocessed/
├── train/
│   ├── images/          (5490 images preprocessed)
│   └── labels/          (5490 labels - copied unchanged)
├── valid/
│   ├── images/          (1107 images preprocessed)
│   └── labels/          (1107 labels - copied unchanged)
├── test/
│   ├── images/          (443 images preprocessed)
│   └── labels/          (443 labels - copied unchanged)
└── data.yaml            (copied from source)
```

## Karakteristik Preprocessing untuk Underwater Images

### Mengapa Underwater Images Sulit?
1. **Color Absorption**: Air menyerap red & orange spectrum
2. **Scattering**: Partikel di air menciptakan haze/backscatter
3. **Low Contrast**: Lighting tidak uniform
4. **Low Visibility**: Detail kurang jelas
5. **Shadows**: Variasi pencahayaan yang ekstrem

### Teknik yang Diterapkan
| Problem | Solusi | Teknik |
|---------|--------|--------|
| Dark images | Brightness increase | Brightness +5 |
| Low contrast | Local contrast enhancement | CLAHE |
| Color imbalance | Red/Green restoration | Color correction |
| Washed out colors | Adaptive equalization | CLAHE + Histogram |
| Noise | Bilateral filtering | Bilateral filter |
| Size variation | Standardization | Resize 640×640 |

## Performance & Kecepatan

- **Processing Speed**: ~17-28 images/second (tergantung CPU)
- **Total Processing Time** (7040 images):
  - Train (5490): ~200-320 detik
  - Valid (1107): ~40-65 detik  
  - Test (443): ~16-26 detik
  - **Total**: ~7-10 menit
- **Disk Space**: Original ≈ 2-3GB, Output ≈ 2-3GB

## Files Generated

### Log Files
- `preprocessing.log`: Detailed log of all operations
- `preprocessing_output.txt`: Console output capture

### Output Dataset
- `combined_dataset_preprocessed/`: Complete preprocessed dataset

### Comparison Visualization
- `preprocessing_comparison.png`: 5 sample images (before/after) untuk visual inspection

## Parameters Fine-tuning

Jika hasil kurang memuaskan, pertimbangkan:

```python
# Untuk gambar lebih gelap:
brightness = 10  # increase from 5
contrast = 1.3   # increase from 1.2

# Untuk mengurangi noise lebih banyak:
CLAHE_CLIP_LIMIT = 3.0  # increase from 2.0
NLM_h = 12  # meningkatkan agresivitas denoising

# Untuk preservasi detail lebih baik:
BLUR_KERNEL_SIZE = (3, 3)  # reduce dari (5, 5)
BLUR_SIGMA = 0.3  # reduce dari 0.5
```

## Integration dengan YOLOv8 Training

```python
# Di training script, gunakan:
data.yaml_path = "combined_dataset_preprocessed/data.yaml"

# YOLOv8 akan load paths:
# - train: combined_dataset_preprocessed/train/images
# - val: combined_dataset_preprocessed/valid/images
# - test: combined_dataset_preprocessed/test/images
```

## Quality Assurance

Untuk memverifikasi hasil preprocessing:

1. **Visual Inspection**:
   - Buka `preprocessing_comparison.png`
   - Bandingkan before/after untuk 5 sampel

2. **Statistics**:
   - Periksa `preprocessing.log` untuk error messages
   - Verify bahwa semua images berhasil diproses
   - Confirm jumlah files match expected count

3. **Disk Space**:
   - Train images: 5490 files di `train/images`
   - Valid images: 1107 files di `valid/images`
   - Test images: 443 files di `test/images`
   - Labels untuk semua splits

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow processing | System load | Run saat CPU usage rendah |
| Corrupted images | Bad source files | Check log, manual review |
| Memory issues | Large batch | Process dalam splits |
| Bad enhancement | Parameter mismatch | Tune brightness/contrast |
| Poor contrast | Insufficient CLAHE | Increase clip_limit |

## References & Methods

- **CLAHE**: Pizer et al., "Adaptive Histogram Equalization and Its Variations" (1987)
- **Bilateral Filtering**: Tomasi & Manduchi (1998)
- **Underwater Imaging**: Iqbal et al., "Underwater Image Enhancement..." (2013)
- **YOLOv8**: Ultralytics implementation

## Author Notes

Preprocessing ini dirancang khusus untuk:
- ✓ Dataset underwater dengan kondisi cahaya rendah
- ✓ Color cast dari penyerapan spektrum merah
- ✓ Noise reduction dengan edge preservation
- ✓ Optimal untuk YOLOv8 Segmentation training
- ✓ Balanced antara enhancement dan detail preservation

---
Generated: 2026-06-01
Dataset: combined_dataset (7040 images)
Output: combined_dataset_preprocessed
