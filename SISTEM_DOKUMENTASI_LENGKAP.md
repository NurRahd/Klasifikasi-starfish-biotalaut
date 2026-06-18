# Dokumentasi Lengkap Sistem Deteksi Biota Laut Underwater

Dokumen ini menjelaskan keseluruhan sistem pada folder `PC_CombinedDataset`, mulai dari dataset, preprocessing, training model YOLOv8, sistem inferensi dual-model, evaluasi, UI lokal, sampai struktur hasil output.

## 1. Ringkasan Sistem

Sistem ini dibuat untuk mendeteksi objek biota laut pada citra underwater menggunakan YOLOv8. Fokus kelas akhir yang digunakan oleh sistem inferensi adalah:

| ID akhir | Kelas | Keterangan |
|---:|---|---|
| 0 | eel | Belut laut |
| 1 | fish | Ikan |
| 2 | jellyfish | Ubur-ubur |
| 3 | starfish | Bintang laut |

Arsitektur inferensi utama memakai dua model YOLOv8:

| Model | File bobot | Kelas yang ditangani | Confidence default |
|---|---|---|---:|
| Model biota 3-class | `trained_biota_3class_model_best.pt` | eel, fish, jellyfish | 0.4 |
| Model starfish | `cascade_model_starfish_best.pt` | starfish | 0.6 |

Kedua model dijalankan pada citra yang sudah dipreprocess. Hasil deteksi kemudian digabung, dikoreksi dengan aturan warna untuk mengurangi salah klasifikasi starfish/jellyfish, disegmentasi dengan OpenCV, diekstraksi fiturnya, lalu disimpan ke folder `results/`.

## 2. Tujuan dan Ruang Lingkup

Tujuan sistem:

- Menggabungkan beberapa dataset underwater menjadi dataset YOLO yang konsisten.
- Melakukan preprocessing khusus citra underwater.
- Melatih model YOLOv8 untuk deteksi objek biota laut.
- Menjalankan inferensi dengan dua model yang saling melengkapi.
- Menyediakan output visual, CSV fitur, dan laporan deteksi.
- Menyediakan web UI lokal sederhana untuk upload gambar dan melihat hasil.

Ruang lingkup utama sistem adalah object detection berbasis bounding box. Beberapa skrip awal juga mendukung konversi format segmentation ke detection dan validasi label YOLO.

## 3. Struktur Folder Penting

```text
PC_CombinedDataset/
├── biota_laut/
│   └── Underwater Marine Species.v2i.yolov8/
├── starfish_only/
├── combined_dataset/
├── combined_dataset_preprocessed/
├── preprocessed_dataset/
├── combined_detection_dataset/
├── preprocessed_detection_dataset/
├── biota_3class_dataset/
├── cascade_starfish_dataset/
├── results/
├── runs/
├── label_backups/
├── dual_model_detection_system.py
├── detection_ui.py
├── create_biota_3class_dataset.py
├── train_biota_3class_model.py
├── train_cascade_starfish_only.py
├── evaluate_map_dataset.py
├── analyze_biota_errors.py
├── deployment_check.py
├── trained_biota_3class_model_best.pt
└── cascade_model_starfish_best.pt
```

Penjelasan folder:

| Folder | Fungsi |
|---|---|
| `biota_laut/Underwater Marine Species.v2i.yolov8/` | Dataset sumber biota laut dari Roboflow/YOLOv8. |
| `starfish_only/` | Dataset sumber khusus starfish. |
| `combined_dataset/` | Gabungan dataset biota dan starfish dalam format YOLO. |
| `combined_dataset_preprocessed/` | Versi `combined_dataset` setelah preprocessing citra. |
| `preprocessed_dataset/` | Dataset preprocessing yang dipakai sebagai sumber pembuatan dataset biota 3-class. |
| `combined_detection_dataset/` | Dataset deteksi gabungan 3 kelas: fish, jellyfish, starfish. |
| `preprocessed_detection_dataset/` | Dataset deteksi hasil preprocessing sebagian. |
| `biota_3class_dataset/` | Dataset khusus kelas eel, fish, jellyfish. Ini dipakai untuk training model biota final. |
| `cascade_starfish_dataset/` | Dataset khusus starfish satu kelas. Ini dipakai untuk training model starfish. |
| `runs/` | Output training Ultralytics YOLO. |
| `results/` | Output inference, evaluasi, visualisasi, CSV, dan hasil UI. |
| `label_backups/` | Backup label saat konversi format label. |

## 4. Statistik Dataset Saat Ini

Jumlah file gambar dan label yang terdeteksi pada workspace:

| Dataset | Train | Valid | Test | Total gambar |
|---|---:|---:|---:|---:|
| `biota_3class_dataset` | 4253 | 821 | 329 | 5403 |
| `combined_dataset` | 5490 | 1107 | 443 | 7040 |
| `combined_dataset_preprocessed` | 5490 | 1107 | 443 | 7040 |
| `preprocessed_dataset` | 5490 | 1107 | 443 | 7040 |
| `combined_detection_dataset` | 33464 | 6846 | 2740 | 43050 |
| `preprocessed_detection_dataset` | 4526 | 0 | 0 | 4526 |
| `cascade_starfish_dataset` | 524 | 204 | 82 | 810 |
| `starfish_only` | 262 | 102 | 41 | 405 |

Catatan:

- `combined_dataset` dan `combined_dataset_preprocessed` memiliki jumlah file sama, tetapi citra pada versi preprocessed sudah ditingkatkan kualitas visualnya.
- `cascade_starfish_dataset` memiliki jumlah dua kali `starfish_only` karena dibuat dari dataset deteksi gabungan dan/atau hasil setup cascade.
- `preprocessed_detection_dataset` saat ini hanya memiliki split train.

## 5. Konfigurasi Dataset YAML

### `biota_3class_dataset/data.yaml`

Dataset ini dipakai oleh `train_biota_3class_model.py`.

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: eel
  1: fish
  2: jellyfish
```

### `combined_detection_dataset/data.yaml`

Dataset ini dipakai oleh pipeline deteksi 3 kelas lama.

```yaml
names:
- fish
- jellyfish
- starfish
nc: 3
test: test/images
train: train/images
val: valid/images
```

### `cascade_starfish_dataset/data.yaml`

Dataset ini dipakai oleh `train_cascade_starfish_only.py`.

```yaml
names:
- starfish
nc: 1
train: .../cascade_starfish_dataset/train/images
val: .../cascade_starfish_dataset/valid/images
test: .../cascade_starfish_dataset/test/images
```

## 6. Alur Besar Sistem

Alur sistem dari data sampai hasil akhir:

```text
Dataset sumber
  -> validasi dan cek format label
  -> gabung dataset / konversi label
  -> preprocessing citra underwater
  -> buat dataset khusus biota dan starfish
  -> training model YOLOv8
  -> inferensi dual-model
  -> merge dan koreksi deteksi
  -> segmentasi objek
  -> ekstraksi fitur
  -> visualisasi dan export hasil
```

## 7. Preprocessing Citra Underwater

Preprocessing utama ada di `preprocess_dataset.py`.

Masalah umum pada citra underwater:

| Masalah | Dampak | Solusi dalam sistem |
|---|---|---|
| Cahaya merah terserap air | Warna objek pudar/kebiruan | Red channel boost |
| Kontras rendah | Batas objek sulit terlihat | CLAHE dan contrast adjustment |
| Noise/backscatter | Deteksi menjadi tidak stabil | Gaussian blur dan bilateral filter |
| Pencahayaan tidak merata | Objek gelap/terlalu terang | Histogram normalization |
| Ukuran gambar bervariasi | Input model tidak seragam | Resize ke 640x640 |

Tahapan preprocessing:

1. Resize gambar ke 640x640.
2. Gaussian blur ringan untuk mengurangi noise frekuensi tinggi.
3. Brightness dan contrast adjustment.
4. CLAHE pada channel L di LAB color space.
5. Koreksi warna underwater dengan penguatan channel red dan green.
6. Bilateral filter untuk denoising sambil menjaga edge.
7. Normalisasi intensitas pada channel V di HSV.

Output preprocessing:

- Gambar hasil preprocessing disimpan di dataset output.
- Label disalin tanpa perubahan.
- Log proses disimpan di `preprocessing.log`.
- Perbandingan visual disimpan di `preprocessing_comparison.png`.

## 8. Konversi dan Persiapan Dataset

### 8.1 Menggabungkan dataset segmentation

Skrip: `combine_datasets.py`

Fungsi:

- Membuat struktur `combined_dataset/train|valid|test`.
- Menyalin gambar dan label dari dataset biota dan starfish.
- Menambahkan prefix pada file starfish agar tidak bentrok nama.
- Menulis `data.yaml`.
- Mendukung remapping label class jika diperlukan.

### 8.2 Validasi format label

Skrip terkait:

- `check_label_formats.py`
- `validate_yolov8_segmentation.py`
- `dataset_readiness_check.py`

Fungsi:

- Mengecek apakah label YOLO valid.
- Membedakan label detection dan segmentation.
- Menghitung distribusi class.
- Mendeteksi label kosong, rusak, tidak berpasangan dengan gambar, atau koordinat invalid.
- Membuat report CSV seperti `label_format_report.csv` dan `dataset_readiness_report.csv`.

### 8.3 Konversi segmentation ke detection

Skrip: `convert_to_yolo_detection.py`

Fungsi:

- Membaca dataset dengan label YOLO detection atau segmentation.
- Jika label berbentuk polygon segmentation, sistem menghitung bounding box minimum.
- Mapping class akhir menjadi `fish`, `jellyfish`, dan `starfish`.
- Menulis output ke `combined_detection_dataset/`.
- Menulis laporan ke `conversion_report.csv`.

Mapping penting di skrip ini:

| Label sumber | Class akhir |
|---|---|
| fish, eel, lionfish, lobster, shark | fish |
| jellyfish, jelly-fish | jellyfish |
| starfish, star-fish, sea_star, sea-star, star | starfish |

### 8.4 Dataset biota 3-class

Skrip: `create_biota_3class_dataset.py`

Fungsi:

- Membaca label dari `preprocessed_dataset`.
- Menyalin hanya gambar yang seluruh objeknya termasuk class `0`, `1`, atau `2`.
- Gambar yang mengandung class lain seperti lionfish, lobster, atau starfish dilewati.
- Menulis output ke `biota_3class_dataset/`.
- Menulis `data.yaml` dengan kelas eel, fish, jellyfish.

### 8.5 Dataset starfish cascade

Skrip terkait:

- `setup_cascade_datasets.py`
- `setup_cascade_starfish_only.py`

Fungsi:

- Membuat dataset khusus starfish.
- Mengubah class starfish menjadi class ID `0` untuk model satu kelas.
- Menulis `cascade_starfish_dataset/data.yaml`.

## 9. Training Model

### 9.1 Training model biota 3-class

Skrip: `train_biota_3class_model.py`

Konfigurasi utama:

| Parameter | Nilai |
|---|---|
| Dataset | `biota_3class_dataset/data.yaml` |
| Base weights | `yolov8n.pt` |
| Epochs | 15 |
| Image size | 640 |
| Batch | 2 |
| Patience | 5 |
| Device | 0 |
| Output final | `trained_biota_3class_model_best.pt` |

Command:

```powershell
.\.venv\Scripts\python.exe train_biota_3class_model.py
```

Setelah training selesai, skrip mencari `runs/detect/biota_3class_train*/weights/best.pt`, lalu menyalinnya menjadi `trained_biota_3class_model_best.pt`.

### 9.2 Training model starfish-only

Skrip: `train_cascade_starfish_only.py`

Konfigurasi utama:

| Parameter | Nilai |
|---|---|
| Dataset | `cascade_starfish_dataset/data.yaml` |
| Base weights | `yolov8n.pt` |
| Epochs | 20 |
| Image size | 640 |
| Batch | 8 |
| Patience | 5 |
| Device | cpu |
| Output final | `cascade_model_starfish_best.pt` |

Command:

```powershell
.\.venv\Scripts\python.exe train_cascade_starfish_only.py
```

### 9.3 Training pipeline deteksi 3 kelas lama

Skrip: `train_yolov8_detection.py`

Fungsi:

- Melatih YOLOv8 detection pada `combined_detection_dataset/data.yaml`.
- Evaluasi pada validation set.
- Inferensi pada `combined_detection_dataset/test/images`.
- Menyimpan hasil ke `results/detection`, `results/validation`, `results/plots`, dan `results/predictions`.

Pipeline ini memakai class `fish`, `jellyfish`, `starfish`, bukan class akhir empat kelas pada sistem dual-model.

## 10. Sistem Inferensi Utama

Skrip inti: `dual_model_detection_system.py`

Komponen utama:

| Komponen | Fungsi |
|---|---|
| `DetectionConfig` | Menyimpan path model, class mapping, threshold, target size, dan folder output. |
| `load_models()` | Memuat model YOLO biota dan starfish. |
| `preprocess_image()` | Membaca gambar dan menerapkan preprocessing input inferensi. |
| `run_biota_detection()` | Menjalankan model eel/fish/jellyfish. |
| `run_starfish_detection()` | Menjalankan model starfish. |
| `run_parallel_detection()` | Menjalankan dua model secara paralel. |
| `merge_detections()` | Menggabungkan hasil dua model dan menangani duplikasi/overlap. |
| `apply_color_class_corrections()` | Koreksi class berbasis profil warna untuk kasus starfish vs jellyfish. |
| `segment_object()` | Segmentasi objek dengan GrabCut/fallback OpenCV. |
| `extract_features()` | Ekstraksi fitur area, perimeter, warna, circularity, aspect ratio, edge count, dan lainnya. |
| `visualize_detections()` | Membuat gambar hasil deteksi dengan bounding box. |
| `visualize_complete_analysis()` | Membuat visualisasi lengkap beberapa subplot. |
| `save_results()` | Menyimpan hasil deteksi, segmentasi, fitur, CSV, dan visualisasi. |
| `process_single_image()` | Pipeline lengkap untuk satu gambar. |
| `process_multiple_images()` | Pipeline batch untuk folder gambar. |

### 10.1 Konfigurasi default

```python
MODEL_BIOTA = "trained_biota_3class_model_best.pt"
MODEL_STARFISH = "cascade_model_starfish_best.pt"
CONF_BIOTA = 0.4
CONF_STARFISH = 0.6
TARGET_SIZE = 640
RESULTS_DIR = Path("results")
```

Class final:

```python
CLASS_NAMES = {
    0: "eel",
    1: "fish",
    2: "jellyfish",
    3: "starfish",
}
```

### 10.2 Alur inferensi single image

```text
Input image
  -> resolve path
  -> load original image
  -> resize/preprocess ke 640x640
  -> YOLO biota detection
  -> YOLO starfish detection
  -> merge detection list
  -> color-based class correction
  -> draw bounding boxes
  -> segment tiap objek
  -> extract object features
  -> save TXT, CSV, images, dan visualisasi lengkap
```

### 10.3 Cara menjalankan inferensi dari Python

```python
from dual_model_detection_system import process_single_image

process_single_image("path/to/image.jpg")
```

Batch folder:

```python
from dual_model_detection_system import process_multiple_images

process_multiple_images("path/to/images")
```

Custom threshold:

```python
from dual_model_detection_system import DetectionConfig, process_single_image

config = DetectionConfig()
config.CONF_BIOTA = 0.3
config.CONF_STARFISH = 0.5

process_single_image("image.jpg", config)
```

## 11. Web UI Lokal

Skrip: `detection_ui.py`

Fungsi:

- Menjalankan HTTP server lokal di `127.0.0.1:7860`.
- Menyediakan halaman upload gambar.
- Mengizinkan upload label `.txt` opsional untuk evaluasi terhadap ground truth.
- Mengatur confidence threshold lewat form.
- Mengaktifkan/mematikan koreksi warna.
- Menampilkan hasil deteksi, ringkasan class, fitur objek, dan metrik evaluasi jika label tersedia.
- Menyimpan hasil UI ke `results/ui/`.

Command:

```powershell
.\.venv\Scripts\python.exe detection_ui.py
```

Buka browser:

```text
http://127.0.0.1:7860
```

Struktur output UI:

```text
results/ui/
├── uploads/
├── outputs/
├── detection/
├── segmentation/
├── features/
├── visualizations/
└── csv/
```

## 12. Evaluasi Model

### 12.1 Evaluasi mAP dual-model

Skrip: `evaluate_map_dataset.py`

Fungsi:

- Menjalankan model dual-model pada dataset berlabel.
- Membaca label YOLO ground truth.
- Melakukan mapping class sesuai dataset.
- Menghitung precision, recall, F1, AP, dan mAP.
- Mendukung dataset tunggal atau semua dataset terpilih.

Contoh command:

```powershell
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset biota_3class_dataset --split test
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset starfish_only --split valid
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset all --split test
```

Mapping dataset pada evaluasi:

| Dataset | Mapping label |
|---|---|
| `biota_3class_dataset` | 0 eel, 1 fish, 2 jellyfish |
| `starfish_only` | semua class label dipetakan ke final class 3 starfish |
| `cascade_starfish_dataset` | semua class label dipetakan ke final class 3 starfish |
| `combined_detection_dataset` | 0 fish, 1 jellyfish, 2 starfish dipetakan ke 1, 2, 3 |

### 12.2 Analisis error model biota

Skrip: `analyze_biota_errors.py`

Fungsi:

- Membandingkan prediksi model biota 3-class dengan ground truth.
- Menghasilkan contoh visual untuk false positive, false negative, dan low IoU.
- Menyimpan `summary.json` dan `cases.csv`.

Contoh command:

```powershell
.\.venv\Scripts\python.exe analyze_biota_errors.py --split test --limit 50
```

Output:

```text
results/biota_error_analysis/<split>/
├── false_positive/
├── false_negative/
├── low_iou/
├── summary.json
└── cases.csv
```

## 13. Output Sistem

Output utama sistem disimpan di `results/`.

```text
results/
├── detection/
├── segmentation/
├── features/
├── visualizations/
├── csv/
├── ui/
├── validation/
├── plots/
├── predictions/
└── biota_error_analysis/
```

Penjelasan:

| Folder | Isi |
|---|---|
| `results/detection/` | File `.txt` hasil deteksi dan ringkasan inference. |
| `results/segmentation/` | Mask dan hasil segmentasi objek. |
| `results/features/` | Detail fitur objek per gambar. |
| `results/visualizations/` | Gambar output dengan bounding box dan analisis lengkap. |
| `results/csv/` | CSV gabungan fitur, biasanya `feature_results.csv`. |
| `results/ui/` | Upload dan output dari web UI lokal. |
| `results/validation/` | Hasil evaluasi model training lama. |
| `results/plots/` | Plot training/evaluasi. |
| `results/predictions/` | Gambar prediksi dari pipeline training lama. |
| `results/biota_error_analysis/` | Sampel visual error model biota. |

### Format CSV fitur

File `results/csv/feature_results.csv` berisi data seperti:

| Kolom | Arti |
|---|---|
| `filename` | Nama file gambar. |
| `object_id` | Nomor objek pada gambar. |
| `class` / `class_name` | Nama class objek. |
| `confidence` | Confidence score dari model. |
| `model` | Model asal deteksi, biota atau starfish. |
| `area` | Luas objek. |
| `perimeter` | Keliling contour. |
| `mean_b`, `mean_g`, `mean_r` | Rata-rata warna BGR. |
| `circularity` | Indikator seberapa membulat bentuk objek. |
| `aspect_ratio` | Rasio lebar/tinggi bounding box atau objek. |
| `edge_count` | Jumlah pixel edge hasil Canny. |

## 14. Skrip Utilitas

| File | Fungsi |
|---|---|
| `example_usage.py` | Contoh pemakaian `process_single_image`, batch, dan custom config. |
| `test_demo.py` | Tes ketersediaan model, folder hasil, config, input image, dan full pipeline. |
| `utilities.py` | Analisis CSV fitur, plot distribusi confidence, export JSON/Excel, cleanup hasil, quality check. |
| `quick_infer_best.py` | Inferensi memakai `best.pt` atau `last.pt` terbaru di `runs/detect/*/weights`. |
| `run_inference_only.py` | Menjalankan inferensi tanpa training ulang. |
| `cascade_inference_combined.py` | Inferensi cascade pada model gabungan/terpisah. |
| `deployment_check.py` | Mengecek versi Python, dependency, file model, file skrip, folder, disk, permission, GPU, dan quick inference. |
| `resume_biota_3class_training.py` | Melanjutkan training model biota jika tersedia checkpoint. |

## 15. Dependency

Dependency Python utama:

```text
ultralytics
opencv-python
numpy
pandas
matplotlib
torch
pyyaml
tqdm
```

Instalasi umum:

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics opencv-python numpy pandas matplotlib pyyaml tqdm
```

Catatan:

- `torch` biasanya ikut terpasang sebagai dependency Ultralytics, tetapi bisa perlu instalasi khusus tergantung CPU/GPU.
- Training GPU membutuhkan konfigurasi CUDA yang cocok dengan PyTorch.

## 16. Command Penting

### Cek kesiapan sistem

```powershell
.\.venv\Scripts\python.exe deployment_check.py
```

### Buat dataset biota 3-class

```powershell
.\.venv\Scripts\python.exe create_biota_3class_dataset.py
```

### Training biota 3-class

```powershell
.\.venv\Scripts\python.exe train_biota_3class_model.py
```

### Training starfish-only

```powershell
.\.venv\Scripts\python.exe train_cascade_starfish_only.py
```

### Jalankan UI

```powershell
.\.venv\Scripts\python.exe detection_ui.py
```

### Evaluasi dual-model

```powershell
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset all --split test
```

### Analisis error biota

```powershell
.\.venv\Scripts\python.exe analyze_biota_errors.py --split test --limit 50
```

## 17. Catatan Kritis dan Konsistensi

Beberapa nama file model di dokumentasi/skrip lama tidak selalu sama:

- `dual_model_detection_system.py` memakai `trained_biota_3class_model_best.pt`.
- Beberapa dokumentasi lama menyebut `trained_3class_model_best.pt`.
- `deployment_check.py` masih mengecek `trained_3class_model_best.pt`, bukan `trained_biota_3class_model_best.pt`.

Rekomendasi:

- Untuk sistem final dual-model, gunakan `trained_biota_3class_model_best.pt` dan `cascade_model_starfish_best.pt`.
- Jika menjalankan `deployment_check.py`, sesuaikan daftar model yang dicek agar cocok dengan konfigurasi terbaru.

Catatan class:

- `combined_detection_dataset` memakai 3 kelas: fish, jellyfish, starfish.
- `biota_3class_dataset` memakai 3 kelas berbeda: eel, fish, jellyfish.
- Sistem final memakai 4 class ID akhir dengan starfish sebagai ID 3 setelah hasil model starfish digabung.

## 18. Troubleshooting

| Masalah | Kemungkinan penyebab | Solusi |
|---|---|---|
| Model file not found | File `.pt` tidak ada di root project | Pastikan `trained_biota_3class_model_best.pt` dan `cascade_model_starfish_best.pt` ada. |
| Tidak ada deteksi | Confidence terlalu tinggi atau gambar terlalu gelap | Turunkan `CONF_BIOTA`/`CONF_STARFISH`, cek preprocessing. |
| Banyak false positive starfish | Objek jellyfish/warna pucat mirip starfish | Aktifkan color correction dan naikkan `CONF_STARFISH`. |
| Training gagal karena CUDA | PyTorch/CUDA tidak cocok | Jalankan dengan CPU atau instal PyTorch CUDA yang sesuai. |
| UI tidak bisa dibuka | Server belum jalan atau port 7860 dipakai | Jalankan `detection_ui.py`, cek terminal, atau ganti port di skrip. |
| Label tidak terbaca | Format bukan YOLO atau koordinat invalid | Jalankan `check_label_formats.py` dan `dataset_readiness_check.py`. |
| CSV fitur kosong | Tidak ada objek terdeteksi/segmentasi gagal | Cek visualisasi hasil dan threshold confidence. |

## 19. Rekomendasi Workflow Final

Untuk penggunaan normal sistem yang sudah siap:

1. Pastikan model final tersedia:
   - `trained_biota_3class_model_best.pt`
   - `cascade_model_starfish_best.pt`
2. Jalankan `deployment_check.py` setelah daftar modelnya disesuaikan.
3. Jalankan UI:
   ```powershell
   .\.venv\Scripts\python.exe detection_ui.py
   ```
4. Upload gambar underwater ke `http://127.0.0.1:7860`.
5. Periksa output di `results/ui/` dan `results/csv/`.

Untuk eksperimen ulang training:

1. Validasi dataset sumber.
2. Jalankan preprocessing jika dataset berubah.
3. Buat ulang `biota_3class_dataset`.
4. Buat ulang `cascade_starfish_dataset` jika data starfish berubah.
5. Training ulang model biota dan starfish.
6. Evaluasi dengan `evaluate_map_dataset.py`.
7. Analisis error dengan `analyze_biota_errors.py`.

## 20. Ringkasan File Utama

| File | Peran |
|---|---|
| `dual_model_detection_system.py` | Core pipeline inferensi dual-model, preprocessing, merge, segmentasi, fitur, visualisasi, save output. |
| `detection_ui.py` | Web UI lokal untuk upload dan uji deteksi. |
| `create_biota_3class_dataset.py` | Membuat dataset bersih eel/fish/jellyfish. |
| `train_biota_3class_model.py` | Training YOLOv8 untuk biota 3-class. |
| `train_cascade_starfish_only.py` | Training YOLOv8 satu kelas starfish. |
| `evaluate_map_dataset.py` | Evaluasi precision/recall/F1/AP/mAP untuk dual-model. |
| `analyze_biota_errors.py` | Analisis visual false positive, false negative, dan low IoU. |
| `preprocess_dataset.py` | Preprocessing citra underwater. |
| `convert_to_yolo_detection.py` | Konversi dataset campuran ke YOLO detection. |
| `dataset_readiness_check.py` | Audit kesiapan dataset sebelum training. |
| `utilities.py` | Analisis dan export hasil fitur. |

## 21. Status Sistem Saat Ini

Berdasarkan file yang ada di workspace:

- Dataset utama dan dataset turunan sudah tersedia.
- Model final biota dan starfish sudah tersedia.
- Sistem inferensi dual-model sudah menjadi pipeline utama.
- UI lokal sudah tersedia di `detection_ui.py`.
- Output hasil eksperimen sudah tersimpan di `results/` dan `runs/`.
- Dokumentasi lama tersedia, tetapi beberapa simbolnya mengalami masalah encoding dan beberapa nama model tidak sepenuhnya sinkron dengan skrip terbaru.

Dokumen ini dapat dijadikan dokumentasi utama yang lebih konsisten untuk menjelaskan keseluruhan sistem.
