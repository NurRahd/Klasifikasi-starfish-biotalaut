# Evaluasi Model Deteksi Biota Laut Underwater

Dokumen ini menjelaskan evaluasi model pada sistem deteksi biota laut underwater. Fokusnya adalah cara membaca metrik, hasil evaluasi yang sudah ada di project, script yang digunakan, dan rekomendasi perbaikan berdasarkan hasil tersebut.

## 1. Ringkasan

Sistem memiliki dua jenis evaluasi:

1. Evaluasi training masing-masing model YOLOv8.
   - Sumber: `runs/detect/*/results.csv`
   - Mengukur performa model pada validation set saat training.

2. Evaluasi end-to-end pipeline dual-model.
   - Sumber: `results/map_evaluation/*.json`
   - Mengukur performa sistem lengkap setelah dua model digabung, class dimapping ke class akhir, confidence threshold diterapkan, dan hasil prediksi dibandingkan dengan label dataset.

Kedua evaluasi ini tidak boleh dibaca sebagai angka yang sama. Evaluasi training mengukur model secara terpisah pada dataset validasi masing-masing. Evaluasi end-to-end mengukur sistem final pada dataset test gabungan, termasuk efek mapping class, threshold, merge detection, dan koreksi warna.

## 2. Model yang Dievaluasi

| Model | File bobot | Dataset training | Kelas |
|---|---|---|---|
| Biota 3-class | `trained_biota_3class_model_best.pt` | `biota_3class_dataset` | eel, fish, jellyfish |
| Starfish-only | `cascade_model_starfish_best.pt` | `cascade_starfish_dataset` | starfish |

Class akhir sistem:

| ID | Nama class |
|---:|---|
| 0 | eel |
| 1 | fish |
| 2 | jellyfish |
| 3 | starfish |

## 3. Metrik Evaluasi

| Metrik | Arti |
|---|---|
| TP | True Positive, prediksi benar dengan class benar dan IoU memenuhi threshold. |
| FP | False Positive, prediksi yang tidak cocok dengan ground truth. |
| FN | False Negative, ground truth yang tidak berhasil dideteksi. |
| Precision | Proporsi prediksi yang benar dari semua prediksi. Formula: `TP / (TP + FP)`. |
| Recall | Proporsi objek ground truth yang berhasil ditemukan. Formula: `TP / (TP + FN)`. |
| F1-score | Rata-rata harmonik precision dan recall. |
| IoU | Intersection over Union antara bounding box prediksi dan ground truth. |
| Mean IoU | Rata-rata IoU dari prediksi yang matched. |
| AP50 | Average Precision pada IoU threshold 0.50. |
| mAP50 | Rata-rata AP50 seluruh class. |
| mAP50-95 | Rata-rata AP pada IoU threshold 0.50 sampai 0.95. Ini lebih ketat dari mAP50. |

Interpretasi cepat:

- Precision rendah berarti banyak false positive.
- Recall rendah berarti banyak objek tidak terdeteksi.
- mAP50 rendah berarti confidence ranking dan kecocokan prediksi terhadap ground truth masih buruk.
- Mean IoU tinggi tetapi precision/recall rendah berarti prediksi yang berhasil match cukup akurat posisinya, tetapi jumlah match sangat sedikit.

## 4. Hasil Validasi Training Model

Angka berikut berasal dari `results.csv` pada folder training YOLOv8.

### 4.1 Model biota 3-class

Sumber:

```text
runs/detect/biota_3class_train-3/results.csv
```

Epoch terakhir:

| Metrik | Nilai |
|---|---:|
| Precision | 0.5809 |
| Recall | 0.5527 |
| mAP50 | 0.5649 |
| mAP50-95 | 0.3589 |
| Train box loss | 1.1521 |
| Train cls loss | 1.2962 |
| Train dfl loss | 1.5608 |
| Val box loss | 1.1954 |
| Val cls loss | 1.4885 |
| Val dfl loss | 1.5584 |

Interpretasi:

- Model biota menunjukkan performa sedang pada validation set training.
- Precision dan recall cukup seimbang, tetapi belum tinggi.
- mAP50-95 lebih rendah dari mAP50, menandakan lokalisasi bounding box masih dapat ditingkatkan pada threshold IoU yang lebih ketat.

### 4.2 Model starfish-only

Sumber:

```text
runs/detect/cascade_starfish_only/results.csv
```

Epoch terakhir:

| Metrik | Nilai |
|---|---:|
| Precision | 0.9687 |
| Recall | 0.9953 |
| mAP50 | 0.9937 |
| mAP50-95 | 0.8569 |
| Train box loss | 0.4097 |
| Train cls loss | 0.3674 |
| Train dfl loss | 1.0664 |
| Val box loss | 0.5987 |
| Val cls loss | 0.4527 |
| Val dfl loss | 1.2093 |

Interpretasi:

- Model starfish sangat kuat pada validation set training.
- Precision, recall, dan mAP tinggi.
- Karena dataset starfish-only lebih sempit, performa validasi tinggi belum tentu langsung sama saat digabung dengan model lain pada dataset campuran.

## 5. Hasil Evaluasi End-to-End Dual-Model

Evaluasi end-to-end dilakukan oleh `evaluate_map_dataset.py`. Script ini memuat dua model, menjalankan pipeline deteksi, memetakan class sesuai dataset, lalu membandingkan prediksi dengan label YOLO.

### 5.1 Evaluasi semua dataset test

Sumber:

```text
results/map_evaluation/map_summary_all_test.json
```

Dataset yang dievaluasi saat `--dataset all`:

- `biota_3class_dataset`
- `starfish_only`
- `combined_detection_dataset`

Hasil:

| Metrik | Nilai |
|---|---:|
| Images | 3110 |
| Ground truth labels | 8011 |
| Predictions | 2985 |
| TP | 602 |
| FP | 2383 |
| FN | 7409 |
| Precision | 0.2017 |
| Recall | 0.0751 |
| F1-score | 0.1095 |
| Mean IoU | 0.8371 |
| mAP50 | 0.0460 |
| mAP50-95 | 0.0347 |

AP per class:

| Class | AP50 | mAP50-95 |
|---|---:|---:|
| eel | 0.0000 | 0.0000 |
| fish | 0.0503 | 0.0319 |
| jellyfish | 0.0180 | 0.0100 |
| starfish | 0.1155 | 0.0966 |

Interpretasi:

- Evaluasi end-to-end masih rendah.
- Recall 0.0751 berarti sebagian besar objek ground truth belum terdeteksi.
- Precision 0.2017 berarti banyak prediksi tidak cocok dengan ground truth.
- Mean IoU 0.8371 cukup tinggi, tetapi hanya dihitung dari prediksi yang berhasil match. Artinya, saat sistem benar-benar match, posisi box cukup baik; masalah utamanya adalah jumlah match yang sedikit.
- Starfish memiliki AP terbaik dibanding class lain pada evaluasi gabungan, selaras dengan validasi training starfish yang memang kuat.
- Eel mendapat AP 0 pada evaluasi gabungan, sehingga perlu dicek apakah masalahnya berasal dari model, mapping class, distribusi data, threshold, atau mismatch label.

### 5.2 Evaluasi khusus `biota_3class_dataset` test

Sumber:

```text
results/map_evaluation/map_summary_biota_3class_dataset_test.json
```

Hasil:

| Metrik | Nilai |
|---|---:|
| Images | 329 |
| Ground truth labels | 1009 |
| Predictions | 1081 |
| TP | 10 |
| FP | 1071 |
| FN | 999 |
| Precision | 0.0093 |
| Recall | 0.0099 |
| F1-score | 0.0096 |
| Mean IoU | 0.5588 |
| mAP50 | 0.0004 |
| mAP50-95 | 0.0001 |

AP per class:

| Class | AP50 | mAP50-95 |
|---|---:|---:|
| eel | 0.0001 | 0.0000 |
| fish | 0.0001 | 0.0000 |
| jellyfish | 0.0010 | 0.0002 |

Interpretasi:

- Hasil khusus biota test sangat rendah walaupun validasi training model biota menunjukkan performa sedang.
- Ini menunjukkan ada perbedaan penting antara evaluasi training dan evaluasi pipeline final.
- Penyebab yang perlu dicek:
  - Dataset evaluasi tidak sama distribusinya dengan validation training.
  - Preprocessing inferensi berbeda dari preprocessing saat training.
  - Label ground truth dan prediksi memakai skala/koordinat yang tidak sepadan.
  - Model yang dimuat saat evaluasi bukan bobot yang diharapkan.
  - Confidence threshold terlalu tinggi/rendah sehingga ranking AP buruk.
  - Class mapping pada dataset biota atau model metadata tidak sesuai.

## 6. Analisis Error Model Biota

Sumber:

```text
results/biota_error_analysis/test/summary.json
```

Ringkasan:

| Metrik | Nilai |
|---|---:|
| Images dianalisis | 10 |
| Ground truth labels | 31 |
| Predictions | 19 |
| TP | 0 |
| False positive | 7 |
| False negative | 31 |
| Low IoU | 12 |

Output visual:

```text
results/biota_error_analysis/test/
├── false_positive/
├── false_negative/
├── low_iou/
├── summary.json
└── cases.csv
```

Interpretasi:

- Pada sampel 10 gambar, belum ada prediksi yang memenuhi syarat true positive.
- Ada 12 kasus low IoU, artinya beberapa prediksi mungkin mendekati objek tetapi bounding box belum cukup overlap dengan ground truth.
- Ada 31 false negative, sehingga banyak objek ground truth tidak ditemukan.
- Folder visual perlu diperiksa untuk melihat apakah masalahnya class salah, box bergeser, ukuran box tidak tepat, atau gambar memang sulit.

## 7. Script Evaluasi

### 7.1 `evaluate_map_dataset.py`

Fungsi:

- Memuat model dual-model dari `dual_model_detection_system.py`.
- Mengambil gambar dari split dataset.
- Membaca label YOLO.
- Memetakan label dataset ke class final sistem.
- Menghitung precision, recall, F1, mean IoU, mAP50, dan mAP50-95.
- Menyimpan summary JSON dan per-image CSV.

Contoh command:

```powershell
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset all --split test --quiet
```

Evaluasi dataset tertentu:

```powershell
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset biota_3class_dataset --split test --quiet
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset starfish_only --split test --quiet
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset combined_detection_dataset --split test --quiet
```

Uji cepat dengan limit gambar:

```powershell
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset all --split test --limit 50 --quiet
```

Uji threshold berbeda:

```powershell
.\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset all --split test --conf-biota 0.25 --conf-starfish 0.7 --quiet
```

Output default:

```text
results/map_evaluation/
├── map_summary_<dataset>_<split>.json
└── map_per_image_<dataset>_<split>.csv
```

### 7.2 `analyze_biota_errors.py`

Fungsi:

- Fokus pada model biota 3-class.
- Membandingkan prediksi dengan label.
- Mengelompokkan error menjadi false positive, false negative, dan low IoU.
- Menghasilkan gambar visual untuk inspeksi manual.

Command:

```powershell
.\.venv\Scripts\python.exe analyze_biota_errors.py --split test --limit 50
```

### 7.3 Evaluasi di `detection_ui.py`

UI lokal juga memiliki fungsi evaluasi.

Jika user upload gambar dan label YOLO `.txt`, UI menghitung:

- precision
- recall
- F1
- mean IoU
- AP50
- mAP50-95
- TP, FP, FN

Jika label tidak tersedia, UI mengambil fallback dari:

1. `results/map_evaluation/map_summary_all_test.json`, jika ada.
2. Rata-rata validation metrics dari `runs/detect/biota_3class_train-3/results.csv` dan `runs/detect/cascade_starfish_only/results.csv`, jika summary dataset tidak ada.

## 8. Kenapa Validasi Training Tinggi tapi Evaluasi Pipeline Rendah?

Perbedaan ini penting dan wajar terjadi pada sistem multi-komponen.

Kemungkinan penyebab:

| Penyebab | Penjelasan |
|---|---|
| Dataset berbeda | Validation saat training memakai dataset model masing-masing, sedangkan evaluasi pipeline memakai dataset gabungan. |
| Class mapping berbeda | Model biota memakai eel/fish/jellyfish, dataset deteksi lama memakai fish/jellyfish/starfish. |
| Preprocessing berbeda | Gambar training dan gambar evaluasi bisa melewati pipeline preprocessing yang tidak sama. |
| Threshold global | Confidence 0.4 dan 0.6 mungkin tidak optimal untuk semua dataset. |
| Domain shift | Dataset test bisa memiliki kondisi pencahayaan, kamera, atau distribusi objek berbeda. |
| Merge logic | Penggabungan dua model dapat menekan atau mempertahankan prediksi tertentu. |
| Label mismatch | Label ground truth bisa berasal dari format atau class schema yang berbeda. |
| Evaluasi lebih ketat | mAP end-to-end memperhitungkan ranking confidence dan IoU threshold. |

## 9. Rekomendasi Perbaikan Evaluasi

Prioritas pemeriksaan:

1. Pastikan bobot model yang dimuat benar.
   - `DetectionConfig.MODEL_BIOTA` harus mengarah ke `trained_biota_3class_model_best.pt`.
   - `DetectionConfig.MODEL_STARFISH` harus mengarah ke `cascade_model_starfish_best.pt`.

2. Evaluasi model biota langsung dengan Ultralytics pada dataset yang sama.
   ```powershell
   .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO('trained_biota_3class_model_best.pt'); m.val(data='biota_3class_dataset/data.yaml', imgsz=640)"
   ```

3. Bandingkan hasil direct YOLO validation dengan `evaluate_map_dataset.py`.
   - Jika direct validation bagus tetapi custom evaluation buruk, masalah kemungkinan ada di preprocessing, mapping, atau koordinat evaluasi.

4. Jalankan evaluasi dengan beberapa threshold.
   - Biota: coba `0.15`, `0.25`, `0.35`, `0.45`.
   - Starfish: coba `0.5`, `0.6`, `0.7`.

5. Periksa visual error.
   - Buka folder `results/biota_error_analysis/test/false_negative`.
   - Buka folder `results/biota_error_analysis/test/low_iou`.
   - Lihat apakah box terlalu kecil, terlalu besar, bergeser, atau class salah.

6. Pisahkan evaluasi per dataset.
   ```powershell
   .\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset biota_3class_dataset --split test --quiet
   .\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset starfish_only --split test --quiet
   .\.venv\Scripts\python.exe evaluate_map_dataset.py --dataset combined_detection_dataset --split test --quiet
   ```

7. Buat confusion/error report per class.
   - Saat ini summary sudah punya AP per class.
   - Untuk diagnosis lebih detail, perlu report jumlah TP/FP/FN per class.

## 10. Kesimpulan Evaluasi Saat Ini

Kesimpulan berdasarkan artefak evaluasi yang ada:

- Model starfish sangat baik pada validation training, dengan mAP50 sekitar 0.9937.
- Model biota cukup sedang pada validation training, dengan mAP50 sekitar 0.5649.
- Pipeline dual-model end-to-end pada semua test dataset masih rendah, dengan mAP50 sekitar 0.0460 dan recall sekitar 0.0751.
- Evaluasi khusus `biota_3class_dataset` test sangat rendah, sehingga bagian biota perlu menjadi prioritas debugging.
- Prediksi yang berhasil match memiliki Mean IoU cukup tinggi pada evaluasi semua dataset, sehingga masalah utama bukan hanya kualitas posisi box, tetapi jumlah prediksi yang berhasil match dan kesesuaian class/mapping.

Rekomendasi paling penting:

1. Validasi ulang model biota secara langsung menggunakan Ultralytics.
2. Cek kesamaan preprocessing training dan inference.
3. Audit mapping label dan class ID.
4. Jalankan threshold sweep untuk mencari konfigurasi confidence yang lebih stabil.
5. Tambahkan evaluasi TP/FP/FN per class agar sumber error lebih jelas.
