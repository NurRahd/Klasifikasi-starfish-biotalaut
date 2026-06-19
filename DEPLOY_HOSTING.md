# Deploy Web Detection UI

Panduan ini untuk deploy `detection_ui.py` sebagai web app Python.

## File penting

- `detection_ui.py` - web server dan UI utama.
- `dual_model_detection_system.py` - pipeline deteksi.
- `trained_biota_3class_model_best.pt` - model eel/fish/jellyfish.
- `cascade_model_starfish_best.pt` - model starfish.
- `requirements.txt` - dependency hosting.
- `Procfile` - start command untuk hosting Heroku-like.
- `render.yaml` - konfigurasi Render.

## Deploy ke Render

1. Push project ini ke GitHub.
2. Buka Render, pilih **New +** lalu **Web Service**.
3. Hubungkan repository GitHub.
4. Gunakan pengaturan berikut:
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python detection_ui.py`
5. Deploy.

Render akan memberi environment variable `PORT` otomatis. Aplikasi sudah membaca `PORT` dan bind ke `0.0.0.0`, jadi tidak perlu ubah kode lagi.

## Cek setelah deploy

- Buka URL Render untuk UI.
- Buka `/healthz` untuk health check sederhana.
- Upload gambar `.jpg`, `.jpeg`, `.png`, `.bmp`, atau `.webp`.

## Catatan resource

Aplikasi ini menjalankan YOLO/PyTorch di server. Hosting gratis bisa lambat atau gagal karena RAM kecil, terutama saat model pertama kali dimuat. Jika deploy gagal karena memory, gunakan plan dengan RAM lebih besar atau VPS.

## Jalankan lokal

```powershell
.\.venv\Scripts\python.exe detection_ui.py
```

Lalu buka:

```text
http://127.0.0.1:7860
```
