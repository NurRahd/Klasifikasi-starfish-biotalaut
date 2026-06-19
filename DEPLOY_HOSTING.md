# Deploy Web Detection UI

Panduan ini untuk deploy web deteksi sebagai aplikasi Streamlit.

## File penting

- `streamlit_app.py` - UI Streamlit utama untuk hosting.
- `detection_ui.py` - helper pipeline dan UI lokal lama.
- `dual_model_detection_system.py` - pipeline deteksi.
- `trained_biota_3class_model_best.pt` - model eel/fish/jellyfish.
- `cascade_model_starfish_best.pt` - model starfish.
- `requirements.txt` - dependency hosting Streamlit.

## Deploy ke Streamlit Community Cloud

1. Push project ini ke GitHub.
2. Buka <https://share.streamlit.io>.
3. Pilih **New app**.
4. Pilih repository `NurRahd/Klasifikasi-starfish-biotalaut`.
5. Pilih branch `main`.
6. Isi **Main file path**:

```text
streamlit_app.py
```

7. Klik **Deploy**.

## Cek setelah deploy

- Buka URL Streamlit yang diberikan.
- Upload gambar `.jpg`, `.jpeg`, `.png`, `.bmp`, atau `.webp`.

## Catatan resource

Aplikasi ini menjalankan YOLO/PyTorch di server. Streamlit Community Cloud bisa lambat saat model pertama kali dimuat. Jika deploy gagal karena memory, gunakan platform dengan RAM lebih besar.

## Jalankan lokal

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

Lalu buka:

```text
http://127.0.0.1:7860
```
