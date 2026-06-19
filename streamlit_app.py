from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from detection_ui import OUTPUT_DIR, UPLOAD_DIR, make_config, run_detection


class StreamlitForm:
    def __init__(self, values: dict[str, object]):
        self.values = values

    def getfirst(self, name: str, default=None):
        return self.values.get(name, default)


st.set_page_config(
    page_title="Deteksi Biota Laut",
    page_icon=":ocean:",
    layout="wide",
)

st.title("Deteksi Bintang Laut dan Biota Laut Dangkal")
st.caption("Upload gambar underwater untuk menjalankan deteksi YOLOv8 dual-model.")

with st.sidebar:
    st.header("Pengaturan")
    conf_biota = st.slider("Confidence biota", 0.05, 0.95, 0.40, 0.05)
    conf_starfish = st.slider("Confidence starfish", 0.05, 0.95, 0.60, 0.05)
    color_correction = st.checkbox("Koreksi warna", value=True)

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Upload gambar untuk mulai deteksi.")
    st.stop()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

suffix = Path(uploaded_file.name).suffix.lower()
upload_path = UPLOAD_DIR / f"streamlit_{int(time.time() * 1000)}{suffix}"
upload_path.write_bytes(uploaded_file.getbuffer())

form = StreamlitForm(
    {
        "conf_biota": conf_biota,
        "conf_starfish": conf_starfish,
        "color_correction": "on" if color_correction else "off",
    }
)
config = make_config(form)

with st.spinner("Menjalankan deteksi..."):
    try:
        result = run_detection(upload_path, config, original_name=uploaded_file.name)
    except Exception as exc:
        st.error(f"Deteksi gagal: {exc}")
        st.stop()

summary = result["summary"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total objek", summary["total_objects"])
col2.metric("Rata-rata confidence", summary["avg_confidence"])
col3.metric("Biota raw", summary["raw_biota"])
col4.metric("Starfish raw", summary["raw_starfish"])

st.subheader("Hasil Deteksi")
tabs = st.tabs(["Gabungan", "Biota", "Starfish", "Original"])

for tab, key in zip(tabs, ["merged", "biota", "starfish", "original"]):
    with tab:
        image_name = Path(result["images"][key]).name
        image_path = OUTPUT_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning("Gambar hasil tidak ditemukan.")

st.subheader("Ringkasan Kelas")
if summary["class_counts"]:
    counts_df = pd.DataFrame(
        [{"class": name, "count": count} for name, count in summary["class_counts"].items()]
    )
    st.dataframe(counts_df, use_container_width=True, hide_index=True)
else:
    st.write("Tidak ada objek terdeteksi.")

st.subheader("Detail Deteksi")
detections = result.get("detections", [])
if detections:
    st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)
else:
    st.write("Tidak ada detail deteksi.")

metrics = result.get("metrics")
if metrics:
    st.subheader("Metrik Dataset")
    st.json(metrics)
