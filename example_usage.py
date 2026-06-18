"""
====================================================
CONTOH PENGGUNAAN SISTEM DETEKSI MULTI-MODEL
====================================================

Panduan penggunaan sistem deteksi underwater dual-model:
1. Model biota laut (fish, jellyfish, starfish)
2. Model starfish

====================================================
"""

from dual_model_detection_system import (
    DetectionConfig, 
    process_single_image,
    process_multiple_images
)
from pathlib import Path


# ====================================================
# CONTOH 1: PROSES SINGLE IMAGE
# ====================================================

def example_single_image():
    """Proses satu image dengan konfigurasi default"""
    
    print("=" * 60)
    print("CONTOH 1: Processing Single Image")
    print("=" * 60)
    
    # Path ke image
    image_path = "combined_dataset/test/images/sample_image.jpg"
    
    # Process dengan konfigurasi default
    if Path(image_path).exists():
        process_single_image(image_path)
    else:
        print(f"Image not found: {image_path}")
        print("Ganti dengan path image Anda yang benar")


# ====================================================
# CONTOH 2: PROSES MULTIPLE IMAGES
# ====================================================

def example_multiple_images():
    """Proses multiple images dalam satu direktori"""
    
    print("=" * 60)
    print("CONTOH 2: Processing Multiple Images")
    print("=" * 60)
    
    # Direktori berisi images
    images_dir = "combined_dataset/test/images"
    
    if Path(images_dir).exists():
        process_multiple_images(images_dir)
    else:
        print(f"Directory not found: {images_dir}")


# ====================================================
# CONTOH 3: CUSTOM CONFIGURATION
# ====================================================

def example_custom_config():
    """Proses dengan konfigurasi custom"""
    
    print("=" * 60)
    print("CONTOH 3: Custom Configuration")
    print("=" * 60)
    
    # Buat config custom
    config = DetectionConfig()
    
    # Ubah confidence threshold jika diperlukan
    config.CONF_BIOTA = 0.3      # Lebih sensitif untuk biota
    config.CONF_STARFISH = 0.5   # Lebih ketat untuk starfish
    
    image_path = "combined_dataset/test/images/sample_image.jpg"
    
    if Path(image_path).exists():
        process_single_image(image_path, config)
    else:
        print(f"Image not found: {image_path}")


# ====================================================
# CONTOH 4: BATCH PROCESSING WITH DIFFERENT CONFIGS
# ====================================================

def example_batch_processing():
    """Batch processing dengan berbagai konfigurasi"""
    
    print("=" * 60)
    print("CONTOH 4: Batch Processing Different Configs")
    print("=" * 60)
    
    configs = [
        {
            'name': 'Strict',
            'biota_conf': 0.5,
            'starfish_conf': 0.7
        },
        {
            'name': 'Balanced',
            'biota_conf': 0.4,
            'starfish_conf': 0.6
        },
        {
            'name': 'Sensitive',
            'biota_conf': 0.3,
            'starfish_conf': 0.5
        }
    ]
    
    image_path = "combined_dataset/test/images/sample_image.jpg"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    for cfg in configs:
        print(f"\nProcessing with {cfg['name']} config:")
        config = DetectionConfig()
        config.CONF_BIOTA = cfg['biota_conf']
        config.CONF_STARFISH = cfg['starfish_conf']
        
        # Tambah suffix ke hasil
        original_results = config.RESULTS_DIR
        config.RESULTS_DIR = original_results / cfg['name'].lower()
        config.RESULTS_DIR.mkdir(exist_ok=True)
        
        process_single_image(image_path, config)


# ====================================================
# QUICK START - PILIH SALAH SATU
# ====================================================

if __name__ == "__main__":
    
    # Pilih satu contoh untuk dijalankan:
    
    # 1. Jalankan single image
    example_single_image()
    
    # 2. Jalankan multiple images
    # example_multiple_images()
    
    # 3. Jalankan dengan custom config
    # example_custom_config()
    
    # 4. Jalankan batch processing
    # example_batch_processing()
    
    print("\n" + "=" * 60)
    print("✓ Processing Complete!")
    print("=" * 60)
    print("\nHasil tersimpan di folder 'results/' dengan struktur:")
    print("  results/")
    print("    ├── detection/        (detection results .txt)")
    print("    ├── segmentation/     (segmentation masks)")
    print("    ├── features/         (feature data)")
    print("    ├── visualizations/   (output images)")
    print("    └── csv/              (feature_results.csv)")
