"""
TEST & DEMO SCRIPT - DUAL MODEL DETECTION SYSTEM

Script ini untuk testing dan demo cepat sistem deteksi.
Gunakan ini untuk:
1. Test apakah model sudah load dengan benar
2. Demo sistem dengan image test
3. Validasi hasil detection
"""

from dual_model_detection_system import DetectionConfig, YOLO
from pathlib import Path
import os


def test_models_available():
    """Test apakah model files tersedia"""
    print("\n" + "=" * 60)
    print("TEST 1: Model Files Availability")
    print("=" * 60)
    
    config = DetectionConfig()
    
    models = [
        config.MODEL_BIOTA,
        config.MODEL_STARFISH
    ]
    
    for model_name in models:
        if Path(model_name).exists():
            size = os.path.getsize(model_name) / 1024 / 1024
            print(f"✓ {model_name} ({size:.1f} MB)")
        else:
            print(f"✗ {model_name} - NOT FOUND")
    
    print("=" * 60)


def test_models_load():
    """Test loading models"""
    print("\n" + "=" * 60)
    print("TEST 2: Loading Models")
    print("=" * 60)
    
    config = DetectionConfig()
    
    try:
        print(f"Loading {config.MODEL_BIOTA}...")
        model_biota = YOLO(config.MODEL_BIOTA)
        print(f"✓ Biota model loaded")
        
        print(f"Loading {config.MODEL_STARFISH}...")
        model_starfish = YOLO(config.MODEL_STARFISH)
        print(f"✓ Starfish model loaded")
        
        print(f"\nModel info:")
        print(f"  Biota - Task: {model_biota.task}, Names: {model_biota.names}")
        print(f"  Starfish - Task: {model_starfish.task}, Names: {model_starfish.names}")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
    
    print("=" * 60)


def test_results_directory():
    """Test hasil directory structure"""
    print("\n" + "=" * 60)
    print("TEST 3: Results Directory Structure")
    print("=" * 60)
    
    config = DetectionConfig()
    
    # Create directories
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.DETECTION_DIR.mkdir(exist_ok=True)
    config.SEGMENTATION_DIR.mkdir(exist_ok=True)
    config.FEATURES_DIR.mkdir(exist_ok=True)
    config.VIZ_DIR.mkdir(exist_ok=True)
    config.CSV_DIR.mkdir(exist_ok=True)
    
    dirs = [
        config.DETECTION_DIR,
        config.SEGMENTATION_DIR,
        config.FEATURES_DIR,
        config.VIZ_DIR,
        config.CSV_DIR
    ]
    
    for dir_path in dirs:
        if dir_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path}")
    
    print("=" * 60)


def test_config():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("TEST 4: Configuration")
    print("=" * 60)
    
    config = DetectionConfig()
    
    print(f"\nModel Settings:")
    print(f"  Biota confidence: {config.CONF_BIOTA}")
    print(f"  Starfish confidence: {config.CONF_STARFISH}")
    print(f"  Target size: {config.TARGET_SIZE}x{config.TARGET_SIZE}")
    
    print(f"\nClass Mapping:")
    for cls_id, cls_name in config.CLASS_NAMES.items():
        print(f"  {cls_id} = {cls_name}")
    
    print(f"\nClass Colors (BGR):")
    for cls_id, color in config.CLASS_COLORS.items():
        cls_name = config.CLASS_NAMES[cls_id]
        print(f"  {cls_name}: {color}")
    
    print(f"\nResult Directories:")
    print(f"  Base: {config.RESULTS_DIR}")
    print(f"  Detection: {config.DETECTION_DIR}")
    print(f"  Segmentation: {config.SEGMENTATION_DIR}")
    print(f"  Features: {config.FEATURES_DIR}")
    print(f"  Visualizations: {config.VIZ_DIR}")
    print(f"  CSV: {config.CSV_DIR}")
    
    print("=" * 60)


def test_image_input():
    """Test image input handling"""
    print("\n" + "=" * 60)
    print("TEST 5: Image Input")
    print("=" * 60)
    
    import cv2
    
    # Try to find test images
    test_paths = [
        "combined_dataset/test/images",
        "combined_detection_dataset/test/images",
        "test_images"
    ]
    
    found_images = []
    for path in test_paths:
        if Path(path).exists():
            images = list(Path(path).glob("*.jpg")) + list(Path(path).glob("*.png"))
            if images:
                found_images = images
                print(f"Found {len(images)} images in {path}")
                for img in images[:3]:
                    print(f"  - {img}")
                break
    
    if not found_images:
        print("⚠ No test images found")
        print("Try these paths:")
        for path in test_paths:
            print(f"  - {path}")
    
    print("=" * 60)


def test_full_pipeline():
    """Test full pipeline dengan dummy image"""
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline (Dummy Image)")
    print("=" * 60)
    
    import cv2
    import numpy as np
    from dual_model_detection_system import preprocess_image
    
    config = DetectionConfig()
    
    # Create dummy underwater-like image (640x640)
    dummy_img = np.random.randint(50, 120, (640, 640, 3), dtype=np.uint8)
    
    # Add some colored circles (simulating objects)
    cv2.circle(dummy_img, (100, 100), 30, (0, 0, 255), -1)  # Red object
    cv2.circle(dummy_img, (300, 300), 40, (0, 255, 0), -1)  # Green object
    cv2.circle(dummy_img, (500, 500), 35, (255, 0, 0), -1)  # Blue object
    
    # Save dummy image
    dummy_path = "dummy_test_image.jpg"
    cv2.imwrite(dummy_path, dummy_img)
    print(f"✓ Created dummy image: {dummy_path}")
    
    # Test preprocessing
    try:
        original, preprocessed = preprocess_image(dummy_path, config)
        print(f"✓ Preprocessing successful")
        print(f"  Original: {original.shape}")
        print(f"  Preprocessed: {preprocessed.shape}")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
    
    # Cleanup
    if Path(dummy_path).exists():
        os.remove(dummy_path)
        print(f"✓ Cleaned up dummy image")
    
    print("=" * 60)


def run_all_tests():
    """Run semua tests"""
    print("\n" + "=" * 80)
    print("DUAL MODEL DETECTION SYSTEM - DIAGNOSTIC TESTS")
    print("=" * 80)
    
    tests = [
        ("Model Availability", test_models_available),
        ("Model Loading", test_models_load),
        ("Directory Structure", test_results_directory),
        ("Configuration", test_config),
        ("Image Input", test_image_input),
        ("Pipeline Test", test_full_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "✓ PASS"))
        except Exception as e:
            results.append((test_name, f"✗ FAIL: {e}"))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    print("=" * 80 + "\n")


# ====================================================
# QUICK DEMO - PROCESS SINGLE IMAGE
# ====================================================

def quick_demo():
    """Quick demo dengan image dari dataset"""
    print("\n" + "=" * 80)
    print("QUICK DEMO - DUAL MODEL DETECTION")
    print("=" * 80)
    
    from dual_model_detection_system import process_single_image
    
    # Cari image test
    test_images = [
        "combined_dataset/test/images",
        "combined_detection_dataset/test/images"
    ]
    
    image_file = None
    for test_dir in test_images:
        images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
        if images:
            image_file = str(images[0])
            break
    
    if image_file:
        print(f"Processing: {image_file}\n")
        process_single_image(image_file)
    else:
        print("No test image found. Please provide an image path.")
        print("\nUsage:")
        print("  from dual_model_detection_system import process_single_image")
        print("  process_single_image('path/to/your/image.jpg')")


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            run_all_tests()
        elif command == "demo":
            quick_demo()
        elif command == "test1":
            test_models_available()
        elif command == "test2":
            test_models_load()
        elif command == "test3":
            test_results_directory()
        elif command == "test4":
            test_config()
        elif command == "test5":
            test_image_input()
        elif command == "test6":
            test_full_pipeline()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  python test_demo.py test         - Run all tests")
            print("  python test_demo.py demo         - Quick demo")
            print("  python test_demo.py test1        - Test model availability")
            print("  python test_demo.py test2        - Test model loading")
            print("  python test_demo.py test3        - Test directory structure")
            print("  python test_demo.py test4        - Test configuration")
            print("  python test_demo.py test5        - Test image input")
            print("  python test_demo.py test6        - Test pipeline")
    else:
        # Default: run all tests
        print("No command specified. Running all tests...\n")
        run_all_tests()
        
        print("=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Verify all tests passed (✓ marks)")
        print("\n2. If models not found:")
        print("   - Check if .pt files exist in current directory")
        print("   - Or update MODEL_BIOTA and MODEL_STARFISH paths in DetectionConfig")
        print("\n3. To run quick demo:")
        print("   python test_demo.py demo")
        print("\n4. To process your image:")
        print("   from dual_model_detection_system import process_single_image")
        print("   process_single_image('path/to/image.jpg')")
        print("\n" + "=" * 80)
