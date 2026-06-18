"""
DEPLOYMENT CHECKLIST & SETUP GUIDE
====================================

Checklist untuk memastikan sistem siap production
"""

from pathlib import Path
import os
import sys


def check_python_version():
    """Check Python version"""
    print("\n" + "=" * 60)
    print("✓ Python Version Check")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    
    print("✓ Python version OK")
    return True


def check_dependencies():
    """Check required packages"""
    print("\n" + "=" * 60)
    print("✓ Checking Dependencies")
    print("=" * 60)
    
    required = {
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"✓ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name} - NOT INSTALLED")
    
    if missing:
        print(f"\nTo install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("✓ All dependencies installed")
    return True


def check_model_files():
    """Check model .pt files"""
    print("\n" + "=" * 60)
    print("✓ Checking Model Files")
    print("=" * 60)
    
    models = [
        "trained_3class_model_best.pt",
        "cascade_model_starfish_best.pt"
    ]
    
    found = []
    missing = []
    
    for model in models:
        if Path(model).exists():
            size = os.path.getsize(model) / 1024 / 1024
            found.append(model)
            print(f"✓ {model} ({size:.1f} MB)")
        else:
            missing.append(model)
            print(f"❌ {model} - NOT FOUND")
    
    if missing:
        print(f"\nMissing models in current directory:")
        for model in missing:
            print(f"  - {model}")
        print("\nExpected location: " + str(Path.cwd()))
        return False
    
    print("✓ All model files found")
    return True


def check_script_files():
    """Check main script files"""
    print("\n" + "=" * 60)
    print("✓ Checking Script Files")
    print("=" * 60)
    
    scripts = [
        "dual_model_detection_system.py",
        "example_usage.py",
        "test_demo.py",
        "utilities.py"
    ]
    
    found = []
    missing = []
    
    for script in scripts:
        if Path(script).exists():
            found.append(script)
            print(f"✓ {script}")
        else:
            missing.append(script)
            print(f"❌ {script} - NOT FOUND")
    
    if missing:
        print(f"\nMissing scripts:")
        for script in missing:
            print(f"  - {script}")
        return False
    
    print("✓ All script files found")
    return True


def check_directory_structure():
    """Check and create result directories"""
    print("\n" + "=" * 60)
    print("✓ Checking Directory Structure")
    print("=" * 60)
    
    from dual_model_detection_system import DetectionConfig
    
    config = DetectionConfig()
    
    dirs = [
        config.RESULTS_DIR,
        config.DETECTION_DIR,
        config.SEGMENTATION_DIR,
        config.FEATURES_DIR,
        config.VIZ_DIR,
        config.CSV_DIR
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")
    
    print("✓ Directory structure ready")
    return True


def check_disk_space():
    """Check available disk space"""
    print("\n" + "=" * 60)
    print("✓ Checking Disk Space")
    print("=" * 60)
    
    import shutil
    
    disk = shutil.disk_usage(".")
    free_gb = disk.free / (1024**3)
    
    print(f"Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 1:
        print(f"⚠ Less than 1 GB available!")
        return False
    
    print(f"✓ Sufficient disk space")
    return True


def check_write_permissions():
    """Check write permissions"""
    print("\n" + "=" * 60)
    print("✓ Checking Write Permissions")
    print("=" * 60)
    
    test_file = Path("_permission_test.txt")
    
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ Write permissions OK")
        return True
    except PermissionError:
        print(f"❌ No write permissions!")
        return False


def check_gpu_availability():
    """Check GPU availability"""
    print("\n" + "=" * 60)
    print("✓ Checking GPU Availability")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"⚠ GPU not available (CPU will be used)")
            return False
    except ImportError:
        print(f"⚠ PyTorch not installed (GPU check skipped)")
        return False


def run_quick_inference_test():
    """Run quick inference test"""
    print("\n" + "=" * 60)
    print("✓ Running Quick Inference Test")
    print("=" * 60)
    
    try:
        from dual_model_detection_system import DetectionConfig, YOLO
        import cv2
        import numpy as np
        
        config = DetectionConfig()
        
        # Create dummy image
        dummy_img = np.random.randint(50, 120, (640, 640, 3), dtype=np.uint8)
        
        print("Loading models...")
        model_biota = YOLO(config.MODEL_BIOTA)
        model_starfish = YOLO(config.MODEL_STARFISH)
        
        print("Running inference on dummy image...")
        result_biota = model_biota(dummy_img, conf=config.CONF_BIOTA, verbose=False)
        result_starfish = model_starfish(dummy_img, conf=config.CONF_STARFISH, verbose=False)
        
        print(f"✓ Biota model inference OK")
        print(f"✓ Starfish model inference OK")
        print(f"✓ Inference test passed")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


# ====================================================
# DEPLOYMENT CHECKLIST
# ====================================================

def full_deployment_check():
    """Run full deployment checklist"""
    print("\n\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  DUAL MODEL DETECTION SYSTEM - DEPLOYMENT CHECKLIST".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Script Files", check_script_files),
        ("Directory Structure", check_directory_structure),
        ("Disk Space", check_disk_space),
        ("Write Permissions", check_write_permissions),
        ("GPU Availability", check_gpu_availability),
        ("Inference Test", run_quick_inference_test),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Error during {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT CHECKLIST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    for check_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("\n✓ SYSTEM READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("  1. Process test image:")
        print("     from dual_model_detection_system import process_single_image")
        print("     process_single_image('test_image.jpg')")
        print("\n  2. View results in results/ directory")
        print("\n  3. Check feature_results.csv in results/csv/")
        return True
    else:
        print("\n❌ System not ready. Fix issues above.")
        return False


# ====================================================
# QUICK START AFTER DEPLOYMENT
# ====================================================

def print_quick_start():
    """Print quick start guide"""
    print("\n\n" + "=" * 60)
    print("QUICK START")
    print("=" * 60)
    
    print("""
1. Process single image:
   python -c "from dual_model_detection_system import process_single_image; process_single_image('image.jpg')"

2. Process multiple images:
   python -c "from dual_model_detection_system import process_multiple_images; process_multiple_images('images/')"

3. Custom configuration:
   from dual_model_detection_system import DetectionConfig, process_single_image
   config = DetectionConfig()
   config.CONF_BIOTA = 0.3
   config.CONF_STARFISH = 0.5
   process_single_image('image.jpg', config)

4. Run tests:
   python test_demo.py test

5. Analyze results:
   from utilities import analyze_features_per_class
   analyze_features_per_class()

Results location:
  - Visualizations: results/visualizations/
  - CSV data: results/csv/feature_results.csv
  - Detection results: results/detection/
    """)


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    import sys
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + "  DUAL MODEL DETECTION SYSTEM".center(58) + "║")
    print("║" + "  Setup & Deployment Check".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-inference":
        print("\nRunning checks without inference test...")
        checks = [
            ("Python Version", check_python_version),
            ("Dependencies", check_dependencies),
            ("Model Files", check_model_files),
            ("Script Files", check_script_files),
            ("Directory Structure", check_directory_structure),
            ("Disk Space", check_disk_space),
            ("Write Permissions", check_write_permissions),
            ("GPU Availability", check_gpu_availability),
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = check_func()
                results.append((check_name, result))
            except Exception as e:
                results.append((check_name, False))
    else:
        # Full deployment check
        success = full_deployment_check()
        sys.exit(0 if success else 1)
    
    print_quick_start()
