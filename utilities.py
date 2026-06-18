"""
UTILITIES & HELPER FUNCTIONS
=============================

Fungsi-fungsi tambahan untuk advanced use cases
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dual_model_detection_system import DetectionConfig


# ====================================================
# IMAGE UTILITIES
# ====================================================

def batch_resize_images(input_dir: str, output_dir: str, size: int = 640):
    """
    Batch resize images untuk preprocessing
    
    Args:
        input_dir: Directory input images
        output_dir: Directory output images
        size: Target size (square)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"Resizing {len(images)} images...")
    
    for img_file in images:
        img = cv2.imread(str(img_file))
        resized = cv2.resize(img, (size, size))
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), resized)
        print(f"  ✓ {img_file.name}")


def compare_images(original_path: str, processed_path: str):
    """
    Banding-bandingkan 2 image side-by-side
    
    Args:
        original_path: Path original image
        processed_path: Path processed image
    """
    import matplotlib.pyplot as plt
    
    orig = cv2.imread(original_path)
    proc = cv2.imread(processed_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Processed")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


# ====================================================
# ANALYSIS UTILITIES
# ====================================================

def load_feature_results(csv_path: str = "results/csv/feature_results.csv"):
    """
    Load feature results CSV
    
    Args:
        csv_path: Path ke CSV file
        
    Returns:
        pd.DataFrame: Feature data
    """
    df = pd.read_csv(csv_path)
    return df


def analyze_features_per_class(csv_path: str = "results/csv/feature_results.csv"):
    """
    Analisis fitur per class
    
    Args:
        csv_path: Path ke CSV file
    """
    df = load_feature_results(csv_path)
    
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS PER CLASS")
    print("=" * 60)
    
    for class_name in df['class'].unique():
        class_data = df[df['class'] == class_name]
        print(f"\n{class_name.upper()} (n={len(class_data)})")
        print("-" * 40)
        print(f"  Area: {class_data['area'].mean():.2f} ± {class_data['area'].std():.2f}")
        print(f"  Perimeter: {class_data['perimeter'].mean():.2f} ± {class_data['perimeter'].std():.2f}")
        print(f"  Circularity: {class_data['circularity'].mean():.4f} ± {class_data['circularity'].std():.4f}")
        print(f"  Aspect Ratio: {class_data['aspect_ratio'].mean():.4f} ± {class_data['aspect_ratio'].std():.4f}")
        print(f"  Avg Confidence: {class_data['confidence'].mean():.4f}")


def analyze_detection_stats(csv_path: str = "results/csv/feature_results.csv"):
    """
    Analisis statistik deteksi
    
    Args:
        csv_path: Path ke CSV file
    """
    df = load_feature_results(csv_path)
    
    print("\n" + "=" * 60)
    print("DETECTION STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal detections: {len(df)}")
    print(f"Unique images: {df['filename'].nunique()}")
    print(f"Average objects per image: {len(df) / df['filename'].nunique():.2f}")
    
    print(f"\nDetections per class:")
    for class_name, count in df['class'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nDetections per model:")
    for model_name, count in df['model'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {model_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nConfidence distribution:")
    print(f"  Mean: {df['confidence'].mean():.4f}")
    print(f"  Std: {df['confidence'].std():.4f}")
    print(f"  Min: {df['confidence'].min():.4f}")
    print(f"  Max: {df['confidence'].max():.4f}")


def export_analysis_summary(csv_path: str = "results/csv/feature_results.csv", 
                           output_path: str = "results/csv/analysis_summary.txt"):
    """
    Export analisis summary ke text file
    
    Args:
        csv_path: Path ke CSV file
        output_path: Path output file
    """
    df = load_feature_results(csv_path)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DUAL MODEL DETECTION ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Images Processed: {df['filename'].nunique()}\n")
        f.write(f"Total Objects Detected: {len(df)}\n")
        f.write(f"Avg Objects per Image: {len(df) / df['filename'].nunique():.2f}\n\n")
        
        f.write("DETECTIONS PER CLASS\n")
        f.write("-" * 40 + "\n")
        for class_name, count in df['class'].value_counts().items():
            f.write(f"{class_name}: {count}\n")
        
        f.write("\nDETECTIONS PER MODEL\n")
        f.write("-" * 40 + "\n")
        for model_name, count in df['model'].value_counts().items():
            f.write(f"{model_name}: {count}\n")
        
        f.write("\nFEATURE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Area (mean): {df['area'].mean():.2f}\n")
        f.write(f"Perimeter (mean): {df['perimeter'].mean():.2f}\n")
        f.write(f"Circularity (mean): {df['circularity'].mean():.4f}\n")
        f.write(f"Aspect Ratio (mean): {df['aspect_ratio'].mean():.4f}\n")
        f.write(f"Confidence (mean): {df['confidence'].mean():.4f}\n")
    
    print(f"Summary exported to: {output_path}")


# ====================================================
# VISUALIZATION UTILITIES
# ====================================================

def plot_confidence_distribution(csv_path: str = "results/csv/feature_results.csv"):
    """
    Plot confidence score distribution
    
    Args:
        csv_path: Path ke CSV file
    """
    import matplotlib.pyplot as plt
    
    df = load_feature_results(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(df['confidence'], bins=20, edgecolor='black')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Per class
    for class_name in df['class'].unique():
        class_conf = df[df['class'] == class_name]['confidence']
        axes[1].hist(class_conf, bins=15, alpha=0.6, label=class_name)
    
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution per Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_comparison(csv_path: str = "results/csv/feature_results.csv"):
    """
    Plot feature comparison per class
    
    Args:
        csv_path: Path ke CSV file
    """
    import matplotlib.pyplot as plt
    
    df = load_feature_results(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    features = ['area', 'perimeter', 'circularity', 'aspect_ratio']
    axes_flat = axes.flatten()
    
    for idx, feature in enumerate(features):
        df.boxplot(column=feature, by='class', ax=axes_flat[idx])
        axes_flat[idx].set_title(f'{feature.capitalize()} Distribution')
        axes_flat[idx].set_xlabel('Class')
        axes_flat[idx].set_ylabel(feature)
    
    plt.suptitle('Feature Comparison per Class')
    plt.tight_layout()
    plt.show()


# ====================================================
# MODEL UTILITIES
# ====================================================

def compare_model_performance(csv_path: str = "results/csv/feature_results.csv"):
    """
    Compare performance between biota and starfish model
    
    Args:
        csv_path: Path ke CSV file
    """
    df = load_feature_results(csv_path)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"  Objects detected: {len(model_data)}")
        print(f"  Avg confidence: {model_data['confidence'].mean():.4f}")
        print(f"  Min confidence: {model_data['confidence'].min():.4f}")
        print(f"  Max confidence: {model_data['confidence'].max():.4f}")
        print(f"  Classes found: {model_data['class'].unique().tolist()}")


# ====================================================
# CONFIG UTILITIES
# ====================================================

def create_custom_config(biota_conf: float, starfish_conf: float, 
                        output_dir: str = "results_custom"):
    """
    Buat custom configuration
    
    Args:
        biota_conf: Biota confidence threshold
        starfish_conf: Starfish confidence threshold
        output_dir: Custom output directory
        
    Returns:
        DetectionConfig: Custom configuration object
    """
    config = DetectionConfig()
    config.CONF_BIOTA = biota_conf
    config.CONF_STARFISH = starfish_conf
    
    custom_path = Path(output_dir)
    config.RESULTS_DIR = custom_path
    config.DETECTION_DIR = custom_path / "detection"
    config.SEGMENTATION_DIR = custom_path / "segmentation"
    config.FEATURES_DIR = custom_path / "features"
    config.VIZ_DIR = custom_path / "visualizations"
    config.CSV_DIR = custom_path / "csv"
    
    # Create directories
    for dir_path in [config.DETECTION_DIR, config.SEGMENTATION_DIR, 
                     config.FEATURES_DIR, config.VIZ_DIR, config.CSV_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Custom config created: {output_dir}")
    print(f"  Biota confidence: {biota_conf}")
    print(f"  Starfish confidence: {starfish_conf}")
    
    return config


# ====================================================
# EXPORT UTILITIES
# ====================================================

def export_to_json(csv_path: str = "results/csv/feature_results.csv",
                  json_path: str = "results/csv/feature_results.json"):
    """
    Export CSV ke JSON
    
    Args:
        csv_path: Path ke CSV file
        json_path: Path output JSON
    """
    df = pd.read_csv(csv_path)
    df.to_json(json_path, orient='records', indent=2)
    print(f"✓ Exported to JSON: {json_path}")


def export_to_excel(csv_path: str = "results/csv/feature_results.csv",
                   excel_path: str = "results/csv/feature_results.xlsx"):
    """
    Export CSV ke Excel
    
    Args:
        csv_path: Path ke CSV file
        excel_path: Path output Excel
    """
    try:
        df = pd.read_csv(csv_path)
        df.to_excel(excel_path, index=False)
        print(f"✓ Exported to Excel: {excel_path}")
    except ImportError:
        print("❌ openpyxl not installed. Install: pip install openpyxl")


# ====================================================
# CLEANUP UTILITIES
# ====================================================

def cleanup_results(keep_csv: bool = True, keep_viz: bool = True):
    """
    Cleanup temporary files
    
    Args:
        keep_csv: Keep CSV files
        keep_viz: Keep visualization files
    """
    import shutil
    
    config = DetectionConfig()
    
    print("\nCleaning up results directory...")
    
    if not keep_csv:
        if config.CSV_DIR.exists():
            shutil.rmtree(config.CSV_DIR)
            print("  ✓ Removed CSV files")
    
    if not keep_viz:
        if config.VIZ_DIR.exists():
            shutil.rmtree(config.VIZ_DIR)
            print("  ✓ Removed visualizations")


# ====================================================
# QUALITY CHECK UTILITIES
# ====================================================

def quality_check(csv_path: str = "results/csv/feature_results.csv"):
    """
    Perform quality checks on results
    
    Args:
        csv_path: Path ke CSV file
    """
    df = load_feature_results(csv_path)
    
    print("\n" + "=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)
    
    issues = []
    
    # Check 1: No NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"⚠ Found {nan_count} NaN values")
    else:
        print("✓ No NaN values")
    
    # Check 2: Confidence range
    if (df['confidence'] < 0).any() or (df['confidence'] > 1).any():
        issues.append("⚠ Confidence values out of range [0, 1]")
    else:
        print("✓ Confidence values in valid range")
    
    # Check 3: Positive areas
    if (df['area'] <= 0).any():
        issues.append("⚠ Found non-positive area values")
    else:
        print("✓ All areas positive")
    
    # Check 4: Circularity range
    if (df['circularity'] < 0).any() or (df['circularity'] > 1).any():
        issues.append("⚠ Circularity values out of typical range")
    else:
        print("✓ Circularity values valid")
    
    # Check 5: Sufficient detections
    if len(df) < 5:
        issues.append(f"⚠ Only {len(df)} detections (expected at least 5)")
    else:
        print(f"✓ Sufficient detections ({len(df)})")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ All quality checks passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("UTILITIES - Usage Examples")
    print("=" * 60)
    
    print("\n1. Analyze features:")
    print("   from utilities import analyze_features_per_class")
    print("   analyze_features_per_class()")
    
    print("\n2. Plot distributions:")
    print("   from utilities import plot_confidence_distribution")
    print("   plot_confidence_distribution()")
    
    print("\n3. Compare models:")
    print("   from utilities import compare_model_performance")
    print("   compare_model_performance()")
    
    print("\n4. Quality check:")
    print("   from utilities import quality_check")
    print("   quality_check()")
    
    print("\n5. Export analysis:")
    print("   from utilities import export_analysis_summary")
    print("   export_analysis_summary()")
    
    print("\n" + "=" * 60)
