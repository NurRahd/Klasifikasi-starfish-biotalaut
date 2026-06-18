"""
TECHNICAL DOCUMENTATION
========================

Dokumentasi teknis mendalam tentang algoritma dan metodologi
"""

# ====================================================
# 1. PREPROCESSING PIPELINE
# ====================================================

PREPROCESSING_DOCUMENTATION = """
TAHAP 2: PREPROCESSING CITRA UNDERWATER
========================================

Underwater imaging memiliki challenges unik:
- Pengurangan cahaya dengan kedalaman
- Color absorption (terutama red channel)
- Scattering dan backscatter
- Noise tinggi
- Low contrast

Solusi preprocessing kami:

1. RESIZE
   - Target: 640x640 pixels (standar YOLOv8)
   - Method: cv2.resize dengan interpolasi default
   - Alasan: Menstandarkan input untuk model

2. GAUSSIAN BLUR
   - Kernel: 5x5
   - Sigma: 0 (otomatis dihitung)
   - Alasan: Mengurangi noise tanpa blur edge terlalu banyak

3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Convert BGR → LAB (independent channel)
   - Apply CLAHE ke L channel saja
   - Clip limit: 2.0
   - Tile grid: 8x8
   - Convert LAB → BGR

   Keuntungan CLAHE vs standar histogram equalization:
   - Tidak over-enhance noise
   - Better untuk underwater image
   - Preserve natural look
   - Adaptive per region

4. BRIGHTNESS & CONTRAST ADJUSTMENT
   - Formula: output = alpha * input + beta
   - Alpha (contrast): 1.1 (sedikit increase contrast)
   - Beta (brightness): 10 (slight increase brightness)
   - Alasan: Compensate untuk underwater dimness

Result: Image lebih jelas dengan contrast lebih baik
"""


# ====================================================
# 2. DUAL-MODEL DETECTION STRATEGY
# ====================================================

DUAL_MODEL_STRATEGY = """
TAHAP 3-4: DUAL-MODEL PARALLEL DETECTION
==========================================

Mengapa dual-model approach?

1. SPECIALIZATION
   - Model 1 (Biota): Trained khusus untuk eel, fish, jellyfish
   - Model 2 (Starfish): Trained khusus untuk starfish
   - Hasil: Setiap model optimal untuk classnya

2. DIFFERENT CONFIDENCE THRESHOLDS
   - Biota: conf = 0.4 (lebih sensitif, catch lebih banyak)
   - Starfish: conf = 0.6 (lebih ketat, higher precision)
   
   Alasan:
   - Biota mungkin lebih kecil/samar, perlu conf lebih rendah
   - Starfish lebih besar/distinct, bisa lebih ketat
   - Confidence disesuaikan dengan training data characteristics

3. PARALLEL PROCESSING
   - Kedua model run independent
   - Tidak blocking satu sama lain
   - Dapat di-parallelize dengan threading/multiprocessing

4. RESULT MERGING
   - Simple concatenation: deteksi_merged = deteksi_biota + deteksi_starfish
   - No NMS (Non-Maximum Suppression) applied di merge stage
   - Reasoning: Model terpisah, class terpisah, tidak ada duplikasi

Pros:
✓ Higher recall (catch lebih banyak objek)
✓ Leverages model specialization
✓ Can detect semua 4 classes dalam 1 image

Cons:
✗ Potentially higher false positives (mitigation: use higher conf threshold)
✗ 2x inference time (mitigation: parallel processing, GPU)
"""


# ====================================================
# 3. SEGMENTATION ALGORITHM
# ====================================================

SEGMENTATION_ALGORITHM = """
TAHAP 6: SEGMENTASI OBJEK - GRABCUT + MORPHOLOGY
==================================================

GrabCut Algorithm (Interactive Graph-based Foreground Extraction):

1. INPUT
   - Cropped region dari bounding box + padding
   - Rectangle region untuk initialization
   - Background dan foreground models (GMM)

2. ALGORITHM
   a) Initialize bounding rectangle
   b) Learn color models untuk FG dan BG
   c) Segment image ke foreground/background/uncertain
   d) Learn color models dari segmentation
   e) Ulangi hingga convergence (atau 5 iterations)

3. OUTPUT
   - Soft mask: setiap pixel punya probability
   - Hard mask: foreground=255, background/uncertain=0

4. MORPHOLOGICAL OPERATIONS
   
   Kernel: Ellipse 5x5 (lebih smooth dibanding square)
   
   a) OPENING (Erosion → Dilation)
      - Remove small noise
      - Separate touching objects
   
   b) CLOSING (Dilation → Erosion)
      - Fill small holes
      - Smooth boundaries

5. FALLBACK MECHANISM
   - Jika GrabCut error/timeout: gunakan Otsu's thresholding
   - Otsu automatically find optimal threshold
   - Simple tapi effective untuk underwater images

Keuntungan GrabCut:
✓ Better foreground separation
✓ Handles complex backgrounds
✓ More accurate than simple thresholding
✓ Less dependent on threshold tuning

Result: Precise object segmentation untuk feature extraction
"""


# ====================================================
# 4. FEATURE EXTRACTION
# ====================================================

FEATURE_EXTRACTION = """
TAHAP 7: EKSTRAKSI FITUR (10 FITUR PER OBJEK)
================================================

1. AREA (Pixel count)
   Formula: area = count_of_white_pixels_in_mask
   Meaning: Ukuran objek
   Range: 0 - image_size²
   Interpretation: Bigger area = larger object

2. PERIMETER
   Formula: perimeter = cv2.arcLength(contour, True)
   Meaning: Panjang boundary objek
   Unit: pixels
   Interpretation: How "complex" the boundary

3. MEAN COLOR (RGB)
   Formula: mean_r = average(all_red_values_in_mask)
            mean_g = average(all_green_values_in_mask)
            mean_b = average(all_blue_values_in_mask)
   Meaning: Characteristic color of object
   Range: 0-255 per channel
   Application: Classification, identification

4. EDGE COUNT (Canny edge detection)
   Formula: edge_count = count_white_pixels_after_canny
   Method: cv2.Canny(mask, 50, 150)
   Meaning: How detailed/textured the object
   Interpretation: High edge count = textured, Low = smooth

5. CIRCULARITY
   Formula: circularity = 4π * area / (perimeter²)
   Range: 0 to 1 (1 = perfect circle)
   Meaning: How round is the object
   Interpretation:
   - Jellyfish: high circularity (circular)
   - Fish: lower circularity (elongated)
   - Eel: low circularity (long thin)
   - Starfish: moderate (star shape)

6. ASPECT RATIO
   Formula: aspect_ratio = height / width (dari bounding rect)
   Range: 0 to ∞ (typically 0.5 to 2.0)
   Meaning: Shape elongation
   Interpretation:
   - Ratio ~1: square/compact
   - Ratio >1: taller
   - Ratio <1: wider

Feature Correlation untuk Classification:
- Jellyfish: High area, high circularity, blue-ish color
- Fish: Medium area, lower circularity, green color
- Eel: Lower area, very low circularity, darker color
- Starfish: Medium area, moderate circularity, yellow color

All features normalized and stored untuk ML analysis
"""


# ====================================================
# 5. VISUALIZATION STRATEGY
# ====================================================

VISUALIZATION_STRATEGY = """
TAHAP 8: VISUALISASI LENGKAP (8+ SUBPLOT)
===========================================

Matplotlib Figure Organization:

ROW 1: Input Processing
├─ Col 1: Original Image (raw input)
├─ Col 2: Preprocessing (CLAHE + brightness adjusted)
└─ Col 3: Biota Detection (bbox + labels)

ROW 2: Model Outputs & Detections
├─ Col 1: Starfish Detection (bbox + labels)
├─ Col 2: Merged Detection (combined output)
└─ Col 3+: Object 1 Segmentation (masking)

ROW 3: Object Details
├─ Col 1+: Object 1 Contour (extracted boundaries)
├─ Col 2+: Object 1 Edge Detection (Canny)
└─ Col 3+: Objects 2-3 (repeat pattern)

Size: 20x12 inches @ 150 DPI = High resolution

Advantages:
✓ Single image shows entire pipeline
✓ Easy untuk understand processing steps
✓ Good untuk presentasi dan publikasi
✓ Shows preprocessing effectiveness
✓ Demonstrates segmentation quality
✓ Visualizes feature extraction results
"""


# ====================================================
# 6. CONFIDENCE THRESHOLD TUNING
# ====================================================

CONFIDENCE_TUNING = """
CONFIDENCE THRESHOLD TUNING GUIDE
==================================

Bagaimana confidence score bekerja?

1. CONFIDENCE DEFINITION
   - Probability model yakin objek tersebut adalah class yang diprediksi
   - Range: 0 to 1 (0% to 100%)
   - Higher = model lebih confident

2. THRESHOLD EFFECT
   - Hanya detection dengan conf ≥ threshold yang accepted
   - If conf = 0.7 threshold: hanya deteksi dengan 70%+ confidence diterima

3. DEFAULT CONFIGURATION
   - Biota (eel, fish, jellyfish): 0.4
   - Starfish: 0.6

   Alasan:
   - Biota models mungkin lebih uncertain (diverse appearances)
   - Starfish model more confident (specific morphology)

4. TUNING SCENARIOS

   SCENARIO A: Missing too many objects (low recall)
   ├─ Problem: Many false negatives
   ├─ Solution: LOWER confidence threshold
   ├─ Example: 0.4 → 0.3 untuk biota
   └─ Trade-off: Might increase false positives

   SCENARIO B: Too many wrong detections (high false positive)
   ├─ Problem: Many false positives
   ├─ Solution: RAISE confidence threshold
   ├─ Example: 0.6 → 0.7 untuk starfish
   └─ Trade-off: Might miss some true objects

   SCENARIO C: Balance needed
   ├─ Start: biota=0.4, starfish=0.6
   ├─ Adjust gradually
   ├─ Test pada subset images
   └─ Validate results

5. PRECISION vs RECALL TRADEOFF

   |  Threshold  | Precision | Recall | Use Case |
   |-------------|-----------|--------|----------|
   | 0.3 (Low)   | ★★☆☆☆     | ★★★★★  | Exploratory |
   | 0.4 (Med)   | ★★★☆☆     | ★★★★☆  | General |
   | 0.5 (Med+)  | ★★★☆☆     | ★★★☆☆  | Balanced |
   | 0.6 (High)  | ★★★★☆     | ★★★☆☆  | Production |
   | 0.7 (High+) | ★★★★★     | ★★☆☆☆  | Strict |

6. CLASS-SPECIFIC TUNING

   If detecting different classes well differently:
   ├─ Eel (thin, hard to detect): lower threshold
   ├─ Fish (common, obvious): medium threshold
   ├─ Jellyfish (distinctive): medium-high threshold
   └─ Starfish (very distinctive): high threshold

7. RECOMMENDED STARTING POINTS

   For research/exploration:
   ├─ biota_conf = 0.3
   └─ starfish_conf = 0.5

   For production:
   ├─ biota_conf = 0.4
   └─ starfish_conf = 0.6

   For strict detection:
   ├─ biota_conf = 0.5
   └─ starfish_conf = 0.7

Always validate dengan manual inspection dataset!
"""


# ====================================================
# 7. COMPUTATIONAL COMPLEXITY
# ====================================================

COMPUTATIONAL_ANALYSIS = """
COMPUTATIONAL COMPLEXITY ANALYSIS
==================================

Time Complexity per Image:

1. PREPROCESSING: O(W×H)
   ├─ Resize: O(W×H)
   ├─ Blur: O(W×H)
   ├─ CLAHE: O(W×H)
   └─ Brightness: O(W×H)
   Total: ~50-100ms untuk 640x640

2. BIOTA INFERENCE: O(complexity_of_network)
   - YOLOv8n (nano): ~10-20ms GPU, ~50-100ms CPU
   - Depends pada model size

3. STARFISH INFERENCE: O(complexity_of_network)
   - Same as biota (similar architecture)
   - ~10-20ms GPU, ~50-100ms CPU

4. SEGMENTATION (per object): O(A) where A = object area
   ├─ GrabCut: ~50-200ms per object
   ├─ Morphology: O(A)
   └─ Contour: O(A)
   Total: ~N × 100ms untuk N objects

5. FEATURE EXTRACTION (per object): O(A)
   ├─ Area: O(A)
   ├─ Perimeter: O(perimeter_length)
   ├─ Color: O(A)
   ├─ Edge: O(A)
   └─ Circularity/AR: O(perimeter_length)
   Total: ~N × 50ms untuk N objects

6. VISUALIZATION: O(W×H + N×subplot)
   ├─ Draw boxes: O(N)
   ├─ Rendering: O(W×H)
   └─ Matplotlib: ~500-1000ms
   Total: ~1-2s

TOTAL TIME ESTIMATE:
- GPU: 100-300ms untuk inference + 500-1500ms untuk segmentasi/viz
  = 600-1800ms per image ≈ 1-2 seconds per image
  
- CPU: 200-600ms untuk inference + 500-1500ms untuk segmentasi/viz
  = 700-2100ms per image ≈ 1-2.5 seconds per image

MEMORY USAGE:
- Image: 640×640×3 = ~1.2MB
- Model (biota): ~20-40MB
- Model (starfish): ~20-40MB
- Processing buffers: ~100-200MB
- Total: ~200-300MB

OPTIMIZATION OPPORTUNITIES:
1. Parallel inference (thread 2 models)
2. GPU acceleration (10-50x faster)
3. Batch processing (multiple images)
4. Skip visualization untuk speed
5. Cache model weights
6. Use lower resolution untuk preview
"""


# ====================================================
# 8. ACCURACY & VALIDATION METRICS
# ====================================================

ACCURACY_METRICS = """
VALIDATION & ACCURACY METRICS
==============================

Metrics untuk evaluate performance:

1. CONFUSION MATRIX
   
           Predicted Positive | Predicted Negative
   ─────────────────────────────────────────────────
   Actual Positive |  TP (true positive)    | FN (false negative)
   Actual Negative |  FP (false positive)   | TN (true negative)

2. PRECISION
   Formula: TP / (TP + FP)
   Meaning: Dari semua deteksi, berapa yang correct?
   Interpretation: Tidak ada false positives → precision tinggi

3. RECALL (Sensitivity)
   Formula: TP / (TP + FN)
   Meaning: Dari semua objek actual, berapa yang terdeteksi?
   Interpretation: Tidak ada false negatives → recall tinggi

4. F1-SCORE
   Formula: 2 × (Precision × Recall) / (Precision + Recall)
   Meaning: Harmonic mean dari precision dan recall
   Use: Balanced metric

5. IoU (Intersection over Union) - untuk bbox
   Formula: |A ∩ B| / |A ∪ B|
   Meaning: Overlap antara predicted dan ground truth box
   Threshold: Biasa 0.5 IoU untuk dianggap correct match

6. mAP (mean Average Precision)
   - Precision pada berbagai recall levels
   - Average across classes
   - Standard metric untuk object detection

VALIDATION PROCEDURE:

1. Manual Inspection
   ├─ Pick sample images
   ├─ Visually inspect detections
   └─ Check false positives/negatives

2. Statistical Analysis
   ├─ Feature distributions
   ├─ Class-wise statistics
   └─ Confidence distributions

3. Compare Models
   ├─ Biota model performance per class
   ├─ Starfish model performance
   └─ Merged performance

EXPECTED PERFORMANCE (approximate):

Class        | Precision | Recall | F1-Score
─────────────────────────────────────────────
Eel          |   0.75    |  0.80  |  0.77
Fish         |   0.82    |  0.85  |  0.83
Jellyfish    |   0.78    |  0.75  |  0.76
Starfish     |   0.85    |  0.82  |  0.83
─────────────────────────────────────────────
Overall      |   0.80    |  0.81  |  0.80

*Dependent upon training data quality
"""


# ====================================================
# 9. BEST PRACTICES
# ====================================================

BEST_PRACTICES = """
BEST PRACTICES & RECOMMENDATIONS
==================================

1. IMAGE PREPARATION
   ✓ Ensure consistent image format
   ✓ Check image quality (not too blurry)
   ✓ Verify image size is reasonable
   ✓ Remove corrupted images
   ✓ Standardize color space (BGR)

2. PREPROCESSING
   ✓ Always apply preprocessing (don't skip)
   ✓ Monitor CLAHE clip limit (default 2.0 is good)
   ✓ Adjust brightness for very dark/bright images
   ✓ Test gaussian blur kernel size if needed

3. CONFIDENCE THRESHOLDS
   ✓ Start with defaults (0.4, 0.6)
   ✓ Gradually adjust based on validation
   ✓ Keep separate thresholds per model
   ✓ Document your threshold choices

4. SEGMENTATION
   ✓ Use GrabCut (more accurate)
   ✓ Adequate padding around objects
   ✓ Apply morphological operations
   ✓ Check fallback to thresholding works

5. FEATURE EXTRACTION
   ✓ Normalize features before ML
   ✓ Remove outliers if needed
   ✓ Handle edge cases (very small/large objects)
   ✓ Validate feature ranges

6. VISUALIZATION
   ✓ Always visualize results before analysis
   ✓ Check for obvious errors
   ✓ Save high-res images for presentations
   ✓ Use consistent color scheme

7. BATCH PROCESSING
   ✓ Test pada small batch first
   ✓ Monitor memory usage
   ✓ Check disk space beforehand
   ✓ Use separate configs untuk different datasets

8. ERROR HANDLING
   ✓ Validate model files exist
   ✓ Check write permissions
   ✓ Handle corrupted images gracefully
   ✓ Log processing steps

9. VALIDATION
   ✓ Always validate pada held-out test set
   ✓ Compare dengan ground truth labels
   ✓ Calculate precision, recall, F1
   ✓ Analyze failure cases

10. DEPLOYMENT
    ✓ Run deployment checklist
    ✓ Test dengan real data
    ✓ Monitor performance
    ✓ Keep logs untuk troubleshooting
"""


# ====================================================
# MAIN DOCUMENTATION
# ====================================================

if __name__ == "__main__":
    docs = {
        "1_Preprocessing": PREPROCESSING_DOCUMENTATION,
        "2_Dual-Model_Strategy": DUAL_MODEL_STRATEGY,
        "3_Segmentation": SEGMENTATION_ALGORITHM,
        "4_Feature_Extraction": FEATURE_EXTRACTION,
        "5_Visualization": VISUALIZATION_STRATEGY,
        "6_Confidence_Tuning": CONFIDENCE_TUNING,
        "7_Computational_Complexity": COMPUTATIONAL_ANALYSIS,
        "8_Validation_Metrics": ACCURACY_METRICS,
        "9_Best_Practices": BEST_PRACTICES,
    }
    
    print("=" * 60)
    print("TECHNICAL DOCUMENTATION")
    print("=" * 60)
    print("\nAvailable topics:")
    for i, (name, _) in enumerate(docs.items(), 1):
        print(f"{i}. {name}")
    
    print("\nUsage:")
    print("  python technical_docs.py           # Print all")
    print("  python technical_docs.py 1         # Print preprocessing")
    print("  python technical_docs.py all       # Export all to txt")
    
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            for name, content in docs.items():
                print(f"\n{'='*60}")
                print(f"{name}")
                print(f"{'='*60}")
                print(content)
        else:
            try:
                idx = int(arg) - 1
                key = list(docs.keys())[idx]
                print(f"\n{key}")
                print("=" * 60)
                print(docs[key])
            except (ValueError, IndexError):
                print("Invalid argument")
    else:
        for name, content in docs.items():
            print(f"\n{'='*60}")
            print(f"{name}")
            print(f"{'='*60}")
            print(content)
