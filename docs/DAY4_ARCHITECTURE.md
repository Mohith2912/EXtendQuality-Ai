# Day 4: Pixel-Level Inspection Engine Architecture

This document describes the actual architecture and pipeline of the Day 4 ExtendQuality AI (EQM) inspection engine.

```
Input Image
     ↓
Image Validation (Format, dimensions, blur, brightness, contrast, channels)
     ↓
Preprocessing (Resize, grayscale conversion, Gaussian denoise, CLAHE, normalize)
     ↓
Pixel-Level Segmentation & Analysis (Adaptive threshold + Circular Bearing Mask + morphology + contours)
     ↓
Feature Extraction (Pixel counts, edge density, regions, intensity stats, contrast ratios)
     ↓
Explainable Quality Scoring (Multi-component penalty deduction: base 100 - penalties)
     ↓
Deterministic Classification (PASS / WARNING / FAIL status mapping)
     ↓
Structured Result Generation (Pydantic model with JSON serialization)
     ↓
Logging & Debug Visualization (Side-by-side original + mask overlay)
```

---

## 1. What Happens to the Image (Pipeline Steps)
Every input image undergoes the following deterministic sequence:
1. **Validation**: Verified for format, dimensions, readability, and basic image quality (not too blurry, not too dark or overexposed, sufficient contrast). Supports both BGR/RGB color and single-channel grayscale inputs.
2. **Preprocessing**:
   - Resized to standard size ($640 \times 640$ pixels by default) using area interpolation.
   - Converted to grayscale.
   - Denoised using a Gaussian filter (default kernel size $5 \times 5$).
   - Enhanced local contrast using **CLAHE** (Contrast Limited Adaptive Histogram Equalization, default clip limit $2.0$).
   - Normalized to full $[0, 255]$ range to standardize brightness.

---

## 2. How Affected Pixels/Regions are Identified
Instead of using deep-learning bounding boxes, Day 4 uses a robust OpenCV-based computer vision segmentation pipeline:
1. **Adaptive Thresholding**: Local mean-gaussian thresholding isolates potential anomalous regions from their immediate neighborhood.
2. **Circular Bearing Masking**: Since the bearing sits in a fixed central region, we apply a circular ring mask (outer radius 252, inner radius 148 at 640x640 scale) to ignore background and center hole pixels. This prevents false detections in non-bearing regions.
3. **Morphological Filtering**: An opening operation cleans small salt-and-pepper noise, followed by a closing operation to merge nearby defect pixels.
4. **Contour/Connected Component Analysis**: Distinct contiguous defect regions are extracted as contours. Contours smaller than the area threshold (default 50 pixels) are filtered out.

---

## 3. Which Features are Extracted
For every inspection, we extract a dictionary of shape and intensity features:
- `total_defect_pixels`: Count of pixels identified as defective.
- `affected_pixel_percentage`: Ratio of defect pixels to total image pixels.
- `edge_density`: Ratio of edge pixels to total image pixels (computed using Canny).
- `region_count`: Number of distinct defect contours.
- `largest_region_area`: Area of the largest individual defect region in pixels.
- `defect_area_total`: Sum of areas of all detected regions.
- `mean_defect_intensity` & `std_defect_intensity`: Average brightness and variance inside the defect regions.
- `mean_normal_intensity`: Average brightness in non-defective regions.
- `contrast_ratio`: Normalized intensity difference between defect and normal regions.

---

## 4. How the Quality Score is Calculated
The quality score is explainable and deterministic, starting at a base of `100.0` and applying weighted deductions:

$$\text{Quality Score} = 100.0 - P_{\text{pixel}} - P_{\text{edge}} - P_{\text{region}} - P_{\text{iq}}$$

- **Pixel Defect Penalty ($P_{\text{pixel}}$)**: Max weight `40.0`. Scales linearly from $0\%$ to $15\%$ affected pixels (critical threshold).
- **Edge Density Penalty ($P_{\text{edge}}$)**: Max weight `15.0`. Scales linearly up to an edge density of `0.25`.
- **Region Count Penalty ($P_{\text{region}}$)**: Max weight `10.0`. Scales linearly up to `20` defect regions.
- **Image Quality Penalty ($P_{\text{iq}}$)**: Max weight `10.0`. Penalizes borderline blur, brightness, and contrast values that are close to validation rejection limits.

The final score is clamped to the range $[0.0, 100.0]$.

---

## 5. How PASS / WARNING / FAIL is Determined
The final quality score is mapped directly to a status classification using configured thresholds (default: PASS $\ge 90.0$, WARNING $\ge 70.0$, FAIL $< 70.0$):
- **PASS**: $\text{score} \ge 90.0$
- **WARNING**: $70.0 \le \text{score} < 90.0$
- **FAIL**: $\text{score} < 70.0$

Boundary conditions are explicitly validated (e.g., exactly 90.0 maps to PASS, 89.9 to WARNING, exactly 70.0 to WARNING, 69.9 to FAIL).

---

## 6. Why Pixel-Level Analysis is Preferred over Bounding Boxes
Bounding boxes outline coarse rectangular areas and over-estimate defect dimensions (e.g. a thin diagonal scratch covering a small area gets a huge bounding box). Pixel-level analysis:
- Measures the **exact surface area** affected by surface wear, scratches, or corrosion.
- Quantifies count, distribution, and geometry of anomalies directly.
- Avoids false scale estimations and is more appropriate for industrial quality standards (like SOP tolerance checks).

---

## 7. Where Canny Edge Detection is Used
Canny is strictly used in `feature_extractor.py` to calculate the overall **edge density** of the preprocessed image. It helps identify surface roughness or scratch complexity. It is **not** used to segment defect regions (which is handled by adaptive thresholding).

---

## 8. How Errors are Handled
The engine provides graceful containment:
- Bad, missing, or corrupted image files are caught at the start of the pipeline.
- Controlled validation/processing errors return an `InspectionResult` with `status="ERROR"` and details in `errors` list.
- Normal execution flow remains uninterrupted. Batch processing is protected so that one bad image doesn't crash the entire run.

---

## 9. How Batch Inspection Works
The batch CLI accepts a directory of images:
1. Validates each image individually.
2. Accumulates individual `InspectionResult` objects.
3. If an image fails to process, it records the error and proceeds to the next image.
4. Generates aggregate statistics (total processed, PASS/WARNING/FAIL counts, average score, average time, error counts) and returns a complete `BatchSummary` serializable to JSON.

---

## 10. How the System is Tested
Our test suite consists of **33 unit and integration tests** verifying:
- **Image Validation**: Handles missing, invalid formats, corrupted files, and grayscale compatibility.
- **Preprocessing**: Resizing, grayscale, denoise, and CLAHE behavior.
- **Segmentation**: Clean vs. anomalous image detection, circular ring masking behavior.
- **Feature Extraction**: Key presence, correct metric math.
- **Scoring**: Boundary conditions, penalty weights, image quality impact.
- **Batch Processing**: Mixed results, error containment.
- **Regression**: All 22 existing Day 1-Day 3 tests pass successfully.
