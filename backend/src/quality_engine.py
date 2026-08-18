from typing import Dict, Any, List, Optional
from backend.src.detector import DetectionResult
from backend.src.scoring import calculate_base_score, aggregate_confidence, calculate_defect_severity
from backend.src.config import settings

def determine_status(score: float) -> str:
    """
    Determines the PASS/WARNING/FAIL status based on the quality score.
    """
    if score >= settings.SCORE_PASS_THRESHOLD:
        return "PASS"
    elif score >= settings.SCORE_WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "FAIL"

def evaluate_quality(detection_result: DetectionResult) -> dict:
    """
    Evaluates the quality of a part based on the DetectionResult.
    Returns a dictionary with the evaluation metrics.
    """
    if not detection_result.model_available or detection_result.status != "INFERENCE_SUCCESS":
        return {
            "status": "ERROR",
            "quality_score": 0.0,
            "confidence": 0.0,
            "defects": [],
            "message": f"Cannot evaluate quality: {detection_result.status}"
        }
        
    score = calculate_base_score(detection_result.detections)
    overall_confidence = aggregate_confidence(detection_result.detections)
    status = determine_status(score)
    
    # Process defects for reporting
    defects = []
    for det in detection_result.detections:
        if det.class_name == "good":
            continue
            
        defects.append({
            "type": det.class_name,
            "confidence": det.confidence,
            "severity": calculate_defect_severity(det),
            "bbox": det.bbox
        })
        
    return {
        "status": status,
        "quality_score": score,
        "confidence": overall_confidence,
        "defects": defects,
        "message": "Quality evaluation successful"
    }


# ===========================================================================
# Day 4: Pixel-level inspection pipeline
# ===========================================================================
import time
import logging
import cv2
from pathlib import Path

from backend.src.image_validator import validate_image
from backend.src.preprocessor import preprocess_image
from backend.src.segmentation import segment_defects
from backend.src.feature_extractor import extract_features
from backend.src.scoring import calculate_pixel_quality_score
from backend.src.inspection_result import InspectionResult
from backend.src.visualizer import save_debug_output

logger = logging.getLogger("eqm.engine")


def run_pixel_inspection(
    image_path: str,
    visualize: bool = False,
    visualization_output_dir: Optional[str] = None,
) -> InspectionResult:
    """
    Day 4: Full pixel-level inspection pipeline.

    Pipeline:
        1. Image validation
        2. Load image
        3. Preprocess (resize, grayscale, denoise, CLAHE, normalize)
        4. Pixel-level segmentation (adaptive threshold + morphology + contours)
        5. Feature extraction (defect pixels, edge density, regions, intensity)
        6. Multi-component quality scoring
        7. PASS/WARNING/FAIL classification
        8. Build structured InspectionResult
        9. Optional visualization

    Args:
        image_path: Path to the image file.
        visualize: Whether to generate debug visualization.
        visualization_output_dir: Override output directory for visualizations.

    Returns:
        InspectionResult with full inspection data.
    """
    start_time = time.time()
    image_name = Path(image_path).name
    warnings_list: list = []
    errors_list: list = []

    logger.info("Inspection started: %s", image_name)

    # --- Step 1: Image validation ---
    logger.debug("Step 1: Validating image")
    validation = validate_image(image_path)

    if not validation["valid"]:
        elapsed = (time.time() - start_time) * 1000.0
        logger.warning("Validation failed for %s: %s", image_name, validation["message"])
        return InspectionResult(
            image_id=image_name,
            status="ERROR",
            errors=[f"Validation failed: {validation['message']}"],
            processing_time_ms=round(elapsed, 2),
        )

    image_quality_metrics = validation.get("metadata", {}).get("quality_metrics", {})
    logger.info("Validation passed: blur=%.1f, brightness=%.1f, contrast=%.1f",
                image_quality_metrics.get("blur_variance", 0),
                image_quality_metrics.get("brightness_mean", 0),
                image_quality_metrics.get("contrast_std", 0))

    # --- Step 2: Load image ---
    logger.debug("Step 2: Loading image")
    image = cv2.imread(image_path)
    if image is None:
        elapsed = (time.time() - start_time) * 1000.0
        return InspectionResult(
            image_id=image_name,
            status="ERROR",
            errors=["Failed to load image after validation"],
            processing_time_ms=round(elapsed, 2),
        )

    # --- Step 3: Preprocess ---
    logger.debug("Step 3: Preprocessing")
    try:
        preprocess_result = preprocess_image(image)
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000.0
        logger.error("Preprocessing failed: %s", str(e))
        return InspectionResult(
            image_id=image_name,
            status="ERROR",
            errors=[f"Preprocessing failed: {str(e)}"],
            processing_time_ms=round(elapsed, 2),
        )

    # --- Step 4: Pixel-level segmentation ---
    logger.debug("Step 4: Pixel-level segmentation")
    try:
        seg_result = segment_defects(preprocess_result.grayscale)
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000.0
        logger.error("Segmentation failed: %s", str(e))
        return InspectionResult(
            image_id=image_name,
            status="ERROR",
            errors=[f"Segmentation failed: {str(e)}"],
            processing_time_ms=round(elapsed, 2),
        )

    # --- Step 5: Feature extraction ---
    logger.debug("Step 5: Feature extraction")
    try:
        features = extract_features(preprocess_result.grayscale, seg_result)
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000.0
        logger.error("Feature extraction failed: %s", str(e))
        return InspectionResult(
            image_id=image_name,
            status="ERROR",
            errors=[f"Feature extraction failed: {str(e)}"],
            processing_time_ms=round(elapsed, 2),
        )

    # --- Step 6: Quality scoring ---
    logger.debug("Step 6: Quality scoring")
    score_result = calculate_pixel_quality_score(features, image_quality_metrics)
    quality_score = score_result["quality_score"]

    # --- Step 7: Classification ---
    logger.debug("Step 7: Classification")
    status = determine_status(quality_score)

    # --- Step 8: Build result ---
    elapsed = (time.time() - start_time) * 1000.0

    result = InspectionResult(
        image_id=image_name,
        status=status,
        quality_score=quality_score,
        affected_pixel_percentage=round(seg_result.affected_percentage, 4),
        defect_area_total=sum(r.area for r in seg_result.regions),
        region_count=len(seg_result.regions),
        features=features,
        analysis_reliability=score_result["analysis_reliability"],
        processing_time_ms=round(elapsed, 2),
        warnings=warnings_list,
        errors=errors_list,
        score_breakdown=score_result["score_breakdown"],
    )

    logger.info("Inspection complete: %s → %s (score=%.1f, time=%.1fms)",
                image_name, status, quality_score, elapsed)

    # --- Step 9: Optional visualization ---
    if visualize or settings.VISUALIZATION_ENABLED:
        logger.debug("Step 9: Generating visualization")
        save_debug_output(
            image_name,
            preprocess_result.image,
            seg_result,
            result,
            output_dir=visualization_output_dir,
        )

    return result
