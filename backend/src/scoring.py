from typing import List, Dict, Any
from backend.src.detector import Detection
from backend.src.config import settings

def calculate_defect_severity(detection: Detection) -> str:
    """
    Calculates the severity of a single defect.
    This can be expanded in the future to consider bounding box area, etc.
    For Day 3, we use simple rules based on class and confidence.
    """
    # Assuming standard classes: good, scratch, dent, rust
    if detection.class_name == "good":
        return "none"
        
    if detection.class_name == "rust":
        return "high" # Rust is always a high severity issue per SOP
        
    if detection.class_name == "dent":
        return "high" if detection.confidence > 0.6 else "medium"
        
    if detection.class_name == "scratch":
        return "medium" if detection.confidence > 0.5 else "low"
        
    return "unknown"

def calculate_base_score(detections: List[Detection]) -> float:
    """
    Calculates a base quality score from 0 to 100 based on defects.
    100 means perfect condition.
    """
    score = 100.0
    
    # Simple scoring logic for Day 3
    for det in detections:
        if det.class_name == "good":
            continue
            
        severity = calculate_defect_severity(det)
        
        if severity == "high":
            score -= 35.0
        elif severity == "medium":
            score -= 15.0
        elif severity == "low":
            score -= 5.0
            
    # Ensure score doesn't drop below 0
    return max(0.0, score)

def aggregate_confidence(detections: List[Detection]) -> float:
    """
    Calculates the overall confidence of the detection result.
    """
    if not detections:
        return 1.0 # If no detections (assuming 'good' part), confidence is nominally 1.0
        
    # Average confidence of all defects
    total_conf = sum(d.confidence for d in detections if d.class_name != "good")
    defect_count = sum(1 for d in detections if d.class_name != "good")
    
    if defect_count == 0:
        return 1.0
        
    return total_conf / defect_count


def calculate_pixel_quality_score(
    features: dict,
    image_quality_metrics: dict = None,
) -> dict:
    """
    Day 4: Multi-component quality scoring from pixel-level features.

    Score = Base (100)
            - pixel_defect_penalty   (scaled by affected_pixel_percentage)
            - edge_density_penalty   (scaled by edge_density)
            - image_quality_penalty  (from validation quality metrics)
            - region_count_penalty   (scaled by number of defect regions)

    All penalties are capped to their maximum weight.
    Final score is clamped to [0, 100].

    Args:
        features: Feature dictionary from extract_features().
        image_quality_metrics: Optional quality metrics from image validation
                               (blur_variance, brightness_mean, contrast_std).

    Returns:
        Dictionary with:
            - quality_score: float (0-100)
            - score_breakdown: dict with each component's contribution
            - analysis_reliability: float (0-1) measuring how reliable the analysis is
    """
    base_score = 100.0

    # --- Pixel defect penalty ---
    affected_pct = features.get("affected_pixel_percentage", 0.0)
    critical_pct = settings.PIXEL_DEFECT_CRITICAL_PERCENT
    max_penalty = settings.QUALITY_WEIGHT_PIXEL_DEFECT

    # Linear scaling: 0% → 0 penalty, critical_pct% → full penalty
    if critical_pct > 0:
        pixel_penalty = min(max_penalty, (affected_pct / critical_pct) * max_penalty)
    else:
        pixel_penalty = max_penalty if affected_pct > 0 else 0.0

    # --- Edge density penalty ---
    edge_density = features.get("edge_density", 0.0)
    critical_edge = settings.EDGE_DENSITY_CRITICAL
    max_edge_penalty = settings.QUALITY_WEIGHT_EDGE_DENSITY

    if critical_edge > 0:
        edge_penalty = min(max_edge_penalty, (edge_density / critical_edge) * max_edge_penalty)
    else:
        edge_penalty = max_edge_penalty if edge_density > 0 else 0.0

    # --- Region count penalty ---
    region_count = features.get("region_count", 0)
    critical_regions = settings.REGION_COUNT_CRITICAL
    max_region_penalty = settings.QUALITY_WEIGHT_REGION_COUNT

    if critical_regions > 0:
        region_penalty = min(max_region_penalty, (region_count / critical_regions) * max_region_penalty)
    else:
        region_penalty = max_region_penalty if region_count > 0 else 0.0

    # --- Image quality penalty ---
    max_iq_penalty = settings.QUALITY_WEIGHT_IMAGE_QUALITY
    iq_penalty = 0.0

    if image_quality_metrics:
        blur_var = image_quality_metrics.get("blur_variance", 200.0)
        brightness = image_quality_metrics.get("brightness_mean", 128.0)
        contrast = image_quality_metrics.get("contrast_std", 50.0)

        # Penalize borderline quality (close to validation thresholds)
        # Blur: threshold is 100. Penalize if between 100 and 200 (just above threshold)
        if blur_var < 200.0:
            iq_penalty += (max_iq_penalty * 0.4) * max(0, (200.0 - blur_var) / 100.0)

        # Brightness: penalize if close to edges of [50, 200] range
        if brightness < 80.0:
            iq_penalty += (max_iq_penalty * 0.3) * max(0, (80.0 - brightness) / 30.0)
        elif brightness > 180.0:
            iq_penalty += (max_iq_penalty * 0.3) * max(0, (brightness - 180.0) / 20.0)

        # Contrast: penalize if low
        if contrast < 40.0:
            iq_penalty += (max_iq_penalty * 0.3) * max(0, (40.0 - contrast) / 20.0)

        iq_penalty = min(max_iq_penalty, iq_penalty)

    # --- Final score ---
    final_score = base_score - pixel_penalty - edge_penalty - region_penalty - iq_penalty
    final_score = max(0.0, min(100.0, final_score))

    # --- Analysis reliability ---
    # Measures how confident we are in the analysis result (NOT ML confidence)
    # Based on image quality conditions that affect analysis accuracy
    reliability = 1.0
    if image_quality_metrics:
        blur_var = image_quality_metrics.get("blur_variance", 200.0)
        if blur_var < 300.0:
            reliability *= min(1.0, blur_var / 300.0)
        contrast = image_quality_metrics.get("contrast_std", 50.0)
        if contrast < 60.0:
            reliability *= min(1.0, contrast / 60.0)

    return {
        "quality_score": round(final_score, 2),
        "score_breakdown": {
            "base_score": base_score,
            "pixel_defect_penalty": round(pixel_penalty, 2),
            "edge_density_penalty": round(edge_penalty, 2),
            "region_count_penalty": round(region_penalty, 2),
            "image_quality_penalty": round(iq_penalty, 2),
        },
        "analysis_reliability": round(reliability, 4),
    }
