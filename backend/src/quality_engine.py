from typing import Dict, Any, List
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
