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
