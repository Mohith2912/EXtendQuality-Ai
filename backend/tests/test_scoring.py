import pytest
from backend.src.detector import Detection
from backend.src.scoring import calculate_base_score, aggregate_confidence, calculate_defect_severity

def test_calculate_defect_severity():
    d1 = Detection(class_id=1, class_name="rust", confidence=0.9, bbox=[0,0,10,10])
    assert calculate_defect_severity(d1) == "high"
    
    d2 = Detection(class_id=2, class_name="scratch", confidence=0.4, bbox=[0,0,10,10])
    assert calculate_defect_severity(d2) == "low"
    
def test_calculate_base_score():
    detections = [
        Detection(class_id=1, class_name="rust", confidence=0.9, bbox=[0,0,10,10]),
        Detection(class_id=2, class_name="scratch", confidence=0.4, bbox=[0,0,10,10])
    ]
    # Rust (-35), Scratch (-5) -> 100 - 40 = 60
    assert calculate_base_score(detections) == 60.0

def test_aggregate_confidence():
    detections = [
        Detection(class_id=1, class_name="rust", confidence=0.9, bbox=[0,0,10,10]),
        Detection(class_id=2, class_name="scratch", confidence=0.4, bbox=[0,0,10,10])
    ]
    assert aggregate_confidence(detections) == 0.65
