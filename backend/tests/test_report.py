import pytest
from backend.src.report_generator import generate_report
from backend.src.detector import DetectionResult

def test_report_generation():
    res = DetectionResult(
        status="MODEL_NOT_AVAILABLE",
        model_available=False,
        model_version="none",
        inference_time_ms=0.0,
        detections=[]
    )
    quality_evaluation = {
        "status": "ERROR",
        "quality_score": 0.0,
        "confidence": 0.0,
        "defects": [],
        "message": "Model not available"
    }
    
    report = generate_report("bearing_0001.jpg", res, quality_evaluation)
    
    assert report["image"] == "bearing_0001.jpg"
    assert "inspection_id" in report
    assert report["status"] == "ERROR"
