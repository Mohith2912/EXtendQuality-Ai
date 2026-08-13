import pytest
from backend.src.detector import Detector, DetectionResult

def test_model_not_available():
    detector = Detector(model_path="non_existent_model.pt")
    res = detector.detect("some_image.jpg")
    
    assert isinstance(res, DetectionResult)
    assert res.status == "MODEL_NOT_AVAILABLE"
    assert res.model_available is False
    assert len(res.detections) == 0
