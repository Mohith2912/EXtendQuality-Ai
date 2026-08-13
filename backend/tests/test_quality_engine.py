import pytest
from backend.src.detector import DetectionResult, Detection
from backend.src.quality_engine import evaluate_quality

def test_quality_engine_pass():
    res = DetectionResult(
        status="INFERENCE_SUCCESS",
        model_available=True,
        model_version="v1",
        inference_time_ms=10.0,
        detections=[Detection(class_id=0, class_name="good", confidence=0.99, bbox=[0,0,10,10])]
    )
    eval_res = evaluate_quality(res)
    assert eval_res["status"] == "PASS"
    assert eval_res["quality_score"] == 100.0

def test_quality_engine_fail():
    res = DetectionResult(
        status="INFERENCE_SUCCESS",
        model_available=True,
        model_version="v1",
        inference_time_ms=10.0,
        detections=[Detection(class_id=1, class_name="rust", confidence=0.9, bbox=[0,0,10,10])]
    )
    eval_res = evaluate_quality(res)
    assert eval_res["status"] == "FAIL"
    assert eval_res["quality_score"] == 65.0

def test_quality_engine_model_not_available():
    res = DetectionResult(
        status="MODEL_NOT_AVAILABLE",
        model_available=False,
        model_version="none",
        inference_time_ms=0.0,
        detections=[]
    )
    eval_res = evaluate_quality(res)
    assert eval_res["status"] == "ERROR"
