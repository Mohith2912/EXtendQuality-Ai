import json
from backend.src.inspection_result import InspectionResult


def test_inspection_result_json():
    res = InspectionResult(
        image_id="test_img.jpg",
        status="PASS",
        quality_score=95.5,
        affected_pixel_percentage=0.5,
        defect_area_total=20,
        region_count=2,
        features={"some_metric": 42.0},
        analysis_reliability=0.98,
        processing_time_ms=12.5,
        warnings=["minor reflection"],
        score_breakdown={"base_score": 100.0, "pixel_defect_penalty": 4.5}
    )
    
    # Test dictionary conversion
    d = res.to_dict()
    assert d["image_id"] == "test_img.jpg"
    assert d["status"] == "PASS"
    assert d["quality_score"] == 95.5
    
    # Test JSON string conversion
    js = res.to_json()
    parsed = json.loads(js)
    assert parsed["image_id"] == "test_img.jpg"
    assert parsed["status"] == "PASS"
    assert parsed["quality_score"] == 95.5
    assert parsed["features"]["some_metric"] == 42.0
    assert parsed["score_breakdown"]["pixel_defect_penalty"] == 4.5
