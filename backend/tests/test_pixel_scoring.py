import pytest
from backend.src.scoring import calculate_pixel_quality_score
from backend.src.quality_engine import determine_status
from backend.src.config import settings


def test_pixel_scoring_logic():
    # Setup test features
    features = {
        "affected_pixel_percentage": 5.0,  # 5% out of critical 15% -> penalty is 5/15 * 40 = 13.33
        "edge_density": 0.05,             # 0.05 out of critical 0.25 -> penalty is 0.05/0.25 * 15 = 3.0
        "region_count": 5,                 # 5 out of critical 20 -> penalty is 5/20 * 10 = 2.5
    }
    
    # 100 - 13.33 - 3.0 - 2.5 = 81.17
    # No image quality metrics provided
    res = calculate_pixel_quality_score(features, image_quality_metrics=None)
    
    assert res["quality_score"] == pytest.approx(81.17, abs=0.1)
    assert res["score_breakdown"]["base_score"] == 100.0
    assert res["score_breakdown"]["pixel_defect_penalty"] == pytest.approx(13.33, abs=0.1)
    assert res["score_breakdown"]["edge_density_penalty"] == pytest.approx(3.0, abs=0.1)
    assert res["score_breakdown"]["region_count_penalty"] == pytest.approx(2.5, abs=0.1)
    assert res["score_breakdown"]["image_quality_penalty"] == 0.0


def test_pixel_scoring_with_borderline_image_quality():
    features = {
        "affected_pixel_percentage": 0.0,
        "edge_density": 0.0,
        "region_count": 0,
    }
    
    # Test borderline blur (Laplacian var = 120, close to 100)
    # Blur penalty: (max_penalty * 0.4) * (200.0 - blur_var) / 100.0 = 4.0 * (80.0 / 100.0) = 3.2
    image_quality_metrics = {
        "blur_variance": 120.0,
        "brightness_mean": 128.0,
        "contrast_std": 50.0
    }
    
    res = calculate_pixel_quality_score(features, image_quality_metrics=image_quality_metrics)
    assert res["quality_score"] == pytest.approx(100.0 - 3.2, abs=0.1)
    assert res["analysis_reliability"] < 1.0 # Should decrease reliability because blur_variance < 300


def test_scoring_boundaries_and_classification():
    # Verify status mapping based on thresholds
    pass_thresh = settings.SCORE_PASS_THRESHOLD
    warn_thresh = settings.SCORE_WARNING_THRESHOLD
    
    # Exactly PASS threshold (90.0)
    assert determine_status(pass_thresh) == "PASS"
    
    # Exactly WARNING threshold (70.0)
    assert determine_status(warn_thresh) == "WARNING"
    
    # Just above PASS threshold (90.1)
    assert determine_status(pass_thresh + 0.1) == "PASS"
    
    # Just below PASS threshold (89.9)
    assert determine_status(pass_thresh - 0.1) == "WARNING"
    
    # Just above WARNING threshold (70.1)
    assert determine_status(warn_thresh + 0.1) == "WARNING"
    
    # Just below WARNING threshold (69.9)
    assert determine_status(warn_thresh - 0.1) == "FAIL"
    
    # Minimum score (0.0)
    assert determine_status(0.0) == "FAIL"
    
    # Maximum score (100.0)
    assert determine_status(100.0) == "PASS"
