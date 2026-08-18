import numpy as np
from backend.src.segmentation import segment_defects
from backend.src.feature_extractor import extract_features


def test_extract_features():
    # Setup image with a clear defect region
    img = np.ones((100, 100), dtype=np.uint8) * 180
    img[45:55, 15:25] = 50 # Defect region (100 pixels) at distance ~30 from center
    
    seg_res = segment_defects(img, adaptive_block_size=35, adaptive_c=10, morph_kernel_size=3, min_region_area=20)
    
    features = extract_features(img, seg_res, canny_low=50, canny_high=150)
    
    expected_keys = [
        "total_defect_pixels",
        "affected_pixel_percentage",
        "edge_density",
        "region_count",
        "largest_region_area",
        "defect_area_total",
        "mean_defect_intensity",
        "std_defect_intensity",
        "mean_normal_intensity",
        "contrast_ratio",
        "edge_pixel_count"
    ]
    
    for key in expected_keys:
        assert key in features
        
    assert features["region_count"] >= 1
    assert features["total_defect_pixels"] > 0
    assert features["mean_defect_intensity"] < features["mean_normal_intensity"]
    assert features["contrast_ratio"] > 0.0
