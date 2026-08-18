import numpy as np
import pytest
from backend.src.segmentation import segment_defects


def test_segment_defects_clean():
    # Clean image: uniform grayscale
    img = np.ones((100, 100), dtype=np.uint8) * 128
    
    # Run segmentation
    res = segment_defects(img, adaptive_block_size=15, adaptive_c=5, morph_kernel_size=3, min_region_area=10)
    
    assert res.total_defect_pixels == 0
    assert len(res.regions) == 0
    assert res.affected_percentage == 0.0


def test_segment_defects_with_anomaly():
    # Create an image with a dark defect region on a light background
    # Adaptive threshold inv will flag dark regions on bright backgrounds (since it behaves as local threshold)
    img = np.ones((100, 100), dtype=np.uint8) * 200
    img[45:55, 15:25] = 50 # Defect region (100 pixels) at distance ~30 from center
    
    res = segment_defects(img, adaptive_block_size=35, adaptive_c=10, morph_kernel_size=3, min_region_area=20)
    
    assert res.total_defect_pixels > 0
    assert len(res.regions) >= 1
    assert res.regions[0].area >= 50 # Should be around 100 pixels
    assert res.affected_percentage > 0.0


def test_segment_defects_invalid():
    with pytest.raises(ValueError):
        segment_defects(None)
        
    with pytest.raises(ValueError):
        segment_defects(np.ones((100, 100, 3), dtype=np.uint8)) # Must be grayscale (2D)
