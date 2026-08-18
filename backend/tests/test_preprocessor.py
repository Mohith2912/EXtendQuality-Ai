import numpy as np
import pytest
import cv2
from backend.src.preprocessor import preprocess_image


def test_preprocess_image_valid():
    # Create a dummy BGR image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Add a pattern
    img[40:60, 40:60] = 50
    
    res = preprocess_image(img, target_size=64, denoise_kernel=3, clahe_clip=2.0, clahe_grid=8)
    
    assert res.image.shape == (64, 64, 3)
    assert res.grayscale.shape == (64, 64)
    assert "resize" in res.metadata["steps_applied"]
    assert "grayscale" in res.metadata["steps_applied"]
    assert "denoise" in res.metadata["steps_applied"]
    assert "clahe" in res.metadata["steps_applied"]
    assert "normalize" in res.metadata["steps_applied"]


def test_preprocess_image_invalid():
    with pytest.raises(ValueError):
        preprocess_image(None)
    
    with pytest.raises(ValueError):
        preprocess_image(np.empty((0, 0, 3), dtype=np.uint8))
