import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from backend.src.image_validator import validate_image

@pytest.fixture
def temp_images(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    
    # 1. Good image (High contrast, sharp, normal brightness)
    good_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    good_path = str(base_dir / "good.jpg")
    cv2.imwrite(good_path, good_img)
    
    # 2. Blurred image
    blur_img = cv2.GaussianBlur(good_img, (21, 21), 0)
    blur_path = str(base_dir / "blurred.jpg")
    cv2.imwrite(blur_path, blur_img)
    
    # 3. Dark image (Sharp but dark)
    dark_img = np.random.randint(0, 30, (100, 100, 3), dtype=np.uint8)
    dark_path = str(base_dir / "dark.jpg")
    cv2.imwrite(dark_path, dark_img)
    
    # 4. Low contrast image (Sharp but low contrast)
    low_cont_img = np.random.randint(120, 136, (100, 100, 3), dtype=np.uint8)
    low_cont_path = str(base_dir / "low_contrast.jpg")
    cv2.imwrite(low_cont_path, low_cont_img)
    
    # 5. Invalid image (corrupted text file masquerading as jpg)
    invalid_path = str(base_dir / "invalid.jpg")
    with open(invalid_path, "w") as f:
        f.write("Not an image")
        
    return {
        "good": good_path,
        "blurred": blur_path,
        "dark": dark_path,
        "low_contrast": low_cont_path,
        "invalid": invalid_path
    }

def test_validate_image_good(temp_images):
    res = validate_image(temp_images["good"])
    assert res["valid"] == True
    assert "quality_metrics" in res["metadata"]

def test_validate_image_blurred(temp_images):
    res = validate_image(temp_images["blurred"])
    assert res["valid"] == False
    assert res["error"] == "IMAGE_BLURRED"

def test_validate_image_dark(temp_images):
    res = validate_image(temp_images["dark"])
    assert res["valid"] == False
    assert res["error"] == "INVALID_BRIGHTNESS"

def test_validate_image_low_contrast(temp_images):
    res = validate_image(temp_images["low_contrast"])
    assert res["valid"] == False
    assert res["error"] == "LOW_CONTRAST"

def test_validate_image_invalid(temp_images):
    res = validate_image(temp_images["invalid"])
    assert res["valid"] == False
    assert res["error"] == "CORRUPTED_IMAGE"
