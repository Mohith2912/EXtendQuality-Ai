import os
import cv2
import numpy as np
import pytest
from backend.src.image_validator import validate_image

def test_missing_image():
    res = validate_image("non_existent_image.jpg")
    assert res["valid"] is False
    assert res["error"] == "FILE_NOT_FOUND"

def test_unsupported_format():
    # create a dummy file
    with open("dummy.txt", "w") as f:
        f.write("hello")
    res = validate_image("dummy.txt")
    assert res["valid"] is False
    assert res["error"] == "UNSUPPORTED_FORMAT"
    os.remove("dummy.txt")

def test_corrupted_image():
    # create a dummy file with supported extension
    with open("dummy.jpg", "w") as f:
        f.write("not an image")
    res = validate_image("dummy.jpg")
    assert res["valid"] is False
    assert res["error"] == "CORRUPTED_IMAGE"
    os.remove("dummy.jpg")


def test_validation_dimension_limits():
    # Create an image that is too small (e.g. 10x10)
    small_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    cv2.imwrite("small.jpg", small_img)
    try:
        res = validate_image("small.jpg")
        assert res["valid"] is False
        assert res["error"] == "INVALID_DIMENSIONS"
    finally:
        if os.path.exists("small.jpg"):
            os.remove("small.jpg")


def test_validation_grayscale_image():
    from unittest.mock import patch
    # Mock cv2.imread to return a 2D grayscale image (high contrast, sharp, normal brightness)
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    gray_img[::2, ::2] = 180
    gray_img[1::2, 1::2] = 80
    
    with patch("os.path.exists", return_value=True), patch("cv2.imread", return_value=gray_img):
        res = validate_image("dummy.jpg")
        assert res["valid"] is True
        assert res["metadata"]["channels"] == 1


def test_validation_unsupported_channels():
    from unittest.mock import patch
    # Mock cv2.imread to return a 5-channel image
    img_5ch = np.ones((100, 100, 5), dtype=np.uint8) * 128
    
    with patch("os.path.exists", return_value=True), patch("cv2.imread", return_value=img_5ch):
        res = validate_image("dummy.jpg")
        assert res["valid"] is False
        assert res["error"] == "UNSUPPORTED_CHANNELS"


