import os
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
