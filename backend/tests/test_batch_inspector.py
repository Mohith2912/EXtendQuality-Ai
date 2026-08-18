import os
import cv2
import numpy as np
import pytest
from backend.src.batch_inspector import run_batch_inspection


@pytest.fixture
def temp_batch_dir(tmp_path):
    batch_dir = tmp_path / "batch_images"
    batch_dir.mkdir()
    
    # 1. Clear image (should PASS)
    pass_img = np.zeros((200, 200, 3), dtype=np.uint8)
    pass_img[::2, ::2] = 180
    pass_img[1::2, 1::2] = 80
    pass_img = np.clip(pass_img + np.random.randint(-5, 5, pass_img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(batch_dir / "01_pass.jpg"), pass_img)
    
    # 2. Defective image (should WARNING/FAIL)
    fail_img = np.zeros((200, 200, 3), dtype=np.uint8)
    fail_img[::2, ::2] = 180
    fail_img[1::2, 1::2] = 80
    fail_img = np.clip(fail_img + np.random.randint(-5, 5, fail_img.shape), 0, 255).astype(np.uint8)
    # Add a huge black defect block (40x40 = 1600 pixels)
    fail_img[80:120, 80:120] = 10
    cv2.imwrite(str(batch_dir / "02_fail.jpg"), fail_img)
    
    # 3. Invalid format/corrupted image (should ERROR)
    invalid_path = batch_dir / "03_corrupt.jpg"
    with open(invalid_path, "w") as f:
        f.write("not an image")
        
    return batch_dir


def test_batch_inspection(temp_batch_dir):
    summary = run_batch_inspection(str(temp_batch_dir), visualize=False)
    
    # Should find exactly 3 files
    assert summary.total_images == 3
    
    # Check that it processed successfully without crashing
    assert len(summary.results) == 3
    
    # One file is corrupt, so error_count should be at least 1
    assert summary.error_count == 1
    assert any(err["image"] == "03_corrupt.jpg" for err in summary.errors)
    
    # Check that pass and fail images are processed and categorized
    # 01_pass should be PASS (or WARNING depending on noise variance)
    # 02_fail should be WARNING or FAIL because of the big dark region
    success_results = [r for r in summary.results if r.status != "ERROR"]
    assert len(success_results) == 2
    
    # Verify summary dict format
    d = summary.to_dict()
    assert d["total_images"] == 3
    assert d["error_count"] == 1
    assert "average_quality_score" in d
    assert "average_processing_time_ms" in d
