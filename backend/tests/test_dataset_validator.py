import pytest
from backend.scripts.dataset_validator import validate_dataset
from backend.src.config import settings

def test_dataset_validator():
    # If the dataset directory exists from Day 2, we expect no errors
    # If it doesn't, we expect errors for missing splits
    stats = validate_dataset(settings.DATASET_DIR)
    
    # We should have the standard keys
    assert "total_images" in stats
    assert "train_images" in stats
    assert "val_images" in stats
    assert "total_labels" in stats
    assert "missing_labels" in stats
    assert "invalid_labels" in stats
    assert "class_distribution" in stats
    assert "resolutions" in stats
    assert "errors" in stats
