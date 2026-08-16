import os
import pytest
from pathlib import Path
from backend.src.dataset_validator import DatasetValidator

@pytest.fixture
def temp_dataset(tmp_path):
    # Setup mock dataset
    base_dir = tmp_path / "dataset"
    base_dir.mkdir()
    
    # data.yaml
    yaml_content = "train: images/train\nval: images/val\nnames: ['good', 'scratch', 'dent', 'rust']"
    (base_dir / "data.yaml").write_text(yaml_content)
    
    # images/labels dirs
    (base_dir / "images" / "train").mkdir(parents=True)
    (base_dir / "labels" / "train").mkdir(parents=True)
    
    return base_dir

def test_dataset_validator_empty_label(temp_dataset):
    validator = DatasetValidator(str(temp_dataset / "data.yaml"))
    
    # Create empty label file
    label_path = temp_dataset / "labels" / "train" / "test1.txt"
    label_path.write_text("")
    
    assert validator.validate_label(label_path) == True
    assert validator.stats["empty_label_images"] == 1

def test_dataset_validator_invalid_class(temp_dataset):
    validator = DatasetValidator(str(temp_dataset / "data.yaml"))
    label_path = temp_dataset / "labels" / "train" / "test2.txt"
    # Class 5 is invalid (only 0-3 allowed)
    label_path.write_text("5 0.5 0.5 0.1 0.1\n")
    
    assert validator.validate_label(label_path) == False
    assert validator.stats["invalid_labels"] == 1

def test_dataset_validator_invalid_coords(temp_dataset):
    validator = DatasetValidator(str(temp_dataset / "data.yaml"))
    label_path = temp_dataset / "labels" / "train" / "test3.txt"
    # Coordinates outside [0,1]
    label_path.write_text("1 1.5 0.5 0.1 0.1\n")
    
    assert validator.validate_label(label_path) == False
    assert validator.stats["invalid_labels"] == 1

def test_dataset_validator_negative_wh(temp_dataset):
    validator = DatasetValidator(str(temp_dataset / "data.yaml"))
    label_path = temp_dataset / "labels" / "train" / "test4.txt"
    # Negative width
    label_path.write_text("1 0.5 0.5 -0.1 0.1\n")
    
    assert validator.validate_label(label_path) == False
    assert validator.stats["invalid_labels"] == 1

def test_dataset_validator_malformed(temp_dataset):
    validator = DatasetValidator(str(temp_dataset / "data.yaml"))
    label_path = temp_dataset / "labels" / "train" / "test5.txt"
    label_path.write_text("1 0.5 abc 0.1 0.1\n")
    
    assert validator.validate_label(label_path) == False
    assert validator.stats["invalid_labels"] == 1

def test_dataset_validator_missing_label(temp_dataset):
    validator = DatasetValidator(str(temp_dataset / "data.yaml"))
    label_path = temp_dataset / "labels" / "train" / "missing.txt"
    
    assert validator.validate_label(label_path) == False
    assert validator.stats["missing_labels"] == 1
