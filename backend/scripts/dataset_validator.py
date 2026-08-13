import os
import glob
from pathlib import Path
from backend.src.config import settings
from backend.src.image_validator import validate_image

def validate_dataset(dataset_dir: str = None) -> dict:
    """
    Validates the YOLO dataset and returns statistics and errors.
    """
    if dataset_dir is None:
        dataset_dir = settings.DATASET_DIR
        
    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "total_labels": 0,
        "missing_labels": 0,
        "invalid_labels": 0,
        "class_distribution": {},
        "resolutions": set(),
        "errors": []
    }
    
    splits = ["train", "val"]
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            stats["errors"].append(f"Missing split directories for {split}")
            continue
            
        images = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in settings.SUPPORTED_IMAGE_FORMATS]
        
        for img_name in images:
            stats["total_images"] += 1
            if split == "train":
                stats["train_images"] += 1
            else:
                stats["val_images"] += 1
                
            img_path = os.path.join(img_dir, img_name)
            lbl_name = Path(img_name).stem + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            # Validate image
            img_val = validate_image(img_path)
            if not img_val["valid"]:
                stats["errors"].append(f"Invalid image {img_name}: {img_val['message']}")
            else:
                res = f"{img_val['metadata']['width']}x{img_val['metadata']['height']}"
                stats["resolutions"].add(res)
                
            # Validate label
            if not os.path.exists(lbl_path):
                stats["missing_labels"] += 1
                stats["errors"].append(f"Missing label for {img_name}")
            else:
                stats["total_labels"] += 1
                try:
                    with open(lbl_path, "r") as f:
                        lines = f.readlines()
                        if not lines:
                            # Empty label is valid in YOLO (background image)
                            stats["class_distribution"]["background"] = stats["class_distribution"].get("background", 0) + 1
                            continue
                            
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                stats["invalid_labels"] += 1
                                stats["errors"].append(f"Invalid label format in {lbl_name}: {line.strip()}")
                                continue
                                
                            class_id = int(parts[0])
                            # basic coord check
                            coords = [float(p) for p in parts[1:]]
                            if any(c < 0.0 or c > 1.0 for c in coords):
                                stats["invalid_labels"] += 1
                                stats["errors"].append(f"Invalid bounding box in {lbl_name}: {line.strip()}")
                                
                            stats["class_distribution"][class_id] = stats["class_distribution"].get(class_id, 0) + 1
                except Exception as e:
                    stats["invalid_labels"] += 1
                    stats["errors"].append(f"Failed to read label {lbl_name}: {str(e)}")
                    
    # convert sets to list for JSON serialization if needed
    stats["resolutions"] = list(stats["resolutions"])
    return stats

if __name__ == "__main__":
    import json
    print(json.dumps(validate_dataset(), indent=2))
