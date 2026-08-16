import os
from pathlib import Path
import cv2
import yaml

class DatasetValidator:
    def __init__(self, dataset_yaml_path: str):
        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.base_dir = self.dataset_yaml_path.parent
        self.stats = {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "total_annotations": 0,
            "empty_label_images": 0,
            "missing_labels": 0,
            "invalid_labels": 0,
            "corrupted_images": 0,
            "class_distribution": {},
            "image_dimensions": set(),
            "image_formats": set()
        }
        self.classes = []
        self._load_config()
        
    def _load_config(self):
        with open(self.dataset_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            self.classes = config.get('names', [])
            self.train_dir = self.base_dir / config.get('train', 'images/train')
            self.val_dir = self.base_dir / config.get('val', 'images/val')

    def validate_image(self, img_path: Path) -> bool:
        if not img_path.exists():
            return False
        if img_path.stat().st_size == 0:
            self.stats["corrupted_images"] += 1
            return False
            
        self.stats["image_formats"].add(img_path.suffix.lower())
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                self.stats["corrupted_images"] += 1
                return False
            
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                self.stats["corrupted_images"] += 1
                return False
                
            self.stats["image_dimensions"].add(f"{w}x{h}")
            return True
        except Exception:
            self.stats["corrupted_images"] += 1
            return False

    def validate_label(self, label_path: Path) -> bool:
        if not label_path.exists():
            self.stats["missing_labels"] += 1
            return False
            
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                
            if not content:
                self.stats["empty_label_images"] += 1
                return True # Empty labels are valid representations of "good"
                
            lines = content.split('\n')
            is_valid = True
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    is_valid = False
                    continue
                    
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= len(self.classes):
                    is_valid = False
                    
                x, y, w, h = map(float, parts[1:])
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    is_valid = False
                if w <= 0 or h <= 0:
                    is_valid = False
                if w * h == 0:
                    is_valid = False
                
                if is_valid:
                    self.stats["class_distribution"][self.classes[cls_id]] = self.stats["class_distribution"].get(self.classes[cls_id], 0) + 1
                    self.stats["total_annotations"] += 1
                    
            if not is_valid:
                self.stats["invalid_labels"] += 1
            return is_valid
        except Exception:
            self.stats["invalid_labels"] += 1
            return False

    def validate_split(self, split_dir: Path, split_name: str):
        if not split_dir.exists():
            return
            
        img_extensions = ['.jpg', '.jpeg', '.png']
        images = []
        for ext in img_extensions:
            images.extend(split_dir.rglob(f"*{ext}"))
            
        for img_path in images:
            self.stats["total_images"] += 1
            self.stats[f"{split_name}_images"] += 1
            
            if self.validate_image(img_path):
                # Construct label path
                rel_path = img_path.relative_to(self.base_dir / "images")
                label_path = self.base_dir / "labels" / rel_path.with_suffix('.txt')
                self.validate_label(label_path)

    def run(self):
        self.validate_split(self.train_dir, "train")
        self.validate_split(self.val_dir, "val")
        return self.stats

    def report(self):
        print("========================================")
        print("EXtendQuality — Dataset Validation Report")
        print("========================================")
        print(f"Total Images: {self.stats['total_images']}")
        print(f"  Train: {self.stats['train_images']}")
        print(f"  Val:   {self.stats['val_images']}")
        print("")
        print(f"Valid empty-label images: {self.stats['empty_label_images']}")
        print(f"Invalid annotation files: {self.stats['invalid_labels']}")
        print(f"Missing label files:      {self.stats['missing_labels']}")
        print(f"Corrupted images:         {self.stats['corrupted_images']}")
        print("")
        print("Class Distribution (annotations):")
        for cls_name, count in self.stats["class_distribution"].items():
            print(f"  {cls_name}: {count}")
        print("")
        print(f"Image Formats:    {', '.join(self.stats['image_formats'])}")
        print(f"Image Dimensions: {', '.join(self.stats['image_dimensions'])}")
        print("========================================")

if __name__ == "__main__":
    validator = DatasetValidator("backend/dataset/data.yaml")
    validator.run()
    validator.report()
