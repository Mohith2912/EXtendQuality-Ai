import cv2
import numpy as np
import os
import random
from pathlib import Path

# Defect Classes
CLASSES = ["good", "scratch", "dent", "rust"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

IMAGE_SIZE = 640
NUM_IMAGES = 100 # Adjust as needed for prototype
TRAIN_SPLIT = 0.8

def create_base_bearing():
    """Create a synthetic image of a good bearing."""
    # Dark industrial background
    bg = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 40
    
    # Outer ring
    center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
    radius_outer = 250
    radius_inner = 150
    cv2.circle(bg, center, radius_outer, (150, 150, 150), -1) # Light gray
    cv2.circle(bg, center, radius_inner, (40, 40, 40), -1)    # Background color inside
    
    # Add some basic lighting/shading (gradient)
    for r in range(radius_inner, radius_outer, 2):
        shade = int(150 - (r - radius_inner) * 0.5)
        cv2.circle(bg, center, r, (shade, shade, shade), 2)
        
    # Draw some "balls" or "rollers" (simplified)
    for angle in range(0, 360, 45):
        rad = np.deg2rad(angle)
        cx = int(center[0] + 200 * np.cos(rad))
        cy = int(center[1] + 200 * np.sin(rad))
        cv2.circle(bg, (cx, cy), 20, (100, 100, 100), -1)

    # Add general noise
    noise = np.random.randint(0, 15, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    bg = cv2.add(bg, noise)
    
    return bg

def add_scratch(img):
    """Add a synthetic scratch."""
    x1, y1 = random.randint(150, 490), random.randint(150, 490)
    x2 = x1 + random.randint(-50, 50)
    y2 = y1 + random.randint(-50, 50)
    
    # Needs to be on the bearing (approximate check)
    center = IMAGE_SIZE // 2
    dist1 = np.sqrt((x1 - center)**2 + (y1 - center)**2)
    
    # Only draw if on the ring roughly
    if 150 <= dist1 <= 250:
        cv2.line(img, (x1, y1), (x2, y2), (220, 220, 220), thickness=random.randint(1, 3))
        # Return bounding box in YOLO format (cx, cy, w, h normalized)
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        # add padding
        min_x, min_y = max(0, min_x - 5), max(0, min_y - 5)
        max_x, max_y = min(IMAGE_SIZE, max_x + 5), min(IMAGE_SIZE, max_y + 5)
        
        cx = ((min_x + max_x) / 2) / IMAGE_SIZE
        cy = ((min_y + max_y) / 2) / IMAGE_SIZE
        w = (max_x - min_x) / IMAGE_SIZE
        h = (max_y - min_y) / IMAGE_SIZE
        return img, [CLASS_TO_ID["scratch"], cx, cy, w, h]
    return img, None

def add_dent(img):
    """Add a synthetic dent."""
    x, y = random.randint(150, 490), random.randint(150, 490)
    center = IMAGE_SIZE // 2
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    if 150 <= dist <= 250:
        r = random.randint(5, 15)
        cv2.circle(img, (x, y), r, (50, 50, 50), -1)
        cv2.circle(img, (x, y), r - 2, (30, 30, 30), -1)
        
        min_x, min_y = max(0, x - r - 2), max(0, y - r - 2)
        max_x, max_y = min(IMAGE_SIZE, x + r + 2), min(IMAGE_SIZE, y + r + 2)
        
        cx = ((min_x + max_x) / 2) / IMAGE_SIZE
        cy = ((min_y + max_y) / 2) / IMAGE_SIZE
        w = (max_x - min_x) / IMAGE_SIZE
        h = (max_y - min_y) / IMAGE_SIZE
        return img, [CLASS_TO_ID["dent"], cx, cy, w, h]
    return img, None

def add_rust(img):
    """Add synthetic rust."""
    x, y = random.randint(150, 490), random.randint(150, 490)
    center = IMAGE_SIZE // 2
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    if 150 <= dist <= 250:
        r = random.randint(15, 40)
        overlay = img.copy()
        cv2.circle(overlay, (x, y), r, (30, 60, 150), -1) # BGR rust color
        # Blend it
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        min_x, min_y = max(0, x - r), max(0, y - r)
        max_x, max_y = min(IMAGE_SIZE, x + r), min(IMAGE_SIZE, y + r)
        
        cx = ((min_x + max_x) / 2) / IMAGE_SIZE
        cy = ((min_y + max_y) / 2) / IMAGE_SIZE
        w = (max_x - min_x) / IMAGE_SIZE
        h = (max_y - min_y) / IMAGE_SIZE
        return img, [CLASS_TO_ID["rust"], cx, cy, w, h]
    return img, None

def generate_dataset(base_dir):
    """Generate the synthetic dataset."""
    print("Generating synthetic dataset...")
    
    dirs = {
        "images/train": os.path.join(base_dir, "images", "train"),
        "images/val": os.path.join(base_dir, "images", "val"),
        "labels/train": os.path.join(base_dir, "labels", "train"),
        "labels/val": os.path.join(base_dir, "labels", "val"),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    for i in range(NUM_IMAGES):
        split = "train" if random.random() < TRAIN_SPLIT else "val"
        img = create_base_bearing()
        labels = []
        
        # Decide if good or defective
        is_defective = random.random() > 0.3 # 70% defective
        
        if is_defective:
            defect_type = random.choice(["scratch", "dent", "rust"])
            success = False
            attempts = 0
            while not success and attempts < 10:
                if defect_type == "scratch":
                    img, label = add_scratch(img)
                elif defect_type == "dent":
                    img, label = add_dent(img)
                elif defect_type == "rust":
                    img, label = add_rust(img)
                    
                if label:
                    labels.append(label)
                    success = True
                attempts += 1
        
        # Save image and label
        img_name = f"bearing_{i:04d}.jpg"
        img_path = os.path.join(dirs[f"images/{split}"], img_name)
        cv2.imwrite(img_path, img)
        
        label_name = f"bearing_{i:04d}.txt"
        label_path = os.path.join(dirs[f"labels/{split}"], label_name)
        
        with open(label_path, "w") as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                
    # Create data.yaml
    yaml_content = f"""
train: ./images/train
val: ./images/val

nc: 4
names: ['good', 'scratch', 'dent', 'rust']
"""
    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())
        
    print(f"Dataset generated at {base_dir}")

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    generate_dataset(dataset_dir)
