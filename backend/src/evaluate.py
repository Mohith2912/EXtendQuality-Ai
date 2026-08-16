import os
from pathlib import Path
import yaml
import argparse
from backend.src.config import settings

def evaluate_model(model_path=r"runs\detect\runs\day3_baseline_5epochs\weights\best.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed.")
        return
        
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    model = YOLO(model_path)
    
    print("\nRunning Validation...")
    metrics = model.val(data=settings.DATASET_YAML, split='val')
    
    print("\n========================================")
    print("EXtendQuality — Validation Results")
    print("========================================")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    # Ultralytics metrics.box provides arrays for p, r, f1 per class
    # We can extract them if available, otherwise just use the overall.
    
    # Run failure analysis on validation set
    print("\nStarting Failure Analysis...")
    with open(settings.DATASET_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    val_dir = Path(settings.DATASET_DIR) / data_config.get('val', 'images/val')
    val_images = list(val_dir.rglob("*.jpg"))
    classes = data_config.get('names', [])
    
    print(f"Evaluating {len(val_images)} validation images for detailed analysis.")
    
    failures = []
    confidence_obs = {"high": 0, "medium": 0, "low": 0}
    
    for img_path in val_images:
        # Ground truth
        rel_path = img_path.relative_to(Path(settings.DATASET_DIR) / "images")
        label_path = Path(settings.DATASET_DIR) / "labels" / rel_path.with_suffix('.txt')
        
        gt_classes = set()
        if label_path.exists():
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    for line in content.split('\n'):
                        parts = line.split()
                        if parts:
                            gt_classes.add(int(parts[0]))
        
        # Predictions
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        pred_classes = set()
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                pred_classes.add(cls_id)
                
                if conf >= 0.7:
                    confidence_obs["high"] += 1
                elif conf >= 0.4:
                    confidence_obs["medium"] += 1
                else:
                    confidence_obs["low"] += 1
                    
        # Compare
        if not gt_classes and not pred_classes:
            continue # True negative (Good bearing)
            
        if not gt_classes and pred_classes:
            failures.append(f"False Positive on {img_path.name}: Predicted {[classes[c] for c in pred_classes]}, Ground Truth was 'good'")
        elif gt_classes and not pred_classes:
            failures.append(f"False Negative (Missed) on {img_path.name}: Ground Truth {[classes[c] for c in gt_classes]}, Predicted nothing")
        elif gt_classes != pred_classes:
            failures.append(f"Mismatch on {img_path.name}: Ground Truth {[classes[c] for c in gt_classes]}, Predicted {[classes[c] for c in pred_classes]}")
            
    print("\nFailure Analysis Results:")
    if not failures:
        print("No obvious class-level failures detected (bounding box IoU not checked here).")
    else:
        for f in failures[:20]: # show up to 20
            print(f" - {f}")
            
    print("\nConfidence Analysis:")
    print(f" - High confidence (>0.7):   {confidence_obs['high']} detections")
    print(f" - Medium confidence (0.4-0.7): {confidence_obs['medium']} detections")
    print(f" - Low confidence (<0.4):    {confidence_obs['low']} detections")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/day3_baseline_5epochs/weights/best.pt")
    args = parser.parse_args()
    evaluate_model(args.model)
