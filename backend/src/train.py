import os
import argparse
from pathlib import Path
from backend.src.config import settings

def train_yolo(model_name="yolov8n.pt", epochs=5, imgsz=640, batch=16, device=""):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Please install it first.")
        return
        
    print(f"Starting YOLO Baseline Training")
    print(f"Model: {model_name}")
    print(f"Dataset: {settings.DATASET_YAML}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    
    # Load a model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=settings.DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name=f"day3_baseline_{epochs}epochs",
        seed=42,
        exist_ok=True
    )
    
    print("\nTraining completed successfully.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXtendQuality Day 3 Baseline Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model to use")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, 0, etc.)")
    
    args = parser.parse_args()
    train_yolo(model_name=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
