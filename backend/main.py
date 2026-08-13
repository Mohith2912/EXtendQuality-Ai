import argparse
import sys
import json
from pathlib import Path

# Add backend directory to path if not running as module
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.image_validator import validate_image
from backend.src.detector import Detector
from backend.src.quality_engine import evaluate_quality
from backend.src.report_generator import generate_report, report_to_json
from backend.scripts.dataset_validator import validate_dataset
from backend.scripts.train_model import train

VERSION = "0.3.0-day3"

def run_inspection(image_path: str):
    """
    Runs the full Day 3 inspection pipeline on a single image.
    """
    print("========================================")
    print("       EXTENDQUALITY INSPECTION")
    print("========================================")
    print(f"Target      : {image_path}\n")

    # 1. Image Validation
    img_val = validate_image(image_path)
    if not img_val["valid"]:
        print(f"ERROR: Image validation failed.")
        print(f"Reason: {img_val['message']}")
        return

    # 2. Detection
    detector = Detector()
    detection_result = detector.detect(image_path)
    
    # 3. Quality Evaluation
    quality_evaluation = evaluate_quality(detection_result)
    
    # 4. Report Generation
    image_name = Path(image_path).name
    report = generate_report(image_name, detection_result, quality_evaluation)
    
    # Print human-readable summary
    if report["status"] == "ERROR" and "MODEL_NOT_AVAILABLE" in report.get("error_message", "") or not detection_result.model_available:
        print("Status      : MODEL_NOT_AVAILABLE")
        print("Score       : N/A")
        print("Confidence  : N/A")
        print("Defects     : N/A")
        print("Processing  : 0 ms")
        print("\nNote: The trained model is not available. Fake predictions have been prevented.")
    else:
        print(f"Image       : {report['image']}")
        print(f"Status      : {report['status']}")
        print(f"Score       : {report['quality_score']}")
        print(f"Confidence  : {report['confidence'] * 100:.1f}%")
        print(f"Defects     : {len(report['defects'])}")
        print(f"Processing  : {report['processing_time_ms']:.1f} ms")
        
        if report["defects"]:
            print("\nDefect Details:")
            for d in report["defects"]:
                print(f"  - [{d['severity'].upper()}] {d['type']} (conf: {d['confidence']:.2f})")
                
    print("========================================")
    
    # (Optional) Return report for external use
    return report

def main():
    parser = argparse.ArgumentParser(description="ExtendQuality ML CLI")
    parser.add_argument("--image", type=str, help="Run inspection pipeline on an image")
    parser.add_argument("--validate-dataset", action="store_true", help="Validate the training dataset and show stats")
    parser.add_argument("--train", action="store_true", help="Run the model training pipeline")
    parser.add_argument("--version", action="store_true", help="Show system version")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"ExtendQuality AI - v{VERSION}")
        sys.exit(0)
        
    if args.validate_dataset:
        print("Validating dataset...")
        stats = validate_dataset()
        print(json.dumps(stats, indent=2))
        sys.exit(0)
        
    if args.train:
        train()
        sys.exit(0)
        
    if args.image:
        run_inspection(args.image)
        sys.exit(0)
        
    parser.print_help()

if __name__ == "__main__":
    main()
