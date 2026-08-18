import argparse
import sys
import json
import logging
from pathlib import Path

# Add backend directory to path if not running as module
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.image_validator import validate_image
from backend.src.detector import Detector
from backend.src.quality_engine import evaluate_quality, run_pixel_inspection
from backend.src.report_generator import generate_report, report_to_json
from backend.src.batch_inspector import run_batch_inspection
from backend.src.config import settings

# Lazy imports for scripts (may have heavy dependencies)
def _import_dataset_validator():
    from backend.scripts.dataset_validator import validate_dataset
    return validate_dataset

def _import_train():
    from backend.scripts.train_model import train
    return train

VERSION = "0.4.0-day4"

def _setup_logging(level_str: str = None):
    """Configure structured logging for the inspection pipeline."""
    level_str = level_str or settings.LOG_LEVEL
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=settings.LOG_FORMAT,
        force=True,
    )

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


def run_pixel_inspection_cli(image_path: str, visualize: bool = False):
    """
    Day 4: Runs the pixel-level inspection pipeline on a single image.
    """
    print("========================================")
    print("    EXTENDQUALITY PIXEL INSPECTION")
    print("========================================")
    print(f"Target      : {image_path}\n")

    result = run_pixel_inspection(image_path, visualize=visualize)

    # Print human-readable summary
    print(f"Image       : {result.image_id}")
    print(f"Status      : {result.status}")
    print(f"Score       : {result.quality_score:.1f}")
    print(f"Reliability : {result.analysis_reliability:.2f}")
    print(f"Affected    : {result.affected_pixel_percentage:.2f}%")
    print(f"Regions     : {result.region_count}")
    print(f"Processing  : {result.processing_time_ms:.1f} ms")

    if result.score_breakdown:
        print("\nScore Breakdown:")
        for k, v in result.score_breakdown.items():
            print(f"  {k}: {v}")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    if result.warnings:
        print(f"\nWarnings: {result.warnings}")

    print("========================================")

    # Print JSON result
    print("\nJSON Result:")
    print(result.to_json())

    return result


def run_batch_cli(directory: str, visualize: bool = False):
    """
    Day 4: Runs batch pixel-level inspection on a directory.
    """
    print("========================================")
    print("    EXTENDQUALITY BATCH INSPECTION")
    print("========================================")
    print(f"Directory   : {directory}\n")

    summary = run_batch_inspection(directory, visualize=visualize)

    print(f"Total Images : {summary.total_images}")
    print(f"PASS         : {summary.pass_count}")
    print(f"WARNING      : {summary.warning_count}")
    print(f"FAIL         : {summary.fail_count}")
    print(f"ERRORS       : {summary.error_count}")
    print(f"Avg Score    : {summary.average_quality_score:.1f}")
    print(f"Avg Time     : {summary.average_processing_time_ms:.1f} ms")
    print(f"Total Time   : {summary.total_processing_time_ms:.1f} ms")

    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for err in summary.errors[:10]:
            print(f"  - {err['image']}: {err['message']}")

    print("========================================")

    # Save batch results as JSON
    batch_json = json.dumps(summary.to_dict(), indent=2, default=str)
    print("\nBatch Summary JSON:")
    print(batch_json)

    return summary


def main():
    parser = argparse.ArgumentParser(description="ExtendQuality ML CLI")
    parser.add_argument("--image", type=str, help="Run inspection pipeline on an image")
    parser.add_argument("--pixel-inspect", type=str, help="Run Day 4 pixel-level inspection on an image")
    parser.add_argument("--batch", type=str, help="Run batch pixel inspection on a directory")
    parser.add_argument("--visualize", action="store_true", help="Enable debug visualization output")
    parser.add_argument("--validate-dataset", action="store_true", help="Validate the training dataset and show stats")
    parser.add_argument("--train", action="store_true", help="Run the model training pipeline")
    parser.add_argument("--version", action="store_true", help="Show system version")
    parser.add_argument("--log-level", type=str, default=None, help="Set logging level (DEBUG, INFO, WARNING)")
    
    args = parser.parse_args()

    # Setup logging
    _setup_logging(args.log_level)
    
    if args.version:
        print(f"ExtendQuality AI - v{VERSION}")
        sys.exit(0)
        
    if args.validate_dataset:
        print("Validating dataset...")
        validate_dataset = _import_dataset_validator()
        stats = validate_dataset()
        print(json.dumps(stats, indent=2))
        sys.exit(0)
        
    if args.train:
        train = _import_train()
        train()
        sys.exit(0)

    if args.pixel_inspect:
        run_pixel_inspection_cli(args.pixel_inspect, visualize=args.visualize)
        sys.exit(0)

    if args.batch:
        run_batch_cli(args.batch, visualize=args.visualize)
        sys.exit(0)
        
    if args.image:
        run_inspection(args.image)
        sys.exit(0)
        
    parser.print_help()

if __name__ == "__main__":
    main()
