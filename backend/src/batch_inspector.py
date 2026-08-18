"""
Day 4: Batch inspection processing.

Processes a directory of images through the full pixel-level inspection pipeline.
Catches per-image errors so one bad image doesn't terminate the batch.
Generates summary statistics.
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.src.config import settings
from backend.src.inspection_result import InspectionResult

logger = logging.getLogger("eqm.batch")


class BatchSummary:
    """Summary statistics for a batch inspection run."""

    def __init__(self):
        self.total_images: int = 0
        self.pass_count: int = 0
        self.warning_count: int = 0
        self.fail_count: int = 0
        self.error_count: int = 0
        self.total_processing_time_ms: float = 0.0
        self.results: List[InspectionResult] = []
        self.errors: List[Dict[str, str]] = []

    @property
    def average_quality_score(self) -> float:
        """Average quality score across non-error results."""
        scored = [r.quality_score for r in self.results if r.status != "ERROR"]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def average_processing_time_ms(self) -> float:
        """Average processing time per image."""
        return self.total_processing_time_ms / self.total_images if self.total_images > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary to dictionary."""
        return {
            "total_images": self.total_images,
            "pass_count": self.pass_count,
            "warning_count": self.warning_count,
            "fail_count": self.fail_count,
            "error_count": self.error_count,
            "average_quality_score": round(self.average_quality_score, 2),
            "average_processing_time_ms": round(self.average_processing_time_ms, 2),
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
        }


def run_batch_inspection(
    directory: str,
    visualize: bool = False,
    output_dir: Optional[str] = None,
) -> BatchSummary:
    """
    Runs pixel-level inspection on all supported images in a directory.

    Args:
        directory: Path to directory containing images.
        visualize: Whether to save debug visualizations.
        output_dir: Override visualization output directory.

    Returns:
        BatchSummary with per-image results and aggregate statistics.
    """
    # Import here to avoid circular imports
    from backend.src.quality_engine import run_pixel_inspection

    summary = BatchSummary()

    if not os.path.isdir(directory):
        logger.error("Batch directory does not exist: %s", directory)
        summary.errors.append({
            "image": directory,
            "error": "DIRECTORY_NOT_FOUND",
            "message": f"Directory not found: {directory}",
        })
        return summary

    # Collect supported images
    supported_exts = set(settings.SUPPORTED_IMAGE_FORMATS)
    image_paths = []
    for f in sorted(os.listdir(directory)):
        ext = Path(f).suffix.lower()
        if ext in supported_exts:
            image_paths.append(os.path.join(directory, f))

    logger.info("Batch inspection: found %d images in %s", len(image_paths), directory)

    for img_path in image_paths:
        summary.total_images += 1
        image_name = Path(img_path).name

        try:
            result = run_pixel_inspection(
                img_path,
                visualize=visualize,
                visualization_output_dir=output_dir,
            )
            summary.results.append(result)
            summary.total_processing_time_ms += result.processing_time_ms

            if result.status == "PASS":
                summary.pass_count += 1
            elif result.status == "WARNING":
                summary.warning_count += 1
            elif result.status == "FAIL":
                summary.fail_count += 1
            elif result.status == "ERROR":
                summary.error_count += 1
                if result.errors:
                    summary.errors.append({
                        "image": image_name,
                        "error": "INSPECTION_ERROR",
                        "message": "; ".join(result.errors),
                    })

        except Exception as e:
            summary.error_count += 1
            error_msg = f"Unexpected error processing {image_name}: {str(e)}"
            logger.error(error_msg)
            summary.errors.append({
                "image": image_name,
                "error": "UNEXPECTED_ERROR",
                "message": str(e),
            })
            # Create an error result so the batch summary is complete
            error_result = InspectionResult(
                image_id=image_name,
                status="ERROR",
                errors=[str(e)],
            )
            summary.results.append(error_result)

    logger.info(
        "Batch complete: %d total, %d PASS, %d WARNING, %d FAIL, %d errors. Avg score: %.1f",
        summary.total_images, summary.pass_count, summary.warning_count,
        summary.fail_count, summary.error_count, summary.average_quality_score,
    )

    return summary
