"""
Day 4: Optional debug visualization for inspection results.

Generates annotated images showing:
- Original image with defect mask overlay
- Quality score, status, and region count annotation
- Saved to configurable output directory

Visualization is OPTIONAL and never runs unless explicitly enabled.
It does not affect normal inspection execution.
"""
import cv2
import numpy as np
import os
import logging
from typing import Optional

from backend.src.config import settings
from backend.src.segmentation import SegmentationResult
from backend.src.inspection_result import InspectionResult

logger = logging.getLogger("eqm.visualizer")

# Status colors (BGR)
STATUS_COLORS = {
    "PASS": (0, 200, 0),       # Green
    "WARNING": (0, 200, 255),  # Orange
    "FAIL": (0, 0, 220),       # Red
    "ERROR": (128, 128, 128),  # Gray
}


def create_visualization(
    original_image: np.ndarray,
    segmentation: SegmentationResult,
    result: InspectionResult,
    output_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Creates a debug visualization image with defect overlay and annotations.

    Args:
        original_image: Original BGR image (resized/preprocessed).
        segmentation: SegmentationResult with defect mask.
        result: InspectionResult with scores and status.
        output_path: If provided, saves the visualization to this path.

    Returns:
        Annotated BGR image, or None if visualization fails.
    """
    try:
        h, w = original_image.shape[:2]

        # Create the overlay: original + red-tinted defect regions
        overlay = original_image.copy()
        defect_colored = np.zeros_like(overlay)
        defect_colored[:, :, 2] = segmentation.defect_mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, defect_colored, 0.3, 0)

        # Draw region contours in yellow
        contour_mask = segmentation.defect_mask.copy()
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        # Side-by-side: original | overlay
        canvas = np.zeros((h + 60, w * 2, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = original_image
        canvas[0:h, w:w*2] = overlay

        # Annotations at the bottom
        status_color = STATUS_COLORS.get(result.status, (255, 255, 255))
        y_text = h + 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        cv2.putText(canvas, f"Status: {result.status}", (10, y_text),
                    font, font_scale, status_color, thickness)
        cv2.putText(canvas, f"Score: {result.quality_score:.1f}", (180, y_text),
                    font, font_scale, (255, 255, 255), thickness)
        cv2.putText(canvas, f"Regions: {result.region_count}", (350, y_text),
                    font, font_scale, (255, 255, 255), thickness)
        cv2.putText(canvas, f"Affected: {result.affected_pixel_percentage:.2f}%", (500, y_text),
                    font, font_scale, (255, 255, 255), thickness)

        y_text2 = h + 45
        cv2.putText(canvas, f"Image: {result.image_id}", (10, y_text2),
                    font, font_scale, (200, 200, 200), thickness)
        cv2.putText(canvas, f"Time: {result.processing_time_ms:.1f}ms", (400, y_text2),
                    font, font_scale, (200, 200, 200), thickness)

        # Labels for the two panels
        cv2.putText(canvas, "Original", (10, 20), font, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, "Defect Overlay", (w + 10, 20), font, 0.6, (0, 255, 255), 1)

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            cv2.imwrite(output_path, canvas)
            logger.info("Visualization saved to: %s", output_path)

        return canvas

    except Exception as e:
        logger.error("Visualization failed: %s", str(e))
        return None


def save_debug_output(
    image_id: str,
    original_image: np.ndarray,
    segmentation: SegmentationResult,
    result: InspectionResult,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Convenience function: creates and saves visualization to the configured output directory.

    Args:
        image_id: Image identifier (used for filename).
        original_image: Original BGR image.
        segmentation: SegmentationResult.
        result: InspectionResult.
        output_dir: Override output directory. Uses config default if None.

    Returns:
        Path to the saved visualization, or None if disabled/failed.
    """
    output_dir = output_dir or settings.VISUALIZATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Clean filename
    safe_name = image_id.replace("/", "_").replace("\\", "_")
    if "." in safe_name:
        safe_name = safe_name.rsplit(".", 1)[0]
    output_path = os.path.join(output_dir, f"{safe_name}_debug.jpg")

    vis = create_visualization(original_image, segmentation, result, output_path)
    return output_path if vis is not None else None
