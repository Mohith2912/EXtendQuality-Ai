"""
Day 4: Pixel-level defect region analysis.

Uses adaptive thresholding + morphological operations + connected component
analysis to identify potential defect regions at the pixel level.

This is NOT semantic segmentation — it is a deterministic image analysis
pipeline that identifies anomalous regions based on intensity patterns.

Canny edge detection is NOT used here. It is used only in feature_extractor.py
as an edge-density metric.
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from backend.src.config import settings

logger = logging.getLogger("eqm.segmentation")


@dataclass
class RegionInfo:
    """Statistics for a single detected defect region."""
    area: int
    centroid: Tuple[int, int]
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    perimeter: float


@dataclass
class SegmentationResult:
    """Container for segmentation output."""
    defect_mask: np.ndarray                # Binary mask (0=normal, 255=defect)
    regions: List[RegionInfo] = field(default_factory=list)
    total_defect_pixels: int = 0
    total_pixels: int = 0
    affected_percentage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def segment_defects(
    grayscale: np.ndarray,
    adaptive_block_size: Optional[int] = None,
    adaptive_c: Optional[int] = None,
    morph_kernel_size: Optional[int] = None,
    min_region_area: Optional[int] = None,
) -> SegmentationResult:
    """
    Performs pixel-level defect region analysis on a preprocessed grayscale image.

    Pipeline:
        1. Adaptive thresholding — separates anomalous regions from local background
        2. Morphological opening — removes small noise specks
        3. Morphological closing — fills small gaps in defect regions
        4. Connected component analysis — identifies and measures distinct regions
        5. Region filtering — removes regions below minimum area threshold

    Args:
        grayscale: Preprocessed grayscale image (uint8, single channel).
        adaptive_block_size: Block size for adaptive threshold. Must be odd.
        adaptive_c: Constant subtracted from mean in adaptive threshold.
        morph_kernel_size: Kernel size for morphological operations.
        min_region_area: Minimum contour area to count as a defect region.

    Returns:
        SegmentationResult with defect mask and region statistics.

    Raises:
        ValueError: If input image is invalid.
    """
    if grayscale is None or grayscale.size == 0:
        raise ValueError("Input grayscale image is None or empty")

    if len(grayscale.shape) != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {grayscale.shape}")

    block_size = adaptive_block_size or settings.SEGMENT_ADAPTIVE_BLOCK_SIZE
    c_val = adaptive_c if adaptive_c is not None else settings.SEGMENT_ADAPTIVE_C
    morph_k = morph_kernel_size or settings.SEGMENT_MORPH_KERNEL_SIZE
    min_area = min_region_area if min_region_area is not None else settings.SEGMENT_MIN_REGION_AREA

    # Ensure block_size is odd and >= 3
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    h, w = grayscale.shape
    total_pixels = h * w

    logger.debug("Segmentation: image %dx%d, block=%d, c=%d, morph=%d, min_area=%d",
                 w, h, block_size, c_val, morph_k, min_area)

    # Step 1: Adaptive thresholding
    # ADAPTIVE_THRESH_GAUSSIAN_C provides smoother thresholding than MEAN_C
    # THRESH_BINARY_INV: bright defects on dark background become white in mask
    thresh = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_val,
    )

    # Step 1b: Apply circular bearing surface mask to ignore background and center hole
    bearing_mask = np.zeros_like(grayscale, dtype=np.uint8)
    center = (w // 2, h // 2)
    # Scale radii according to resized dimension (standard configuration is based on 640x640)
    scale = w / 640.0
    r_outer = int(252 * scale)  # Slightly padded to capture edges
    r_inner = int(148 * scale)  # Slightly padded
    cv2.circle(bearing_mask, center, r_outer, 255, thickness=-1)
    cv2.circle(bearing_mask, center, r_inner, 0, thickness=-1)
    
    thresh = cv2.bitwise_and(thresh, bearing_mask)

    # Step 2: Morphological opening — removes small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 3: Morphological closing — fills small gaps in defect regions
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 4: Connected component analysis via contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Filter and analyze regions
    regions: List[RegionInfo] = []
    defect_mask = np.zeros_like(grayscale, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Draw this region onto the defect mask
        cv2.drawContours(defect_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Compute region statistics
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        x, y, rw, rh = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        regions.append(RegionInfo(
            area=int(area),
            centroid=(cx, cy),
            bounding_rect=(x, y, rw, rh),
            perimeter=float(perimeter),
        ))

    total_defect_pixels = int(np.count_nonzero(defect_mask))
    affected_percentage = (total_defect_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    # Sort regions by area (largest first)
    regions.sort(key=lambda r: r.area, reverse=True)

    logger.info("Segmentation: %d regions found, %.2f%% affected (%d/%d pixels)",
                len(regions), affected_percentage, total_defect_pixels, total_pixels)

    return SegmentationResult(
        defect_mask=defect_mask,
        regions=regions,
        total_defect_pixels=total_defect_pixels,
        total_pixels=total_pixels,
        affected_percentage=affected_percentage,
        metadata={
            "adaptive_block_size": block_size,
            "adaptive_c": c_val,
            "morph_kernel_size": morph_k,
            "min_region_area": min_area,
            "contours_found": len(contours),
            "contours_filtered": len(contours) - len(regions),
        },
    )
