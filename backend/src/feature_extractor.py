"""
Day 4: Feature extraction from segmentation results.

Extracts measurable quality features from the defect mask and original image.

Canny edge detection is used HERE as an edge-density metric (ratio of edge
pixels to total pixels). This measures surface complexity / texture roughness.
Canny is NOT used as a segmentation method.
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any

from backend.src.config import settings
from backend.src.segmentation import SegmentationResult

logger = logging.getLogger("eqm.feature_extractor")


def extract_features(
    grayscale: np.ndarray,
    segmentation: SegmentationResult,
    canny_low: int = None,
    canny_high: int = None,
) -> Dict[str, Any]:
    """
    Extracts quality-relevant features from segmentation output.

    Features extracted:
        - total_defect_pixels: count of pixels classified as defective
        - affected_pixel_percentage: percentage of image that is defective
        - edge_density: ratio of edge pixels to total pixels (via Canny)
        - region_count: number of distinct defect regions
        - largest_region_area: area of the largest defect region (pixels)
        - mean_defect_intensity: average pixel intensity within defect regions
        - std_defect_intensity: intensity variation within defect regions
        - mean_normal_intensity: average pixel intensity in non-defect regions
        - contrast_ratio: intensity contrast between defect and normal regions
        - defect_area_total: sum of all region areas

    Args:
        grayscale: Preprocessed grayscale image.
        segmentation: SegmentationResult from segment_defects().
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.

    Returns:
        Dictionary of extracted feature values.
    """
    canny_low = canny_low or settings.FEATURE_CANNY_LOW
    canny_high = canny_high or settings.FEATURE_CANNY_HIGH

    total_pixels = segmentation.total_pixels

    # --- Edge density via Canny ---
    edges = cv2.Canny(grayscale, canny_low, canny_high)
    edge_pixel_count = int(np.count_nonzero(edges))
    edge_density = edge_pixel_count / total_pixels if total_pixels > 0 else 0.0

    # --- Region statistics ---
    region_count = len(segmentation.regions)
    largest_region_area = segmentation.regions[0].area if region_count > 0 else 0
    defect_area_total = sum(r.area for r in segmentation.regions)

    # --- Intensity analysis ---
    defect_mask = segmentation.defect_mask
    has_defect_pixels = segmentation.total_defect_pixels > 0
    has_normal_pixels = total_pixels > segmentation.total_defect_pixels

    if has_defect_pixels:
        defect_pixels = grayscale[defect_mask > 0]
        mean_defect_intensity = float(np.mean(defect_pixels))
        std_defect_intensity = float(np.std(defect_pixels))
    else:
        mean_defect_intensity = 0.0
        std_defect_intensity = 0.0

    if has_normal_pixels:
        normal_pixels = grayscale[defect_mask == 0]
        mean_normal_intensity = float(np.mean(normal_pixels))
    else:
        mean_normal_intensity = 0.0

    # Contrast ratio: how different defect regions are from normal regions
    if has_defect_pixels and has_normal_pixels and mean_normal_intensity > 0:
        contrast_ratio = abs(mean_defect_intensity - mean_normal_intensity) / mean_normal_intensity
    else:
        contrast_ratio = 0.0

    features = {
        "total_defect_pixels": segmentation.total_defect_pixels,
        "affected_pixel_percentage": round(segmentation.affected_percentage, 4),
        "edge_density": round(edge_density, 6),
        "region_count": region_count,
        "largest_region_area": largest_region_area,
        "defect_area_total": defect_area_total,
        "mean_defect_intensity": round(mean_defect_intensity, 2),
        "std_defect_intensity": round(std_defect_intensity, 2),
        "mean_normal_intensity": round(mean_normal_intensity, 2),
        "contrast_ratio": round(contrast_ratio, 4),
        "edge_pixel_count": edge_pixel_count,
    }

    logger.info("Features extracted: %d defect pixels (%.2f%%), %d regions, edge_density=%.4f",
                features["total_defect_pixels"],
                features["affected_pixel_percentage"],
                features["region_count"],
                features["edge_density"])

    return features
