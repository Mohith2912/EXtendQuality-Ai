"""
Day 4: Deterministic image preprocessing pipeline.

Each step has a clear purpose:
- Resize: normalize input dimensions for consistent analysis
- Grayscale: simplify to single channel for threshold-based segmentation
- Denoise: reduce sensor noise that creates false defect detections
- CLAHE: enhance local contrast to make subtle defects more visible
- Normalize: scale pixel values to consistent range
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional

from backend.src.config import settings

logger = logging.getLogger("eqm.preprocessor")


class PreprocessingResult:
    """Container for preprocessing output and metadata."""

    def __init__(self, image: np.ndarray, grayscale: np.ndarray, metadata: Dict[str, Any]):
        self.image = image          # Preprocessed BGR image
        self.grayscale = grayscale  # Preprocessed grayscale image
        self.metadata = metadata    # Details of operations applied


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[int] = None,
    denoise_kernel: Optional[int] = None,
    clahe_clip: Optional[float] = None,
    clahe_grid: Optional[int] = None,
) -> PreprocessingResult:
    """
    Runs the deterministic preprocessing pipeline on a loaded image.

    Args:
        image: Input BGR image (numpy array from cv2.imread).
        target_size: Target dimension for resize. Uses config default if None.
        denoise_kernel: Gaussian blur kernel size. Uses config default if None.
        clahe_clip: CLAHE clip limit. Uses config default if None.
        clahe_grid: CLAHE grid size. Uses config default if None.

    Returns:
        PreprocessingResult with preprocessed images and metadata.

    Raises:
        ValueError: If input image is None or empty.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    target_size = target_size or settings.PREPROCESS_RESIZE
    denoise_kernel = denoise_kernel or settings.PREPROCESS_DENOISE_KERNEL
    clahe_clip = clahe_clip or settings.PREPROCESS_CLAHE_CLIP_LIMIT
    clahe_grid = clahe_grid or settings.PREPROCESS_CLAHE_GRID_SIZE

    metadata: Dict[str, Any] = {
        "original_shape": image.shape,
        "steps_applied": [],
    }

    processed = image.copy()
    original_h, original_w = processed.shape[:2]

    # Step 1: Resize to target dimensions (preserves aspect ratio with padding)
    if original_h != target_size or original_w != target_size:
        processed = cv2.resize(processed, (target_size, target_size), interpolation=cv2.INTER_AREA)
        metadata["steps_applied"].append("resize")
        metadata["resized_to"] = (target_size, target_size)
        logger.debug("Resized from %dx%d to %dx%d", original_w, original_h, target_size, target_size)

    # Step 2: Convert to grayscale for analysis
    if len(processed.shape) == 3 and processed.shape[2] == 3:
        grayscale = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    elif len(processed.shape) == 2:
        grayscale = processed.copy()
    else:
        # Single-channel image in 3D array
        grayscale = processed[:, :, 0]
    metadata["steps_applied"].append("grayscale")

    # Step 3: Gaussian noise reduction
    # Kernel must be odd and positive
    k = denoise_kernel if denoise_kernel % 2 == 1 else denoise_kernel + 1
    grayscale = cv2.GaussianBlur(grayscale, (k, k), 0)
    metadata["steps_applied"].append("denoise")
    metadata["denoise_kernel"] = k

    # Step 4: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    grayscale = clahe.apply(grayscale)
    metadata["steps_applied"].append("clahe")
    metadata["clahe_clip_limit"] = clahe_clip
    metadata["clahe_grid_size"] = clahe_grid

    # Step 5: Normalize to 0-255 (ensures full dynamic range utilization)
    grayscale = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX)
    metadata["steps_applied"].append("normalize")

    metadata["final_shape"] = grayscale.shape
    logger.info("Preprocessing complete: %d steps applied", len(metadata["steps_applied"]))

    return PreprocessingResult(
        image=processed,
        grayscale=grayscale,
        metadata=metadata,
    )
