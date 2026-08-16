import os
import cv2
import numpy as np
from pathlib import Path
from backend.src.config import settings

class ImageValidationError(Exception):
    pass

def validate_image(image_path: str) -> dict:
    """
    Validates an image before processing.
    Returns a structured dictionary indicating success or failure.
    Includes Day 3 Image Quality Conditions.
    """
    if not os.path.exists(image_path):
        return {"valid": False, "error": "FILE_NOT_FOUND", "message": f"Image file not found: {image_path}"}
    
    ext = Path(image_path).suffix.lower()
    if ext not in settings.SUPPORTED_IMAGE_FORMATS:
        return {"valid": False, "error": "UNSUPPORTED_FORMAT", "message": f"Unsupported image format: {ext}"}
        
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return {"valid": False, "error": "CORRUPTED_IMAGE", "message": "Failed to decode image"}
            
        height, width = img.shape[:2]
        if height == 0 or width == 0:
            return {"valid": False, "error": "INVALID_DIMENSIONS", "message": "Image dimensions are invalid"}
            
        # --- Day 3 Image Quality Conditions ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur detection (Variance of Laplacian)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < 100.0:
            return {"valid": False, "error": "IMAGE_BLURRED", "message": f"Image is too blurry (Laplacian var: {blur_val:.2f})"}
            
        # 2. Brightness (Mean intensity)
        brightness = np.mean(gray)
        if brightness < 50.0 or brightness > 200.0:
            return {"valid": False, "error": "INVALID_BRIGHTNESS", "message": f"Image brightness ({brightness:.2f}) is out of acceptable range [50, 200]"}
            
        # 3. Contrast (Intensity standard deviation)
        contrast = np.std(gray)
        if contrast < 20.0:
            return {"valid": False, "error": "LOW_CONTRAST", "message": f"Image contrast ({contrast:.2f}) is too low"}
            
        return {
            "valid": True,
            "error": None,
            "message": "Image is valid",
            "metadata": {
                "width": width,
                "height": height,
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
                "quality_metrics": {
                    "blur_variance": float(blur_val),
                    "brightness_mean": float(brightness),
                    "contrast_std": float(contrast)
                }
            }
        }
    except Exception as e:
        return {"valid": False, "error": "PROCESSING_ERROR", "message": str(e)}
