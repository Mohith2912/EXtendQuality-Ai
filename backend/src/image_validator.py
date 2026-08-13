import os
import cv2
from pathlib import Path
from backend.src.config import settings

class ImageValidationError(Exception):
    pass

def validate_image(image_path: str) -> dict:
    """
    Validates an image before processing.
    Returns a structured dictionary indicating success or failure.
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
            
        return {
            "valid": True,
            "error": None,
            "message": "Image is valid",
            "metadata": {
                "width": width,
                "height": height,
                "channels": img.shape[2] if len(img.shape) > 2 else 1
            }
        }
    except Exception as e:
        return {"valid": False, "error": "PROCESSING_ERROR", "message": str(e)}
