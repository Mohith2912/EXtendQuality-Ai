import os
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Base paths
    BASE_DIR: str = str(Path(__file__).parent.parent.resolve())
    DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    # Model configuration
    MODEL_PATH: str = os.path.join(MODELS_DIR, "best.pt")
    DATASET_YAML: str = os.path.join(DATASET_DIR, "data.yaml")
    
    # Image configuration
    SUPPORTED_IMAGE_FORMATS: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    IMAGE_SIZE: int = 640
    MIN_IMAGE_DIMENSION: int = 32
    MAX_IMAGE_DIMENSION: int = 8192
    
    # Detection configuration
    CONFIDENCE_THRESHOLD: float = 0.40
    IOU_THRESHOLD: float = 0.45
    
    # Quality scoring thresholds
    SCORE_PASS_THRESHOLD: float = 90.0
    SCORE_WARNING_THRESHOLD: float = 70.0
    
    # --- Day 4: Preprocessing configuration ---
    PREPROCESS_RESIZE: int = 640
    PREPROCESS_DENOISE_KERNEL: int = 5
    PREPROCESS_CLAHE_CLIP_LIMIT: float = 2.0
    PREPROCESS_CLAHE_GRID_SIZE: int = 8
    
    # --- Day 4: Segmentation configuration ---
    SEGMENT_ADAPTIVE_BLOCK_SIZE: int = 35
    SEGMENT_ADAPTIVE_C: int = 10
    SEGMENT_MORPH_KERNEL_SIZE: int = 3
    SEGMENT_MIN_REGION_AREA: int = 50
    
    # --- Day 4: Feature extraction ---
    FEATURE_CANNY_LOW: int = 50
    FEATURE_CANNY_HIGH: int = 150
    
    # --- Day 4: Pixel-level quality scoring weights ---
    # Each weight controls how much that component affects the final score.
    # The base quality starts at 100, then penalties are subtracted.
    QUALITY_WEIGHT_PIXEL_DEFECT: float = 40.0
    QUALITY_WEIGHT_EDGE_DENSITY: float = 15.0
    QUALITY_WEIGHT_IMAGE_QUALITY: float = 10.0
    QUALITY_WEIGHT_REGION_COUNT: float = 10.0
    
    # Thresholds for pixel-level penalty scaling
    PIXEL_DEFECT_CRITICAL_PERCENT: float = 15.0
    EDGE_DENSITY_CRITICAL: float = 0.25
    REGION_COUNT_CRITICAL: int = 20
    
    # --- Day 4: Visualization ---
    VISUALIZATION_ENABLED: bool = False
    VISUALIZATION_OUTPUT_DIR: str = os.path.join(BASE_DIR, "debug_output")
    
    # --- Day 4: Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    model_config = ConfigDict(env_prefix="EQ_")

settings = Settings()
