import os
from pathlib import Path
from pydantic_settings import BaseSettings

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
    
    # Detection configuration
    CONFIDENCE_THRESHOLD: float = 0.40
    IOU_THRESHOLD: float = 0.45
    
    # Quality scoring thresholds
    SCORE_PASS_THRESHOLD: float = 90.0
    SCORE_WARNING_THRESHOLD: float = 70.0
    
    class Config:
        env_prefix = "EQ_"

settings = Settings()
