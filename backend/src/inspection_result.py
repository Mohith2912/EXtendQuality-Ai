"""
Day 4: Structured inspection result model.

Standardizes the output of every inspection into a serializable Pydantic model.
Supports JSON serialization via model_dump() / model_dump_json().
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import uuid


class InspectionResult(BaseModel):
    """Standardized result for a single image inspection."""

    # Identification
    inspection_id: str = Field(default_factory=lambda: f"EQM-{str(uuid.uuid4())[:8].upper()}")
    image_id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Classification
    status: str = "UNKNOWN"          # PASS, WARNING, FAIL, ERROR
    quality_score: float = 0.0       # 0-100, higher is better

    # Defect measurements
    affected_pixel_percentage: float = 0.0
    defect_area_total: int = 0
    region_count: int = 0

    # Extracted features
    features: Dict[str, Any] = Field(default_factory=dict)

    # Analysis reliability
    # This is a deterministic metric based on image quality conditions,
    # NOT a fake ML confidence score. It measures how reliable the
    # pixel-level analysis is likely to be for this particular image.
    analysis_reliability: float = 1.0

    # Performance
    processing_time_ms: float = 0.0

    # Diagnostics
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    # Score breakdown (explainability)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to formatted JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()
