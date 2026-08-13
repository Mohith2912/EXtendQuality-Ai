import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

def generate_report(image_name: str, detection_result: Any, quality_evaluation: dict) -> dict:
    """
    Generates a structured JSON inspection report.
    """
    inspection_id = f"EQM-{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Base report structure
    report = {
        "inspection_id": inspection_id,
        "image": image_name,
        "timestamp": timestamp,
        "model_version": getattr(detection_result, "model_version", "unknown"),
        "processing_time_ms": getattr(detection_result, "inference_time_ms", 0.0)
    }
    
    if quality_evaluation.get("status") == "ERROR":
        report.update({
            "status": quality_evaluation["status"],
            "quality_score": 0.0,
            "confidence": 0.0,
            "defects": [],
            "error_message": quality_evaluation.get("message", "Unknown error")
        })
    else:
        report.update({
            "status": quality_evaluation.get("status", "UNKNOWN"),
            "quality_score": round(quality_evaluation.get("quality_score", 0.0), 2),
            "confidence": round(quality_evaluation.get("confidence", 0.0), 3),
            "defects": quality_evaluation.get("defects", [])
        })
        
    return report

def report_to_json(report: dict) -> str:
    """
    Converts a report dictionary to a formatted JSON string.
    """
    return json.dumps(report, indent=2)
