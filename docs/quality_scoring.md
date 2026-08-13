# Quality Scoring Framework

The Quality Engine converts raw bounding box detections into actionable quality insights (PASS/WARNING/FAIL).

## Detection Severities
- **good**: none
- **scratch**: low (confidence < 0.5) to medium (confidence > 0.5)
- **dent**: medium (confidence < 0.6) to high (confidence > 0.6)
- **rust**: high (always)

## Base Score
The score starts at 100.
- High severity defect: -35
- Medium severity defect: -15
- Low severity defect: -5

## Status Thresholds
* **PASS**: 90 - 100
* **WARNING**: 70 - 89
* **FAIL**: 0 - 69

*Note: These thresholds are development-stage rules and are configurable in `backend/src/config.py`.*
