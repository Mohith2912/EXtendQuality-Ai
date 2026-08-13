# Model Pipeline Architecture

## The Data Flow

```mermaid
graph TD
    A[Input Image] --> B(Image Validator)
    B -->|Fail| H(Error Report)
    B -->|Pass| C(Detector Abstraction)
    
    C -->|MODEL_NOT_AVAILABLE| I(Unavailable State)
    C -->|INFERENCE_SUCCESS| D(Standard DetectionResult)
    
    D --> E(Quality Scoring Engine)
    E --> F(PASS / WARNING / FAIL)
    F --> G(Structured JSON Report)
```

## Independence from YOLO
The `Detector` class is the only component that directly interacts with YOLO (or any future model implementation). It outputs a `DetectionResult` containing a list of `Detection` objects. The `Quality Engine` consumes this standard contract, meaning the model can be swapped out without affecting the quality scoring rules.
