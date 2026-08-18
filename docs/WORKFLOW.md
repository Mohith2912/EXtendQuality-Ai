# EXtendQuality — Development Workflow

This document tracks the day-by-day development progress of the EXtendQuality project.

---

## Day 1 — Project Initialization (2026-08-10)

### Objective
Initialize the project repository, define the vision, and establish the foundational README.

### Work Completed
- Created the GitHub repository (`EXtendQuality-Ai`)
- Authored the initial `README.md` with:
  - Vision and problem statement
  - Core innovation (selective orchestration of OpenCV → YOLO → VLM)
  - System architecture diagram
  - Complete workflow description
  - Prototype scope (bearings only)
  - Frontend and backend vision
  - Technology stack selection
  - Project structure outline
  - Core module descriptions
  - Inspection record data model
  - 14-day prototype plan
  - Future roadmap
  - Metrics to prove value
  - Risks and mitigations
  - Development principles
  - Demo goal

### Key Decisions
- Bearing-focused prototype scope
- 4 defect classes: `good`, `scratch`, `dent`, `rust`
- Selective VLM escalation architecture (not every image goes to VLM)
- React + FastAPI + Ultralytics YOLO stack

### Commits
- `aba5a5d` — Day 1 of Development

---

## Day 2 — Architecture & Scope Definition (2026-08-11 – 2026-08-12)

### Objective
Finalize the project architecture, define inspection rules, establish the dataset configuration, and build the initial backend scaffolding.

### Work Completed
- Restructured the project file layout (`d48dd71`)
- Created documentation:
  - `docs/architecture.md` — System architecture with Mermaid diagrams
  - `docs/scope_and_rules.md` — Defect classes, inspection rules, escalation conditions, SOP
  - `docs/model_pipeline.md` — Data flow from image input to structured report
  - `docs/quality_scoring.md` — Scoring framework (PASS/WARNING/FAIL)
- Built initial backend components:
  - `backend/src/config.py` — Centralized settings (thresholds, paths, image config)
  - `backend/src/detector.py` — Detector abstraction with `DetectionResult` schema
  - `backend/src/image_validator.py` — Basic image validation (exists, format, readable, dimensions)
  - `backend/src/quality_engine.py` — Quality Intelligence Engine placeholder
  - `backend/src/scoring.py` — Quality scoring engine (severity mapping, base score deductions)
  - `backend/src/report_generator.py` — Structured JSON report generation
- Built initial test suite:
  - `backend/tests/test_detector.py`
  - `backend/tests/test_validator.py`
  - `backend/tests/test_scoring.py`
  - `backend/tests/test_quality_engine.py`
  - `backend/tests/test_report.py`
- Prepared the bearing dataset:
  - `backend/dataset/data.yaml` — YOLO dataset configuration (4 classes)
  - `backend/dataset/images/train/` — 78 training images (640×640, .jpg)
  - `backend/dataset/images/val/` — 22 validation images
  - `backend/dataset/labels/train/` — Training annotations (YOLO format)
  - `backend/dataset/labels/val/` — Validation annotations
- Created backend virtual environment (`backend/venv/`)

### Key Decisions
- Confidence threshold: 0.40
- IoU threshold (NMS): 0.45
- Blur detection: Variance of Laplacian > 100
- Brightness range: Mean intensity 50–200
- Contrast: RMS > 20
- Quality score: starts at 100, deducted per defect severity
- PASS ≥ 90, WARNING ≥ 70, FAIL < 70
- Rust is always high severity regardless of confidence

### Commits
- `d48dd71` — Changed the file structure
- `a9fdcc2` — Initial commit in README.md file
- `7799fb4` — Proceeded with Day 2 work

---

## Day 3 — Baseline YOLO Training & Validation (2026-08-13 – 2026-08-16)

### Objective
Establish a reproducible baseline YOLO defect-detection training and validation pipeline.

### Work Completed

#### Dependencies
- Added `ultralytics`, `pandas`, `matplotlib`, `pytest` to `backend/requirements.txt`
- Installed all dependencies in `backend/venv`
- Verified Ultralytics v8.4.120 installation

#### Dataset Validation
- Created `backend/src/dataset_validator.py`:
  - Validates image integrity (exists, readable, non-zero, valid dimensions)
  - Validates YOLO annotations (class IDs, coordinate ranges, bbox dimensions)
  - Distinguishes empty labels (valid `good` class) from invalid labels
  - Reports class distribution, image formats, and dimensions
- Dataset audit results:
  - 100 images total (78 train, 22 val)
  - All 640×640 .jpg format
  - 34 empty-label images (representing `good` bearings)
  - 0 invalid annotations, 0 missing labels, 0 corrupted images
  - Class distribution: scratch (26), rust (24), dent (16)

#### Image Quality Conditions
- Updated `backend/src/image_validator.py` with Day 3 quality checks:
  - Blur detection (Variance of Laplacian, threshold > 100)
  - Brightness validation (mean intensity range [50, 200])
  - Contrast validation (intensity std > 20)
  - Returns quality metrics in validation result metadata

#### Training Pipeline
- Created `backend/src/train.py`:
  - Configurable: model, epochs, image size, batch size, device, seed
  - Uses Ultralytics YOLO API
  - Supports CLI arguments (`--epochs`, `--model`, `--imgsz`, `--batch`, `--device`)
  - Fixed random seed (42) for reproducibility

#### Evaluation Pipeline
- Created `backend/src/evaluate.py`:
  - Runs YOLO validation on val set (Precision, Recall, mAP50, mAP50-95)
  - Performs failure analysis (False Positives, False Negatives, class mismatches)
  - Analyzes confidence distribution (high/medium/low buckets)

#### Tests
- Created `backend/tests/test_dataset_validator.py`:
  - Tests empty labels, invalid class IDs, invalid coordinates, negative w/h, malformed annotations, missing labels
- Created `backend/tests/test_image_validator_day3.py`:
  - Tests good image, blurred image, dark image, low-contrast image, corrupted image
- All 22 tests passing

#### Baseline Training Runs
- **Smoke test (1 epoch):** Pipeline validated end-to-end on CPU
- **5-epoch baseline:** Completed in ~0.026 hours (CPU)
  - All metrics near zero (expected for 5 epochs)
  - Only `rust` showed minimal detection signal (P=0.00056, R=0.333)
- **25-epoch baseline:** In progress for meaningful metrics

#### Evaluation Results (5-epoch model)
- Precision: 0.000187
- Recall: 0.111
- mAP50: 0.0001
- mAP50-95: 0.0001
- Per-class: scratch (0/0/0/0), dent (0/0/0/0), rust (0.00056/0.333/0.000228/0.00016)
- Failure analysis: 12/12 defective validation images were False Negatives
- Confidence analysis: 0 high, 0 medium, 0 low confidence detections

### Key Decisions
- YOLOv8n (Nano) selected as baseline model for prototype speed
- CPU training (AMD Ryzen 7 7435HS) — no GPU available
- 5 epochs validated as pipeline smoke test; longer runs needed for real baseline
- Empty labels confirmed as valid representation of `good` class (not invalid)

### Files Created
- `backend/src/dataset_validator.py`
- `backend/src/train.py`
- `backend/src/evaluate.py`
- `backend/tests/test_dataset_validator.py`
- `backend/tests/test_image_validator_day3.py`
- `docs/DAY3_PROGRESS.md`
- `docs/WORKFLOW.md` (this file)

### Files Modified
- `backend/requirements.txt` — Added YOLO and analysis dependencies
- `backend/src/image_validator.py` — Added blur/brightness/contrast checks

### Commits
- `bebd517` — Day 3 work of Extend Quality
- `5aa2806` — Day 3 Work
- `8801a07` — Day 3 work
- `84ab977` — Updated Day 3 work (2)

---

## Day 4 — Pixel-Level Inspection Engine Implementation (2026-08-18)

### Objective
Transform the Day 3 prototype into a robust pixel-level inspection engine using OpenCV region analysis, feature extraction, multi-component quality scoring, and batch processing.

### Work Completed
- **Extended Configuration**: Updated `config.py` with default parameters for preprocessing, segmentation, feature Canny limits, and multi-component scoring weights. Fixed Pydantic V2 class Config warning.
- **Preprocessor Module**: Created `preprocessor.py` for deterministic resizing, grayscale conversion, Gaussian denoising, CLAHE contrast enhancement, and normalization.
- **Segmentation Module**: Created `segmentation.py` implementing adaptive thresholding, morphological opening/closing, contour connected component analysis, and a custom circular bearing surface mask to exclude background and center hole noise.
- **Feature Extractor Module**: Created `feature_extractor.py` to extract metrics like affected pixel counts, region count, largest region area, edge density (via Canny), and intensity/contrast statistics.
- **Inspection Result Schema**: Created `inspection_result.py` with standard Pydantic schema supporting JSON serialization and explaining score breakdown and deterministic analysis reliability.
- **Scoring & Classification Improvements**: Implemented explainable multi-component pixel scoring in `scoring.py` (deducting weighted penalties from 100.0) and tested boundary cases.
- **Batch Processing**: Created `batch_inspector.py` to process image directories, containing per-image errors gracefully and returning summary statistics.
- **Debug Visualization**: Created `visualizer.py` to save side-by-side annotated original vs defect overlay images.
- **CLI Enhancements**: Updated `main.py` adding `--pixel-inspect`, `--batch`, `--visualize`, and `--log-level` flags.
- **Comprehensive Testing**: Added 11 unit and integration tests across all modules.

### Key Decisions
- Restrict defect analysis strictly to the bearing surface using circular masking based on scaling dimensions.
- Use Canny edge detection solely for texture roughness/edge density metrics, not semantic segmentation.
- Coexistence of Day 3 YOLO detection structure and Day 4 pixel segmentation structure.

### Files Created
- `backend/src/preprocessor.py`
- `backend/src/segmentation.py`
- `backend/src/feature_extractor.py`
- `backend/src/inspection_result.py`
- `backend/src/visualizer.py`
- `backend/src/batch_inspector.py`
- `backend/tests/test_preprocessor.py`
- `backend/tests/test_segmentation.py`
- `backend/tests/test_feature_extractor.py`
- `backend/tests/test_inspection_result.py`
- `backend/tests/test_pixel_scoring.py`
- `backend/tests/test_batch_inspector.py`
- `docs/DAY4_ARCHITECTURE.md`

### Files Modified
- `backend/src/config.py`
- `backend/src/image_validator.py`
- `backend/src/quality_engine.py`
- `backend/src/scoring.py`
- `backend/main.py`
- `docs/WORKFLOW.md`

---

## Day 5 — (Planned)

### Objective
Build OpenCV preprocessing and measurement pipeline.

---

## Days 6–7 — (Planned)

### Objective
Build FastAPI inspection endpoint and integrate OpenCV + YOLO.

---

## Day 8 — (Planned)

### Objective
Build React Live Inspection screen.

---

## Day 9 — (Planned)

### Objective
Build Quality Intelligence Engine and escalation rules.

---

## Days 10–14 — (Planned)

### Objective
VLM integration, inspection history, analytics, end-to-end testing, demo hardening, and documentation.
