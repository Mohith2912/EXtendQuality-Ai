# Scope and Inspection Rules

## Defect Classes
For the initial prototype, we are focusing on four specific bearing classifications:

1. **`good`**: A normal bearing with no visible defects.
2. **`scratch`**: A linear surface mark on the bearing.
3. **`dent`**: A physical depression or chip on the bearing surface.
4. **`rust`**: Oxidation/discoloration on the metal surface.

These classes will be used for our YOLO detection model.

## Inspection Rules

### 1. Image Quality Validation (OpenCV Layer)
Before an image is processed for defects, it must pass basic image quality checks. Images failing these checks will be rejected or escalated immediately without further processing.

- **Blur Detection**: Variance of Laplacian threshold > 100. (Images below this are considered too blurry).
- **Brightness**: Mean pixel intensity between 50 and 200. (Images outside this range are too dark or overexposed).
- **Contrast**: RMS contrast > 20. (Requires sufficient contrast to detect surface anomalies).

### 2. Defect Detection Rules (YOLO Layer)
- **Confidence Threshold**: Minimum 0.40 confidence to register a bounding box as a defect.
- **IoU Threshold (NMS)**: 0.45 to prevent duplicate overlapping bounding boxes for the same defect.

### 3. Quality Intelligence Engine (Escalation Rules)
The Quality Intelligence Engine decides whether to take the fast path (automated decision) or the escalation path (VLM + Human review).

**Fast Path Conditions (Automated Accept/Reject):**
- **Routine Accept**: 0 defects detected by YOLO, and OpenCV checks pass perfectly.
- **Routine Reject**: Defect confidence > 0.85, defect area > 5% of bearing area, and image quality is optimal.

**Escalation Conditions (Route to VLM):**
- **Ambiguous Confidence**: Defect detected with confidence between 0.40 and 0.85.
- **Multiple Minor Defects**: > 3 defects detected, but all with low individual confidence.
- **Borderline Image Quality**: OpenCV metrics are close to the threshold (e.g., Blur Variance between 100 and 120).
- **Complex Geometry**: Defect bounding box overlaps with bearing edges (e.g., inner ring or outer ring boundary), making it hard to distinguish from normal part geometry.

### 4. Standard Operating Procedure (SOP) for VLM Context
When an image is escalated, the VLM will be provided with this SOP context:
- "A bearing is ACCEPTABLE if minor scratches are purely cosmetic and do not exceed 2mm in length."
- "A bearing must be REJECTED if any rust or dent is present, regardless of size."
- "If contamination (dust/oil) is suspected rather than a scratch, recommend CLEAN AND REINSPECT."
