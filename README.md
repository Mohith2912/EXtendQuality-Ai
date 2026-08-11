# EXtendQuality

**EXtendQuality** is an AI-assisted Industrial Quality Intelligence Platform designed to extend traditional computer-vision inspection systems with contextual AI reasoning for post-detection quality decisions.

---

## Vision and Problem Statement

Manufacturing quality inspection is increasingly automated for visual defect detection, but automated detection often does not fully answer the operational question: **what should happen next?**

A standard detector can usually answer what defect was detected, where it is located, and with what confidence. However, quality teams often still need to decide:
- Is the defect severe or borderline?
- Is it contamination or an actual defect?
- Should the part be accepted, rejected, reworked, cleaned, or reinspected?
- Which Standard Operating Procedure (SOP) rule applies?
- Do similar historical cases support the same decision?

This creates a real gap between machine vision detection and quality decision-making, especially in high-volume environments, ambiguous surface conditions, or scenarios where human expertise is limited. **EXtendQuality** bridges this gap as an AI Quality Copilot, evaluating image quality, routing routine cases to a fast decision path, escalating ambiguous cases to a Vision Language Model (VLM), and preserving human-in-the-loop quality approval.

---

## Core Innovation

The core innovation of EXtendQuality is not the isolated use of OpenCV, YOLO, or VLMs, but the **selective orchestration** of these layers. It supports contextual industrial quality reasoning with structured evidence, SOP context, history, and human review.

Most cases follow a **fast path**:
`Camera → OpenCV → YOLO → Quality Intelligence Engine → Fast Decision`

Only ambiguous, low-confidence, or context-dependent cases follow the **escalation path**:
`Camera → OpenCV → YOLO → Quality Intelligence Engine → VLM → Recommendation → Human Confirmation`

This selective approach ensures lower latency, reduced multimodal compute usage, better cost control, privacy-sensitive deployments, and realistic industrial feasibility.

---

## System Architecture

```text
Camera / Image Source
        ↓
OpenCV (Image Processing + Quality Checks + Measurements)
        ↓
YOLO (Fast Defect Detection)
        ↓
Quality Intelligence Engine
        ├── Routine / High Confidence → Fast Decision
        └── Ambiguous / Low Confidence → VLM
                                             ↓
                                   AI Recommendation
                                             ↓
                                   Human Confirmation / Override
                                             ↓
                                    Inspection History
                                             ↓
                                         Analytics
```

---

## Complete Workflow

1. Camera captures a bearing image.
2. Image-quality validation is performed.
3. OpenCV preprocessing and measurement pipeline runs.
4. YOLO performs defect detection.
5. The Quality Intelligence Engine evaluates the evidence.
6. If routine: the system produces a fast decision.
7. If ambiguous: the system prepares structured VLM context.
8. The VLM evaluates the image together with SOP and evidence.
9. A structured recommendation is generated.
10. A human confirms or overrides the recommendation.
11. The full inspection record is stored.
12. Dashboard and analytics are updated.

---

## Prototype Scope

The first prototype is intentionally restricted to **bearings** as a scope-control strategy. 

**Goals:** Support a manageable number of bearing defect classes, use a controlled imaging setup, demonstrate repeatable image capture, show the full inspection and decision path, preserve traceability, and visually prove selective VLM escalation.

**Out of Scope:** Inspecting every product category, claiming universal industrial generalization, replacing human quality engineers entirely, or claiming every image requires a VLM.

---

## Frontend Vision

The frontend behaves as an **industrial quality console**. 
- Dark, premium, industrial control-room UI
- Data-dense but clean dashboard layout
- Clear pipeline visibility and live inspection experience
- Explainable AI decision cards with strong evidence display
- Searchable history and meaningful analytics
- Human review visibility

---

## Backend Vision

The backend acts as a clean orchestration and service layer between the frontend and the AI components.
- Isolates frontend from implementation details
- Exposes clean structured APIs
- Supports modular services (OpenCV/YOLO/VLM replacement without frontend redesign)
- Stores structured history and analytics

---

## Technology Stack

- **Frontend:** React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Lucide React, Recharts, React Router, Axios
- **Backend:** Python, FastAPI, OpenCV, Ultralytics YOLO, Pydantic, SQLite, SQLAlchemy
- **Future Additions:** PostgreSQL, Redis, local VLM deployment, optional knowledge retrieval for SOP/history

---

## Project Structure

```text
EXtendQuality/
├── README.md
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   ├── inspection/
│   │   │   ├── intelligence/
│   │   │   ├── analytics/
│   │   │   ├── history/
│   │   │   └── ui/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── store/
│   │   └── types/
└── backend/
    ├── requirements.txt
    ├── main.py
    └── app/
        ├── api/routes/
        ├── core/
        ├── models/
        ├── schemas/
        ├── services/
        │   ├── opencv/
        │   ├── yolo/
        │   ├── intelligence/
        │   ├── vlm/
        │   ├── history/
        │   └── analytics/
        └── database/
```

---

## Core Modules

- **OpenCV Layer:** Image normalization, blur detection, brightness/contrast checks, contour support, image suitability validation.
- **YOLO Layer:** Defect localization, class prediction, confidence scoring, bounding box generation.
- **Quality Intelligence Engine:** Combines confidence, measurements, quality signals, and SOP conditions to decide routine vs ambiguous routing.
- **VLM Layer:** Receives structured context, reasons on selected cases, returns structured recommendation and explanation.
- **Human Review Module:** Preserves operator authority, supports confirm/override, records traceability notes.
- **Analytics Module:** Tracks escalation rates, defect trends, override rates, throughput, and quality statistics.

---

## Inspection Record (Data Model)

```typescript
interface InspectionRecord {
  inspection_id: string;
  timestamp: string;
  product_type: string;
  image_reference: string;
  opencv: {
    brightness?: string;
    blur?: string;
    contrast?: string;
    defect_area?: number;
    measurements?: Record<string, unknown>;
  };
  yolo: {
    defect_class?: string;
    confidence?: number;
    bounding_box?: { x: number; y: number; w: number; h: number; };
  };
  intelligence: {
    escalation_required: boolean;
    route: "fast_path" | "vlm";
    reason: string;
  };
  vlm?: {
    assessment?: string;
    recommendation?: string;
    confidence?: number;
    evidence_considered?: string[];
  };
  ai_recommendation?: string;
  final_decision?: string;
  override_reason?: string;
}
```

---

## Key Pages and Product Experience

- **Dashboard:** Operational home screen presenting system status and production overview.
- **Live Inspection:** Visually demonstrates the AI pipeline, detection, evidence, escalation, and recommendation states in real time.
- **Intelligence View:** Shows why the Quality Engine took the fast path or escalated, displaying evidence and ambiguity indicators.
- **VLM View:** Displays evidence considered, structured recommendation, reasoning summary, uncertainty, and SOP rule references.
- **Human Decision Panel:** Allows confirmation or override of AI recommendations and captures override reasons.
- **Inspection History:** Searchable records proving traceability of images, defects, routing paths, and final decisions.
- **Analytics:** Defect trends, model performance, VLM usage, human override rate, and quality performance trends.
- **SOP and Product Management:** Shows active bearing rules and product support architecture.

---

## 14-Day Prototype Plan

| Days | Planned Work |
|------|--------------|
| 1–2 | Finalize scope, select bearing defect classes, obtain/clean dataset, define inspection rules, freeze architecture. |
| 3–4 | Baseline YOLO training/validation, evaluate precision/recall/mAP, establish image-quality conditions. |
| 5 | Build OpenCV preprocessing and measurement pipeline. |
| 6–7 | Build FastAPI inspection endpoint and integrate OpenCV + YOLO. |
| 8 | Build React Live Inspection screen and structured result display. |
| 9 | Build Quality Intelligence Engine and escalation rules. |
| 10 | Integrate VLM with structured input/output on a controlled test set. |
| 11 | Build inspection history and save complete records. |
| 12 | Build analytics and VLM escalation visualization. |
| 13 | End-to-end testing, failure cases, latency and cost measurements. |
| 14 | Demo hardening, UI polish, documentation, PPT, and evidence collection. |

---

## Future Roadmap

- **Month 1:** Production-grade workflow, validation dataset discipline, model management.
- **Month 2:** Multiple bearing variants, second product category.
- **Month 3:** Local/on-premise VLM deployment, security, model lifecycle management.
- **Month 4:** MES/QMS/ERP integration, enterprise reporting.
- **Month 5:** Advanced analytics, validated retrieval architecture for SOP/history, additional justified camera modalities.
- **Month 6:** Multi-line deployment, predictive quality intelligence, broader product support.

---

## Metrics to Prove Value

- YOLO precision, recall, and mAP (with false positive/negative rates)
- Average processing latency & end-to-end inspection latency
- VLM escalation rate & VLM latency on escalated cases
- Human override rate & agreement between AI and reference decisions
- Inspection throughput & image-quality rejection rate
- Decision traceability completeness

---

## Risks and Mitigations

- **Risks:** YOLO false positives/misses, VLM hallucination, domain mismatch, reflective metal surface issues, data scarcity, latency, cloud cost, privacy constraints, trust concerns, overclaiming novelty.
- **Mitigations:** Validate YOLO on representative data, constrain VLM outputs, use SOP context, retain human approval, limit defect classes initially, use controlled lighting, measure latency/cost, support local deployment paths, and preserve absolute auditability.

---

## Claims We Should Not Make

We do **not** claim:
- OpenCV, YOLO, or VLMs are individually novel.
- The exact architecture is globally unique without prior-art validation.
- The prototype solves every manufacturing inspection use case.
- The bearing solution trivially generalizes to all other products.
- Every image needs a VLM.
- The system fully removes the need for human quality engineers.

**Correct Positioning:** EXtendQuality explores a selective AI quality decision-support architecture for industrial inspection, demonstrating its feasibility through a bearing-focused prototype.

---

## Development Principles

- Build modularly and keep frontend/backend cleanly separated.
- Preserve typed contracts.
- Start with mocks where useful, replacing them incrementally with real services.
- Prioritize the critical path over peripheral features.
- Track visible progress with daily commits.
- Optimize for demo reliability over excessive feature breadth.

---

## Demo Goal

The final demonstration must execute the following flow:
1. Place or upload a bearing image
2. Run inspection and show OpenCV checks
3. Show YOLO detection
4. Show routing to fast path or VLM
5. Display evidence and AI recommendation
6. Confirm or override decision
7. Save inspection
8. Open history and analytics to prove the system is selective, explainable, and traceable.

---

## Final Project Definition

**EXtendQuality** is an AI Quality Copilot for manufacturing that extends fast computer-vision inspection with selective multimodal reasoning. OpenCV prepares and measures the image, YOLO performs high-speed defect detection, a Quality Intelligence Engine decides whether additional reasoning is needed, and a VLM analyzes only ambiguous cases using visual evidence, inspection guidelines, and relevant history. The system returns a structured recommendation while retaining human oversight, inspection traceability, and analytics.