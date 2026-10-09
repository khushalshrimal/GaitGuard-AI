# GaitGuard AI — Hackathon Final Presentation (12 Slides)

---

## Slide 1: Title & Project Identification
* **Headline**: GaitGuard AI: AI-Assisted Cattle Gait & Lameness-Risk Screening System
* **Subtitle**: Video-Based Biomechanical Movement Analysis & Explainable AI Triage
* **Presenter**: GaitGuard AI Engineering Team
* **Disclaimer Note**: *AI-assisted screening tool. Not a veterinary diagnostic device.*

---

## Slide 2: Problem Statement & Impact
* **Headline**: Bovine Lameness — A Silent Crisis in Dairy & Beef Farming
* **Key Points**:
  - Affects up to 25–35% of commercial dairy cattle worldwide.
  - Causes severe animal pain, reduced milk production, and high culling costs.
  - Traditional locomotion scoring is manual, subjective, and catches lameness late.
* **Our Solution**: Automated, mobile-first smartphone video screening providing quantitative gait risk triage within seconds.

---

## Slide 3: Proposed Solution & User Journey
* **Headline**: Simple Mobile-First Screening Workflow
* **User Flow**:
  1. Open React PWA (Offline-ready, API connected indicator).
  2. Select or Drag-and-Drop Cattle Walking Video (or Record via Camera).
  3. Automated Quality Gate Validation (Blur, Lighting, Framing check).
  4. Instant Calibrated Triage Result (NORMAL / LAMENESS RISK / INCONCLUSIVE).
  5. Detailed SHAP Biomechanical Evidence & Next Steps.

---

## Slide 4: Real Application Interface
* **Headline**: Production-Hardened SaaS User Experience
* **Interface Highlights**:
  - Clear 3-way triage badges (`NORMAL`, `LAMENESS_RISK`, `INCONCLUSIVE`).
  - Calibrated probability gauge with uncertainty bounds ($\pm 10\%$).
  - Level 1 & Level 2 SHAP biomechanical feature attributions.
  - Zero PII collection & guaranteed zero temporary video storage.

---

## Slide 5: System Architecture & Data Flow
* **Headline**: Modular Microservice Architecture
* **Pipeline Layers**:
  - **Frontend**: React 18 + TypeScript + Tailwind CSS PWA.
  - **Backend API**: FastAPI + Uvicorn + Request ID correlation (`req_...`).
  - **Security Gate**: Magic-byte verification (`ftyp`, `RIFF`) & path traversal safety.
  - **Quality Gate**: OpenCV motion blur & keypoint coverage validation.
  - **Core ML Engine**: PyTorch BiLSTM + Sigmoid Calibrator + SHAP Explainer.

---

## Slide 6: Keypoint Extraction & Gait Feature Engineering
* **Headline**: 17 Anatomical Keypoints $\rightarrow$ 76 Biomechanical Features
* **Key Features**:
  - 17 Anatomical Keypoints: Withers, spine elevation points, hocks, fetlocks, hoofs.
  - Savitzky-Golay trajectory filtering eliminates tracking jitter.
  - Torso-length normalization makes gait metrics scale and distance invariant.
  - 76 engineered features per frame capturing back-arch curvature, stance timing asymmetry, stride length variance, and head nodding amplitude.

---

## Slide 7: BiLSTM Neural Network & Sequence Modeling
* **Headline**: Temporal Sequence Dynamics over 128-Frame Windows
* **Model Highlights**:
  - Bidirectional LSTM architecture captures past and future temporal context.
  - Input tensor shape: `(128, 76)`.
  - Analyzes cyclic walking rhythm and subtle gait asymmetries impossible to detect in static single frames.

---

## Slide 8: Evaluation Results & Leak-Free GroupKFold
* **Headline**: Rigorous Internal Evaluation (0 Cow Overlap)
* **CV Results**:
  - Accuracy: **81.97%**
  - F1-Score: **0.7973**
  - Out-of-Fold (OOF) ROC-AUC: **0.9016**
  - Screening Recall ($\tau = 0.34$): **89.15%**
* **Validation Rigor**: 5-Fold animal-level `GroupKFold` guarantees 0 cow overlap between training and validation splits.

---

## Slide 9: Calibration, Triage & Explainability
* **Headline**: Calibrated Probabilities & SHAP Attributions
* **Key Mechanics**:
  - **Platt Sigmoid Calibration**: Maps raw logits into true probability estimates.
  - **3-Way Triage**: Screening threshold $\tau = 0.34$, inconclusive band $[0.24, 0.44]$.
  - **Deep SHAP Evidence**: Highlights specific kinematic features (e.g. back arch curvature) as risk contributors or neutralizers.

---

## Slide 10: Privacy, Security & Data Retention
* **Headline**: Zero Temporary Video Persistence Guarantee
* **Security Standards**:
  - Uploaded videos deleted inside deterministic `finally` execution blocks.
  - Raw magic-byte validation prevents malicious file ingestion.
  - 0 user PII collected or stored on disk.
  - Non-diagnostic disclaimer displayed across all API responses.

---

## Slide 11: Responsible Use & External Validation Roadmap
* **Headline**: Responsible AI Deployment Policy
* **Key Mandates**:
  - Internal CV results are NOT external clinical validation.
  - External field validation remains `BLOCKED — DATA COLLECTION REQUIRED`.
  - Preregistered 6-step field validation protocol ready for blinded farm testing.
  - AI decision support tool to aid, not replace, veterinary professionals.

---

## Slide 12: Conclusion & Project Summary
* **Headline**: GaitGuard AI — Ready for Live Field Demonstration
* **Summary Achievements**:
  - Sub-67ms pipeline model inference latency.
  - 239 backend tests & 13 frontend tests (100% pass rate).
  - Production PWA build & non-root Docker deployment container ready.
  - **Live Demo & Q&A**.
