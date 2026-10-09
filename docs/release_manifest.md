# GaitGuard AI — Phase 18 Official Release Manifest

## 1. System Metadata & Release Identifiers

* **Release Name**: GaitGuard AI — Phase 18 Final Release
* **Execution Date**: 2026-10-09
* **Git Branch**: `main`
* **Git Commit**: `7b16133`
* **Git Tags**: `phase-18-final-release-complete` (Phase 18), `phase-17-final-e2e-demo-complete` (Phase 17)
* **API Version**: `v1`
* **Model Version**: `bilstm-mode-d-f76`
* **Pipeline Version**: `phase-12-integrated`
* **Feature Schema**: `schema-76-v1` (76 features per frame)

---

## 2. Frozen Scientific Parameters

* **Screening Threshold ($\tau$)**: `0.34` (FROZEN)
* **Uncertainty Margin ($\Delta$)**: `0.10` (FROZEN)
* **Inconclusive Interval**: `[0.24, 0.44]` (FROZEN)
* **Temporal Sequence Length**: `128 frames` (FROZEN)
* **Keypoint Schema**: `17 anatomical cattle keypoints` (FROZEN)

---

## 3. Test Suite & Build Verification Manifest

| Test Layer | Test Suite Location | Tests Executed | Passed | Failed | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Python Backend Unittest** | `tests/` (15 test modules) | 239 | 239 | 0 | **100% PASS** |
| **Frontend Vitest Suite** | `frontend/src/__tests__/` | 13 | 13 | 0 | **100% PASS** |
| **Vite PWA Build** | `frontend/dist/` | 1 Build | 1 | 0 | **100% PASS** |

---

## 4. Environment & Dependency Manifest

* **Python Runtime**: `3.10+`
* **Node.js Runtime**: `18.0+`
* **Core Python Dependencies**: `torch>=2.0.0`, `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `pydantic>=2.0`, `scikit-learn>=1.2.0`, `shap>=0.41.0`, `opencv-python>=4.7.0`, `python-pptx>=1.0.2`
* **Core Frontend Dependencies**: `react@18.3.1`, `typescript@5.5.3`, `vite@6.4.4`, `tailwindcss@3.4.1`, `lucide-react@0.344.0`

---

## 5. Artifacts Generated in Phase 18

1. `README.md` — Updated master project README.
2. `docs/architecture.md` — System architecture and sequence diagrams.
3. `docs/model_evaluation.md` — GroupKFold internal cross-validation report.
4. `docs/api_documentation.md` — OpenAPI REST API specification.
5. `docs/privacy_and_security.md` — Privacy audit & zero persistence verification.
6. `docs/dataset_licensing_audit.md` — Asset provenance & license audit.
7. `docs/live_demo_runbook.md` — 4–5 minute hackathon presentation runbook.
8. `docs/viva_preparation.md` — 40+ technical viva Q&A guide.
9. `docs/limitations_and_validation.md` — Responsible-use policy & field protocol.
10. `docs/release_manifest.md` — Release manifest.
11. `presentation/GaitGuard_AI_Final_Presentation.pptx` — 12-slide PowerPoint presentation deck.
12. `presentation/GaitGuard_AI_Final_Presentation.md` — Markdown slide outline deck.
