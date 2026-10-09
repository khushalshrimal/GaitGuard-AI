# GaitGuard AI — AI-Assisted Cattle Gait & Lameness-Risk Screening System

[![Phase 18 Release](https://img.shields.io/badge/Phase-18--Release-emerald.svg)](https://github.com/khushalshrimal/GaitGuard-AI)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React PWA](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB.svg)](https://reactjs.org/)
[![License Audit](https://img.shields.io/badge/License%20Audit-Completed-amber.svg)](docs/dataset_licensing_audit.md)

**GaitGuard AI** is an end-to-end, privacy-hardened software system designed for mobile-first, video-based cattle gait and lameness-risk screening. It processes walking cattle videos through an automated Video Quality Gate, 17-keypoint anatomical pose extraction, 76 biomechanical gait feature engineering, a Bidirectional LSTM (BiLSTM) temporal neural network, Sigmoid probability calibration, a 3-way triage engine, and SHAP (SHapley Additive exPlanations) attribution evidence.

> [!IMPORTANT]
> **MEDICAL & VETERINARY DISCLAIMER**: GaitGuard AI is an **AI-assisted screening decision-support tool**, NOT a veterinary diagnostic device. It flags biomechanical movement anomalies relative to training baselines to support routine herd triage and does not replace visual or physical evaluation by a licensed veterinarian.

---

## Architecture Overview

```
                                  [ USER / MOBILE DEVICE ]
                                             │
                                   Upload / Live Camera Video
                                             │
                                             ▼
                                 [ FASTAPI REST BACKEND ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ VIDEO QUALITY GATE ]                      [ MAGIC BYTE & SECURITY ]
          (Blur / Coverage / Motion)                    (ftyp / RIFF / Path Traversal)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                                  [ POSE KEYPOINT PIPELINE ]
                               (17 Anatomical Cattle Keypoints)
                                             │
                                             ▼
                                 [ FEATURE ENGINEERING ]
                            (76 Biomechanical & Motion Features)
                                             │
                                             ▼
                                 [ Phase 7 BiLSTM MODEL ]
                            (Temporal Movement Pattern Analysis)
                                             │
                                             ▼
                                  [ Phase 8 CALIBRATION ]
                               (Platt Sigmoid Probabilities)
                                             │
                                             ▼
                                   [ Phase 8 TRIAGE ]
                        (Threshold τ = 0.34, Margin ± 0.10)
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     ▼                       ▼                       ▼
            [ NORMAL OUTCOME ]     [ LAMENESS RISK OUTCOME ]   [ INCONCLUSIVE OUTCOME ]
             (Prob < 0.24)             (Prob > 0.44)          (0.24 <= Prob <= 0.44)
                     │                       │                       │
                     └───────────────────────┼───────────────────────┘
                                             ▼
                                  [ Phase 11 SHAP EVIDENCE ]
                           (Level 1 Attributions & Level 2 Metrics)
                                             │
                                             ▼
                                 [ REACT / PWA FRONTEND ]
                           (Result Cards, Evidence & Disclaimer)
```

---

## Key Features Implemented

1. **Mobile-First React PWA Frontend**: Modern SaaS user interface built with React 18, TypeScript, Tailwind CSS, Vite, and Lucide icons featuring drag-and-drop file upload, live camera recording controls, and offline readiness.
2. **Production-Hardened FastAPI Backend**: Asynchronous REST service powered by Uvicorn, Pydantic v2 schemas, CORS middleware, and request correlation ID tracking (`X-Request-ID: req_...`).
3. **Video Quality Gate & Capture Coach**: Automated assessment of motion blur, spatial framing, keypoint coverage, and frame rate with actionable user retry guidance.
4. **Frozen Temporal BiLSTM Neural Network**: Evaluates sequence movement dynamics over 128-frame windows using a 76-feature representation (coordinates, velocities, and biomechanical indicators).
5. **Calibrated Probability & 3-Way Triage**: Platt Sigmoid scaling maps raw neural network outputs into calibrated screening probabilities, triaging outcomes into `NORMAL`, `LAMENESS_RISK`, or `INCONCLUSIVE` (uncertainty band $[0.24, 0.44]$ at threshold $\tau = 0.34$).
6. **Explainable AI (SHAP Evidence Layer)**: Computes Level 1 feature attributions and Level 2 domain gait metrics to highlight biomechanical risk contributors and neutralizers.
7. **Zero Temporary Video Retention**: Guaranteed post-inference deletion of temporary upload files inside deterministic `finally` blocks across all success, quality rejection, exception, and timeout paths.

---

## Dataset Provenance & Internal Validation

* **Primary Benchmark Dataset**: Russello et al. cattle keypoint sequence dataset (272 video sequence samples, 98 unique cows).
* **Cross-Validation Protocol**: Leak-free 5-Fold animal-level `GroupKFold` (0 cow overlap between training and validation folds).
* **Internal Performance Metrics**:
  * Accuracy: ~81.97%
  * Recall: ~78.96%
  * F1-Score: ~0.7973
  * Out-of-Fold (OOF) ROC-AUC: ~0.9016
  * Thresholded OOF Screening Recall ($\tau = 0.34$): ~89.15%

> [!WARNING]
> **EXTERNAL FIELD VALIDATION STATUS**: Independent external field validation on unseen farm datasets remains `BLOCKED — DATA COLLECTION REQUIRED` per Phase 15 protocol. Reported accuracy metrics represent internal cross-validation results and must not be cited as clinical sensitivity/specificity.

---

## System Requirements

* **Operating System**: Windows 10/11, macOS, or Linux
* **Python**: Version 3.10 or higher
* **Node.js**: Version 18.0 or higher (npm 9+)
* **Docker** (Optional): Docker Desktop 4.0+ for containerized execution

---

## Quick Startup Guide

### 1. Repository Setup & Environment
```powershell
# Clone repository
git clone https://github.com/khushalshrimal/GaitGuard-AI.git
cd GaitGuard-AI

# Copy environment template
Copy-Item .env.example .env
```

### 2. Backend Startup (FastAPI + Uvicorn)
```powershell
# Install Python dependencies
py -3 -m pip install -r requirements.txt

# Start production API server
$env:PYTHONPATH="."
py -3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* API Health Check: `http://localhost:8000/health`
* OpenAPI Documentation: `http://localhost:8000/docs`

### 3. Frontend Startup (React + Vite PWA)
```powershell
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
* Access Application: `http://localhost:5173`

---

## Automated Test Execution

GaitGuard AI includes complete test suites across both backend and frontend layers:

```powershell
# Run complete Python backend test suite (239 tests across 15 modules)
$env:PYTHONPATH="."
py -3 -m unittest discover -s tests

# Run Vitest frontend test suite (13 tests)
cd frontend
npm test -- --run

# Run Vite frontend production build
npm run build
```

---

## Docker Containerization

```powershell
# Build non-root production container
docker build -t gaitguard-ai:phase-18 .

# Run with Docker Compose
docker-compose up -d
```

---

## Project Documentation Index

| Topic | Document Path |
| :--- | :--- |
| **System Architecture** | [docs/architecture.md](docs/architecture.md) |
| **Model Evaluation & Metrics** | [docs/model_evaluation.md](docs/model_evaluation.md) |
| **API Contract & Schema** | [docs/api_documentation.md](docs/api_documentation.md) |
| **Privacy & Security Audit** | [docs/privacy_and_security.md](docs/privacy_and_security.md) |
| **Dataset & License Audit** | [docs/dataset_licensing_audit.md](docs/dataset_licensing_audit.md) |
| **Live Demo Runbook** | [docs/live_demo_runbook.md](docs/live_demo_runbook.md) |
| **Technical Viva Preparation (40+ Q&A)**| [docs/viva_preparation.md](docs/viva_preparation.md) |
| **Limitations & Validation Policy** | [docs/limitations_and_validation.md](docs/limitations_and_validation.md) |
| **Release Manifest** | [docs/release_manifest.md](docs/release_manifest.md) |

---

## License & Attribution

* Software License: Open Source (MIT License for application code).
* Dataset & Asset Provenance: Russello et al. cattle keypoint dataset. See [docs/dataset_licensing_audit.md](docs/dataset_licensing_audit.md) for full license audit details.
