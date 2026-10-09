# GaitGuard AI — Dataset, Asset & Software Licensing Audit

## 1. Audit Overview

This document summarizes the provenance, license status, attribution requirements, and usage scope for all datasets, pre-trained model weights, and software dependencies integrated into GaitGuard AI.

---

## 2. Dataset Provenance Audit

| Resource Name | Original Source / Citation | Intended Usage | License / Status | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| **Russello et al. Keypoint Dataset** | Russello et al., *Cattle Gait Analysis via Keypoint Tracking* | Primary benchmark model training & 5-fold CV evaluation | Academic / Research Use | `VERIFIED — RESEARCH USE ONLY` |
| **CowScreeningDB** | CowScreeningDB Public Lameness Database | Exploratory gait feature validation & EDA baselines | Open Academic License | `VERIFIED — RESEARCH USE ONLY` |
| **LSTM Lameness Dataset** | Keypoint csv sequences | Feature benchmark baseline comparisons | Public Repository | `VERIFIED — RESEARCH USE ONLY` |

---

## 3. Pre-Trained Model Weights & Software Assets

| Asset Name | Component Purpose | Source / License | Distribution Status |
| :--- | :--- | :--- | :--- |
| **Phase 7 BiLSTM Weights** | `gaitguard/models/bilstm_model.pt` | Trained internally on Russello dataset | `VERIFIED — INCLUDED IN REPO` |
| **Platt Sigmoid Calibrator**| `gaitguard/triage/calibrator.py` | Cross-fitted on internal OOF predictions | `VERIFIED — INCLUDED IN REPO` |
| **OpenCV (`opencv-python`)**| Video decoding & frame sampling | Apache 2.0 License | `VERIFIED — OPEN SOURCE` |
| **PyTorch (`torch`)** | BiLSTM neural network evaluation | BSD-style License | `VERIFIED — OPEN SOURCE` |
| **FastAPI / Uvicorn** | REST API Backend server | MIT License | `VERIFIED — OPEN SOURCE` |
| **React / Vite / Tailwind** | PWA Frontend application | MIT License | `VERIFIED — OPEN SOURCE` |

---

## 4. Compliance & Commercial Scope Notice

> [!WARNING]
> **COMMERCIAL USE AUDIT REQUIREMENT**:
> While application source code is distributed under the MIT License, dataset assets and pre-trained model weights trained on academic datasets (Russello et al.) are restricted to **non-commercial academic research, educational demonstrations, and hackathon evaluation**. Any commercial deployment requires independent model training on commercially licensed cattle video datasets.
