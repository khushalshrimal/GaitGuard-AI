# GaitGuard AI — Live Demo Runbook & Hackathon Walkthrough

## 1. Demo Walkthrough Overview

* **Target Duration**: 4 to 5 Minutes
* **Objective**: Live demonstration of GaitGuard AI running from actual code and inference backend.
* **Presenter Role**: Presenter demonstrates the complete user journey: launching app, reviewing privacy/disclaimer, uploading sample cattle walking video, quality gate inspection, BiLSTM inference, calibrated triage outcome, and SHAP explainability.

---

## 2. Pre-Demo Environment Checklist (5 Minutes Before Demo)

- [ ] **Backend Server Active**: Run `$env:PYTHONPATH="."; py -3 -m uvicorn app.main:app --port 8000`
- [ ] **Backend Health Check**: Open `http://localhost:8000/health` in browser — assert `{"status": "ok", "model_loaded": true}`
- [ ] **Frontend Active**: Run `npm run dev` in `frontend/` directory.
- [ ] **Frontend Loaded**: Open `http://localhost:5173` in browser — assert green "API Connected" badge in Header.
- [ ] **Sample Video Ready**: Have test sample video ready on local desktop.

---

## 3. Minute-by-Minute Live Presentation Script

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TIMELINE OUTLINE                                                        │
│ 0:00–0:30  Problem & Vision                                             │
│ 0:30–1:00  App Launch & Privacy-First Framing                           │
│ 1:00–2:00  Video Upload & Preview                                       │
│ 2:00–3:00  Quality Gate & BiLSTM Inference                              │
│ 3:00–4:00  Screening Result & SHAP Evidence                             │
│ 4:00–5:00  Technical Architecture & Responsible Use Summary             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 0:00–0:30 — Problem & Vision
* **Action**: Show Title Slide on screen.
* **Speaker Script**:
  > *"Bovine lameness is one of the top economic and welfare challenges in dairy farming, leading to reduced milk yield and acute pain. Traditional detection relies on manual visual scoring, which is subjective and often catches lameness too late. GaitGuard AI provides an AI-assisted screening decision-support tool that transforms standard smartphone video recordings into quantitative biomechanical gait analysis."*

### 0:30–1:00 — Product Launch & Privacy-First Framing
* **Action**: Switch to browser showing `http://localhost:5173`. Click **About & Privacy** tab.
* **Speaker Script**:
  > *"Here is the live GaitGuard AI web application. Notice the green 'API Connected' indicator at the top. Before running a screening, let's look at Privacy: GaitGuard enforces a strict zero-persistence data policy. Uploaded videos are processed in isolated memory and immediately deleted after keypoint extraction. No video files or farmer PII are stored on disk. Also notice our disclaimer: GaitGuard is an AI-assisted screening tool, not a veterinary diagnosis."*

### 1:00–2:00 — Video Input & Preview
* **Action**: Click **Screening** tab. Select or drag-and-drop a sample cattle walking video file.
* **Speaker Script**:
  > *"We go to the Screening workflow. Farmers can record live cattle video on mobile devices or drag-and-drop existing files. I will select a sample walking video. The client instantly validates the format and file size, rendering an HTML5 video preview so the user can verify the recording before submitting."*

### 2:00–3:00 — Live Screening Execution
* **Action**: Click **Screen This Video** button.
* **Speaker Script**:
  > *"When I click 'Screen This Video', the stream is transmitted to our FastAPI backend. The server checks magic bytes to prevent spoofing, runs an automated Video Quality Gate assessing camera stability and keypoint coverage, extracts 17 anatomical keypoints, and passes 76 biomechanical features to our 5-fold cross-validated BiLSTM neural network."*

### 3:00–4:00 — Screening Result & SHAP Evidence
* **Action**: Scroll down to display the Screening Result Card and SHAP Evidence Panel.
* **Speaker Script**:
  > *"Within milliseconds, GaitGuard returns a calibrated screening result. Here we see the calibrated probability percentage, confidence level, and uncertainty margin. Our 3-way triage engine categorizes outcomes into NORMAL, LAMENESS_RISK, or INCONCLUSIVE for samples within our uncertainty band [0.24, 0.44]. Below, our SHAP Explainability panel breaks down Level 1 feature attributions—such as back-arch elevation or stance asymmetry—explaining model reasoning without claiming medical causality."*

### 4:00–5:00 — Technical Summary & Responsible Use
* **Action**: Click **How It Works** tab or switch to summary slide.
* **Speaker Script**:
  > *"In summary, GaitGuard AI combines computer vision, temporal neural networks, Platt calibration, and explainability into a responsive, privacy-preserving microservice. Our pipeline model latency is under 67 ms. Reported accuracy is 81.97% on internal animal-level GroupKFold cross-validation. We emphasize that external field validation remains the essential next step before commercial deployment. Thank you!"*

---

## 4. Fallback & Troubleshooting Procedure

| Issue / Failure | Immediate Recovery Step |
| :--- | :--- |
| **API Offline Badge in Header** | Verify Uvicorn terminal is running. Restart via `$env:PYTHONPATH="."; py -3 -m uvicorn app.main:app --port 8000`. |
| **Quality Gate Rejection (HTTP 422)**| Show `QualityRetry` UI as a feature! Explain that the Quality Gate correctly intercepted low-quality video to prevent false predictions. |
| **Camera Permission Denied** | Use the "Upload Existing Video File" button directly. |
