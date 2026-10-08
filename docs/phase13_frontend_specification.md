# GaitGuard AI — Phase 13 Frontend Technical Specification

## 1. Overview
The GaitGuard AI Phase 13 frontend is a mobile-first, high-contrast Progressive Web Application (PWA) built with **React 19**, **TypeScript**, **Vite**, **Tailwind CSS**, and **Lucide Icons**. It serves as the primary field interface for farmers, herd managers, and veterinarians to record or upload cattle walking videos and receive real-time, explainable gait screening results.

## 2. Architecture & Design Principles
- **Strict Separation of Concerns**: All ML keypoint extraction, BiLSTM sequence classification, probability calibration, and SHAP explainability execute strictly on the Phase 12 FastAPI backend (`/api/v1/screen`). Zero ML inference logic resides in the client bundle.
- **Mobile-First Outdoor Usability**: Features large tap targets (`min-height: 48px`), high-contrast color palettes (Emerald/Rose/Amber), and readable typography for sunlight readability.
- **Real-Time API Health Monitoring**: Header polls `GET /health` every 15 seconds to display real-time connection state (`Online` / `Offline`).
- **Four Core UI Outcomes**:
  1. `NORMAL`: Emerald badge, calibrated risk level, recommended monitoring schedule.
  2. `LAMENESS_RISK`: Rose badge, elevated risk level, immediate physical inspection protocol.
  3. `INCONCLUSIVE`: Amber badge, probability near uncertainty threshold (0.50 ± 0.08), re-recording guidance.
  4. `RETRY`: Quality Gate intercept, itemized quality issues (lighting, distance, angle), capture coach guidance.

## 3. Core Component Hierarchy
```text
src/
├── App.tsx                     # Main workflow state machine (IDLE, CAPTURING, PREVIEW, SUBMITTING, RESULT, RETRY, ERROR)
├── main.tsx                    # React DOM root render
├── index.css                   # Tailwind imports & outdoor utility styles
├── api/
│   └── gaitguard.ts            # GaitGuardApiClient (getHealth, getVersion, screenVideo)
├── types/
│   └── api.ts                  # TypeScript interfaces matching Phase 12 schemas
└── components/
    ├── Header.tsx              # Sticky header with live API status badge
    ├── CaptureGuide.tsx        # Outdoor camera placement & walking guidelines
    ├── VideoRecorder.tsx       # HTML5 getUserMedia live recorder with fallback
    ├── VideoPreview.tsx        # Recorded/uploaded video player with metadata
    ├── ProcessingState.tsx     # Animated 4-stage screening loader
    ├── QualityRetry.tsx        # Phase 10 Quality Gate retry panel & guidance
    ├── ScreeningResult.tsx     # Result banner, risk gauge, confidence & action items
    ├── EvidencePanel.tsx       # Level 1 SHAP attributions & Level 2 Gait metrics
    └── Disclaimer.tsx          # Non-diagnostic screening legal disclaimer
```

## 4. Screening API Endpoint Schema (`POST /api/v1/screen`)
- **Input**: `multipart/form-data` containing `video` (file), optional `animal_id` (string), and optional `session_id` (string).
- **Response**: `ScreeningResponse` containing:
  - `status`: `"success" | "retry" | "error"`
  - `video_quality`: Quality metrics and retry guidance.
  - `inference`: Decision (`NORMAL` | `LAMENESS_RISK` | `INCONCLUSIVE`), calibrated probability, confidence (`HIGH` | `MEDIUM` | `LOW`), threshold, and uncertainty margin.
  - `explanation`: Level 1 top contributors and Level 2 derived gait metrics.
  - `disclaimer`: Strict screening non-diagnostic disclaimer.

## 5. Non-Diagnostic Language Enforcement
All components enforce explicit non-diagnostic phrasing:
- *"GaitGuard AI is an AI-assisted screening tool. This output is not a veterinary diagnosis."*
- *"Elevated lameness risk indicated based on gait analysis."*
