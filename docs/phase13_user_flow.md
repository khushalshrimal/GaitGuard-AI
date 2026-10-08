# GaitGuard AI — Phase 13 Farmer User Flow & State Machine

## 1. Primary User Journey

```mermaid
graph TD
    A[Open GaitGuard PWA] --> B[View Capture Guide & Enter Animal ID]
    B --> C{Select Input Method}
    C -->|Record Camera| D[Live Camera View & Recording]
    C -->|Upload Video| E[Select File from Device]
    D --> F[Preview Recorded Video]
    E --> F
    F --> G[Click 'Start AI Gait Screening']
    G --> H[Processing State: 4-Stage Animation]
    H --> I{Backend API Response}
    I -->|Quality Retry| J[Quality Retry Screen: Issue Details & Tips]
    I -->|Screening Result| K[Screening Result Screen]
    J -->|Click 'Record Again'| B
    K --> L[View Level 1 SHAP Attributions & Level 2 Gait Metrics]
    K --> M[Read Non-Diagnostic Disclaimer]
    K -->|Click 'Screen Another Cow'| B
```

## 2. State Machine Transitions

| State | Trigger | Next State | Action / Visual |
| :--- | :--- | :--- | :--- |
| `IDLE` | App load | `IDLE` | Displays header, health indicator, capture guide, cow ID input, and primary action buttons. |
| `IDLE` | Click 'Record Video' | `CAPTURING` | Activates camera feed via `navigator.mediaDevices.getUserMedia`. |
| `IDLE` | File selected | `PREVIEW` | Loads selected video file into HTML5 preview player. |
| `CAPTURING` | Record & stop | `PREVIEW` | Creates recorded video Blob file and transitions to preview. |
| `PREVIEW` | Click 'Start AI Gait Screening' | `SUBMITTING` | Sends `POST /api/v1/screen` multipart payload to backend. |
| `SUBMITTING` | Backend returns `status: "retry"` | `RETRY` | Displays Phase 10 Quality Gate issues and capture tips. |
| `SUBMITTING` | Backend returns `status: "success"` | `RESULT` | Renders screening decision (`NORMAL`, `LAMENESS_RISK`, `INCONCLUSIVE`), calibrated risk %, and SHAP evidence. |
| `SUBMITTING` | Network / API failure | `ERROR` | Shows user-friendly error card with retry button. |
| `RETRY` / `RESULT` / `ERROR` | Click 'Reset / Screen Another' | `IDLE` | Clears video state and resets form for next animal. |
