# GaitGuard AI — Technical Architecture & Component Specification

## 1. System Architecture Diagram

```mermaid
graph TD
    A[Client UI / React PWA] -->|POST /api/v1/screen| B[FastAPI REST API / App Layer]
    B --> C[Request ID Middleware]
    C --> D[Video Validation & Magic Bytes Check]
    D -->|Invalid Magic Bytes / File Type| E[HTTP 400 Bad Request]
    D -->|File Size > 100MB| F[HTTP 413 Payload Too Large]
    D -->|Valid Upload| G[Save Temporary Video File]
    G --> H[Phase 10 Video Quality Gate]
    H -->|Quality Score < 50.0| I[HTTP 422 / Retry Response + Guidance]
    H -->|Passed Quality Gate| J[Phase 9 Keypoint Pose Extraction]
    J --> K[Phase 3 Cleaning & Normalization]
    K --> L[Phase 5 Gait Feature Engineering]
    L --> M[Phase 7 BiLSTM Temporal Neural Net]
    M --> N[Phase 8 Platt Sigmoid Calibration]
    N --> O[Phase 8 3-Way Triage Engine]
    O --> P[Phase 11 SHAP Explainer Layer]
    P --> Q[Assemble Screening Response JSON]
    Q --> R[Finally Block: Delete Temp Video File]
    R --> S[Return Screening Response to Client]
```

---

## 2. Component Layer Responsibilities

### 2.1 Client Layer (`frontend/src/`)
* **React PWA**: Single Page Application written in React 18 and TypeScript with Tailwind CSS.
* **State Machine**: Controls flow across states (`IDLE`, `CAPTURING`, `PREVIEW`, `SUBMITTING`, `RESULT`, `RETRY`, `ERROR`).
* **Client Validation**: Enforces extension checks (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) and 100MB max file size limits prior to HTTP transfer.
* **Memory Management**: Revokes local video preview object URLs (`URL.revokeObjectURL`) to prevent browser RAM leaks.

### 2.2 API Server Layer (`app/`)
* **FastAPI Application (`app/main.py`)**: Asynchronous web framework exposing `/health`, `/readiness`, `/version`, and `/api/v1/screen`.
* **Request Correlation Middleware**: Attaches `X-Request-ID: req_<uuid4_hex>` header to all incoming HTTP requests and log entries.
* **Video Upload Service (`app/services/video_service.py`)**: Verifies raw file magic byte signatures (`ftyp`, `RIFF`, `WebM`), sanitizes filenames, enforces path traversal containment within `TEMP_DIR`, streams chunks with 100MB cap, and guarantees temporary file deletion in a deterministic `finally` block.
* **Inference Orchestration Service (`app/services/inference_service.py`)**: Connects Quality Gate, keypoint extraction, BiLSTM model evaluation, Sigmoid calibration, 3-way triage, and SHAP explainability.

### 2.3 Machine Learning Pipeline (`gaitguard/`)
* **Phase 10 Quality Gate (`gaitguard/quality/`)**: Evaluates spatial framing, motion blur indicators, keypoint coverage, and frame rate.
* **Phase 9 Keypoint & Feature Pipeline (`gaitguard/inference/`)**: Formats 17 cattle keypoints across 128 frames into exact training-compatible 76-feature representation.
* **Phase 7 BiLSTM Model (`gaitguard/models/`)**: Bidirectional LSTM neural network evaluating temporal sequence dynamics over sequence windows.
* **Phase 8 Calibration & Triage (`gaitguard/triage/`)**: Platt Sigmoid scaling mapping raw predictions to calibrated probabilities; 3-way triage engine evaluating threshold $\tau = 0.34$ and inconclusive bounds $[0.24, 0.44]$.
* **Phase 11 SHAP Explainer (`gaitguard/explainability/`)**: Exact Deep SHAP attribution engine calculating Level 1 feature attributions and Level 2 domain gait metrics.

---

## 3. Screening Request Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Farmer / Field Worker
    participant FE as React PWA Frontend
    participant API as FastAPI Backend
    participant VS as Video Service
    participant QG as Quality Gate
    participant ML as PyTorch BiLSTM & SHAP
    
    User->>FE: Select / Record Cattle Video
    FE->>FE: Client Format & Size Check (Max 100MB)
    FE->>API: POST /api/v1/screen (multipart video)
    API->>VS: Save Temp File & Verify Magic Bytes
    alt Magic Bytes Mismatch / Unsupported Extension
        VS-->>API: Raise HTTPException(400)
        API-->>FE: HTTP 400 Bad Request JSON
    else Valid Video Stream
        VS-->>API: Safe Temp File Path
        API->>QG: Run Quality Gate Analysis
        alt Quality Score < 50.0
            QG-->>API: Quality Gate Failed (Rejection Reasons)
            API->>VS: Cleanup Temp Video File
            API-->>FE: HTTP 200 / 422 Retry Response + Guidance
        else Quality Gate Passed
            QG-->>API: Quality Gate Passed
            API->>ML: Forward Keypoints to BiLSTM & SHAP
            ML-->>API: Calibrated Probability + Triage Decision + SHAP
            API->>VS: Cleanup Temp Video File (Guaranteed finally)
            API-->>FE: HTTP 200 Screening Response JSON
        end
    end
    FE->>User: Display Screening Outcome & SHAP Evidence
```

---

## 4. Execution Scope: Client vs Server

| Execution Task | Execution Location | Technology |
| :--- | :--- | :--- |
| **UI Rendering & State Machine** | Client Browser | React 18 + TypeScript + Tailwind |
| **Video Recording & Preview** | Client Browser | HTML5 MediaRecorder & `<video>` |
| **Client Format & Size Validation**| Client Browser | JavaScript `File` API |
| **Magic Byte Header Inspection** | Backend Server | Python `VideoService` Binary Reader |
| **Path Traversal Protection** | Backend Server | Python `os.path.realpath` Containment |
| **Video Decoding & Quality Gate** | Backend Server | OpenCV (`cv2`) & NumPy |
| **Keypoint Normalization & Features**| Backend Server | NumPy Vectorized Matrix Math |
| **BiLSTM Forward Pass Inference** | Backend Server | PyTorch (`torch.nn.Module`) CPU/GPU |
| **Sigmoid Calibration & Triage** | Backend Server | Scikit-Learn / Closed-Form Math |
| **SHAP Feature Attribution** | Backend Server | SHAP DeepExplainer |
| **Temporary Video Cleanup** | Backend Server | Python `os.remove` inside `finally` block |
