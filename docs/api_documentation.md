# GaitGuard AI — REST API Documentation (v1)

## 1. Overview

The GaitGuard AI REST API provides automated cattle gait screening, video quality gate validation, system version metadata, and SHAP explainability insights.

* **Base URL**: `http://localhost:8000`
* **API Version**: `v1`
* **Content Types**: `application/json` (standard requests/responses), `multipart/form-data` (video uploads).

---

## 2. Global Headers

### Request Correlation ID Header
All API responses include a unique Request Correlation ID header for distributed log tracking:
```http
X-Request-ID: req_8a9f31c2d0e44b
X-Process-Time-MS: 68.58
```

---

## 3. Endpoints Specification

### 3.1 Liveness Check
`GET /health`

Checks API server liveness. Lightweight endpoint suitable for load balancers and container orchestrators.

**Response (200 OK)**:
```json
{
  "status": "ok",
  "service": "gaitguard-api",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

### 3.2 Readiness Check
`GET /readiness`

Verifies that PyTorch BiLSTM model weights, calibration parameters, and SHAP background samples are loaded and ready in memory.

**Response (200 OK)**:
```json
{
  "status": "ready",
  "service": "gaitguard-api",
  "version": "1.0.0",
  "model_version": "bilstm-mode-d-f76",
  "pipeline_version": "phase-12-integrated"
}
```

**Response (503 Service Unavailable)**:
```json
{
  "detail": "Screening model dependencies are initializing or unavailable."
}
```

---

### 3.3 System Version Metadata
`GET /version`

Exposes system version metadata, feature schema versioning, and phase status.

**Response (200 OK)**:
```json
{
  "api_version": "v1",
  "model_version": "bilstm-mode-d-f76",
  "pipeline_version": "phase-12-integrated",
  "feature_schema_version": "schema-76-v1",
  "phase": 18
}
```

---

### 3.4 Execute Cattle Gait Screening
`POST /api/v1/screen`

Accepts cattle walking video upload (`multipart/form-data`) and returns 3-way screening triage outcome, calibrated risk probability, quality gate metrics, and SHAP feature attributions.

**Form Parameters**:
* `video` (file stream, required): Video file stream (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, max 100MB).
* `animal_id` (string, optional): Cow identifier or ear tag number (e.g., `COW-9812`).
* `session_id` (string, optional): Recording session identifier (e.g., `SESS-2026-10-09`).

#### Successful Screening Response (200 OK — NORMAL Outcome):
```json
{
  "status": "success",
  "request_id": "req_8a9f31c2d0e44b",
  "video_quality": {
    "status": "READY",
    "quality_score": 88.5,
    "keypoint_coverage": 0.94,
    "motion_quality": 85.0,
    "blur_indicator": 12.4,
    "framing_quality": 90.0,
    "issues": [],
    "user_guidance": []
  },
  "inference": {
    "raw_probability": 0.1021,
    "calibrated_probability": 0.1250,
    "decision": "NORMAL",
    "confidence": "HIGH",
    "threshold": 0.34,
    "uncertainty_margin": 0.10
  },
  "explanation": {
    "available": true,
    "top_contributors": [
      {
        "feature_name": "back_arch_curvature",
        "feature_index": 2,
        "attribution": -0.142,
        "direction": "DECREASES_RISK",
        "modality": "biomechanical",
        "body_region": "spine"
      },
      {
        "feature_name": "stance_timing_asymmetry",
        "feature_index": 5,
        "attribution": -0.089,
        "direction": "DECREASES_RISK",
        "modality": "biomechanical",
        "body_region": "limbs"
      }
    ],
    "derived_gait_evidence": [
      {
        "name": "Back Arch Curvature Index",
        "value": -0.0123,
        "relative_contribution": 0.35,
        "direction": "NORMAL",
        "interpretation": "Spinal curvature elevation remains within normal baseline limits."
      }
    ]
  },
  "metadata": {
    "request_id": "req_8a9f31c2d0e44b",
    "animal_id": "COW-9812",
    "session_id": null,
    "pipeline_version": "phase-12-integrated",
    "total_processing_time_sec": 0.068
  },
  "result_summary": "Normal gait movement pattern detected. No significant biomechanical anomalies found.",
  "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
}
```

#### Quality Gate Rejection Response (200 OK / 422 Unprocessable Entity):
```json
{
  "status": "retry",
  "request_id": "req_7f8a9b0c1d2e",
  "video_quality": {
    "status": "RETRY",
    "quality_score": 35.0,
    "keypoint_coverage": 0.42,
    "motion_quality": 30.0,
    "blur_indicator": 85.2,
    "framing_quality": 40.0,
    "issues": ["EXCESSIVE_CAMERA_BLUR", "LOW_KEYPOINT_CONFIDENCE"],
    "user_guidance": [
      "Keep camera steady using a tripod or firm two-handed grip",
      "Ensure full side profile of cow is visible from head to hoofs"
    ]
  },
  "inference": null,
  "explanation": null,
  "metadata": {
    "request_id": "req_7f8a9b0c1d2e",
    "animal_id": null,
    "session_id": null,
    "pipeline_version": "phase-12-integrated",
    "total_processing_time_sec": 0.018
  },
  "result_summary": "Video quality insufficient for reliable gait screening.",
  "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
}
```

---

## 4. Standard HTTP Error Responses

| Status Code | Detail Code | Cause / Remedy |
| :--- | :--- | :--- |
| **400 Bad Request** | `UNSUPPORTED_FILE_EXTENSION` | File extension is not `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm`. |
| **400 Bad Request** | `MAGIC_BYTE_MISMATCH` | Header binary bytes fail video signature validation (`ftyp`/`RIFF`). |
| **413 Payload Too Large** | `PAYLOAD_TOO_LARGE` | Video file size exceeds 100MB limit. |
| **422 Unprocessable Entity**| `QUALITY_GATE_REJECTED` | Quality Gate score $< 50.0$. Follow Capture Coach guidance. |
| **500 Internal Server Error**| `INTERNAL_SERVER_ERROR` | Internal server exception. Sanitized error returned with Request ID. |
