# GaitGuard AI — Phase 16 API Contract Specification

## 1. Overview

This document specifies the frozen REST API contract for the GaitGuard AI inference backend (version `v1`).

* Base URL: `/api/v1`
* Protocol: HTTP/1.1 or HTTP/2 over TLS
* Content Types: `multipart/form-data` (upload), `application/json` (responses)

---

## 2. Endpoints

### 2.1 GET /health
**Description**: Liveness endpoint for load balancers and orchestrators (Kubernetes/Docker).

**Response (200 OK)**:
```json
{
  "status": "ok",
  "service": "gaitguard-api",
  "model_loaded": true,
  "request_id": "req_12345678"
}
```

---

### 2.2 GET /readiness
**Description**: Readiness endpoint checking model pipeline initialization.

**Response (200 OK / 503 Service Unavailable)**:
```json
{
  "status": "ready",
  "service": "gaitguard-api",
  "model_loaded": true,
  "torch_device": "cpu",
  "request_id": "req_12345678"
}
```

---

### 2.3 GET /version
**Description**: System component version metadata.

**Response (200 OK)**:
```json
{
  "api_version": "v1",
  "phase": 16,
  "model_version": "bilstm-mode-d-f76",
  "screening_threshold": 0.34,
  "inconclusive_interval": [0.24, 0.44]
}
```

---

### 2.4 POST /api/v1/screen
**Description**: Primary endpoint for video screening. Accepts cattle walking video upload.

**Request Payload**:
* Form field: `video` (file stream, `.mp4`, `.avi`, or `.mov`, max 100MB)

**Successful Response (200 OK)**:
```json
{
  "request_id": "req_8a9f31c2d0e44b",
  "decision": "NORMAL",
  "confidence": 0.8842,
  "calibrated_prob": 0.1250,
  "raw_prob": 0.1021,
  "quality_score": 88.5,
  "quality_gate": {
    "passed": true,
    "quality_score": 88.5,
    "fps": 30.0,
    "duration_sec": 4.5,
    "total_frames": 135,
    "keypoint_coverage": 0.94,
    "motion_blur_score": 120.4,
    "rejection_reasons": []
  },
  "top_features": [
    {"feature": "back_arch_curvature", "shap_value": 0.142, "description": "Back arch curvature within normal range"},
    {"feature": "stance_asymmetry", "shap_value": -0.089, "description": "Symmetrical limb stance timing"}
  ],
  "disclaimer": "AI-assisted screening tool. Not a veterinary diagnosis.",
  "latency_ms": 68.58
}
```

**Quality Gate Rejection Response (422 Unprocessable Entity)**:
```json
{
  "error": "QUALITY_GATE_REJECTED",
  "message": "Video quality insufficient for gait screening",
  "quality_gate": {
    "passed": false,
    "rejection_reasons": ["EXCESSIVE_CAMERA_BLUR", "LOW_KEYPOINT_CONFIDENCE"]
  },
  "disclaimer": "AI-assisted screening tool. Not a veterinary diagnosis."
}
```

---

## 3. Standard HTTP Error Codes

| Status Code | Code / Meaning | Description |
| :--- | :--- | :--- |
| **400** | `INVALID_FILE_TYPE` | Non-video file or magic bytes mismatch |
| **413** | `PAYLOAD_TOO_LARGE` | Video file size exceeds 100MB |
| **422** | `UNPROCESSABLE_ENTITY` | Video failed Quality Gate |
| **429** | `TOO_MANY_REQUESTS` | Rate limit exceeded |
| **500** | `INTERNAL_SERVER_ERROR` | Generic server error with safe error message (no stack trace) |
| **503** | `SERVICE_UNAVAILABLE` | Server starting up or model uninitialized |
