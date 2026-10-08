# GaitGuard AI — Phase 12 Technical Specification: Production Inference API

## 1. Overview & Architecture
The GaitGuard AI backend inference API exposes a production-oriented REST service built with **FastAPI** and **Pydantic v2**. The service connects:

```
[ HTTP Multipart Video Upload ]
              │
              ▼
[ File Validation & Temp Lifecycle ] ──> (Rejects size > 100MB / path traversal)
              │
              ▼
[ Phase 10 Video Quality Gate ] ────────> (Intercepts RETRY status; zero ML execution)
              │
              ▼
[ Phase 9 Pose & Keypoint Extraction ] ─> (17 keypoints -> 128x76 tensor)
              │
              ▼
[ Phase 7 BiLSTM Gait Classifier ] ──────> (Torch eval mode, no_grad)
              │
              ▼
[ Phase 8 Calibration & 3-Way Triage ] ──> (Sigmoid scaling -> NORMAL / LAMENESS_RISK / INCONCLUSIVE)
              │
              ▼
[ Phase 11 SHAP Explainability Layer ] ──> (Level 1 Attributions & Level 2 Derived Gait Evidence)
              │
              ▼
[ Pydantic Structured API Response ]
```

---

## 2. API Endpoints

### 1. `GET /health`
- **Summary**: Service Liveness & Model Loading Check.
- **Inference**: None.
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "service": "gaitguard-api",
  "version": "1.0.0",
  "model_loaded": true
}
```

### 2. `GET /version`
- **Summary**: Reproducibility & Pipeline Metadata.
- **Response `200 OK`**:
```json
{
  "api_version": "v1",
  "model_version": "bilstm-mode-d-f76",
  "pipeline_version": "phase-12-integrated",
  "feature_schema_version": "schema-76-v1",
  "phase": 12
}
```

### 3. `POST /api/v1/screen`
- **Summary**: End-to-End Cattle Gait Screening & Explanation.
- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `video`: File (`UploadFile`, mandatory)
  - `animal_id`: String (optional)
  - `session_id`: String (optional)
- **Response `200 OK` (Quality Gate Pass - Success)**:
```json
{
  "status": "success",
  "request_id": "req_8fa66c77c20f",
  "video_quality": {
    "status": "READY",
    "quality_score": 75.0,
    "keypoint_coverage": 0.95,
    "motion_quality": 0.25,
    "blur_indicator": 140.0,
    "framing_quality": 0.35,
    "issues": [],
    "user_guidance": []
  },
  "inference": {
    "raw_probability": 0.7245,
    "calibrated_probability": 0.7812,
    "decision": "LAMENESS_RISK",
    "confidence": "HIGH",
    "threshold": 0.34,
    "uncertainty_margin": 0.10
  },
  "explanation": {
    "available": true,
    "top_contributors": [
      {
        "feature_name": "velocity_LHHoof_X",
        "feature_index": 46,
        "attribution": 0.0142,
        "direction": "INCREASES_RISK",
        "modality": "velocity",
        "body_region": "Hindlimbs"
      }
    ],
    "derived_gait_evidence": [
      {
        "name": "normalized_walking_speed",
        "value": 0.0893,
        "relative_contribution": -0.0125,
        "direction": "INCREASES_RISK",
        "interpretation": "Slower walking speed observed in video sequence."
      }
    ]
  },
  "metadata": {
    "request_id": "req_8fa66c77c20f",
    "animal_id": "cow_99",
    "session_id": "sess_01",
    "pipeline_version": "phase-12-integrated",
    "total_processing_time_sec": 0.2450
  },
  "result_summary": "Elevated lameness risk indicated by the screening model.",
  "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
}
```

- **Response `200 OK` (Quality Gate Failure - Retry)**:
```json
{
  "status": "retry",
  "request_id": "req_01db0b8c51ef",
  "video_quality": {
    "status": "RETRY",
    "quality_score": 30.0,
    "keypoint_coverage": 0.0,
    "motion_quality": 0.0,
    "blur_indicator": 0.0,
    "framing_quality": 0.0,
    "issues": ["VIDEO_UNREADABLE"],
    "user_guidance": ["Video file could not be read. Please record a new video and upload again."]
  },
  "inference": null,
  "explanation": null,
  "metadata": {
    "request_id": "req_01db0b8c51ef",
    "animal_id": null,
    "session_id": null,
    "pipeline_version": "phase-12-integrated",
    "total_processing_time_sec": 0.0120
  },
  "result_summary": "Video quality insufficient for ML prediction.",
  "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
}
```

---

## 3. Security, Privacy & File Lifecycle Rules
- **Extension Safety**: Only `.mp4`, `.avi`, `.mov`, `.mkv` extensions allowed.
- **Path Traversal Protection**: Uploaded filenames are sanitized using `os.path.basename` and prepended with random UUID prefixes.
- **Maximum File Size**: 100 MB max size enforced via streaming chunk counter (`HTTP 413 Entity Too Large`).
- **Temporary File Lifecycle**: Uploaded video files are stored temporarily under `tmp_uploads/` during processing and **always deleted in a `finally:` block** post-request. No video files persist on the server.
