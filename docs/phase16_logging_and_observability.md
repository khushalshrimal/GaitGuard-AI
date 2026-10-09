# GaitGuard AI — Phase 16 Structured Logging & Observability Guide

## 1. Structured Logging Overview

GaitGuard AI uses JSON-formatted structured logging for all production events. Structured log outputs allow seamless ingestion into observability stacks such as Grafana Loki, Datadog, AWS CloudWatch, or ELK (Elasticsearch / Logstash / Kibana).

---

## 2. Request Correlation ID Architecture

Every incoming HTTP request to the GaitGuard backend is assigned a unique Request Correlation ID:
* Header format: `X-Request-ID: req_<uuid4_hex>`
* Middleware propagation: `app/main.py` attaches `X-Request-ID` to request state and response headers.
* Context propagation: All downstream loggers, error handlers, and pipeline execution stages include `request_id` in their log record payload.

---

## 3. Log Event Schema

```json
{
  "timestamp": "2026-10-09T08:30:00.123456Z",
  "level": "INFO",
  "logger": "app.routes.screen",
  "request_id": "req_8a9f31c2d0e44b",
  "method": "POST",
  "path": "/api/v1/screen",
  "client_ip": "192.168.1.50",
  "status_code": 200,
  "process_time_ms": 68.58,
  "event": "inference_completed",
  "decision": "NORMAL",
  "confidence": 0.8842,
  "calibrated_prob": 0.1250,
  "quality_score": 88.5
}
```

---

## 4. Key Metric Indicators & Log Triggers

### 4.1 System Health & Readiness Metrics
* `GET /health` -> Liveness check (200 OK)
* `GET /readiness` -> Readiness check (200 OK when model weights loaded; 503 Service Unavailable if uninitialized)

### 4.2 Quality Gate & Inference Logs
* `quality_gate_passed`: Quality score >= 50.0, FPS, motion blur indicators.
* `quality_gate_rejected`: HTTP 422 return with diagnostic code (`LOW_KEYPOINT_CONFIDENCE`, `EXCESSIVE_CAMERA_BLUR`, etc.).
* `inference_concluded`: Triage decision (`NORMAL`, `LAMENESS_RISK`, `INCONCLUSIVE`), confidence score, execution latency in milliseconds.

### 4.3 Privacy & File Cleanup Events
* `temp_file_saved`: Path, size in bytes.
* `temp_file_deleted`: Target file successfully removed from `TEMP_DIR`.
