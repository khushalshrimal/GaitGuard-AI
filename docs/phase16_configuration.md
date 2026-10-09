# GaitGuard AI — Phase 16 Configuration & Environment Management

## Overview
GaitGuard AI uses environment-based configuration for deployment readiness across development, testing, and production environments. Critical settings (such as upload limits, timeouts, CORS origins, and concurrency bounds) are dynamically configured via environment variables.

## Environment Variable Reference
| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `GAITGUARD_ENV` | `production` | Deployment mode (`development`, `testing`, `production`). |
| `GAITGUARD_DEBUG` | `False` | Enables verbose stack traces in development only. |
| `MAX_UPLOAD_SIZE_BYTES` | `104857600` ($100\text{ MB}$) | Max streaming upload payload limit. |
| `MAX_VIDEO_DURATION_SEC` | `30.0` | Max permitted cattle video duration in seconds. |
| `MAX_DECODED_FRAMES` | `900` | Max total frame budget per uploaded video. |
| `REQUEST_TIMEOUT_SEC` | `60` | Server processing timeout. |
| `MAX_CONCURRENT_INFERENCE` | `5` | Concurrency limit for simultaneous ML inference requests. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000,...` | Comma-separated list of allowed CORS origin URLs. |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend API endpoint target. |

## Fail-Safe Startup Rules
- If required model weight files are missing or unreadable, the application logs a critical initialization error and fails readiness checks (`/readiness` returns 503) without exposing private filesystem paths to clients.
