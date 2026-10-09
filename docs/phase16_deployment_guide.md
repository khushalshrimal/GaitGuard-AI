# GaitGuard AI — Phase 16 Deployment & Reverse Proxy Guide

## 1. Production Architecture Overview

```
 [ Client / PWA ] ──(HTTPS)──> [ Reverse Proxy (Nginx / Caddy) ]
                                            │
                                            ▼
                               [ Uvicorn / FastAPI App ]
                                            │
                                  ┌─────────┴─────────┐
                                  ▼                   ▼
                          [ Quality Gate ]     [ BiLSTM Model ]
```

---

## 2. Docker & Container Deployment

### 2.1 Building the Container
```bash
docker build -t gaitguard-ai:phase-16 .
```

### 2.2 Running with Docker Compose
```bash
docker-compose up -d
```

---

## 3. Nginx Reverse Proxy Configuration Example

```nginx
server {
    listen 443 ssl http2;
    server_name gaitguard.example.com;

    ssl_certificate /etc/letsencrypt/live/gaitguard.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gaitguard.example.com/privkey.pem;

    client_max_body_size 100M;
    client_body_timeout 60s;

    # CORS Headers handled by Uvicorn middleware or Nginx
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Static PWA Frontend Hosting
    location /app {
        alias /var/www/gaitguard-pwa;
        try_files $uri $uri/ /app/index.html;
    }
}
```

---

## 4. Environment Variables Checklist

Ensure the following variables are configured in `.env` or production secret store:

```ini
GAITGUARD_ENV=production
DEBUG=False
CORS_ALLOWED_ORIGINS=https://gaitguard.example.com
MAX_UPLOAD_SIZE_BYTES=104857600
MAX_VIDEO_DURATION_SEC=15.0
MAX_CONCURRENT_INFERENCE=4
REQUEST_TIMEOUT_SEC=60.0
```

---

## 5. Security & Network Hardening
1. **Non-Root Container**: Dockerfile executes as unprivileged `gaitguard` user (UID 10001).
2. **TLS 1.3**: Mandate HTTPS for all client traffic.
3. **Rate Limiting**: Apply per-IP rate limits at the Nginx layer (e.g., 10 req/min for `/api/v1/screen`).
4. **Header Hardening**: Send `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Content-Security-Policy`.
