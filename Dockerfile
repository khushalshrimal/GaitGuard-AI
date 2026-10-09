# GaitGuard AI — Phase 16 Dockerfile
# Production-hardened container build with non-root unprivileged execution

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GAITGUARD_ENV=production \
    PORT=8000

# Install system dependencies (OpenCV requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create unprivileged non-root user and set permissions
RUN useradd -m -u 10001 gaitguard && \
    mkdir -p /tmp/gaitguard_uploads && \
    chown -R gaitguard:gaitguard /app /tmp/gaitguard_uploads

USER gaitguard

EXPOSE 8000

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn application server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
