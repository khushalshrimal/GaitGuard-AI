"""
GaitGuard AI - Application Environment & Settings Configuration (Phase 12 & Phase 16)
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "GaitGuard AI - Production Inference API"
    API_VERSION: str = "v1"
    APP_VERSION: str = "1.0.0"
    MODEL_VERSION: str = "bilstm-mode-d-f76"
    PIPELINE_VERSION: str = "phase-12-integrated"
    FEATURE_SCHEMA_VERSION: str = "schema-76-v1"
    
    # Environment Settings
    ENVIRONMENT: str = os.getenv("GAITGUARD_ENV", "production")
    DEBUG: bool = os.getenv("GAITGUARD_DEBUG", "False").lower() in ("true", "1", "yes")
    
    # Upload & Security limits
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    MAX_VIDEO_DURATION_SEC: float = float(os.getenv("MAX_VIDEO_DURATION_SEC", "30.0"))
    MAX_DECODED_FRAMES: int = int(os.getenv("MAX_DECODED_FRAMES", "900"))
    REQUEST_TIMEOUT_SEC: int = int(os.getenv("REQUEST_TIMEOUT_SEC", "60"))
    MAX_CONCURRENT_INFERENCE: int = int(os.getenv("MAX_CONCURRENT_INFERENCE", "5"))
    
    ALLOWED_EXTENSIONS: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ALLOWED_MIME_TYPES: list[str] = [
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "application/octet-stream"
    ]
    
    # Path & Logging configuration
    TEMP_DIR: str = os.getenv("GAITGUARD_TEMP_DIR", os.path.join("tmp_uploads"))
    LOG_LEVEL: str = os.getenv("GAITGUARD_LOG_LEVEL", "INFO")
    
    # Model & Pipeline configuration
    SCREENING_THRESHOLD: float = 0.34
    MARGIN_DELTA: float = 0.10
    
    # CORS Origins
    CORS_ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
