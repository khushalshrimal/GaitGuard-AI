"""
GaitGuard AI - Application Environment & Settings Configuration (Phase 12)
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
    
    # Upload & Security limits
    MAX_UPLOAD_MB: int = 100
    ALLOWED_EXTENSIONS: list[str] = [".mp4", ".avi", ".mov", ".mkv"]
    ALLOWED_MIME_TYPES: list[str] = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska", "application/octet-stream"]
    
    # Path & Logging configuration
    TEMP_DIR: str = os.path.join("tmp_uploads")
    LOG_LEVEL: str = "INFO"
    
    # Model & Pipeline configuration
    SCREENING_THRESHOLD: float = 0.34
    MARGIN_DELTA: float = 0.10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
