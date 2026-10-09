"""
GaitGuard AI - Central Configuration & Threshold Registry (Phase 3 & Phase 16)
Contains validated system constants, preprocessing thresholds, pipeline settings,
and production environment configuration.
"""

import os

# Video & Coordinate Parameters
FRAME_WIDTH = 1920.0
FRAME_HEIGHT = 1080.0
KEYPOINT_COUNT = 17

# Temporal Parameters
TARGET_SEQUENCE_LENGTH = 128
MIN_SEQUENCE_LENGTH = 90      # PROVISIONAL — REQUIRES VALIDATION
MAX_SEQUENCE_LENGTH = 207     # PROVISIONAL — REQUIRES VALIDATION

# Trajectory Smoothing Parameters
SAVGOL_WINDOW_LENGTH = 5      # Must be odd integer; smooths 5-frame local window
SAVGOL_POLYORDER = 2          # Quadratic polynomial fitting for trajectory smoothing

# ML & Cross-Validation Parameters
N_SPLITS = 5
RANDOM_SEED = 42

# Phase 16 Production & Security Environment Configuration
ENVIRONMENT = os.getenv("GAITGUARD_ENV", "production")
DEBUG = os.getenv("GAITGUARD_DEBUG", "False").lower() in ("true", "1", "yes")

# Resource Limits & File Security
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))  # 100 MB
MAX_VIDEO_DURATION_SEC = float(os.getenv("MAX_VIDEO_DURATION_SEC", "30.0"))
MAX_DECODED_FRAMES = int(os.getenv("MAX_DECODED_FRAMES", "900"))
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "60"))
MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "5"))

# Allowed Formats
ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
ALLOWED_MIME_TYPES = [
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/webm"
]

# CORS Settings
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]
