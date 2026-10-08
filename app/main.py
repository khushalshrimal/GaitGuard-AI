"""
GaitGuard AI - Main FastAPI Application Instance (Phase 12)
Configures lifespan startup events, CORS, exception handlers, and API router inclusions.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import initialize_services
from app.routes import health, version, inference

# Setup Logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gaitguard.app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler.
    Pre-loads PyTorch BiLSTM model & SHAP explainer ONCE during app startup.
    """
    logger.info("Initializing GaitGuard AI services and pre-loading PyTorch model & SHAP explainer...")
    initialize_services()
    logger.info("GaitGuard AI backend services initialized successfully.")
    yield
    logger.info("Shutting down GaitGuard AI backend services.")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.APP_VERSION,
    description=(
        "Production-oriented backend inference API for GaitGuard AI. "
        "Provides video quality gating, quadruped keypoint estimation, BiLSTM temporal lameness-risk screening, "
        "Platt probability calibration, and SHAP Explainable AI attributions. "
        "This system is an AI-assisted screening tool, NOT a veterinary diagnostic system."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler protecting against stack trace leakage
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred while processing your request.",
            "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
        }
    )

# Include API Routers
app.include_router(health.router)
app.include_router(version.router)
app.include_router(inference.router)
