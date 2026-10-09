"""
GaitGuard AI - Main FastAPI Application Instance (Phase 12 & Phase 16)
Configures lifespan startup events, CORS, Request ID middleware, exception handlers, and router inclusions.
"""

import uuid
import time
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
    try:
        initialize_services()
        logger.info("GaitGuard AI backend services initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize GaitGuard AI services: {e}", exc_info=True)
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

# Request ID & Observability Middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time_ms = (time.time() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"
    return response

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global Exception Handler protecting against stack trace leakage
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "req_unknown")
    logger.error(f"[{req_id}] Unhandled Exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"X-Request-ID": req_id},
        content={
            "status": "error",
            "request_id": req_id,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": f"Server Error: {type(exc).__name__}: {str(exc)}",
            "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
        }
    )

# Include API Routers
app.include_router(health.router)
app.include_router(version.router)
app.include_router(inference.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)

