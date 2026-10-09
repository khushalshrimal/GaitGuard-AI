"""
GaitGuard AI - Health & Readiness Endpoint Route (Phase 12 & Phase 16)
"""

from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas import HealthResponse
from app.config import settings
from app.dependencies import get_inference_service
from app.services.inference_service import InferenceService

router = APIRouter(tags=["System"])

@router.get("/health", response_model=HealthResponse, summary="Service Liveness Check")
def health_check(service: InferenceService = Depends(get_inference_service)):
    """
    Returns API liveness status. Does NOT execute model inference or expensive checks.
    """
    return HealthResponse(
        status="ok",
        service="gaitguard-api",
        version=settings.APP_VERSION,
        model_loaded=service is not None
    )

@router.get("/readiness", summary="Service Readiness Check")
def readiness_check(service: InferenceService = Depends(get_inference_service)):
    """
    Verifies that required model artifacts and inference services are loaded and ready.
    Returns 200 OK when ready, or 503 Service Unavailable if model dependencies are missing.
    """
    if service is None or not getattr(service, "initialized", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Screening model dependencies are initializing or unavailable."
        )
        
    return {
        "status": "ready",
        "service": "gaitguard-api",
        "version": settings.APP_VERSION,
        "model_version": settings.MODEL_VERSION,
        "pipeline_version": settings.PIPELINE_VERSION
    }
