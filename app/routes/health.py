"""
GaitGuard AI - Health Endpoint Route (Phase 12)
"""

from fastapi import APIRouter
from app.schemas import HealthResponse
from app.config import settings

router = APIRouter(tags=["System"])

@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
def health_check():
    """
    Returns API health status. Does NOT execute model inference.
    """
    return HealthResponse(
        status="ok",
        service="gaitguard-api",
        version=settings.APP_VERSION,
        model_loaded=True
    )
