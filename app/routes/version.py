"""
GaitGuard AI - Version & Reproducibility Information Route (Phase 12)
"""

from fastapi import APIRouter
from app.schemas import VersionResponse
from app.config import settings

router = APIRouter(tags=["System"])

@router.get("/version", response_model=VersionResponse, summary="Pipeline Version & Reproducibility Metadata")
def version_info():
    """
    Exposes pipeline, model, and feature schema version metadata for reproducibility audit.
    """
    return VersionResponse(
        api_version=settings.API_VERSION,
        model_version=settings.MODEL_VERSION,
        pipeline_version=settings.PIPELINE_VERSION,
        feature_schema_version=settings.FEATURE_SCHEMA_VERSION,
        phase=12
    )
