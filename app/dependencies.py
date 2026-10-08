"""
GaitGuard AI - FastAPI Dependency Injection (Phase 12)
Provides singleton instances of GaitGuardInferencePipeline and InferenceService.
Guarantees PyTorch BiLSTM model & SHAP explainer load ONCE at startup.
"""

from typing import Optional
from app.services.inference_service import InferenceService

_inference_service_instance: Optional[InferenceService] = None

def get_inference_service() -> InferenceService:
    """
    FastAPI dependency returning singleton InferenceService instance.
    """
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService()
    return _inference_service_instance

def initialize_services():
    """
    Initializes model and explainer singletons during app lifespan startup.
    """
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService()
    return _inference_service_instance
