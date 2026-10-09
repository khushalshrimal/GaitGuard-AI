"""
GaitGuard AI - Pydantic Request & Response Schemas (Phase 12)
Defines strict Pydantic v2 data validation schemas for all backend API endpoints.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

# Health & Version Schemas
class HealthResponse(BaseModel):
    status: str = Field(default="ok", example="ok")
    service: str = Field(default="gaitguard-api", example="gaitguard-api")
    version: str = Field(default="1.0.0", example="1.0.0")
    model_loaded: bool = Field(default=True, example=True)

class VersionResponse(BaseModel):
    api_version: str = Field(default="v1", example="v1")
    model_version: str = Field(default="bilstm-mode-d-f76", example="bilstm-mode-d-f76")
    pipeline_version: str = Field(default="phase-12-integrated", example="phase-12-integrated")
    feature_schema_version: str = Field(default="schema-76-v1", example="schema-76-v1")
    phase: int = Field(default=12, example=12)

# Quality Gate Schemas
class QualityDetails(BaseModel):
    status: str = Field(..., description="READY, RETRY, or INCONCLUSIVE")
    quality_score: float = Field(..., ge=0.0, le=100.0)
    keypoint_coverage: float = Field(..., ge=0.0, le=1.0)
    motion_quality: float = Field(...)
    blur_indicator: float = Field(...)
    framing_quality: float = Field(...)
    issues: List[str] = Field(default_factory=list)
    user_guidance: List[str] = Field(default_factory=list)

# Inference Details Schemas
class InferenceDetails(BaseModel):
    raw_probability: float = Field(..., ge=0.0, le=1.0)
    calibrated_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="NORMAL, LAMENESS_RISK, or INCONCLUSIVE")
    confidence: str = Field(..., description="HIGH, MEDIUM, or LOW")
    threshold: float = Field(default=0.34)
    uncertainty_margin: float = Field(default=0.10)

# Explanation Schemas
class TopContributor(BaseModel):
    feature_name: str
    feature_index: int
    attribution: float
    direction: str = Field(..., description="INCREASES_RISK, DECREASES_RISK, or LOW_CONTRIBUTION")
    modality: str
    body_region: str

class DerivedGaitEvidenceItem(BaseModel):
    name: str
    value: float
    relative_contribution: float
    direction: str
    interpretation: str

from typing import Optional, List, Dict, Any

class ExplanationDetails(BaseModel):
    available: bool = Field(...)
    top_contributors: Optional[List[TopContributor]] = Field(default_factory=list)
    modality_attribution: Optional[Dict[str, Any]] = None
    body_region_attribution: Optional[Dict[str, float]] = None
    temporal_attribution: Optional[Dict[str, float]] = None
    derived_gait_evidence: Optional[List[DerivedGaitEvidenceItem]] = Field(default_factory=list)
    reason: Optional[str] = None

# Metadata Schemas
class MetadataDetails(BaseModel):
    request_id: str
    animal_id: Optional[str] = None
    session_id: Optional[str] = None
    pipeline_version: str = Field(default="phase-12-integrated")
    total_processing_time_sec: float

# Main Screening Response Schema
class ScreeningResponse(BaseModel):
    status: str = Field(..., description="success, retry, or error")
    request_id: str
    video_quality: Optional[QualityDetails] = None
    inference: Optional[InferenceDetails] = None
    explanation: Optional[ExplanationDetails] = None
    metadata: MetadataDetails
    result_summary: str
    disclaimer: str = Field(
        default="AI-assisted screening tool. This output is not a veterinary diagnosis."
    )
