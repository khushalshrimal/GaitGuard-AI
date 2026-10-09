"""
GaitGuard AI - Screening Endpoint Route (Phase 12)
"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status

from app.schemas import ScreeningResponse
from app.services.video_service import VideoService
from app.services.inference_service import InferenceService
from app.dependencies import get_inference_service

logger = logging.getLogger("gaitguard.routes.inference")

router = APIRouter(prefix="/api/v1", tags=["Screening"])

@router.post("/screen", response_model=ScreeningResponse, summary="Execute Cattle Gait Screening & Explainability")
async def screen_video(
    video: UploadFile = File(..., description="Cattle walking video file (.mp4, .avi, .mov, .mkv)"),
    animal_id: Optional[str] = Form(None, description="Optional animal identification string"),
    session_id: Optional[str] = Form(None, description="Optional recording session identifier"),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Executes end-to-end cattle gait lameness risk screening:
    1. Validates video upload extension, MIME type, and size limits.
    2. Runs Phase 10 Video Quality Gate. Intercepts corrupted/low-quality recordings.
    3. Runs Phase 9 Pose Pipeline -> Phase 7 BiLSTM -> Phase 8 Calibration & Triage.
    4. Runs Phase 11 SHAP Explainability & Evidence Layer.
    5. Safely cleans up temporary video file resources.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    temp_path = None
    
    try:
        # Save upload to secure temporary file
        temp_path = await VideoService.save_temp_file(video)
        
        # Execute screening pipeline
        result_dict = service.process_screening(
            video_path=temp_path,
            animal_id=animal_id,
            session_id=session_id,
            request_id=request_id
        )
        
        return result_dict
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        err_detail = f"Internal error during video screening: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{request_id}] {err_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_detail
        )
    finally:
        if temp_path:
            VideoService.cleanup_temp_file(temp_path)
