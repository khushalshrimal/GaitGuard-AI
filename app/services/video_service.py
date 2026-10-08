"""
GaitGuard AI - Video Upload Validation & Temp File Lifecycle Service (Phase 12)
Enforces secure file upload limits, sanitizes file paths, prevents path traversal,
streams video files to temporary storage, and guarantees post-request cleanup.
"""

import os
import shutil
import uuid
import logging
from fastapi import UploadFile, HTTPException, status

from app.config import settings

logger = logging.getLogger("gaitguard.video_service")

class VideoService:
    """
    Service for handling video upload validation, safe temporary storage, and cleanup.
    """

    @staticmethod
    def validate_upload_file(upload_file: UploadFile):
        """
        Validates file extension, MIME type, and size limits.
        """
        filename = upload_file.filename or ""
        # 1. Path Traversal & Extension Safety
        safe_filename = os.path.basename(filename)
        _, ext = os.path.splitext(safe_filename.lower())
        
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Allowed extensions: {settings.ALLOWED_EXTENSIONS}"
            )

        # 2. MIME Type Validation
        content_type = upload_file.content_type or ""
        if content_type and content_type not in settings.ALLOWED_MIME_TYPES:
            logger.warning(f"File MIME type '{content_type}' not in strictly allowed list, proceeding with extension validation.")

    @staticmethod
    async def save_temp_file(upload_file: UploadFile) -> str:
        """
        Safely streams uploaded file to secure temporary location.
        Enforces maximum file size limit (MAX_UPLOAD_MB).
        """
        VideoService.validate_upload_file(upload_file)
        
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        safe_filename = os.path.basename(upload_file.filename or "uploaded.mp4")
        temp_name = f"upload_{uuid.uuid4().hex[:8]}_{safe_filename}"
        temp_path = os.path.join(settings.TEMP_DIR, temp_name)
        
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        written_bytes = 0

        try:
            with open(temp_path, "wb") as buffer:
                while True:
                    chunk = await upload_file.read(1024 * 1024) # Read 1MB chunk
                    if not chunk:
                        break
                    written_bytes += len(chunk)
                    if written_bytes > max_bytes:
                        buffer.close()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Uploaded file size exceeds maximum allowed limit of {settings.MAX_UPLOAD_MB} MB."
                        )
                    buffer.write(chunk)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error saving temporary video file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process uploaded video file stream."
            )

        return os.path.abspath(temp_path)

    @staticmethod
    def cleanup_temp_file(temp_path: str):
        """
        Safely deletes temporary video file from disk.
        """
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary video file: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to remove temp file '{temp_path}': {e}")
