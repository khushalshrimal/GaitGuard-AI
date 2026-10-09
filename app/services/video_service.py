"""
GaitGuard AI - Video Upload Validation & Temp File Lifecycle Service (Phase 12 & Phase 16)
Enforces magic-byte file signature validation, path traversal prevention,
streaming size limits, empty file rejection, and post-inference temporary file cleanup.
"""

import os
import uuid
import logging
from fastapi import UploadFile, HTTPException, status
from app.config import settings

logger = logging.getLogger("gaitguard.video_service")

# Common video file magic signatures
MAGIC_SIGNATURES = [
    b"ftyp",      # MP4 / MOV
    b"RIFF",      # AVI
    b"\x1a\x45\xdf\xa3", # WebM / MKV
]

class VideoService:
    """
    Service for handling video upload validation, magic-byte inspection, safe temporary storage, and cleanup.
    """

    @staticmethod
    def validate_filename_and_extension(filename: str) -> str:
        """
        Sanitizes filename and validates file extension against allowed list.
        Prevents path traversal attempts.
        """
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename cannot be empty."
            )
            
        safe_filename = os.path.basename(filename)
        _, ext = os.path.splitext(safe_filename.lower())
        
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Allowed extensions: {settings.ALLOWED_EXTENSIONS}"
            )
        return ext

    @staticmethod
    def validate_magic_bytes(header_bytes: bytes):
        """
        Inspects raw file header bytes for valid video magic signature.
        """
        if len(header_bytes) < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File header too short to verify video format."
            )
            
        has_magic = any(sig in header_bytes for sig in MAGIC_SIGNATURES)
        if not has_magic:
            logger.warning("Uploaded file failed magic byte signature check.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file content does not match a valid video stream header."
            )

    @staticmethod
    async def save_temp_file(upload_file: UploadFile) -> str:
        """
        Safely streams uploaded file to secure temporary location.
        Enforces:
        - Extension validation & path traversal prevention
        - Empty file check
        - Magic byte format verification
        - Streaming maximum file size limit (MAX_UPLOAD_MB)
        """
        filename = upload_file.filename or ""
        ext = VideoService.validate_filename_and_extension(filename)
        
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        temp_name = f"upload_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.abspath(os.path.join(settings.TEMP_DIR, temp_name))
        
        # Verify path traversal containment
        real_temp_dir = os.path.abspath(settings.TEMP_DIR)
        if not temp_path.startswith(real_temp_dir):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path traversal attempt detected."
            )

        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        written_bytes = 0
        header_read = False

        try:
            with open(temp_path, "wb") as buffer:
                while True:
                    chunk = await upload_file.read(1024 * 1024) # Read 1MB chunk
                    if not chunk:
                        break
                        
                    if not header_read:
                        VideoService.validate_magic_bytes(chunk[:64])
                        header_read = True

                    written_bytes += len(chunk)
                    if written_bytes > max_bytes:
                        buffer.close()
                        VideoService.cleanup_temp_file(temp_path)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Uploaded file size exceeds maximum allowed limit of {settings.MAX_UPLOAD_MB} MB."
                        )
                    buffer.write(chunk)

            if written_bytes == 0:
                VideoService.cleanup_temp_file(temp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty (0 bytes)."
                )

        except Exception as e:
            VideoService.cleanup_temp_file(temp_path)
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error saving temporary video file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process uploaded video file stream."
            )

        return temp_path

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
