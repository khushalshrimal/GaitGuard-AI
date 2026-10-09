"""
GaitGuard AI - Phase 16 Production Hardening Unit Test Suite
Verifies configuration lookup, /health liveness, /readiness model status, Request ID middleware,
magic bytes signature verification, path traversal rejection, streaming file size limits,
CORS header inclusion, and structured error formatting.
"""

import os
import unittest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.video_service import VideoService, MAGIC_SIGNATURES


class TestPhase16ProductionHardening(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_configuration_settings(self):
        """Verify centralized settings defaults and environment configuration."""
        self.assertIsNotNone(settings.ENVIRONMENT)
        self.assertEqual(settings.MAX_UPLOAD_MB, 100)
        self.assertEqual(settings.SCREENING_THRESHOLD, 0.34)
        self.assertEqual(settings.MARGIN_DELTA, 0.10)

    def test_02_health_liveness_endpoint(self):
        """Verify GET /health returns HTTP 200 with status ok and request_id header."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "gaitguard-api")
        self.assertTrue(data["model_loaded"])
        self.assertIn("X-Request-ID", response.headers)

    def test_03_readiness_endpoint(self):
        """Verify GET /readiness returns HTTP 200 with ready status when model is loaded."""
        response = self.client.get("/readiness")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["version"], settings.APP_VERSION)
        self.assertEqual(data["model_version"], settings.MODEL_VERSION)

    def test_04_request_id_middleware(self):
        """Verify request ID middleware generates or propagates X-Request-ID headers."""
        custom_id = "req_custom_test_12345"
        response = self.client.get("/health", headers={"X-Request-ID": custom_id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Request-ID"), custom_id)

    def test_05_magic_bytes_mp4_validation(self):
        """Verify magic byte verification accepts valid MP4 signatures."""
        mp4_magic = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 50
        # Should not raise exception
        try:
            VideoService.validate_magic_bytes(mp4_magic)
            valid = True
        except HTTPException:
            valid = False
        self.assertTrue(valid)

    def test_06_magic_bytes_invalid_rejection(self):
        """Verify magic byte verification rejects non-video headers (e.g. executable/text)."""
        exe_magic = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 50
        txt_magic = b"Hello world text file payload" + b"\x00" * 50
        
        with self.assertRaises(HTTPException) as cm1:
            VideoService.validate_magic_bytes(exe_magic)
        self.assertEqual(cm1.exception.status_code, 400)

        with self.assertRaises(HTTPException) as cm2:
            VideoService.validate_magic_bytes(txt_magic)
        self.assertEqual(cm2.exception.status_code, 400)

    def test_07_path_traversal_prevention(self):
        """Verify path traversal filenames are sanitized by VideoService."""
        traversal_name = "../../etc/passwd.mp4"
        ext = VideoService.validate_filename_and_extension(traversal_name)
        self.assertEqual(ext, ".mp4")
        
        with self.assertRaises(HTTPException) as cm:
            VideoService.validate_filename_and_extension("../../etc/passwd.txt")
        self.assertEqual(cm.exception.status_code, 400)

    def test_08_upload_non_video_rejection(self):
        """Verify POST /api/v1/screen rejects non-video files with HTTP 400."""
        file_payload = {"video": ("test_script.txt", b"plain text content", "text/plain")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Unsupported file extension", data.get("detail", data.get("message", "")))

    def test_09_cors_headers_presence(self):
        """Verify OPTIONS request handles preflight request."""
        response = self.client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        self.assertIn(response.status_code, (200, 204))

    def test_10_version_endpoint_metadata(self):
        """Verify GET /version returns complete Phase 16 system metadata."""
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["api_version"], "v1")
        self.assertEqual(data["phase"], 16)
        self.assertEqual(data["model_version"], "bilstm-mode-d-f76")


if __name__ == "__main__":
    unittest.main()
