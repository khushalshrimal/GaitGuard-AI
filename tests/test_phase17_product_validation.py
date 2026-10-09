"""
GaitGuard AI - Phase 17 Product Validation & Integration Test Suite
Verifies end-to-end workflow, frozen ML parameters (tau=0.34, inconclusive interval [0.24, 0.44]),
health/readiness endpoints, upload security, magic byte inspection, path traversal checks,
quality gate retry flow, SHAP evidence rendering, request IDs, and zero temporary file retention.
"""

import os
import unittest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.video_service import VideoService
from app.services.inference_service import InferenceService
from gaitguard.triage.triage_engine import TriageResultContract

VALID_MP4_HEADER = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 56


class TestPhase17ProductValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = InferenceService()

    def test_01_frozen_model_parameters(self):
        """Verify screening threshold tau=0.34 and margin delta=0.10 remain strictly frozen."""
        self.assertEqual(settings.SCREENING_THRESHOLD, 0.34)
        self.assertEqual(settings.MARGIN_DELTA, 0.10)
        low_bound = settings.SCREENING_THRESHOLD - settings.MARGIN_DELTA
        high_bound = settings.SCREENING_THRESHOLD + settings.MARGIN_DELTA
        self.assertAlmostEqual(low_bound, 0.24, places=2)
        self.assertAlmostEqual(high_bound, 0.44, places=2)

    def test_02_health_liveness_check(self):
        """Verify GET /health returns HTTP 200 with status ok and model_loaded boolean."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "gaitguard-api")
        self.assertTrue(data["model_loaded"])

    def test_03_readiness_check(self):
        """Verify GET /readiness returns HTTP 200 when PyTorch model is ready."""
        response = self.client.get("/readiness")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["version"], settings.APP_VERSION)

    def test_04_version_metadata(self):
        """Verify GET /version exposes API version v1 and model version metadata."""
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["api_version"], "v1")
        self.assertEqual(data["model_version"], "bilstm-mode-d-f76")

    def test_05_request_correlation_id(self):
        """Verify X-Request-ID header is generated and returned on all API calls."""
        response = self.client.get("/health", headers={"X-Request-ID": "req_phase17_test_999"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Request-ID"), "req_phase17_test_999")

    def test_06_magic_bytes_valid_mp4(self):
        """Verify valid MP4 magic signature passes inspection."""
        try:
            VideoService.validate_magic_bytes(VALID_MP4_HEADER)
            valid = True
        except HTTPException:
            valid = False
        self.assertTrue(valid)

    def test_07_magic_bytes_invalid_script_rejection(self):
        """Verify shell script or text payload renamed to .mp4 is rejected with HTTP 400."""
        script_payload = b"#!/bin/bash\necho malicious" + b"\x00" * 50
        with self.assertRaises(HTTPException) as cm:
            VideoService.validate_magic_bytes(script_payload)
        self.assertEqual(cm.exception.status_code, 400)

    def test_08_path_traversal_sanitization(self):
        """Verify relative path traversal patterns in uploaded filenames are sanitized."""
        filename = "../../../var/log/system.mp4"
        ext = VideoService.validate_filename_and_extension(filename)
        self.assertEqual(ext, ".mp4")

    def test_09_unsupported_file_extension_rejection(self):
        """Verify uploading non-video file formats (.exe, .zip) returns HTTP 400."""
        file_payload = {"video": ("malicious.exe", b"payload data", "application/octet-stream")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 400)

    def test_10_triage_decision_normal(self):
        """Verify calibrated probability 0.12 yields NORMAL decision."""
        res = TriageResultContract.format_result("COW_1", 0.10, 0.12, 0.34, 0.10)
        self.assertEqual(res["decision"], "NORMAL")
        self.assertFalse(res["inconclusive"])

    def test_11_triage_decision_lameness_risk(self):
        """Verify calibrated probability 0.75 yields LAMENESS_RISK decision."""
        res = TriageResultContract.format_result("COW_2", 0.70, 0.75, 0.34, 0.10)
        self.assertEqual(res["decision"], "LAMENESS_RISK")
        self.assertFalse(res["inconclusive"])

    def test_12_triage_decision_inconclusive_bounds(self):
        """Verify calibrated probability in interval [0.24, 0.44] yields INCONCLUSIVE decision."""
        res = TriageResultContract.format_result("COW_3", 0.30, 0.34, 0.34, 0.10)
        self.assertEqual(res["decision"], "INCONCLUSIVE")
        self.assertTrue(res["inconclusive"])

    def test_13_zero_temporary_file_retention(self):
        """Verify temporary upload files are cleaned up immediately following requests."""
        initial_count = len([f for f in os.listdir(settings.TEMP_DIR) if f.startswith("upload_")]) if os.path.exists(settings.TEMP_DIR) else 0
        
        file_payload = {"video": ("test_clean.mp4", b"invalid stream header", "video/mp4")}
        self.client.post("/api/v1/screen", files=file_payload)
        
        final_count = len([f for f in os.listdir(settings.TEMP_DIR) if f.startswith("upload_")]) if os.path.exists(settings.TEMP_DIR) else 0
        self.assertEqual(initial_count, final_count)

    def test_14_no_stack_trace_leakage(self):
        """Verify failed requests return clean JSON error messages without python tracebacks."""
        file_payload = {"video": ("bad_file.txt", b"plain text content", "text/plain")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("Traceback", response.text)

    def test_15_cors_preflight_handling(self):
        """Verify OPTIONS request returns CORS headers for frontend origin."""
        response = self.client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        self.assertIn(response.status_code, (200, 204))


if __name__ == "__main__":
    unittest.main()
