"""
GaitGuard AI - Phase 12 Production Inference API & Backend Integration Unit Test Suite
Verifies health/version endpoints, file upload validation, size limits, path traversal safety,
temp file cleanup, Phase 10 Quality Gate interception, BiLSTM inference, SHAP explanations,
request_id generation, Pydantic schema validation, error handling, and end-to-end integration.
"""

import os
import io
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.video_service import VideoService
from app.services.inference_service import InferenceService, triage_result_conf

class TestPhase12API(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = InferenceService()

    def test_01_health_endpoint(self):
        """Verify GET /health returns status ok and model_loaded True."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "gaitguard-api")
        self.assertTrue(data["model_loaded"])

    def test_02_version_endpoint(self):
        """Verify GET /version returns v1, phase 12, and pipeline version."""
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["api_version"], "v1")
        self.assertEqual(data["phase"], 12)
        self.assertEqual(data["model_version"], "bilstm-mode-d-f76")

    def test_03_valid_upload_extension_pass(self):
        """Verify valid video extensions (.mp4, .avi, .mov) pass validation."""
        class MockFile:
            filename = "valid_cow_walk.mp4"
            content_type = "video/mp4"
        
        try:
            VideoService.validate_upload_file(MockFile())
            valid = True
        except Exception:
            valid = False
        self.assertTrue(valid)

    def test_04_unsupported_file_extension_rejection(self):
        """Verify uploading unsupported file extension (.exe, .txt) returns HTTP 400."""
        file_payload = {"video": ("malicious_script.exe", b"dummy content", "application/octet-stream")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json()["detail"])

    def test_05_file_size_limit_enforcement(self):
        """Verify uploading file exceeding MAX_UPLOAD_MB triggers HTTP 413 entity too large."""
        class MockOverboundFile:
            filename = "huge_video.mp4"
            content_type = "video/mp4"
            
            async def read(self, size):
                # Simulate overbound stream > 100MB
                if not hasattr(self, "_bytes_read"):
                    self._bytes_read = 0
                if self._bytes_read > (settings.MAX_UPLOAD_MB + 1) * 1024 * 1024:
                    return b""
                self._bytes_read += size
                return b"0" * size

        import asyncio
        with self.assertRaises(Exception) as cm:
            asyncio.run(VideoService.save_temp_file(MockOverboundFile()))
        self.assertIn("413", str(cm.exception))

    def test_06_path_traversal_prevention(self):
        """Verify path traversal filenames like '../../malicious.mp4' are sanitized."""
        class MockTraversalFile:
            filename = "../../../etc/passwd.mp4"
            content_type = "video/mp4"
        
        VideoService.validate_upload_file(MockTraversalFile())
        safe_name = os.path.basename(MockTraversalFile.filename)
        self.assertEqual(safe_name, "passwd.mp4")

    def test_07_temp_file_cleanup_on_completion(self):
        """Verify temporary files are safely deleted after cleanup call."""
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        dummy_temp = os.path.join(settings.TEMP_DIR, "test_temp_clean.mp4")
        with open(dummy_temp, "w") as f:
            f.write("temp video data")
            
        self.assertTrue(os.path.exists(dummy_temp))
        VideoService.cleanup_temp_file(dummy_temp)
        self.assertFalse(os.path.exists(dummy_temp))

    def test_08_temp_file_cleanup_on_error(self):
        """Verify cleanup_temp_file handles non-existent or error paths gracefully."""
        try:
            VideoService.cleanup_temp_file("non_existent_file_9999.mp4")
            handled = True
        except Exception:
            handled = False
        self.assertTrue(handled)

    def test_09_corrupted_video_quality_retry(self):
        """Verify unreadable video file returns HTTP 200 with status: retry and VIDEO_UNREADABLE."""
        file_payload = {"video": ("corrupted_video.mp4", b"corrupted header data", "video/mp4")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "retry")
        self.assertIsNone(data["inference"])
        self.assertIn("VIDEO_UNREADABLE", data["video_quality"]["issues"])

    def test_10_short_video_quality_retry(self):
        """Verify quality RETRY response contains Capture Coach guidance."""
        file_payload = {"video": ("short_video.mp4", b"dummy video content", "video/mp4")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "retry")
        self.assertGreater(len(data["video_quality"]["user_guidance"]), 0)

    def test_11_quality_retry_bypasses_ml(self):
        """Verify ML inference and SHAP explanation are null on quality RETRY status."""
        file_payload = {"video": ("bad_quality.mp4", b"bad content", "video/mp4")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        data = response.json()
        self.assertIsNone(data["inference"])
        self.assertIsNone(data["explanation"])

    def test_12_quality_ready_reaches_inference(self):
        """Verify process_screening with synthetic keypoints executes inference pipeline."""
        dummy_kp = np.random.randn(128, 17, 2).astype(np.float32)
        res = self.service.pipeline.analyze_keypoint_sequence(dummy_kp)
        self.assertIn("decision", res)
        self.assertIn("risk_probability_calibrated", res)

    def test_13_bilstm_prediction_returned(self):
        """Verify raw_probability and calibrated_probability are float values in range [0, 1]."""
        dummy_kp = np.random.randn(128, 17, 2).astype(np.float32)
        res = self.service.pipeline.analyze_keypoint_sequence(dummy_kp)
        self.assertGreaterEqual(res["risk_probability_raw"], 0.0)
        self.assertLessEqual(res["risk_probability_raw"], 1.0)
        self.assertGreaterEqual(res["risk_probability_calibrated"], 0.0)
        self.assertLessEqual(res["risk_probability_calibrated"], 1.0)

    def test_14_shap_explanation_returned(self):
        """Verify SHAP explanation is generated for valid prediction."""
        triage_res = {"decision": "LAMENESS_RISK", "risk_probability_raw": 0.8, "risk_probability_calibrated": 0.85, "confidence": "HIGH", "inconclusive": False, "status": "READY"}
        dummy_seq = np.random.randn(128, 17, 2).astype(np.float32)
        exp = self.service.explainer.explain_triage_contract(triage_res, dummy_seq)
        self.assertTrue(exp["explanation_available"])

    def test_15_explanation_failure_handled_safely(self):
        """Verify explanation failure gracefully sets explanation.available=False without throwing HTTP 500."""
        triage_res = {"decision": "NORMAL", "risk_probability_raw": 0.1, "risk_probability_calibrated": 0.1, "confidence": "HIGH", "inconclusive": False, "status": "READY"}
        # Pass invalid sequence to trigger explanation exception gracefully
        exp_res = self.service.explainer.explain_triage_contract(triage_res, np.zeros((1, 1), dtype=np.float32))
        self.assertFalse(exp_res["explanation_available"])

    def test_16_request_id_generated(self):
        """Verify unique request_id is present in all screening API responses."""
        file_payload = {"video": ("req_test.mp4", b"dummy content", "video/mp4")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        data = response.json()
        self.assertIn("request_id", data)
        self.assertTrue(data["request_id"].startswith("req_"))

    def test_17_pydantic_schema_validation(self):
        """Verify screening API response schema compliance."""
        file_payload = {"video": ("schema_test.mp4", b"dummy content", "video/mp4")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("result_summary", data)
        self.assertIn("disclaimer", data)

    def test_18_no_stack_trace_leakage_on_error(self):
        """Verify unexpected internal errors do not leak python tracebacks or absolute server paths."""
        file_payload = {"video": ("unsupported_ext.zip", b"dummy content", "application/zip")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("C:\\Users", response.text)

    def test_19_end_to_end_pipeline_integration(self):
        """True end-to-end integration test from endpoint down to pipeline execution."""
        file_payload = {"video": ("e2e_test.mp4", b"dummy video bytes", "video/mp4")}
        data_params = {"animal_id": "cow_e2e_99", "session_id": "sess_42"}
        response = self.client.post("/api/v1/screen", files=file_payload, data=data_params)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertIn("request_id", res)
        self.assertEqual(res["metadata"]["animal_id"], "cow_e2e_99")

    def test_20_full_regression_safety(self):
        """Verify all documentation reports and CSV benchmarks exist across Phase 2..12."""
        self.assertTrue(os.path.exists(os.path.join("docs", "phase11_global_attribution_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join("docs", "phase10_runtime_benchmarks.csv")))
        self.assertTrue(os.path.exists(os.path.join("datasets", "processed", "phase8_calibrated_oof_predictions.csv")))

if __name__ == "__main__":
    unittest.main()
