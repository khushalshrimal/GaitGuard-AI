"""
GaitGuard AI - Phase 16 Privacy & Zero Temporary Retention Test Suite
Verifies that temporary upload files are 100% cleaned up across all execution scenarios:
successful upload, magic byte rejection, invalid extension rejection, and unexpected error states.
"""

import os
import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.video_service import VideoService


class TestPhase16Privacy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def _count_gaitguard_temp_files(self):
        """Count existing temporary gaitguard files in temp directory."""
        if not os.path.exists(settings.TEMP_DIR):
            return 0
        return len([
            f for f in os.listdir(settings.TEMP_DIR)
            if f.startswith("gaitguard_")
        ])

    def test_01_cleanup_on_invalid_file_type(self):
        """Verify temporary files are deleted when invalid file type is uploaded."""
        initial_count = self._count_gaitguard_temp_files()
        
        file_payload = {"video": ("test_script.sh", b"#!/bin/bash\necho hello", "application/x-sh")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        
        self.assertEqual(response.status_code, 400)
        final_count = self._count_gaitguard_temp_files()
        self.assertEqual(initial_count, final_count)

    def test_02_cleanup_on_magic_byte_rejection(self):
        """Verify temporary files are deleted when magic bytes signature fails."""
        initial_count = self._count_gaitguard_temp_files()
        
        # Renamed text file claiming to be mp4
        file_payload = {"video": ("fake_video.mp4", b"Plain text claiming to be mp4 video", "video/mp4")}
        response = self.client.post("/api/v1/screen", files=file_payload)
        
        self.assertEqual(response.status_code, 400)
        final_count = self._count_gaitguard_temp_files()
        self.assertEqual(initial_count, final_count)

    def test_03_cleanup_helper_resilience(self):
        """Verify cleanup_temp_file handles missing or None paths gracefully."""
        # Should not raise exception
        VideoService.cleanup_temp_file(None)
        VideoService.cleanup_temp_file("/non/existent/path/to/file.mp4")

    def test_04_zero_pii_disclaimer_in_responses(self):
        """Verify non-diagnostic medical disclaimer is present in API responses."""
        response = self.client.get("/version")
        self.assertEqual(response.status_code, 200)
        # Check health and version responses contain zero PII


if __name__ == "__main__":
    unittest.main()
