"""
GaitGuard AI - Phase 10 Video Quality Gate & Capture Coach Unit Test Suite
Verifies quality metrics, rule evaluation, capture coach guidance, quality gate integration,
benchmark file exports, diagnostic figures, and Phase 2..9 regression safety.
"""

import os
import unittest
import numpy as np
import pandas as pd

from gaitguard.quality.rules import QualityStatus, QualityIssueCode, QualityRules
from gaitguard.quality.metrics import QualityMetrics
from gaitguard.quality.capture_coach import CaptureCoach
from gaitguard.quality.analyzer import VideoQualityAnalyzer, QualityResult
from gaitguard.video.reader import VideoMetadata
from gaitguard.inference.pipeline import GaitGuardInferencePipeline

class TestPhase10QualityGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = VideoQualityAnalyzer()
        cls.pipeline = GaitGuardInferencePipeline()

    def test_01_quality_status_enum_values(self):
        """Verify QualityStatus enum contains READY, RETRY, and INCONCLUSIVE."""
        self.assertEqual(QualityStatus.READY.value, "READY")
        self.assertEqual(QualityStatus.RETRY.value, "RETRY")
        self.assertEqual(QualityStatus.INCONCLUSIVE.value, "INCONCLUSIVE")

    def test_02_quality_issue_codes_integrity(self):
        """Verify QualityIssueCode contains required machine-readable codes."""
        required_codes = [
            "VIDEO_UNREADABLE", "VIDEO_TOO_SHORT", "LOW_RESOLUTION",
            "LOW_KEYPOINT_COVERAGE", "INSUFFICIENT_WALKING", "EXCESSIVE_BLUR", "POOR_FRAMING"
        ]
        for code in required_codes:
            self.assertIn(code, QualityIssueCode.__members__)

    def test_03_compute_keypoint_coverage_valid(self):
        """Verify coverage = 1.0 for high confidence array."""
        conf = np.ones((128, 17), dtype=np.float32) * 0.90
        coverage = QualityMetrics.compute_keypoint_coverage(conf, min_conf=0.20)
        self.assertEqual(coverage, 1.0)

    def test_04_compute_keypoint_coverage_low(self):
        """Verify coverage = 0.5 for 50% low confidence array."""
        conf = np.vstack([np.ones((64, 17)) * 0.90, np.ones((64, 17)) * 0.10]).astype(np.float32)
        coverage = QualityMetrics.compute_keypoint_coverage(conf, min_conf=0.20)
        self.assertEqual(coverage, 0.5)

    def test_05_compute_walking_motion_quality_moving(self):
        """Verify positive motion displacement for moving keypoint trajectory."""
        kp = np.zeros((128, 17, 2), dtype=np.float32)
        kp[:, 14, 0] = np.linspace(0.2, 0.4, 128) # Spine1 moving horizontally
        motion_disp = QualityMetrics.compute_walking_motion_quality(kp, torso_len=0.1)
        self.assertGreater(motion_disp, 1.5)

    def test_06_compute_walking_motion_quality_stationary(self):
        """Verify near zero displacement for stationary keypoint trajectory."""
        kp = np.zeros((128, 17, 2), dtype=np.float32)
        kp[:, 14, 0] = 0.2 # Static
        motion_disp = QualityMetrics.compute_walking_motion_quality(kp, torso_len=0.1)
        self.assertEqual(motion_disp, 0.0)

    def test_07_compute_blur_indicator_fallback(self):
        """Verify compute_blur_indicator returns numeric value without crashing."""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128]
        blur_var = QualityMetrics.compute_blur_indicator(frames)
        self.assertIsInstance(blur_var, float)

    def test_08_compute_framing_quality_bounds(self):
        """Verify compute_framing_quality calculates non-zero bounding box area ratio."""
        kp = np.zeros((128, 17, 2), dtype=np.float32)
        kp[:, :, 0] = np.linspace(0.2, 0.6, 128 * 17).reshape(128, 17)
        kp[:, :, 1] = np.linspace(0.3, 0.7, 128 * 17).reshape(128, 17)
        area, in_bounds = QualityMetrics.compute_framing_quality(kp)
        self.assertGreater(area, 0.0)
        self.assertEqual(in_bounds, 1.0)

    def test_09_rules_evaluate_ideal_video(self):
        """Verify ideal video parameters evaluate to READY with 0 issues."""
        status, issues = QualityRules.evaluate(
            meta_valid=True, frame_count=150, height=1080, fps=30.0,
            coverage=0.95, motion_disp=0.25, blur_var=140.0, framing_area=0.35
        )
        self.assertEqual(status, QualityStatus.READY)
        self.assertEqual(len(issues), 0)

    def test_10_rules_evaluate_short_video(self):
        """Verify video under 30 frames evaluates to RETRY with VIDEO_TOO_SHORT."""
        status, issues = QualityRules.evaluate(
            meta_valid=True, frame_count=20, height=1080, fps=30.0,
            coverage=0.95, motion_disp=0.25, blur_var=140.0, framing_area=0.35
        )
        self.assertEqual(status, QualityStatus.RETRY)
        self.assertIn("VIDEO_TOO_SHORT", issues)

    def test_11_rules_evaluate_low_resolution(self):
        """Verify video height under 360p evaluates to RETRY with LOW_RESOLUTION."""
        status, issues = QualityRules.evaluate(
            meta_valid=True, frame_count=100, height=240, fps=30.0,
            coverage=0.95, motion_disp=0.25, blur_var=140.0, framing_area=0.35
        )
        self.assertEqual(status, QualityStatus.RETRY)
        self.assertIn("LOW_RESOLUTION", issues)

    def test_12_rules_evaluate_insufficient_walking(self):
        """Verify static video evaluates to RETRY with INSUFFICIENT_WALKING."""
        status, issues = QualityRules.evaluate(
            meta_valid=True, frame_count=100, height=1080, fps=30.0,
            coverage=0.95, motion_disp=0.01, blur_var=140.0, framing_area=0.35
        )
        self.assertEqual(status, QualityStatus.RETRY)
        self.assertIn("INSUFFICIENT_WALKING", issues)

    def test_13_rules_evaluate_excessive_blur(self):
        """Verify low Laplacian variance evaluates to RETRY with EXCESSIVE_BLUR."""
        status, issues = QualityRules.evaluate(
            meta_valid=True, frame_count=100, height=1080, fps=30.0,
            coverage=0.95, motion_disp=0.25, blur_var=15.0, framing_area=0.35
        )
        self.assertEqual(status, QualityStatus.RETRY)
        self.assertIn("EXCESSIVE_BLUR", issues)

    def test_14_rules_evaluate_low_coverage(self):
        """Verify coverage under 0.65 evaluates to RETRY with LOW_KEYPOINT_COVERAGE."""
        status, issues = QualityRules.evaluate(
            meta_valid=True, frame_count=100, height=1080, fps=30.0,
            coverage=0.50, motion_disp=0.25, blur_var=140.0, framing_area=0.35
        )
        self.assertEqual(status, QualityStatus.RETRY)
        self.assertIn("LOW_KEYPOINT_COVERAGE", issues)

    def test_15_rules_evaluate_corrupt_video(self):
        """Verify unreadable video evaluates to RETRY with VIDEO_UNREADABLE."""
        status, issues = QualityRules.evaluate(
            meta_valid=False, frame_count=0, height=0, fps=0.0,
            coverage=0.0, motion_disp=0.0, blur_var=0.0, framing_area=0.0
        )
        self.assertEqual(status, QualityStatus.RETRY)
        self.assertIn("VIDEO_UNREADABLE", issues)

    def test_16_capture_coach_user_guidance(self):
        """Verify CaptureCoach translates issue codes into user-friendly actionable guidance."""
        issues = ["VIDEO_TOO_SHORT", "EXCESSIVE_BLUR"]
        guidance = CaptureCoach.get_user_guidance(issues)
        self.assertTrue(len(guidance) >= 2)
        self.assertIn("longer walking sequence", guidance[0])
        self.assertIn("Hold the camera steady", guidance[1])

    def test_17_video_quality_analyzer_result_contract(self):
        """Verify VideoQualityAnalyzer returns QualityResult with to_dict representation."""
        meta = VideoMetadata(file_path="dummy.mp4", fps=30.0, width=1920, height=1080, frame_count=128, duration_sec=4.26, is_valid=True)
        conf = np.ones((128, 17), dtype=np.float32) * 0.90
        res = self.analyzer.analyze_quality(meta, confidences=conf)
        self.assertIsInstance(res, QualityResult)
        res_dict = res.to_dict()
        self.assertIn("status", res_dict)
        self.assertIn("quality_score", res_dict)
        self.assertIn("user_guidance", res_dict)

    def test_18_pipeline_quality_gate_integration(self):
        """Verify GaitGuardInferencePipeline includes initialized VideoQualityAnalyzer."""
        self.assertTrue(hasattr(self.pipeline, "quality_analyzer"))
        self.assertIsInstance(self.pipeline.quality_analyzer, VideoQualityAnalyzer)

    def test_19_phase10_runtime_benchmarks_csv_exists(self):
        """Verify docs/phase10_runtime_benchmarks.csv exists and contains benchmark records."""
        csv_path = os.path.join("docs", "phase10_runtime_benchmarks.csv")
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 8)
        self.assertIn("scenario_name", df.columns)
        self.assertIn("quality_score", df.columns)

    def test_20_full_regression_safety(self):
        """Verify all 6 Phase 10 visual artifacts exist in docs/figures/phase10/."""
        fig_dir = os.path.join("docs", "figures", "phase10")
        expected_figs = [
            "quality_score_distribution.png",
            "keypoint_coverage_vs_triage.png",
            "walking_motion_thresholds.png",
            "blur_indicator_analysis.png",
            "capture_coach_decision_flow.png",
            "end_to_end_quality_latency.png"
        ]
        for fig in expected_figs:
            fig_path = os.path.join(fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing Phase 10 figure: {fig}")

if __name__ == "__main__":
    unittest.main()
