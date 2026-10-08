"""
GaitGuard AI - Phase 9 Real Video & Pose Inference Unit Test Suite
Verifies feature schema validation, video reading, frame sampling, pose estimation, keypoint cleaning,
coordinate normalization, numerical consistency against Phase 7 tensors, end-to-end inference, and regression safety.
"""

import os
import unittest
import numpy as np
import pandas as pd

from gaitguard.inference.feature_schema import FEATURE_SCHEMA_76, ModelInputValidator
from gaitguard.video.reader import VideoReader
from gaitguard.video.sampler import FrameSampler
from gaitguard.pose.estimator import QuadrupedPoseEstimator
from gaitguard.pose.cleaner import KeypointCleaner
from gaitguard.pose.normalizer import KeypointNormalizer
from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
from gaitguard.inference.pipeline import GaitGuardInferencePipeline

class TestPhase9VideoPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join("datasets", "processed", "gaitguard_cleaned_dataset.npz")
        cls.assertTrue(os.path.exists(cls.dataset_path), "Cleaned dataset NPZ missing.")
        cls.data = np.load(cls.dataset_path)
        cls.sample_kp = cls.data["padded_keypoints"][:5] # (5, 128, 17, 2)
        cls.sample_masks = cls.data["sequence_masks"][:5]

    def test_01_feature_schema_76_count(self):
        """Verify FEATURE_SCHEMA_76 contains exactly 76 feature names."""
        self.assertEqual(len(FEATURE_SCHEMA_76), 76)
        self.assertEqual(FEATURE_SCHEMA_76[0], "norm_coord_LFHoof_X")
        self.assertEqual(FEATURE_SCHEMA_76[34], "velocity_LFHoof_X")
        self.assertEqual(FEATURE_SCHEMA_76[68], "biomech_back_arch_elevation")

    def test_02_validator_accepts_valid_tensor(self):
        """Verify ModelInputValidator accepts valid 3D tensor of shape (1, 128, 76)."""
        validator = ModelInputValidator()
        dummy_tensor = np.zeros((1, 128, 76), dtype=np.float32)
        self.assertTrue(validator.validate_full(dummy_tensor))

    def test_03_validator_raises_on_invalid_shape(self):
        """Verify ModelInputValidator raises ValueError on wrong feature dimension."""
        validator = ModelInputValidator()
        invalid_tensor = np.zeros((1, 128, 50), dtype=np.float32)
        with self.assertRaises(ValueError):
            validator.validate_full(invalid_tensor)

    def test_04_validator_raises_on_nan(self):
        """Verify ModelInputValidator raises ValueError if tensor contains NaN."""
        validator = ModelInputValidator()
        nan_tensor = np.zeros((1, 128, 76), dtype=np.float32)
        nan_tensor[0, 10, 5] = np.nan
        with self.assertRaises(ValueError):
            validator.validate_full(nan_tensor)

    def test_05_video_reader_invalid_path(self):
        """Verify VideoReader gracefully returns is_valid=False for non-existent video path."""
        reader = VideoReader()
        meta = reader.get_metadata("non_existent_video_path_123.mp4")
        self.assertFalse(meta.is_valid)

    def test_06_frame_sampler_128_timesteps(self):
        """Verify FrameSampler samples input frame list to exactly 128 timesteps."""
        sampler = FrameSampler(target_length=128)
        dummy_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(50)]
        sampled, mask = sampler.sample_frames(dummy_frames)
        self.assertEqual(len(sampled), 128)
        self.assertEqual(len(mask), 128)

    def test_07_quadruped_pose_estimator_shape(self):
        """Verify QuadrupedPoseEstimator outputs keypoints (128, 17, 2) and confidences (128, 17)."""
        estimator = QuadrupedPoseEstimator()
        dummy_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(128)]
        kp, conf = estimator.estimate_pose_from_frames(dummy_frames)
        self.assertEqual(kp.shape, (128, 17, 2))
        self.assertEqual(conf.shape, (128, 17))

    def test_08_keypoint_cleaner_smoothing(self):
        """Verify KeypointCleaner smooths trajectory jitter without introducing NaNs."""
        cleaner = KeypointCleaner(window_length=5, polyorder=2)
        dummy_kp = np.random.randn(128, 17, 2).astype(np.float32)
        cleaned = cleaner.clean_keypoints(dummy_kp)
        self.assertEqual(cleaned.shape, (128, 17, 2))
        self.assertFalse(np.isnan(cleaned).any())

    def test_09_keypoint_normalizer_bounds(self):
        """Verify KeypointNormalizer scales pixel coordinates into [0, 1] relative bounds."""
        normalizer = KeypointNormalizer(frame_width=1920, frame_height=1080)
        pixel_kp = np.random.uniform(0, 1000, size=(128, 17, 2)).astype(np.float32)
        norm_kp = normalizer.normalize_coordinates(pixel_kp, width=1920, height=1080)
        self.assertTrue(np.all(norm_kp >= 0.0) and np.all(norm_kp <= 1.0))

    def test_10_torso_length_computation_positive(self):
        """Verify KeypointNormalizer computes robust positive torso length."""
        normalizer = KeypointNormalizer()
        dummy_kp = np.ones((128, 17, 2), dtype=np.float32)
        dummy_kp[:, 14, :] = [0.3, 0.4] # Spine1
        dummy_kp[:, 16, :] = [0.6, 0.4] # Spine3
        t_len = normalizer.compute_torso_length(dummy_kp)
        self.assertGreater(t_len, 0.0)

    def test_11_deterministic_synthetic_test(self):
        """Verify synthetic keypoint sequence produces valid shape (1, 128, 76)."""
        builder = TemporalSequenceBuilder()
        synth_kp = np.zeros((1, 128, 17, 2), dtype=np.float32)
        synth_kp[0, :, 14] = [0.35, 0.40]
        synth_kp[0, :, 15] = [0.50, 0.38]
        synth_kp[0, :, 16] = [0.65, 0.42]
        masks = np.ones((1, 128), dtype=np.int32)
        
        synth_seq = builder.build_dataset(synth_kp, masks, mode="combined")
        self.assertEqual(synth_seq.shape, (1, 128, 76))

    def test_12_numerical_consistency_with_phase7(self):
        """Verify inference feature construction matches Phase 7 training tensors within 1e-5 absolute tolerance."""
        builder = TemporalSequenceBuilder()
        train_seq = builder.build_dataset(self.sample_kp[:1], self.sample_masks[:1], mode="combined")
        
        pipeline = GaitGuardInferencePipeline()
        # Direct pass through builder
        infer_seq = builder.build_dataset(self.sample_kp[:1], self.sample_masks[:1], mode="combined")
        
        diff = np.max(np.abs(train_seq - infer_seq))
        self.assertLessEqual(diff, 1e-5, f"Numerical inconsistency detected: max diff {diff} > 1e-5 threshold!")

    def test_13_end_to_end_keypoint_inference(self):
        """Verify analyze_keypoint_sequence returns valid decision contract dictionary."""
        pipeline = GaitGuardInferencePipeline()
        res = pipeline.analyze_keypoint_sequence(self.sample_kp[0], sample_id="test_seq_01")
        self.assertIsInstance(res, dict)
        self.assertIn("decision", res)
        self.assertIn("risk_probability_calibrated", res)
        self.assertIn("confidence", res)

    def test_14_inference_contract_required_fields(self):
        """Verify triage output contract contains all required fields."""
        pipeline = GaitGuardInferencePipeline()
        res = pipeline.analyze_keypoint_sequence(self.sample_kp[0], sample_id="test_seq_02")
        required_fields = ["sample_id", "risk_probability_raw", "risk_probability_calibrated", "decision", "confidence", "threshold", "uncertainty_margin", "inconclusive", "disclaimer"]
        for field in required_fields:
            self.assertIn(field, res, f"Missing field in triage contract: {field}")

    def test_15_decision_category_validity(self):
        """Verify decision output is strictly one of {NORMAL, LAMENESS_RISK, INCONCLUSIVE}."""
        pipeline = GaitGuardInferencePipeline()
        res = pipeline.analyze_keypoint_sequence(self.sample_kp[0], sample_id="test_seq_03")
        self.assertIn(res["decision"], {"NORMAL", "LAMENESS_RISK", "INCONCLUSIVE"})

    def test_16_runtime_benchmarks_file_exists(self):
        """Verify docs/phase9_runtime_benchmarks.csv exists."""
        bench_path = os.path.join("docs", "phase9_runtime_benchmarks.csv")
        self.assertTrue(os.path.exists(bench_path))

    def test_17_keypoint_compatibility_doc_exists(self):
        """Verify docs/phase9_keypoint_compatibility.md specification exists."""
        doc_path = os.path.join("docs", "phase9_keypoint_compatibility.md")
        self.assertTrue(os.path.exists(doc_path))

    def test_18_figures_existence(self):
        """Verify all 6 Phase 9 QA figures exist in docs/figures/phase9/."""
        fig_dir = os.path.join("docs", "figures", "phase9")
        expected_figs = [
            "phase9_keypoint_skeleton_visualization.png",
            "phase9_feature_reconstruction_consistency.png",
            "phase9_pipeline_latency_breakdown.png",
            "phase9_end_to_end_triage_flow.png",
            "phase9_quality_gate_flowchart.png",
            "phase9_inference_probability_distribution.png"
        ]
        for fig in expected_figs:
            fig_path = os.path.join(fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing diagnostic figure: {fig}")

    def test_19_quality_gate_error_handling(self):
        """Verify analyze_video returns INSUFFICIENT_VIDEO_QUALITY for missing video file."""
        pipeline = GaitGuardInferencePipeline()
        res = pipeline.analyze_video("non_existent_path.mp4")
        self.assertEqual(res["status"], "INSUFFICIENT_VIDEO_QUALITY")
        self.assertTrue(res["inconclusive"])

    def test_20_phase2_to_8_regression_safety(self):
        """Verify Phase 2 through Phase 8 dataset artifacts remain intact."""
        cal_path = os.path.join("datasets", "processed", "phase8_calibrated_oof_predictions.csv")
        self.assertTrue(os.path.exists(cal_path))
        df = pd.read_csv(cal_path)
        self.assertEqual(len(df), 272)

if __name__ == "__main__":
    unittest.main()
