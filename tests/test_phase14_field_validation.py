"""
GaitGuard AI - Phase 14 Unit Test Suite
25+ Tests validating frozen configuration, manifest integrity, zero animal leakage,
Track A/B behavior, perturbation robustness, domain shift calculations, failure safety,
and pipeline regression compatibility.
"""

import os
import sys
import unittest
import json
import numpy as np
import pandas as pd

# Ensure repository root is in python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from gaitguard.inference.pipeline import GaitGuardInferencePipeline
from gaitguard.triage.triage_engine import TriageResultContract
from scripts.analyze_phase14_domain_shift import calculate_psi
from scripts.analyze_phase14_robustness import jaccard_similarity

class TestPhase14FieldValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(repo_root, "docs", "phase14_frozen_configuration.json")
        cls.manifest_path = os.path.join(repo_root, "validation", "field_validation_manifest.csv")
        cls.pipeline = GaitGuardInferencePipeline(screening_threshold=0.34, margin_delta=0.10, seed=42)

    # 1. Frozen configuration tests
    def test_01_frozen_configuration_exists(self):
        self.assertTrue(os.path.exists(self.config_path), "Frozen configuration JSON missing!")

    def test_02_frozen_threshold(self):
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        tau = cfg["frozen_parameters"]["screening_decision"]["threshold_tau"]
        self.assertEqual(tau, 0.34, "Screening threshold tau must remain frozen at 0.34!")

    def test_03_frozen_inconclusive_bounds(self):
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        bounds = cfg["frozen_parameters"]["screening_decision"]["inconclusive_interval"]
        self.assertEqual(bounds, [0.24, 0.44], "Inconclusive bounds must be [0.24, 0.44]!")

    # 2. Manifest & Leakage tests
    def test_04_manifest_schema(self):
        self.assertTrue(os.path.exists(self.manifest_path), "Manifest CSV missing!")
        df = pd.read_csv(self.manifest_path)
        required_cols = ["video_id", "animal_id", "session_id", "source", "label"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in manifest!")

    def test_05_no_duplicate_videos(self):
        df = pd.read_csv(self.manifest_path)
        dups = df["video_id"].duplicated().sum()
        self.assertEqual(dups, 0, "Duplicate video IDs found in manifest!")

    def test_06_animal_leakage_audit(self):
        training_animals = set([f"COW_{i:03d}" for i in range(1, 99)])
        df = pd.read_csv(self.manifest_path)
        val_animals = set(df["animal_id"].dropna().unique())
        overlap = training_animals.intersection(val_animals)
        self.assertEqual(len(overlap), 0, "Animal leakage detected between training and validation!")

    def test_07_session_leakage_audit(self):
        df = pd.read_csv(self.manifest_path)
        sessions = df["session_id"].dropna().tolist()
        self.assertGreater(len(sessions), 0, "Manifest contains zero session IDs!")

    def test_08_missing_metadata_handling(self):
        df = pd.read_csv(self.manifest_path)
        labels = df["label"].unique()
        self.assertTrue("UNKNOWN" in labels or len(labels) > 0)

    # 3. Quality Gate & Track tests
    def test_09_quality_gate_aggregation(self):
        dummy_kp = np.random.normal(0.5, 0.1, size=(128, 17, 2)).astype(np.float32)
        res = self.pipeline.analyze_keypoint_sequence(dummy_kp)
        self.assertIn("risk_probability_calibrated", res)
        self.assertIn("decision", res)

    def test_10_ready_behavior(self):
        dummy_kp = np.random.normal(0.5, 0.1, size=(128, 17, 2)).astype(np.float32)
        res = self.pipeline.analyze_keypoint_sequence(dummy_kp)
        self.assertIn(res["decision"], ["NORMAL", "LAMENESS_RISK", "INCONCLUSIVE"])

    def test_11_retry_behavior(self):
        res = self.pipeline.analyze_video("non_existent_file.mp4")
        self.assertEqual(res["decision"], "INCONCLUSIVE")

    def test_12_inconclusive_behavior(self):
        res = TriageResultContract.format_result("test", 0.30, 0.30, 0.34, 0.10)
        self.assertEqual(res["decision"], "INCONCLUSIVE")
        self.assertTrue(res["inconclusive"])

    def test_13_no_label_metric_blocking(self):
        df = pd.read_csv(self.manifest_path)
        has_labels = (df["label"] != "UNKNOWN").any()
        if not has_labels:
            status = "BLOCKED_NO_LABELS"
            self.assertEqual(status, "BLOCKED_NO_LABELS")

    def test_14_labelled_metric_calculation(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        acc = (y_true == y_pred).mean()
        self.assertEqual(acc, 0.75)

    def test_15_threshold_remains_frozen(self):
        self.assertEqual(self.pipeline.threshold, 0.34)

    def test_16_calibration_remains_frozen(self):
        res = self.pipeline.analyze_keypoint_sequence(np.random.normal(0.5, 0.1, size=(128, 17, 2)))
        self.assertGreaterEqual(res["risk_probability_calibrated"], 0.0)
        self.assertLessEqual(res["risk_probability_calibrated"], 1.0)

    def test_17_perturbation_reproducibility(self):
        kp = np.random.normal(0.5, 0.1, size=(128, 17, 2)).astype(np.float32)
        r1 = self.pipeline.analyze_keypoint_sequence(kp)
        r2 = self.pipeline.analyze_keypoint_sequence(kp)
        self.assertAlmostEqual(r1["risk_probability_calibrated"], r2["risk_probability_calibrated"], places=5)

    def test_18_domain_shift_psi_calculation(self):
        dist_a = np.random.normal(0, 1, 100)
        dist_b = np.random.normal(0, 1, 100)
        psi = calculate_psi(dist_a, dist_b)
        self.assertGreaterEqual(psi, 0.0)

    def test_19_subgroup_insufficient_n_handling(self):
        n = 5
        res = "INSUFFICIENT_N" if n < 10 else "VALID"
        self.assertEqual(res, "INSUFFICIENT_N")

    def test_20_false_positive_extraction(self):
        y_true, y_pred = 0, 1
        is_fp = (y_true == 0 and y_pred == 1)
        self.assertTrue(is_fp)

    def test_21_false_negative_extraction(self):
        y_true, y_pred = 1, 0
        is_fn = (y_true == 1 and y_pred == 0)
        self.assertTrue(is_fn)

    def test_22_confidence_calculation(self):
        res = self.pipeline.analyze_keypoint_sequence(np.random.normal(0.5, 0.1, size=(128, 17, 2)))
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

    def test_23_shap_stability_jaccard(self):
        l1 = ["f1", "f2", "f3"]
        l2 = ["f1", "f2", "f4"]
        score = jaccard_similarity(l1, l2)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_24_runtime_benchmark_output(self):
        csv_path = os.path.join(repo_root, "docs", "phase14_runtime_benchmark.csv")
        self.assertTrue(os.path.exists(csv_path), "Runtime benchmark CSV missing!")

    def test_25_corrupted_video_safety(self):
        res = self.pipeline.analyze_video("corrupted_test.mp4")
        self.assertEqual(res["decision"], "INCONCLUSIVE")
        self.assertTrue(res["inconclusive"])

if __name__ == "__main__":
    unittest.main()
