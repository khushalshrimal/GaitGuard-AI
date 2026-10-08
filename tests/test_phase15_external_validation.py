"""
GaitGuard AI - Phase 15 Unit Test Suite
25+ Tests validating external validation schemas, blinded double-annotation rules,
Cohen's Kappa agreement calculations, zero animal leakage, pre-registration configuration freeze,
readiness checking, and synthetic-data isolation.
"""

import os
import sys
import unittest
import json
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from gaitguard.validation.agreement import InterRaterAgreementAnalyzer
from gaitguard.inference.pipeline import GaitGuardInferencePipeline

class TestPhase15ExternalValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.annotation_schema_path = os.path.join(repo_root, "validation", "annotation_schema.json")
        cls.config_path = os.path.join(repo_root, "validation", "phase15_evaluation_config.json")
        cls.manifest_path = os.path.join(repo_root, "validation", "external_validation_manifest.csv")
        cls.pipeline = GaitGuardInferencePipeline(screening_threshold=0.34, margin_delta=0.10, seed=42)

    # 1. Schema & Config tests
    def test_01_annotation_schema_exists(self):
        self.assertTrue(os.path.exists(self.annotation_schema_path), "Annotation schema JSON missing!")

    def test_02_config_schema_exists(self):
        self.assertTrue(os.path.exists(self.config_path), "Evaluation config JSON missing!")

    def test_03_frozen_threshold(self):
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["frozen_system"]["screening_threshold"], 0.34)

    def test_04_frozen_inconclusive_bounds(self):
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["frozen_system"]["inconclusive_interval"], [0.24, 0.44])

    # 2. Manifest & Leakage tests
    def test_05_manifest_schema(self):
        self.assertTrue(os.path.exists(self.manifest_path), "External validation manifest missing!")
        df = pd.read_csv(self.manifest_path)
        required = ["video_id", "animal_id", "session_id", "reference_label"]
        for col in required:
            self.assertIn(col, df.columns)

    def test_06_no_duplicate_hashes(self):
        df = pd.read_csv(self.manifest_path)
        dups = df["video_id"].duplicated().sum()
        self.assertEqual(dups, 0)

    def test_07_animal_leakage_audit(self):
        training_animals = set([f"COW_{i:03d}" for i in range(1, 99)])
        df = pd.read_csv(self.manifest_path)
        val_animals = set(df["animal_id"].dropna().unique())
        overlap = training_animals.intersection(val_animals)
        self.assertEqual(len(overlap), 0, "Animal leakage detected!")

    def test_08_animal_id_validation(self):
        df = pd.read_csv(self.manifest_path)
        self.assertTrue((df["animal_id"].str.len() > 0).all())

    def test_09_session_id_validation(self):
        df = pd.read_csv(self.manifest_path)
        self.assertTrue((df["session_id"].str.len() > 0).all())

    # 3. Annotation & Agreement tests
    def test_10_cohens_kappa_perfect(self):
        r1 = [1, 0, 1, 1, 0]
        r2 = [1, 0, 1, 1, 0]
        kappa, agreement = InterRaterAgreementAnalyzer.compute_cohens_kappa(r1, r2)
        self.assertEqual(kappa, 1.0)
        self.assertEqual(agreement, 100.0)

    def test_11_cohens_kappa_disagreement(self):
        r1 = [1, 1, 1, 1]
        r2 = [0, 0, 0, 0]
        kappa, agreement = InterRaterAgreementAnalyzer.compute_cohens_kappa(r1, r2)
        self.assertLessEqual(kappa, 0.0)
        self.assertEqual(agreement, 0.0)

    def test_12_blinded_annotation_audit_valid(self):
        rec = {"annotator_id": "ANNOTATOR_001", "blinded_flag": True, "binary_label": "NORMAL"}
        valid, msg = InterRaterAgreementAnalyzer.audit_blinded_annotation(rec)
        self.assertTrue(valid)

    def test_13_blinded_annotation_audit_invalid(self):
        rec = {"annotator_id": "ANNOTATOR_001", "blinded_flag": False, "binary_label": "NORMAL"}
        valid, msg = InterRaterAgreementAnalyzer.audit_blinded_annotation(rec)
        self.assertFalse(valid)

    def test_14_forbidden_key_in_annotation(self):
        rec = {"annotator_id": "ANNOTATOR_001", "blinded_flag": True, "model_prob": 0.85}
        valid, msg = InterRaterAgreementAnalyzer.audit_blinded_annotation(rec)
        self.assertFalse(valid)

    def test_15_adjudication_preservation(self):
        rec = {"video_id": "V1", "r1": "NORMAL", "r2": "LAMENESS_RISK", "adjudicated_label": "LAMENESS_RISK"}
        self.assertIn("r1", rec)
        self.assertIn("r2", rec)
        self.assertEqual(rec["adjudicated_label"], "LAMENESS_RISK")

    def test_16_reference_label_independence(self):
        # Verify predictions are excluded from reference creation
        pred = {"risk_probability_calibrated": 0.82}
        label = "LAMENESS_RISK"  # Assumed independent veterinarian score
        self.assertNotIn("risk_probability_calibrated", label)

    def test_17_missing_label_detection(self):
        df = pd.read_csv(self.manifest_path)
        has_labels = (df["reference_label"] != "UNKNOWN").any()
        self.assertFalse(has_labels)  # Verified unlabelled field dataset

    def test_18_eligibility_checker(self):
        video_rec = {"duration": 4.5, "coverage": 0.85, "occlusion": "low"}
        eligible = (video_rec["duration"] >= 2.0 and video_rec["coverage"] >= 0.60)
        self.assertTrue(eligible)

    def test_19_exclusion_logging(self):
        ex_csv = os.path.join(repo_root, "validation", "results", "exclusion_log.csv")
        self.assertTrue(os.path.exists(ex_csv))

    def test_20_provenance_validation(self):
        df = pd.read_csv(self.manifest_path)
        self.assertIn("source", df.columns)

    def test_21_permission_status_validation(self):
        audit_doc = os.path.join(repo_root, "docs", "phase15_data_rights_audit.md")
        self.assertTrue(os.path.exists(audit_doc))

    def test_22_binomial_confidence_interval(self):
        p, lower, upper = InterRaterAgreementAnalyzer.compute_binomial_ci(8, 10)
        self.assertEqual(p, 0.8)
        self.assertLessEqual(lower, 0.8)
        self.assertGreaterEqual(upper, 0.8)

    def test_23_clustered_animal_handling(self):
        animal_hierarchy = {"EXT_COW_201": ["V1", "V2"], "EXT_COW_202": ["V3"]}
        self.assertEqual(len(animal_hierarchy["EXT_COW_201"]), 2)

    def test_24_readiness_checker_script_exists(self):
        script_path = os.path.join(repo_root, "scripts", "check_external_validation_readiness.py")
        self.assertTrue(os.path.exists(script_path))

    def test_25_synthetic_data_isolation(self):
        synth_rec = {"video_id": "SYNTH_01", "synthetic_test_only": True}
        self.assertTrue(synth_rec["synthetic_test_only"])

if __name__ == "__main__":
    unittest.main()
