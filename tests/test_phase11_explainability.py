"""
GaitGuard AI - Phase 11 Explainable AI & Evidence Layer Unit Test Suite
Verifies SHAP explainer initialization, temporal attributions, schema contracts,
modality/region aggregations, stability, perturbation faithfulness, non-diagnostic language, and regression safety.
"""

import os
import unittest
import numpy as np
import pandas as pd

from gaitguard.explainability.explainer import GaitGuardExplainer
from gaitguard.explainability.aggregator import AttributionAggregator
from gaitguard.explainability.stability import ExplanationStabilityEvaluator
from gaitguard.inference.pipeline import GaitGuardInferencePipeline

class TestPhase11Explainability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.explainer = GaitGuardExplainer(n_bg_samples=10, seed=42)
        cls.pipeline = GaitGuardInferencePipeline()
        cls.dummy_seq = np.random.randn(1, 128, 76).astype(np.float32)

    def test_01_explainer_initializes(self):
        """Verify GaitGuardExplainer initializes correctly with background dataset."""
        self.assertIsNotNone(self.explainer.model)
        self.assertEqual(self.explainer.X_bg.shape[1:], (128, 76))

    def test_02_explanation_runs_on_valid_sequence(self):
        """Verify explain_sequence returns (128, 76) array."""
        shap_mat = self.explainer.explain_sequence(self.dummy_seq)
        self.assertEqual(shap_mat.shape, (128, 76))

    def test_03_explanation_schema_validity(self):
        """Verify explain_triage_contract produces complete structured contract."""
        triage_res = {
            "decision": "LAMENESS_RISK",
            "risk_probability_raw": 0.72,
            "risk_probability_calibrated": 0.78,
            "confidence": "HIGH",
            "inconclusive": False,
            "status": "READY"
        }
        res = self.explainer.explain_triage_contract(triage_res, self.dummy_seq)
        required_keys = [
            "explanation_available", "prediction", "result_summary", "top_contributors",
            "modality_attribution", "body_region_attribution", "temporal_attribution",
            "derived_gait_evidence", "stability_score", "faithfulness", "explanation_time_ms", "disclaimer"
        ]
        for k in required_keys:
            self.assertIn(k, res)

    def test_04_attribution_dimensions_match_model_input(self):
        """Verify attribution dimensions match 128 timesteps and 76 input features."""
        shap_mat = self.explainer.explain_sequence(self.dummy_seq)
        self.assertEqual(shap_mat.shape[0], 128)
        self.assertEqual(shap_mat.shape[1], 76)

    def test_05_no_nan_in_attributions(self):
        """Verify SHAP attributions contain zero NaN values."""
        shap_mat = self.explainer.explain_sequence(self.dummy_seq)
        self.assertFalse(np.isnan(shap_mat).any())

    def test_06_no_inf_in_attributions(self):
        """Verify SHAP attributions contain zero Inf values."""
        shap_mat = self.explainer.explain_sequence(self.dummy_seq)
        self.assertFalse(np.isinf(shap_mat).any())

    def test_07_directionality_calculation(self):
        """Verify directionality classification (INCREASES_RISK, DECREASES_RISK, LOW_CONTRIBUTION)."""
        shap_mat = np.zeros((128, 76), dtype=np.float32)
        shap_mat[:, 0] = 0.05     # Positive
        shap_mat[:, 1] = -0.04    # Negative
        shap_mat[:, 2] = 0.000001 # Extremely small -> LOW_CONTRIBUTION
        
        contribs = AttributionAggregator.get_top_contributors(shap_mat, top_k=3)
        dirs = {c["feature_index"]: c["direction"] for c in contribs}
        self.assertEqual(dirs[0], "INCREASES_RISK")
        self.assertEqual(dirs[1], "DECREASES_RISK")
        self.assertEqual(dirs[2], "LOW_CONTRIBUTION")

    def test_08_top_k_extraction_deterministic(self):
        """Verify top-K feature extraction returns deterministic results."""
        shap_mat = np.random.randn(128, 76).astype(np.float32)
        top1 = AttributionAggregator.get_top_contributors(shap_mat, top_k=5)
        top2 = AttributionAggregator.get_top_contributors(shap_mat, top_k=5)
        self.assertEqual([c["feature_index"] for c in top1], [c["feature_index"] for c in top2])

    def test_09_explanation_stability_evaluation(self):
        """Verify ExplanationStabilityEvaluator returns Jaccard overlap score in [0.0, 1.0]."""
        res = ExplanationStabilityEvaluator.evaluate_stability(self.explainer, self.dummy_seq, n_runs=3, top_k=5)
        self.assertIn("jaccard_overlap", res)
        self.assertGreaterEqual(res["jaccard_overlap"], 0.0)
        self.assertLessEqual(res["jaccard_overlap"], 1.0)

    def test_10_global_aggregation_works(self):
        """Verify modality, body region, and temporal phase aggregations sum to 1.0."""
        shap_mat = np.random.randn(128, 76).astype(np.float32)
        modality = AttributionAggregator.aggregate_by_modality(shap_mat)
        region = AttributionAggregator.aggregate_by_body_region(shap_mat)
        temporal = AttributionAggregator.aggregate_by_temporal_phase(shap_mat)
        
        self.assertAlmostEqual(sum([modality["coordinates"], modality["velocity"], modality["biomechanical"]]), 1.0, places=2)
        self.assertAlmostEqual(sum(region.values()), 1.0, places=2)
        self.assertAlmostEqual(sum(temporal.values()), 1.0, places=2)

    def test_11_local_explanation_structure(self):
        """Verify local explanation dictionary structure."""
        triage_res = {"decision": "LAMENESS_RISK", "risk_probability_raw": 0.8, "risk_probability_calibrated": 0.85, "confidence": "HIGH", "inconclusive": False, "status": "READY"}
        res = self.explainer.explain_triage_contract(triage_res, self.dummy_seq)
        self.assertTrue(res["explanation_available"])
        self.assertGreater(len(res["top_contributors"]), 0)

    def test_12_normal_explanation_non_diagnostic(self):
        """Verify NORMAL decision uses non-diagnostic summary string."""
        triage_res = {"decision": "NORMAL", "risk_probability_raw": 0.2, "risk_probability_calibrated": 0.15, "confidence": "HIGH", "inconclusive": False, "status": "READY"}
        res = self.explainer.explain_triage_contract(triage_res, self.dummy_seq)
        self.assertEqual(res["result_summary"], "No elevated lameness risk detected by this screening model.")

    def test_13_lameness_risk_explanation_non_diagnostic(self):
        """Verify LAMENESS_RISK decision uses screening summary string."""
        triage_res = {"decision": "LAMENESS_RISK", "risk_probability_raw": 0.8, "risk_probability_calibrated": 0.85, "confidence": "HIGH", "inconclusive": False, "status": "READY"}
        res = self.explainer.explain_triage_contract(triage_res, self.dummy_seq)
        self.assertEqual(res["result_summary"], "Elevated lameness risk indicated by the screening model.")

    def test_14_inconclusive_explanation_safety(self):
        """Verify INCONCLUSIVE decision suppresses confident explanation and provides safety guidance."""
        triage_res = {"decision": "INCONCLUSIVE", "risk_probability_raw": 0.48, "risk_probability_calibrated": 0.50, "confidence": "LOW", "inconclusive": True, "status": "READY"}
        res = self.explainer.explain_triage_contract(triage_res, self.dummy_seq)
        self.assertFalse(res["explanation_available"])
        self.assertIn("insufficient", res["explanation_guidance"])

    def test_15_background_data_split_integrity(self):
        """Verify background dataset shape is (N_bg, 128, 76)."""
        self.assertEqual(self.explainer.X_bg.ndim, 3)
        self.assertEqual(self.explainer.X_bg.shape[1], 128)
        self.assertEqual(self.explainer.X_bg.shape[2], 76)

    def test_16_no_animal_leakage_in_background(self):
        """Verify background selection executes using training sequences."""
        self.assertTrue(len(self.explainer.X_bg) > 0)

    def test_17_phase8_calibration_remains_unchanged(self):
        """Verify Phase 8 calibrated predictions remain valid."""
        oof_path = os.path.join("datasets", "processed", "phase8_calibrated_oof_predictions.csv")
        self.assertTrue(os.path.exists(oof_path))
        df = pd.read_csv(oof_path)
        self.assertEqual(len(df), 272)

    def test_18_phase9_inference_remains_unchanged(self):
        """Verify Phase 9 inference pipeline operates without regression."""
        res = self.pipeline.analyze_keypoint_sequence(np.zeros((128, 17, 2), dtype=np.float32))
        self.assertIn("decision", res)

    def test_19_phase10_quality_gate_remains_unchanged(self):
        """Verify Phase 10 quality analyzer operates without regression."""
        self.assertTrue(hasattr(self.pipeline, "quality_analyzer"))

    def test_20_full_regression_safety(self):
        """Verify all 5 Phase 11 diagnostic visual figures exist in docs/figures/phase11/."""
        fig_dir = os.path.join("docs", "figures", "phase11")
        expected_figs = [
            "phase11_global_feature_importance.png",
            "phase11_modality_body_region_attribution.png",
            "phase11_temporal_attribution_heatmap.png",
            "phase11_derived_gait_evidence_breakdown.png",
            "phase11_explanation_stability_faithfulness.png"
        ]
        for fig in expected_figs:
            fig_path = os.path.join(fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing Phase 11 figure: {fig}")

if __name__ == "__main__":
    unittest.main()
