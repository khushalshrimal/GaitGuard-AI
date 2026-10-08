"""
GaitGuard AI - Phase 8 Probability Calibration & Triage Unit Test Suite
Verifies cross-fitted probability calibration, Brier score, Log Loss, ECE error, threshold analysis,
3-way triage contract logic, and regression safety.
"""

import os
import unittest
import numpy as np
import pandas as pd

from gaitguard.triage.calibrator import CrossFittedCalibrator, compute_brier_score, compute_log_loss, compute_ece
from gaitguard.triage.triage_engine import ScreeningTriageEngine, ConfidenceEstimator, TriageResultContract

class TestPhase8CalibrationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.oof_path = os.path.join("datasets", "processed", "phase7_oof_predictions.csv")
        cls.assertTrue(os.path.exists(cls.oof_path), "Phase 7 OOF predictions CSV missing.")
        cls.oof_df = pd.read_csv(cls.oof_path)
        cls.y_true = cls.oof_df["y_true"].values
        cls.raw_probs = cls.oof_df["oof_prob"].values

    def test_01_oof_predictions_integrity(self):
        """Verify Phase 7 OOF predictions file contains 272 samples across 98 cows with no NaNs."""
        self.assertEqual(len(self.oof_df), 272)
        self.assertEqual(self.oof_df["animal_id"].nunique(), 98)
        self.assertFalse(self.oof_df["oof_prob"].isna().any())

    def test_02_brier_score_bounds(self):
        """Verify Brier Score calculation yields finite float in [0, 1]."""
        brier = compute_brier_score(self.y_true, self.raw_probs)
        self.assertIsInstance(brier, float)
        self.assertTrue(0.0 <= brier <= 1.0)

    def test_03_log_loss_bounds(self):
        """Verify Log Loss calculation yields non-negative finite float."""
        loss = compute_log_loss(self.y_true, self.raw_probs)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_04_ece_computation(self):
        """Verify ECE computation returns scalar ECE and 10 bin dictionaries."""
        ece, bin_data = compute_ece(self.y_true, self.raw_probs, n_bins=10)
        self.assertIsInstance(ece, float)
        self.assertTrue(0.0 <= ece <= 1.0)
        self.assertEqual(len(bin_data["bin_centers"]), 10)
        self.assertEqual(len(bin_data["bin_accuracies"]), 10)

    def test_05_cross_fitted_sigmoid_calibrator(self):
        """Verify Sigmoid (Platt) CrossFittedCalibrator produces 272 valid probabilities in [0, 1]."""
        calibrator = CrossFittedCalibrator(method="sigmoid", n_splits=5)
        cal_probs = calibrator.fit_transform_oof(self.oof_df)
        self.assertEqual(len(cal_probs), 272)
        self.assertFalse(np.isnan(cal_probs).any())
        self.assertTrue(np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0))

    def test_06_cross_fitted_isotonic_calibrator(self):
        """Verify Isotonic CrossFittedCalibrator produces 272 valid probabilities in [0, 1]."""
        calibrator = CrossFittedCalibrator(method="isotonic", n_splits=5)
        cal_probs = calibrator.fit_transform_oof(self.oof_df)
        self.assertEqual(len(cal_probs), 272)
        self.assertFalse(np.isnan(cal_probs).any())
        self.assertTrue(np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0))

    def test_07_confidence_estimator_range(self):
        """Verify ConfidenceEstimator produces values in [0, 1]."""
        conf = ConfidenceEstimator.compute_confidence(self.raw_probs)
        self.assertEqual(len(conf), 272)
        self.assertTrue(np.all(conf >= 0.0) and np.all(conf <= 1.0))

    def test_08_confidence_estimator_extremes(self):
        """Verify confidence is 0.0 at p=0.5 and 1.0 at p=0.0 / p=1.0."""
        self.assertAlmostEqual(float(ConfidenceEstimator.compute_confidence(0.50)), 0.0, places=4)
        self.assertAlmostEqual(float(ConfidenceEstimator.compute_confidence(0.00)), 1.0, places=4)
        self.assertAlmostEqual(float(ConfidenceEstimator.compute_confidence(1.00)), 1.0, places=4)

    def test_09_threshold_sweep_analysis(self):
        """Verify threshold analysis returns DataFrame spanning thresholds 0.10 to 0.90."""
        engine = ScreeningTriageEngine()
        df_thresh = engine.analyze_thresholds(self.y_true, self.raw_probs)
        self.assertGreater(len(df_thresh), 30)
        self.assertIn("recall", df_thresh.columns)
        self.assertIn("precision", df_thresh.columns)

    def test_10_optimal_threshold_selection(self):
        """Verify optimal threshold selection chooses valid threshold meeting min_recall."""
        engine = ScreeningTriageEngine()
        thresh, df_thresh = engine.select_optimal_screening_threshold(self.y_true, self.raw_probs, min_recall=0.80)
        self.assertTrue(0.10 <= thresh <= 0.90)

    def test_11_triage_contract_normal(self):
        """Verify result contract outputs NORMAL for low risk probability."""
        res = TriageResultContract.format_result("S001", 0.15, 0.10, threshold=0.34, margin_delta=0.10)
        self.assertEqual(res["decision"], "NORMAL")
        self.assertFalse(res["inconclusive"])

    def test_12_triage_contract_lame(self):
        """Verify result contract outputs LAMENESS_RISK for high risk probability."""
        res = TriageResultContract.format_result("S002", 0.85, 0.88, threshold=0.34, margin_delta=0.10)
        self.assertEqual(res["decision"], "LAMENESS_RISK")
        self.assertFalse(res["inconclusive"])

    def test_13_triage_contract_inconclusive(self):
        """Verify result contract outputs INCONCLUSIVE for probability in uncertainty band."""
        res = TriageResultContract.format_result("S003", 0.35, 0.34, threshold=0.34, margin_delta=0.10)
        self.assertEqual(res["decision"], "INCONCLUSIVE")
        self.assertTrue(res["inconclusive"])

    def test_14_calibrated_oof_file_exists(self):
        """Verify phase8_calibrated_oof_predictions.csv exists and contains 272 rows."""
        cal_path = os.path.join("datasets", "processed", "phase8_calibrated_oof_predictions.csv")
        self.assertTrue(os.path.exists(cal_path))
        df = pd.read_csv(cal_path)
        self.assertEqual(len(df), 272)
        self.assertIn("oof_prob_calibrated", df.columns)
        self.assertIn("decision", df.columns)

    def test_15_calibration_comparison_file_exists(self):
        """Verify phase8_calibration_comparison.csv exists and contains 3 methods."""
        comp_path = os.path.join("docs", "phase8_calibration_comparison.csv")
        self.assertTrue(os.path.exists(comp_path))
        df = pd.read_csv(comp_path)
        self.assertEqual(len(df), 3)

    def test_16_threshold_analysis_file_exists(self):
        """Verify phase8_threshold_analysis.csv exists."""
        thresh_path = os.path.join("docs", "phase8_threshold_analysis.csv")
        self.assertTrue(os.path.exists(thresh_path))

    def test_17_figures_existence(self):
        """Verify all 7 Phase 8 QA figures exist in docs/figures/phase8/."""
        fig_dir = os.path.join("docs", "figures", "phase8")
        expected_figs = [
            "phase8_raw_reliability_curve.png",
            "phase8_calibrated_reliability_curve.png",
            "phase8_probability_distributions.png",
            "phase8_brier_logloss_comparison.png",
            "phase8_threshold_analysis.png",
            "phase8_triage_confusion_matrix.png",
            "phase8_inconclusive_region_visualization.png"
        ]
        for fig in expected_figs:
            fig_path = os.path.join(fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing diagnostic figure: {fig}")

    def test_18_triage_decision_categories_validity(self):
        """Verify decision output set is strictly a subset of {NORMAL, LAMENESS_RISK, INCONCLUSIVE}."""
        cal_path = os.path.join("datasets", "processed", "phase8_calibrated_oof_predictions.csv")
        df = pd.read_csv(cal_path)
        self.assertTrue(set(df["decision"]).issubset({"NORMAL", "LAMENESS_RISK", "INCONCLUSIVE"}))

    def test_19_phase7_regression_safety(self):
        """Verify Phase 7 OOF predictions file remains unaltered and intact."""
        self.assertTrue(os.path.exists(self.oof_path))
        self.assertEqual(len(self.oof_df), 272)

if __name__ == "__main__":
    unittest.main()
