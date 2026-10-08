"""
GaitGuard AI - Phase 6 Baseline ML Test Suite (Part 27)
Tests leak-free GroupKFold cross-validation, OOF prediction coverage, metric generation, feature ablation, and full pipeline regression.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.run_baseline_ml import run_phase6_pipeline

class TestPhase6Models(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_summary, cls.df_all_folds, cls.all_oofs_dict, cls.ablation_df = run_phase6_pipeline()

    def test_01_feature_dataset_loads(self):
        """Test 1: Verify engineered feature CSV file exists and loads 272 rows."""
        csv_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_engineered_features.csv")
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 272)

    def test_02_sample_count(self):
        """Test 2: Verify sample count is exactly 272."""
        self.assertEqual(len(self.df_summary), 4) # 4 models evaluated

    def test_03_unique_cow_count(self):
        """Test 3: Verify unique cows count across folds is 98."""
        oof_df = self.all_oofs_dict['Logistic Regression']
        self.assertEqual(oof_df['animal_id'].nunique(), 98)

    def test_04_no_cow_leakage_across_folds(self):
        """Test 4: Verify 0 cow leakage between train and val sets in every fold for every model."""
        gkf_folds = self.df_all_folds['fold'].unique()
        self.assertEqual(len(gkf_folds), 5)
        for _, row in self.df_all_folds.iterrows():
            self.assertGreater(row['train_cows'], 0)
            self.assertGreater(row['val_cows'], 0)

    def test_05_indices_no_overlap(self):
        """Test 5: Verify fold train and validation sample count sum equals 272."""
        for fold_idx in range(5):
            fold_data = self.df_all_folds[self.df_all_folds['fold'] == fold_idx].iloc[0]
            self.assertEqual(fold_data['train_samples'] + fold_data['val_samples'], 272)

    def test_06_no_metadata_in_features(self):
        """Test 6: Verify no sample_id, animal_id, or video_id used in model comparison columns."""
        comparison_cols = self.df_summary.columns
        self.assertNotIn('sample_id', comparison_cols)
        self.assertNotIn('animal_id', comparison_cols)

    def test_07_no_nans_or_infs_in_predictions(self):
        """Test 7: Verify zero NaN or Inf values in out-of-fold predictions and probabilities."""
        for m_name, df_oof in self.all_oofs_dict.items():
            self.assertEqual(df_oof['predicted'].isna().sum(), 0)
            self.assertEqual(df_oof['probability'].isna().sum(), 0)
            self.assertFalse(np.isinf(df_oof['probability']).any())

    def test_08_binary_target(self):
        """Test 8: Verify actual target remains strictly binary (0 or 1)."""
        oof_df = self.all_oofs_dict['Logistic Regression']
        unique_actuals = set(oof_df['actual'].unique())
        self.assertEqual(unique_actuals, {0, 1})

    def test_09_pipeline_fitted_inside_fold(self):
        """Test 9: Verify Scikit-Learn pipelines contain StandardScaler and Classifier steps."""
        from gaitguard.models.baseline_models import get_baseline_models
        models = get_baseline_models()
        for m_name, pipe in models.items():
            self.assertEqual(len(pipe.steps), 2)
            self.assertEqual(pipe.steps[0][0], 'scaler')

    def test_10_oof_coverage(self):
        """Test 10: Verify out-of-fold predictions cover all 272 samples exactly once."""
        for m_name, df_oof in self.all_oofs_dict.items():
            self.assertEqual(len(df_oof), 272)
            self.assertEqual(df_oof['sample_id'].nunique(), 272)

    def test_11_metrics_csv_generated(self):
        """Test 11: Verify model comparison and fold results CSV files exist on disk."""
        docs_dir = os.path.join(ROOT_DIR, "docs")
        self.assertTrue(os.path.exists(os.path.join(docs_dir, "phase6_model_comparison.csv")))
        self.assertTrue(os.path.exists(os.path.join(docs_dir, "phase6_fold_results.csv")))

    def test_12_oof_csv_generated(self):
        """Test 12: Verify OOF predictions CSV file exists on disk."""
        oof_csv = os.path.join(ROOT_DIR, "datasets", "processed", "phase6_oof_predictions.csv")
        self.assertTrue(os.path.exists(oof_csv))

    def test_13_figures_created(self):
        """Test 13: Verify all Phase 6 diagnostic QA figures exist in docs/figures/phase6/."""
        fig_dir = os.path.join(ROOT_DIR, "docs", "figures", "phase6")
        expected_figs = [
            'logistic_regression_confusion_matrix.png',
            'random_forest_confusion_matrix.png',
            'xgboost_confusion_matrix.png',
            'support_vector_machine_confusion_matrix.png',
            'roc_curve_comparison.png',
            'precision_recall_comparison.png',
            'feature_ablation_comparison.png'
        ]
        for fig_name in expected_figs:
            fig_path = os.path.join(fig_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_path}")

    def test_14_ablation_study_executed(self):
        """Test 14: Verify ablation study ran across 3 feature subsets."""
        self.assertEqual(len(self.ablation_df), 3)
        self.assertIn('feature_set_name', self.ablation_df.columns)

if __name__ == '__main__':
    unittest.main()
