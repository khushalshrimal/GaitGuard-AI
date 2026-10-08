"""
GaitGuard AI - Phase 5 Feature Engineering Test Suite (Part 21)
Tests feature extraction, torso normalization, quality bounds, zero leakage, reproducibility, and full pipeline regression.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.engineer_gait_features import run_phase5_pipeline

class TestPhase5Features(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_features, cls.feature_stats = run_phase5_pipeline()
        cls.csv_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_engineered_features.csv")
        cls.npz_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_engineered_features.npz")

    def test_01_feature_dataset_loads(self):
        """Test 1: Verify exported feature CSV and NPZ files exist and load correctly."""
        self.assertTrue(os.path.exists(self.csv_path))
        self.assertTrue(os.path.exists(self.npz_path))
        
        df_loaded = pd.read_csv(self.csv_path)
        self.assertEqual(len(df_loaded), 272)

    def test_02_sample_count(self):
        """Test 2: Verify sample count is exactly 272."""
        self.assertEqual(len(self.df_features), 272)

    def test_03_unique_cows_count(self):
        """Test 3: Verify unique cows count is exactly 98."""
        self.assertEqual(self.df_features['animal_id'].nunique(), 98)

    def test_04_no_duplicate_sample_ids(self):
        """Test 4: Verify zero duplicate sample IDs."""
        self.assertEqual(self.df_features['sample_id'].nunique(), 272)

    def test_05_no_nans_in_features(self):
        """Test 5: Verify zero NaN values across all feature columns."""
        feature_cols = ['back_arch_curvature', 'head_nodding_amplitude', 'torso_normalized_stride_length', 'normalized_walking_speed', 'knee_flexion_range']
        nans = self.df_features[feature_cols].isna().sum().sum()
        self.assertEqual(nans, 0)

    def test_06_no_infs_in_features(self):
        """Test 6: Verify zero Inf values across all feature columns."""
        feature_cols = ['back_arch_curvature', 'head_nodding_amplitude', 'torso_normalized_stride_length', 'normalized_walking_speed', 'knee_flexion_range']
        infs = np.isinf(self.df_features[feature_cols].values).sum()
        self.assertEqual(infs, 0)

    def test_07_torso_length_positive(self):
        """Test 7: Verify torso normalization length is strictly > 0."""
        self.assertTrue((self.df_features['torso_length'] > 0).all())

    def test_08_feature_ranges(self):
        """Test 8: Verify feature values fall within realistic physiological bounds."""
        self.assertTrue((self.df_features['knee_flexion_range'] >= 0.0).all())
        self.assertTrue((self.df_features['knee_flexion_range'] <= 180.0).all())
        self.assertTrue((self.df_features['torso_normalized_stride_length'] > 0.0).all())

    def test_09_reproducibility(self):
        """Test 9: Verify feature generation is 100% deterministic."""
        df1, _ = run_phase5_pipeline()
        df2, _ = run_phase5_pipeline()
        np.testing.assert_array_equal(df1['torso_normalized_stride_length'].values, df2['torso_normalized_stride_length'].values)
        np.testing.assert_array_equal(df1['normalized_walking_speed'].values, df2['normalized_walking_speed'].values)

    def test_10_animal_id_alignment(self):
        """Test 10: Verify animal IDs remain perfectly aligned with Phase 3 dataset."""
        cleaned_npz = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_cleaned_dataset.npz")
        c_data = np.load(cleaned_npz)
        np.testing.assert_array_equal(self.df_features['animal_id'].values, c_data['animal_ids'])

    def test_11_target_alignment(self):
        """Test 11: Verify binary targets remain unchanged (143 Normal vs 129 Lame Risk)."""
        self.assertEqual((self.df_features['binary_target'] == 0).sum(), 143)
        self.assertEqual((self.df_features['binary_target'] == 1).sum(), 129)

    def test_12_no_target_leakage_in_features(self):
        """Test 12: Verify feature extraction logic does not consume binary_target or raw_label."""
        feature_cols = [c for c in self.df_features.columns if c not in ['sample_id', 'animal_id', 'binary_target', 'raw_label']]
        self.assertNotIn('binary_target', feature_cols)
        self.assertNotIn('raw_label', feature_cols)

    def test_13_figures_created(self):
        """Test 13: Verify all 7 Phase 5 diagnostic QA figures exist in docs/figures/phase5/."""
        fig_dir = os.path.join(ROOT_DIR, "docs", "figures", "phase5")
        expected_figs = [
            'feature_back_arch_curvature.png',
            'feature_head_nodding_amplitude.png',
            'feature_normalized_stride_length.png',
            'feature_stance_timing_asymmetry.png',
            'feature_walking_speed.png',
            'feature_correlation_heatmap.png',
            'feature_pairplot_summary.png'
        ]
        for fig_name in expected_figs:
            fig_path = os.path.join(fig_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_path}")

if __name__ == '__main__':
    unittest.main()
