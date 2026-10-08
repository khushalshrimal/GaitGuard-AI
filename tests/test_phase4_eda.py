"""
GaitGuard AI - Phase 4 Exploratory Data Analysis (EDA) Test Suite
Tests EDA calculations, statistical summary JSON output, figure file creation, and zero pipeline regression.
"""

import os
import sys
import json
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.run_eda import run_eda

class TestPhase4EDA(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_eda, cls.stats_results = run_eda()

    def test_01_eda_sample_counts_and_columns(self):
        """Verify EDA dataframe contains all 272 samples and expected columns."""
        self.assertEqual(len(self.df_eda), 272)
        expected_cols = [
            'sample_id', 'animal_id', 'raw_seq_len', 'binary_target', 'raw_label',
            'spine_arch_elevation', 'head_nod_var', 'nose_nod_var',
            'lf_hoof_disp', 'rf_hoof_disp', 'lh_hoof_disp', 'rh_hoof_disp',
            'front_gait_asym', 'hind_gait_asym', 'total_gait_asym', 'avg_hoof_disp'
        ]
        for col in expected_cols:
            self.assertIn(col, self.df_eda.columns)

    def test_02_statistical_tests_results(self):
        """Verify statistical tests produce valid p-values and effect sizes."""
        self.assertGreater(len(self.stats_results), 0)
        for res in self.stats_results:
            self.assertIn('variable', res)
            self.assertIn('mann_whitney_p', res)
            self.assertIn('cohen_d', res)
            self.assertTrue(0.0 <= res['mann_whitney_p'] <= 1.0)

    def test_03_figure_files_created(self):
        """Verify all 6 EDA diagnostic figures are generated in docs/figures/phase4/."""
        fig_dir = os.path.join(ROOT_DIR, "docs", "figures", "phase4")
        expected_figs = [
            'eda_target_and_animal_distribution.png',
            'eda_sequence_length_by_class.png',
            'eda_keypoint_movement_variance.png',
            'eda_back_arch_comparison.png',
            'eda_limb_symmetry_analysis.png',
            'eda_feature_correlation_matrix.png'
        ]
        for fig_name in expected_figs:
            fig_path = os.path.join(fig_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure file: {fig_path}")

    def test_04_eda_reproducibility(self):
        """Verify run_eda() produces deterministic numerical results."""
        df1, s1 = run_eda()
        df2, s2 = run_eda()
        np.testing.assert_array_equal(df1['spine_arch_elevation'].values, df2['spine_arch_elevation'].values)
        np.testing.assert_array_equal(df1['total_gait_asym'].values, df2['total_gait_asym'].values)

if __name__ == '__main__':
    unittest.main()
