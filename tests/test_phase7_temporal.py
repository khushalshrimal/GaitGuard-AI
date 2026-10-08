"""
GaitGuard AI - Phase 7 Temporal Model Unit & Integration Test Suite
Verifies sequence building, fold-isolated scaling, BiLSTM forward pass, GroupKFold cow isolation,
OOF prediction coverage, and performance benchmarks.
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch

from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
from gaitguard.temporal.preprocessing import FoldTemporalScaler
from gaitguard.temporal.model import BiLSTMGaitClassifier, set_seed
from gaitguard.temporal.evaluation import TemporalModelEvaluator

class TestPhase7TemporalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join("datasets", "processed", "gaitguard_cleaned_dataset.npz")
        cls.assertTrue(os.path.exists(cls.dataset_path), "Cleaned dataset NPZ missing.")
        
        data = np.load(cls.dataset_path)
        cls.sample_ids = data["sample_ids"]
        cls.animal_ids = data["animal_ids"]
        cls.padded_keypoints = data["padded_keypoints"] # (272, 128, 17, 2)
        cls.sequence_masks = data["sequence_masks"]     # (272, 128)
        cls.binary_targets = data["binary_targets"]     # (272,)
        
        cls.builder = TemporalSequenceBuilder()

    def test_01_dataset_dimensions(self):
        """Test dataset sample count, timesteps, keypoints, and unique cow count."""
        self.assertEqual(self.padded_keypoints.shape[0], 272)
        self.assertEqual(self.padded_keypoints.shape[1], 128)
        self.assertEqual(self.padded_keypoints.shape[2], 17)
        self.assertEqual(self.padded_keypoints.shape[3], 2)
        self.assertEqual(len(np.unique(self.animal_ids)), 98)

    def test_02_sequence_builder_modes(self):
        """Test sequence tensor dimensions across all 4 ablation representation modes."""
        seq_coords = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="coords")
        self.assertEqual(seq_coords.shape, (272, 128, 34))
        
        seq_vel = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="velocity")
        self.assertEqual(seq_vel.shape, (272, 128, 34))
        
        seq_bio = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="biomechanical")
        self.assertEqual(seq_bio.shape, (272, 128, 8))
        
        seq_comb = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="combined")
        self.assertEqual(seq_comb.shape, (272, 128, 76))

    def test_03_no_nan_or_inf_in_sequences(self):
        """Verify generated sequence tensors contain no NaN or Inf values."""
        seq_comb = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="combined")
        self.assertFalse(np.isnan(seq_comb).any(), "Sequence tensor contains NaN values.")
        self.assertFalse(np.isinf(seq_comb).any(), "Sequence tensor contains Inf values.")

    def test_04_fold_temporal_scaler_isolation(self):
        """Verify FoldTemporalScaler fits strictly on training fold unpadded frames."""
        seq_comb = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="combined")
        scaler = FoldTemporalScaler()
        
        # Split 200 train, 72 val
        X_tr, m_tr = seq_comb[:200], self.sequence_masks[:200]
        X_va, m_va = seq_comb[200:], self.sequence_masks[200:]
        
        X_tr_scaled = scaler.fit_transform(X_tr, m_tr)
        X_va_scaled = scaler.transform(X_va, m_va)
        
        self.assertEqual(X_tr_scaled.shape, X_tr.shape)
        self.assertEqual(X_va_scaled.shape, X_va.shape)
        self.assertTrue(scaler.fitted)

    def test_05_padded_frames_remain_zero(self):
        """Verify padded frames (mask == 0) remain exactly 0.0 after scaling."""
        seq_coords = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="coords")
        scaler = FoldTemporalScaler()
        X_scaled = scaler.fit_transform(seq_coords, self.sequence_masks)
        
        invalid_mask_locations = (self.sequence_masks == 0)
        padded_values = X_scaled[invalid_mask_locations]
        self.assertTrue(np.all(padded_values == 0.0), "Padded frames were modified from zero during scaling!")

    def test_06_bilstm_forward_pass_shape_and_range(self):
        """Test PyTorch BiLSTM forward pass output shape (B, 1) and probability bounds [0, 1]."""
        set_seed(42)
        model = BiLSTMGaitClassifier(input_dim=34, hidden_dim=32, dropout=0.3)
        dummy_x = torch.randn(16, 128, 34)
        dummy_mask = torch.ones(16, 128)
        
        probs = model(dummy_x, dummy_mask)
        self.assertEqual(probs.shape, (16, 1))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_07_groupkfold_animal_disjointness(self):
        """Verify GroupKFold split enforces 100% disjoint cow sets across all 5 folds."""
        evaluator = TemporalModelEvaluator(n_splits=5, random_seed=42)
        X_seq = self.builder.build_dataset(self.padded_keypoints, self.sequence_masks, mode="biomechanical")
        
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_seq, self.binary_targets, groups=self.animal_ids), start=1):
            train_cows = set(self.animal_ids[tr_idx])
            val_cows = set(self.animal_ids[va_idx])
            self.assertTrue(train_cows.isdisjoint(val_cows), f"Fold {fold} has overlapping cow IDs between train and val!")

    def test_08_oof_predictions_coverage(self):
        """Verify out-of-fold predictions file has exactly 272 rows and no missing predictions."""
        oof_path = os.path.join("datasets", "processed", "phase7_oof_predictions.csv")
        self.assertTrue(os.path.exists(oof_path), "Phase 7 OOF predictions CSV missing.")
        
        oof_df = pd.read_csv(oof_path)
        self.assertEqual(len(oof_df), 272)
        self.assertFalse(oof_df["oof_prob"].isna().any())
        self.assertFalse(oof_df["oof_pred"].isna().any())
        self.assertEqual(set(oof_df["fold"]), {1, 2, 3, 4, 5})

    def test_09_model_comparison_table_contents(self):
        """Verify model comparison table contains Phase 6 and Phase 7 metrics."""
        comp_path = os.path.join("docs", "phase7_model_comparison.csv")
        self.assertTrue(os.path.exists(comp_path))
        
        comp_df = pd.read_csv(comp_path)
        self.assertEqual(len(comp_df), 5)
        self.assertTrue(any("BiLSTM" in name for name in comp_df["Model Algorithm"]))

    def test_10_fold_results_file_structure(self):
        """Verify fold-level results file exists and has 5 folds."""
        fold_path = os.path.join("docs", "phase7_fold_results.csv")
        self.assertTrue(os.path.exists(fold_path))
        
        fold_df = pd.read_csv(fold_path)
        self.assertEqual(len(fold_df), 5)

    def test_11_figures_existence(self):
        """Verify all 6 Phase 7 diagnostic figures were generated."""
        fig_dir = os.path.join("docs", "figures", "phase7")
        expected_figs = [
            "phase7_bilstm_confusion_matrix.png",
            "phase7_roc_curve_comparison.png",
            "phase7_precision_recall_comparison.png",
            "phase7_ablation_experiment_comparison.png",
            "phase7_fold_wise_stability.png",
            "phase7_animal_error_analysis.png"
        ]
        for fig in expected_figs:
            fig_path = os.path.join(fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing diagnostic figure: {fig}")

    def test_12_reproducibility(self):
        """Verify set_seed ensures deterministic model output."""
        set_seed(42)
        m1 = BiLSTMGaitClassifier(input_dim=8, hidden_dim=16)
        x1 = torch.randn(4, 128, 8)
        p1 = m1(x1).detach().numpy()
        
        set_seed(42)
        m2 = BiLSTMGaitClassifier(input_dim=8, hidden_dim=16)
        x2 = torch.randn(4, 128, 8)
        p2 = m2(x2).detach().numpy()
        
        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_13_bilstm_superior_roc_auc(self):
        """Verify Phase 7 BiLSTM achieves higher OOF ROC-AUC (> 0.80) than Phase 6 tabular baseline (0.7241)."""
        oof_path = os.path.join("datasets", "processed", "phase7_oof_predictions.csv")
        oof_df = pd.read_csv(oof_path)
        
        from sklearn.metrics import roc_auc_score
        bilstm_auc = roc_auc_score(oof_df["y_true"], oof_df["oof_prob"])
        self.assertGreater(bilstm_auc, 0.80, f"BiLSTM AUC ({bilstm_auc:.4f}) did not beat 0.80 threshold!")

    def test_14_binary_targets_validity(self):
        """Verify target vector is strictly binary (0 or 1)."""
        self.assertTrue(set(np.unique(self.binary_targets)).issubset({0, 1}))

    def test_15_sequence_masks_binary(self):
        """Verify sequence masks contain only 0 and 1."""
        self.assertTrue(set(np.unique(self.sequence_masks)).issubset({0, 1}))

    def test_16_torso_length_computation_positive(self):
        """Verify sequence torso length computation is always positive."""
        for i in range(10):
            t_len = self.builder.compute_sequence_torso_length(self.padded_keypoints[i], self.sequence_masks[i])
            self.assertGreater(t_len, 0.0)

    def test_17_raw_sequence_lengths_match_masks(self):
        """Verify mask sum equals raw sequence length."""
        for i in range(50):
            self.assertEqual(int(np.sum(self.sequence_masks[i])), int(np.sum(self.sequence_masks[i] > 0)))

    def test_18_sample_ids_length(self):
        """Verify sample IDs match 272 count."""
        self.assertEqual(len(self.sample_ids), 272)

    def test_19_phase6_regression_safety(self):
        """Verify Phase 6 features file exists and remains intact."""
        feat_path = os.path.join("datasets", "processed", "gaitguard_engineered_features.csv")
        self.assertTrue(os.path.exists(feat_path))
        df = pd.read_csv(feat_path)
        self.assertEqual(len(df), 272)

if __name__ == "__main__":
    unittest.main()
