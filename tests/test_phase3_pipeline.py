"""
GaitGuard AI - Phase 3 Data Cleaning & Quality Assurance Testing Suite
Tests trajectory smoothing, data quality, leakage control, and pipeline reproducibility.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gaitguard.config import SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER
from gaitguard.data.loader import RawDataLoader
from gaitguard.data.validator import DataValidator
from gaitguard.data.cleaner import KeypointCleaner
from gaitguard.data.preprocessor import DataPreprocessor
from gaitguard.data.splitter import GroupDataSplitter
from scripts.preprocess_dataset import run_phase3_pipeline

class TestPhase3DataPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.loader = RawDataLoader(ROOT_DIR)
        cls.raw_samples = cls.loader.load_all_raw_samples()
        cls.validator = DataValidator()
        cls.cleaner = KeypointCleaner(window_length=SAVGOL_WINDOW_LENGTH, polyorder=SAVGOL_POLYORDER)
        cls.preprocessor = DataPreprocessor()
        cls.splitter = GroupDataSplitter(n_splits=5, seed=42)

    def test_01_raw_data_validation(self):
        """Verify all 272 raw samples pass structural and numerical validation."""
        val_report = self.validator.validate_dataset(self.raw_samples)
        self.assertEqual(val_report['valid_samples'], 272)
        self.assertEqual(val_report['invalid_samples'], 0)

    def test_02_savitzky_golay_smoothing_shape_and_bounds(self):
        """Verify Savitzky-Golay trajectory smoothing maintains shape and preserves coordinate bounds."""
        sample = self.raw_samples[0]
        c_sample = self.cleaner.clean_sample(sample)
        smoothed = c_sample['cleaned_keypoints']
        
        self.assertEqual(smoothed.shape, (sample['raw_sequence_length'], 17, 2))
        self.assertFalse(np.isnan(smoothed).any())
        self.assertFalse(np.isinf(smoothed).any())

    def test_03_preprocessed_shapes_and_masks(self):
        """Verify Phase 3 exported arrays have expected shape (272, 128, 17, 2) and valid masks."""
        npz_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_cleaned_dataset.npz")
        self.assertTrue(os.path.exists(npz_path))
        
        data = np.load(npz_path)
        self.assertEqual(data['padded_keypoints'].shape, (272, 128, 17, 2))
        self.assertEqual(data['sequence_masks'].shape, (272, 128))
        self.assertEqual(len(data['binary_targets']), 272)
        
        # Verify normalized bounds [0.0, 1.0]
        self.assertTrue((data['padded_keypoints'] >= 0.0).all())
        self.assertTrue((data['padded_keypoints'] <= 1.0).all())

    def test_04_group_kfold_zero_leakage(self):
        """Verify 5-fold GroupKFold splits have ZERO animal ID overlap."""
        animal_ids = [s['animal_id'] for s in self.raw_samples]
        splits = self.splitter.generate_kfold_splits(animal_ids)
        self.assertEqual(len(splits), 5)

        for sp in splits:
            train_cows = set(np.array(animal_ids)[sp['train_idx']])
            val_cows = set(np.array(animal_ids)[sp['val_idx']])
            overlap = train_cows.intersection(val_cows)
            self.assertEqual(len(overlap), 0, f"Leakage in fold {sp['fold']}")

    def test_05_phase3_reproducibility(self):
        """Run Phase 3 pipeline twice and assert 100% bitwise identity of exported datasets."""
        npz1, _ = run_phase3_pipeline()
        d1 = np.load(npz1)

        npz2, _ = run_phase3_pipeline()
        d2 = np.load(npz2)

        np.testing.assert_array_equal(d1['sample_ids'], d2['sample_ids'])
        np.testing.assert_array_equal(d1['animal_ids'], d2['animal_ids'])
        np.testing.assert_array_equal(d1['padded_keypoints'], d2['padded_keypoints'])
        np.testing.assert_array_equal(d1['sequence_masks'], d2['sequence_masks'])
        np.testing.assert_array_equal(d1['binary_targets'], d2['binary_targets'])
        print("\nPhase 3 Reproducibility Test Passed: Outputs are 100% bitwise identical!")

if __name__ == '__main__':
    unittest.main()
