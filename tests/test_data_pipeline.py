"""
GaitGuard AI - Automated Testing Suite (Phase 2)
Tests Loader, Validator, Preprocessor, Leakage Prevention, and Pipeline Reproducibility.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gaitguard.data.loader import RawDataLoader
from gaitguard.data.validator import DataValidator
from gaitguard.data.preprocessor import DataPreprocessor
from gaitguard.data.splitter import GroupDataSplitter
from scripts.prepare_dataset import run_pipeline

class TestGaitGuardDataPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.loader = RawDataLoader(ROOT_DIR)
        cls.raw_samples = cls.loader.load_all_raw_samples()
        cls.validator = DataValidator()
        cls.preprocessor = DataPreprocessor()
        cls.splitter = GroupDataSplitter(n_splits=5, seed=42)

    def test_01_loader_count_and_structure(self):
        """Verify loader retrieves all 272 raw samples with expected keys."""
        self.assertEqual(len(self.raw_samples), 272)
        sample = self.raw_samples[0]
        self.assertIn('sample_id', sample)
        self.assertIn('animal_id', sample)
        self.assertIn('raw_keypoints', sample)
        self.assertIn('raw_label', sample)
        self.assertEqual(sample['raw_keypoints'].shape[1], 17) # 17 keypoints
        self.assertEqual(sample['raw_keypoints'].shape[2], 3)  # x, y, likelihood

    def test_02_validation_logic(self):
        """Verify validator accepts valid samples and rejects malformed samples."""
        val_report = self.validator.validate_dataset(self.raw_samples)
        self.assertEqual(val_report['valid_samples'], 272)
        self.assertEqual(val_report['invalid_samples'], 0)

        # Test dummy invalid sample rejection
        invalid_sample = {
            'sample_id': '999',
            'animal_id': -5, # invalid ID
            'raw_sequence_length': 100,
            'raw_keypoints': np.ones((100, 10, 3)), # 10 keypoints instead of 17
            'raw_label': 9 # invalid label
        }
        ok, errs = self.validator.validate_sample(invalid_sample)
        self.assertFalse(ok)
        self.assertGreater(len(errs), 0)

    def test_03_preprocessing_shapes_and_bounds(self):
        """Verify preprocessor produces normalized coordinates in [0, 1] and padded shape (128, 17, 2)."""
        p_sample = self.preprocessor.process_sample(self.raw_samples[0])
        self.assertEqual(p_sample['padded_keypoints'].shape, (128, 17, 2))
        self.assertEqual(p_sample['sequence_mask'].shape, (128,))
        
        # Check normalized coordinate bounds [0.0, 1.0]
        norm_kp = p_sample['normalized_keypoints']
        self.assertTrue((norm_kp >= 0.0).all())
        self.assertTrue((norm_kp <= 1.0).all())

    def test_04_label_mapping(self):
        """Verify label mapping: Score 1 -> 0 (Normal), Scores 2..4 -> 1 (Lame Risk)."""
        self.assertEqual(self.preprocessor.map_label(1), 0)
        self.assertEqual(self.preprocessor.map_label(2), 1)
        self.assertEqual(self.preprocessor.map_label(3), 1)
        self.assertEqual(self.preprocessor.map_label(4), 1)

    def test_05_group_split_zero_leakage(self):
        """Verify 5-fold GroupKFold splits have ZERO animal ID overlap between train and val."""
        animal_ids = [s['animal_id'] for s in self.raw_samples]
        splits = self.splitter.generate_kfold_splits(animal_ids)
        self.assertEqual(len(splits), 5)

        for sp in splits:
            train_cows = set(np.array(animal_ids)[sp['train_idx']])
            val_cows = set(np.array(animal_ids)[sp['val_idx']])
            overlap = train_cows.intersection(val_cows)
            self.assertEqual(len(overlap), 0, f"Leakage detected in fold {sp['fold']}!")

    def test_06_pipeline_reproducibility(self):
        """Run pipeline twice and assert 100% bitwise identity of exported datasets."""
        npz_path1, json_path1 = run_pipeline()
        data1 = np.load(npz_path1)

        npz_path2, json_path2 = run_pipeline()
        data2 = np.load(npz_path2)

        np.testing.assert_array_equal(data1['sample_ids'], data2['sample_ids'])
        np.testing.assert_array_equal(data1['animal_ids'], data2['animal_ids'])
        np.testing.assert_array_equal(data1['padded_keypoints'], data2['padded_keypoints'])
        np.testing.assert_array_equal(data1['sequence_masks'], data2['sequence_masks'])
        np.testing.assert_array_equal(data1['binary_targets'], data2['binary_targets'])
        print("\nReproducibility Test Passed: Outputs are 100% bitwise identical!")

if __name__ == '__main__':
    unittest.main()
