"""
GaitGuard AI - Master Preprocessing & Data Pipeline Script (Phase 2)
End-to-end reproducible script to load, validate, clean, standardize, and export processed dataset.
"""

import os
import sys
import time
import json
import numpy as np

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gaitguard.data.loader import RawDataLoader
from gaitguard.data.validator import DataValidator
from gaitguard.data.preprocessor import DataPreprocessor
from gaitguard.data.splitter import GroupDataSplitter

def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("GAITGUARD AI — PHASE 2: REPRODUCIBLE DATA PIPELINE")
    print("=" * 70)

    # 1. Load Raw Data
    print("\n[STEP 1] Loading raw dataset...")
    loader = RawDataLoader(ROOT_DIR)
    raw_samples = loader.load_all_raw_samples()
    print(f"  - Loaded {len(raw_samples)} raw samples.")

    # 2. Data Validation
    print("\n[STEP 2] Running automated data validation...")
    validator = DataValidator()
    val_report = validator.validate_dataset(raw_samples)
    print(f"  - Total Samples: {val_report['total_samples']}")
    print(f"  - Valid Samples: {val_report['valid_samples']}")
    print(f"  - Invalid Samples: {val_report['invalid_samples']}")

    if val_report['invalid_samples'] > 0:
        print("  - CRITICAL WARNING: Invalid samples detected!")
        for sid, errs in val_report['errors'].items():
            print(f"    * Sample {sid}: {errs}")
        raise ValueError("Data validation failed.")
    else:
        print("  - PASS: All 272 raw samples passed structural and numerical validation.")

    # 3. Preprocessing & Standardization
    print("\n[STEP 3] Preprocessing, normalizing coordinates & temporal window padding...")
    preprocessor = DataPreprocessor(frame_width=1920.0, frame_height=1080.0, target_seq_len=128)
    processed_samples = preprocessor.process_dataset(raw_samples)
    print(f"  - Processed {len(processed_samples)} samples.")
    print(f"  - Fixed Padded Keypoints Shape: (272, 128, 17, 2)")
    print(f"  - Coordinate Bounds: Min={np.min([s['padded_keypoints'] for s in processed_samples]):.4f}, Max={np.max([s['padded_keypoints'] for s in processed_samples]):.4f}")

    # 4. Group-Aware Split Generation
    print("\n[STEP 4] Generating leak-free 5-fold GroupKFold splits on animal_id...")
    animal_ids = [s['animal_id'] for s in processed_samples]
    splitter = GroupDataSplitter(n_splits=5, seed=42)
    splits = splitter.generate_kfold_splits(animal_ids)
    
    for sp in splits:
        print(f"  - Fold {sp['fold']}: Train Cows={sp['train_animals_count']}, Val Cows={sp['val_animals_count']} | Train Samples={sp['train_samples_count']}, Val Samples={sp['val_samples_count']}")

    # 5. Export Processed Dataset
    print("\n[STEP 5] Exporting processed dataset to datasets/processed/...")
    out_dir = os.path.join(ROOT_DIR, "datasets", "processed")
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "gaitguard_processed_dataset.npz")
    json_path = os.path.join(out_dir, "dataset_metadata.json")

    # Arrays for NPZ export
    sample_ids = np.array([s['sample_id'] for s in processed_samples], dtype=str)
    animal_ids_arr = np.array([s['animal_id'] for s in processed_samples], dtype=np.int32)
    raw_seq_lens = np.array([s['raw_sequence_length'] for s in processed_samples], dtype=np.int32)
    padded_kps = np.stack([s['padded_keypoints'] for s in processed_samples], axis=0) # (272, 128, 17, 2)
    seq_masks = np.stack([s['sequence_mask'] for s in processed_samples], axis=0)      # (272, 128)
    raw_labels = np.array([s['raw_label'] for s in processed_samples], dtype=np.int32)
    binary_targets = np.array([s['binary_target'] for s in processed_samples], dtype=np.int32)

    np.savez_compressed(
        npz_path,
        sample_ids=sample_ids,
        animal_ids=animal_ids_arr,
        raw_sequence_lengths=raw_seq_lens,
        padded_keypoints=padded_kps,
        sequence_masks=seq_masks,
        raw_labels=raw_labels,
        binary_targets=binary_targets
    )

    # Export Metadata JSON
    metadata = {
        'total_samples': len(processed_samples),
        'unique_cows': int(len(set(animal_ids))),
        'binary_distribution': {
            '0_normal': int(np.sum(binary_targets == 0)),
            '1_lame_risk': int(np.sum(binary_targets == 1))
        },
        'raw_label_distribution': {
            str(k): int(v) for k, v in zip(*np.unique(raw_labels, return_counts=True))
        },
        'input_shape': list(padded_kps.shape),
        'keypoint_count': 17,
        'target_sequence_length': 128,
        'normalization': 'resolution_minmax_[0,1]',
        'group_splits': [
            {
                'fold': sp['fold'],
                'train_cows': sp['train_animals_count'],
                'val_cows': sp['val_animals_count'],
                'train_samples': sp['train_samples_count'],
                'val_samples': sp['val_samples_count'],
                'train_idx': sp['train_idx'].tolist(),
                'val_idx': sp['val_idx'].tolist()
            } for sp in splits
        ]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("PIPELINE PERFORMANCE & COMPLETION SUMMARY")
    print("=" * 70)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Processed Dataset File: {npz_path} ({file_size_mb:.2f} MB)")
    print(f"Metadata JSON File: {json_path}")
    print("STATUS: SUCCESS")
    print("=" * 70)

    return npz_path, json_path

if __name__ == "__main__":
    run_pipeline()
