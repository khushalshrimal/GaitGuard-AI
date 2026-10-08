"""
GaitGuard AI - Data Validation Module (Phase 2)
Automated structural, numerical, metadata, and label verification.
"""

import numpy as np

class DataValidator:
    def __init__(self, expected_keypoint_count=17):
        self.expected_keypoint_count = expected_keypoint_count

    def validate_sample(self, sample):
        errors = []
        
        # 1. Structural Checks
        sample_id = sample.get('sample_id')
        if not sample_id:
            errors.append("Missing sample_id")

        kp = sample.get('raw_keypoints')
        if kp is None or not isinstance(kp, np.ndarray):
            errors.append("raw_keypoints is missing or not a numpy ndarray")
        else:
            if kp.ndim != 3:
                errors.append(f"Expected 3D keypoint array (T, 17, 3), got shape {kp.shape}")
            else:
                T, K, C = kp.shape
                if K != self.expected_keypoint_count:
                    errors.append(f"Expected {self.expected_keypoint_count} keypoints, got {K}")
                if C != 3:
                    errors.append(f"Expected 3 channels (x, y, likelihood), got {C}")
                if T <= 0:
                    errors.append(f"Invalid sequence length T={T}")
                if T != sample.get('raw_sequence_length'):
                    errors.append(f"Sequence length mismatch: Array T={T} vs metadata length={sample.get('raw_sequence_length')}")

        # 2. Numerical Sanity Checks
        if kp is not None and isinstance(kp, np.ndarray):
            if np.isnan(kp).any():
                errors.append("NaN values detected in keypoints")
            if np.isinf(kp).any():
                errors.append("Inf values detected in keypoints")
            if (kp[:, :, 0] < 0).any() or (kp[:, :, 1] < 0).any():
                errors.append("Negative coordinate values detected")

        # 3. Metadata Checks
        animal_id = sample.get('animal_id')
        if animal_id is None or animal_id <= 0:
            errors.append(f"Invalid animal_id: {animal_id}")

        # 4. Label Checks
        raw_label = sample.get('raw_label')
        if raw_label not in [1, 2, 3, 4]:
            errors.append(f"Invalid raw_label value: {raw_label} (Expected integer 1, 2, 3, or 4)")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_dataset(self, samples):
        total = len(samples)
        valid_count = 0
        invalid_count = 0
        all_errors = {}

        for s in samples:
            s_id = s.get('sample_id', 'UNKNOWN')
            ok, errs = self.validate_sample(s)
            if ok:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors[s_id] = errs

        report = {
            'total_samples': total,
            'valid_samples': valid_count,
            'invalid_samples': invalid_count,
            'errors': all_errors
        }
        return report
