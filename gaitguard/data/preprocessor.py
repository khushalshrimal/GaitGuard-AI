"""
GaitGuard AI - Data Preprocessor & Standardization Module (Phase 2)
Handles coordinate normalization, temporal window padding/masking, and binary label mapping.
"""

import numpy as np

FRAME_WIDTH = 1920.0
FRAME_HEIGHT = 1080.0
TARGET_SEQUENCE_LENGTH = 128

class DataPreprocessor:
    def __init__(self, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, target_seq_len=TARGET_SEQUENCE_LENGTH):
        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)
        self.target_seq_len = int(target_seq_len)

    def standardize_keypoints(self, raw_keypoints):
        """
        Extract (x, y) coordinates from (T, 17, 3) raw keypoints.
        Returns float32 array of shape (T, 17, 2).
        """
        xy_coords = raw_keypoints[:, :, :2].astype(np.float32)
        return xy_coords

    def normalize_coordinates(self, xy_coords):
        """
        Normalize pixel coordinates to [0, 1] relative to frame dimensions (1920x1080).
        Preserves spatial aspect ratio and relative joint distances.
        """
        norm_coords = np.zeros_like(xy_coords, dtype=np.float32)
        norm_coords[:, :, 0] = xy_coords[:, :, 0] / self.frame_width
        norm_coords[:, :, 1] = xy_coords[:, :, 1] / self.frame_height
        
        # Clip to ensure valid range [0.0, 1.0]
        norm_coords = np.clip(norm_coords, 0.0, 1.0)
        return norm_coords

    def pad_temporal_sequence(self, norm_coords):
        """
        Pads or clips a variable-length sequence (T, 17, 2) to a uniform window (target_seq_len, 17, 2).
        Generates a boolean sequence mask of shape (target_seq_len,) where True indicates valid frame data.
        """
        T, K, C = norm_coords.shape
        padded_kp = np.zeros((self.target_seq_len, K, C), dtype=np.float32)
        mask = np.zeros((self.target_seq_len,), dtype=bool)

        if T >= self.target_seq_len:
            # Truncate to target sequence length
            padded_kp[:self.target_seq_len, :, :] = norm_coords[:self.target_seq_len, :, :]
            mask[:self.target_seq_len] = True
        else:
            # Pad remaining frames with 0.0
            padded_kp[:T, :, :] = norm_coords
            mask[:T] = True

        return padded_kp, mask

    def map_label(self, raw_label):
        """
        Canonical binary label mapping:
        Score 1 (Normal) -> 0
        Scores 2, 3, 4 (Lame Risk) -> 1
        """
        if raw_label <= 1:
            return 0
        else:
            return 1

    def process_sample(self, raw_sample):
        raw_kp = raw_sample['raw_keypoints']
        std_kp = self.standardize_keypoints(raw_kp)
        norm_kp = self.normalize_coordinates(std_kp)
        padded_kp, mask = self.pad_temporal_sequence(norm_kp)
        binary_target = self.map_label(raw_sample['raw_label'])

        processed_sample = {
            'sample_id': raw_sample['sample_id'],
            'animal_id': int(raw_sample['animal_id']),
            'video_id': raw_sample['video_id'],
            'raw_sequence_length': int(raw_sample['raw_sequence_length']),
            'normalized_keypoints': norm_kp, # (T, 17, 2)
            'padded_keypoints': padded_kp,   # (128, 17, 2)
            'sequence_mask': mask,           # (128,)
            'raw_label': int(raw_sample['raw_label']),
            'binary_target': int(binary_target),
            'split_group': int(raw_sample['animal_id'])
        }
        return processed_sample

    def process_dataset(self, raw_samples):
        processed_samples = []
        for s in raw_samples:
            p_sample = self.process_sample(s)
            processed_samples.append(p_sample)
        return processed_samples
