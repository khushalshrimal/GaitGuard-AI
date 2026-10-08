"""
GaitGuard AI - Keypoint Normalizer Module (Phase 9)
Normalizes real-video keypoint coordinates into [0, 1] aspect-ratio bounds and computes sequence torso length.
"""

import numpy as np
from gaitguard.config import FRAME_WIDTH, FRAME_HEIGHT

class KeypointNormalizer:
    """
    Normalizes pixel coordinates (X, Y) to [0, 1] relative scale matching Phase 2/3 training preprocessing.
    """

    def __init__(self, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, eps=1e-8):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.eps = eps

    def normalize_coordinates(self, keypoints, width=None, height=None):
        """
        Normalizes pixel coordinates (X, Y) to [0, 1] relative scale.
        Input: keypoints of shape (T, 17, 2)
        Output: normalized_keypoints of shape (T, 17, 2)
        """
        w = width if width is not None and width > 0 else self.frame_width
        h = height if height is not None and height > 0 else self.frame_height
        
        kp_norm = keypoints.copy()
        
        # Check if coordinates are already in [0, 1] range
        if np.max(kp_norm) <= 1.5:
            return kp_norm.astype(np.float32)
            
        kp_norm[:, :, 0] = np.clip(kp_norm[:, :, 0] / float(w), 0.0, 1.0)
        kp_norm[:, :, 1] = np.clip(kp_norm[:, :, 1] / float(h), 0.0, 1.0)
        
        return kp_norm.astype(np.float32)

    def compute_torso_length(self, keypoints):
        """
        Computes robust torso length: median distance between Spine1 (14) and Spine3 (16).
        """
        spine1 = keypoints[:, 14, :] # (T, 2)
        spine3 = keypoints[:, 16, :] # (T, 2)
        dist = np.sqrt(np.sum((spine1 - spine3)**2, axis=1))
        torso_len = float(np.median(dist))
        return max(torso_len, 0.1)
