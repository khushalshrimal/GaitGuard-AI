"""
GaitGuard AI - Keypoint Trajectory Cleaner & Smoother Module (Phase 3)
Applies Savitzky-Golay filtering to remove high-frequency tracking jitter while preserving physical gait signals.
"""

import numpy as np
from scipy.signal import savgol_filter
from gaitguard.config import SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER

class KeypointCleaner:
    def __init__(self, window_length=SAVGOL_WINDOW_LENGTH, polyorder=SAVGOL_POLYORDER):
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        # Ensure window_length is odd
        if self.window_length % 2 == 0:
            self.window_length += 1

    def smooth_trajectory(self, xy_coords):
        """
        Applies Savitzky-Golay 1D filtering across time frames for each keypoint and coordinate axis.
        Input: xy_coords of shape (T, 17, 2)
        Output: smoothed xy_coords of shape (T, 17, 2)
        """
        T, K, C = xy_coords.shape
        smoothed = np.copy(xy_coords).astype(np.float32)

        # If sequence is shorter than window_length, adjust window_length
        effective_window = self.window_length
        if T < effective_window:
            effective_window = T if T % 2 != 0 else T - 1

        # If effective_window <= polyorder, fall back to un-smoothed coordinates
        if effective_window <= self.polyorder:
            return smoothed

        for k in range(K):
            for c in range(C):
                trajectory = xy_coords[:, k, c]
                smoothed_traj = savgol_filter(
                    trajectory,
                    window_length=effective_window,
                    polyorder=self.polyorder,
                    mode='nearest'
                )
                smoothed[:, k, c] = smoothed_traj

        return smoothed

    def clean_sample(self, raw_sample):
        """
        Cleans and smooths keypoint coordinates for a single sample.
        """
        raw_kp = raw_sample['raw_keypoints'] # (T, 17, 3)
        xy_coords = raw_kp[:, :, :2].astype(np.float32)
        
        # Apply Savitzky-Golay trajectory smoothing
        smoothed_xy = self.smooth_trajectory(xy_coords)

        cleaned_sample = dict(raw_sample)
        cleaned_sample['cleaned_keypoints'] = smoothed_xy # (T, 17, 2)
        return cleaned_sample

    def clean_dataset(self, raw_samples):
        cleaned_samples = []
        for s in raw_samples:
            c_sample = self.clean_sample(s)
            cleaned_samples.append(c_sample)
        return cleaned_samples
