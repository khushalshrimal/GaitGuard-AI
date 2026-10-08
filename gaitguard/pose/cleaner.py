"""
GaitGuard AI - Keypoint Cleaner & Smoother Module (Phase 9)
Performs short-gap keypoint linear interpolation, confidence thresholding, and Savitzky-Golay trajectory smoothing.
"""

import numpy as np
from scipy.signal import savgol_filter
from gaitguard.config import SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER

class KeypointCleaner:
    """
    Cleans raw keypoint trajectories: interpolates missing keypoint coordinates and smooths motion jitter.
    """

    def __init__(self, window_length=SAVGOL_WINDOW_LENGTH, polyorder=SAVGOL_POLYORDER, min_confidence=0.20):
        self.window_length = window_length
        self.polyorder = polyorder
        self.min_confidence = min_confidence

    def interpolate_short_gaps(self, kp, conf, max_gap=10):
        """
        Interpolates low-confidence or missing keypoint coordinates for gaps <= max_gap.
        Inputs:
            kp: (T, 17, 2)
            conf: (T, 17)
        """
        T, K, C = kp.shape
        kp_cleaned = kp.copy()
        
        for k in range(K):
            valid_mask = conf[:, k] >= self.min_confidence
            valid_idx = np.where(valid_mask)[0]
            
            if len(valid_idx) == 0:
                continue # All low confidence, keep raw
                
            if len(valid_idx) < T:
                for c in range(C):
                    kp_cleaned[:, k, c] = np.interp(
                        np.arange(T),
                        valid_idx,
                        kp[valid_idx, k, c]
                    )
                    
        return kp_cleaned

    def smooth_trajectories(self, kp):
        """
        Applies Savitzky-Golay trajectory smoothing across frames dimension.
        """
        T, K, C = kp.shape
        if T < self.window_length:
            return kp
            
        kp_smoothed = np.zeros_like(kp)
        for k in range(K):
            for c in range(C):
                kp_smoothed[:, k, c] = savgol_filter(
                    kp[:, k, c],
                    window_length=self.window_length,
                    polyorder=self.polyorder
                )
                
        return kp_smoothed

    def clean_keypoints(self, keypoints, confidences=None):
        """
        Full cleaning pipeline: interpolation + Savitzky-Golay trajectory smoothing.
        """
        if confidences is None:
            confidences = np.ones((keypoints.shape[0], keypoints.shape[1]), dtype=np.float32)
            
        kp_interp = self.interpolate_short_gaps(keypoints, confidences)
        kp_smoothed = self.smooth_trajectories(kp_interp)
        
        return kp_smoothed.astype(np.float32)
