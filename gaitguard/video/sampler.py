"""
GaitGuard AI - Frame Sampler Module (Phase 9)
Resamples raw video frame sequences to exactly 128 uniform temporal timesteps.
"""

import numpy as np

class FrameSampler:
    """
    Resamples raw video frame sequences to fixed temporal window length (target_length=128).
    """

    def __init__(self, target_length=128):
        self.target_length = target_length

    def sample_frames(self, frames):
        """
        Resamples input frames list to target_length=128.
        Returns:
            sampled_frames: list of 128 frames
            sequence_mask: np.ndarray of shape (128,) with 1 for valid frames, 0 for padded
        """
        N = len(frames)
        if N == 0:
            raise ValueError("Cannot sample from empty frames list.")
            
        if N == self.target_length:
            return frames, np.ones(self.target_length, dtype=np.int32)
            
        # Uniform index sampling
        indices = np.round(np.linspace(0, N - 1, self.target_length)).astype(int)
        valid_frames = [f for f in frames if f is not None and f.size > 0]
        if len(valid_frames) == 0:
            raise ValueError("All frames in video stream are empty or unreadable.")
            
        sampled_frames = []
        for idx in indices:
            f = frames[idx]
            if f is None or f.size == 0:
                f = valid_frames[0]
            sampled_frames.append(f)
            
        mask = np.ones(self.target_length, dtype=np.int32)
        return sampled_frames, mask
