"""
GaitGuard AI - Temporal Sequence Builder (Phase 7)
Constructs sequence-level temporal feature tensors (N, T, F) from keypoint trajectories.
"""

import numpy as np

class TemporalSequenceBuilder:
    """
    Builds temporal feature sequence tensors from raw keypoint trajectories (N, T, 17, 2).
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute_sequence_torso_length(self, sample_kp, sample_mask):
        """
        Computes robust torso length for a single sequence: median distance between Spine1 (14) and Spine3 (16).
        """
        valid_idx = np.where(sample_mask > 0)[0]
        if len(valid_idx) == 0:
            return 0.1
        valid_kp = sample_kp[valid_idx] # (valid_T, 17, 2)
        spine1 = valid_kp[:, 14, :]
        spine3 = valid_kp[:, 16, :]
        dist = np.sqrt(np.sum((spine1 - spine3)**2, axis=1))
        torso_len = float(np.median(dist))
        return max(torso_len, 0.1)

    def extract_normalized_coords(self, sample_kp, sample_mask, torso_len):
        """
        Extracts frame-by-frame torso-center normalized coordinates (T, 34).
        """
        T = sample_kp.shape[0]
        spine1 = sample_kp[:, 14, :] # (T, 2)
        spine3 = sample_kp[:, 16, :] # (T, 2)
        torso_center = 0.5 * (spine1 + spine3) # (T, 2)
        
        # Center coordinates relative to frame torso center and normalize by torso length
        centered = (sample_kp - torso_center[:, np.newaxis, :]) / (torso_len + self.eps) # (T, 17, 2)
        
        # Zero out invalid padded frames
        invalid_idx = np.where(sample_mask == 0)[0]
        if len(invalid_idx) > 0:
            centered[invalid_idx] = 0.0
            
        return centered.reshape(T, 34)

    def extract_velocity_features(self, sample_kp, sample_mask, torso_len):
        """
        Extracts frame-level keypoint displacement / velocity vectors (T, 34).
        V_t = (KP_t - KP_{t-1}) / torso_len
        """
        T = sample_kp.shape[0]
        vel = np.zeros((T, 17, 2), dtype=np.float32)
        
        # First-order difference along frame dimension
        delta = sample_kp[1:] - sample_kp[:-1] # (T-1, 17, 2)
        vel[1:] = delta / (torso_len + self.eps)
        
        # Zero out invalid padded frames
        invalid_idx = np.where(sample_mask == 0)[0]
        if len(invalid_idx) > 0:
            vel[invalid_idx] = 0.0
            
        return vel.reshape(T, 34)

    def extract_biomechanical_signals(self, sample_kp, sample_mask, torso_len):
        """
        Extracts 8 compact frame-level biomechanical signals (T, 8):
        0: Back Arch Elevation (Y_spine2 relative to spine1-spine3 baseline)
        1: Head Vertical Elevation (Y_headcentroid)
        2: Left Hoof Stride Separation (LFHoof - LHHoof distance)
        3: Right Hoof Stride Separation (RFHoof - RHHoof distance)
        4: Front Hoof Stance Width (LFHoof - RFHoof distance)
        5: Hind Hoof Stance Width (LHHoof - RHHoof distance)
        6: Left Knee Elevation (LFKnee Y)
        7: Right Knee Elevation (RFKnee Y)
        """
        T = sample_kp.shape[0]
        signals = np.zeros((T, 8), dtype=np.float32)
        
        # Keypoint indices
        lf_hoof, rf_hoof, lh_hoof, rh_hoof = 0, 3, 6, 9
        lf_knee, rf_knee = 2, 5
        nose, head_top = 12, 13
        spine1, spine2, spine3 = 14, 15, 16
        
        # 0: Back arch elevation (positive = arched up)
        baseline_y = 0.5 * (sample_kp[:, spine1, 1] + sample_kp[:, spine3, 1])
        signals[:, 0] = (baseline_y - sample_kp[:, spine2, 1]) / (torso_len + self.eps)
        
        # 1: Head vertical elevation
        head_y = 0.5 * (sample_kp[:, nose, 1] + sample_kp[:, head_top, 1])
        signals[:, 1] = head_y / (torso_len + self.eps)
        
        # 2-5: Hoof relative distances
        signals[:, 2] = np.linalg.norm(sample_kp[:, lf_hoof, :] - sample_kp[:, lh_hoof, :], axis=1) / (torso_len + self.eps)
        signals[:, 3] = np.linalg.norm(sample_kp[:, rf_hoof, :] - sample_kp[:, rh_hoof, :], axis=1) / (torso_len + self.eps)
        signals[:, 4] = np.linalg.norm(sample_kp[:, lf_hoof, :] - sample_kp[:, rf_hoof, :], axis=1) / (torso_len + self.eps)
        signals[:, 5] = np.linalg.norm(sample_kp[:, lh_hoof, :] - sample_kp[:, rh_hoof, :], axis=1) / (torso_len + self.eps)
        
        # 6-7: Knee elevations
        signals[:, 6] = sample_kp[:, lf_knee, 1] / (torso_len + self.eps)
        signals[:, 7] = sample_kp[:, rf_knee, 1] / (torso_len + self.eps)
        
        # Mask invalid frames
        invalid_idx = np.where(sample_mask == 0)[0]
        if len(invalid_idx) > 0:
            signals[invalid_idx] = 0.0
            
        return signals

    def build_dataset(self, keypoints, masks, mode='coords'):
        """
        Builds sequence tensor dataset (N, T, F) for a specified representation mode.
        Modes:
        - 'coords': Normalized coordinates (N, T, 34)
        - 'velocity': Keypoint velocity vectors (N, T, 34)
        - 'biomechanical': Frame-level gait signals (N, T, 8)
        - 'combined': Coords + Velocity + Biomechanical (N, T, 76)
        """
        N, T, _, _ = keypoints.shape
        sequences = []
        
        for i in range(N):
            sample_kp = keypoints[i] # (T, 17, 2)
            sample_mask = masks[i]   # (T,)
            torso_len = self.compute_sequence_torso_length(sample_kp, sample_mask)
            
            if mode == 'coords':
                feat = self.extract_normalized_coords(sample_kp, sample_mask, torso_len)
            elif mode == 'velocity':
                feat = self.extract_velocity_features(sample_kp, sample_mask, torso_len)
            elif mode == 'biomechanical':
                feat = self.extract_biomechanical_signals(sample_kp, sample_mask, torso_len)
            elif mode == 'combined':
                coords = self.extract_normalized_coords(sample_kp, sample_mask, torso_len)
                vel = self.extract_velocity_features(sample_kp, sample_mask, torso_len)
                bio = self.extract_biomechanical_signals(sample_kp, sample_mask, torso_len)
                feat = np.hstack([coords, vel, bio])
            else:
                raise ValueError(f"Unknown sequence representation mode: {mode}")
                
            sequences.append(feat)
            
        return np.array(sequences, dtype=np.float32)
