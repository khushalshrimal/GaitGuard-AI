"""
GaitGuard AI - Gait Feature Extractor Module (Phase 5)
Extracts torso-normalized biomechanical gait features from keypoint trajectories.
"""

import numpy as np
import pandas as pd

# Keypoint Index Mapping (Phase 2 Canonical Schema)
# 0: LFHoof, 1: LFAnkle, 2: LFKnee
# 3: RFHoof, 4: RFAnkle, 5: RFKnee
# 6: LHHoof, 7: LHAnkle, 8: LHKnee
# 9: RHHoof, 10: RHAnkle, 11: RHKnee
# 12: Nose, 13: HeadTop
# 14: Spine1, 15: Spine2, 16: Spine3

class GaitFeatureExtractor:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute_torso_length(self, kp_seq):
        """
        Computes robust sequence-level torso length: median distance between Spine1 (14) and Spine3 (16).
        Input: kp_seq of shape (valid_len, 17, 2)
        Output: scalar float32 torso_length
        """
        spine1 = kp_seq[:, 14, :]
        spine3 = kp_seq[:, 16, :]
        dist = np.sqrt(np.sum((spine1 - spine3)**2, axis=1))
        torso_len = np.median(dist)
        if torso_len <= self.eps:
            torso_len = 0.1 # Fallback safe torso length to prevent divide by zero
        return float(torso_len)

    def compute_back_arch_curvature(self, kp_seq, torso_len):
        """
        Computes Back Arch Curvature Index: vertical Y-elevation of Spine2 (15) relative to Spine1-Spine3 baseline,
        normalized by torso_len. Positive value indicates upward back arching.
        """
        spine1_y = kp_seq[:, 14, 1]
        spine2_y = kp_seq[:, 15, 1]
        spine3_y = kp_seq[:, 16, 1]
        
        baseline_y = 0.5 * (spine1_y + spine3_y)
        # Note: Y increases downwards in pixel space; smaller Y means higher elevation / arched back
        elevation = (baseline_y - spine2_y) / (torso_len + self.eps)
        arch_index = float(np.mean(elevation))
        return arch_index

    def compute_head_nodding_amplitude(self, kp_seq, torso_len):
        """
        Computes Head Nodding Amplitude: peak-to-peak vertical oscillation of combined head centroid
        (0.5 * (HeadTop + Nose)), normalized by torso_len.
        """
        head_y = 0.5 * (kp_seq[:, 13, 1] + kp_seq[:, 12, 1])
        amp = (np.max(head_y) - np.min(head_y)) / (torso_len + self.eps)
        return float(amp)

    def compute_torso_normalized_stride_length(self, kp_seq, torso_len):
        """
        Computes Torso-Normalized Stride Length: mean displacement of 4 hooves during walking,
        normalized by torso_len.
        """
        hoof_indices = [0, 3, 6, 9] # LFHoof, RFHoof, LHHoof, RHHoof
        hoof_disps = []

        for h_idx in hoof_indices:
            dx = np.diff(kp_seq[:, h_idx, 0])
            dy = np.diff(kp_seq[:, h_idx, 1])
            disp = np.sum(np.sqrt(dx**2 + dy**2))
            hoof_disps.append(disp)

        mean_disp = np.mean(hoof_disps)
        norm_stride = float(mean_disp / (torso_len + self.eps))
        return norm_stride

    def compute_stance_timing_asymmetry(self, kp_seq):
        """
        Computes Stance-Phase Timing Asymmetry Ratio: ratio of low-velocity stance duration between left and right limb pairs.
        """
        hoof_indices = {'LF': 0, 'RF': 3, 'LH': 6, 'RH': 9}
        stance_frames = {}

        for name, h_idx in hoof_indices.items():
            dx = np.diff(kp_seq[:, h_idx, 0])
            dy = np.diff(kp_seq[:, h_idx, 1])
            vel = np.sqrt(dx**2 + dy**2)
            
            # Low velocity frames (< 25th percentile of velocity) represent stance ground contact
            vel_threshold = np.percentile(vel, 25)
            is_stance = vel <= vel_threshold
            stance_frames[name] = np.sum(is_stance)

        # Front asymmetry: |LF - RF| / (LF + RF)
        lf_s, rf_s = stance_frames['LF'], stance_frames['RF']
        front_asym = abs(lf_s - rf_s) / (lf_s + rf_s + self.eps)

        # Hind asymmetry: |LH - RH| / (LH + RH)
        lh_s, rh_s = stance_frames['LH'], stance_frames['RH']
        hind_asym = abs(lh_s - rh_s) / (lh_s + rh_s + self.eps)

        total_asym = float(0.5 * (front_asym + hind_asym))
        return total_asym

    def compute_walking_speed(self, kp_seq, torso_len):
        """
        Computes Torso-Normalized Walking Speed: trunk forward displacement per valid frame.
        """
        valid_len = len(kp_seq)
        trunk_x = 0.5 * (kp_seq[:, 14, 0] + kp_seq[:, 16, 0])
        trunk_y = 0.5 * (kp_seq[:, 14, 1] + kp_seq[:, 16, 1])

        dx = np.diff(trunk_x)
        dy = np.diff(trunk_y)
        total_disp = np.sum(np.sqrt(dx**2 + dy**2))
        
        speed = float(total_disp / (valid_len * torso_len + self.eps))
        return speed

    def compute_knee_flexion_range(self, kp_seq):
        """
        Computes Knee Flexion Range of Motion (in degrees) averaged across 4 limbs.
        Angle formed by Hoof -> Ankle -> Knee.
        """
        limb_triplets = [
            (0, 1, 2),  # LFHoof, LFAnkle, LFKnee
            (3, 4, 5),  # RFHoof, RFAnkle, RFKnee
            (6, 7, 8),  # LHHoof, LHAnkle, LHKnee
            (9, 10, 11) # RHHoof, RHAnkle, RHKnee
        ]
        ranges = []

        for h, a, k in limb_triplets:
            vec_a_h = kp_seq[:, h, :] - kp_seq[:, a, :] # Ankle to Hoof
            vec_a_k = kp_seq[:, k, :] - kp_seq[:, a, :] # Ankle to Knee

            dot_prod = np.sum(vec_a_h * vec_a_k, axis=1)
            norm_ah = np.linalg.norm(vec_a_h, axis=1) + self.eps
            norm_ak = np.linalg.norm(vec_a_k, axis=1) + self.eps

            cos_angle = np.clip(dot_prod / (norm_ah * norm_ak), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(cos_angle))
            flexion_range = float(np.max(angles_deg) - np.min(angles_deg))
            ranges.append(flexion_range)

        mean_range = float(np.mean(ranges))
        return mean_range

    def extract_sample_features(self, sample):
        """
        Extracts all 7 gait features for a single sample.
        Input sample dictionary containing padded_keypoint array and raw_sequence_length.
        """
        valid_len = sample['raw_sequence_length']
        kp_seq = sample['padded_keypoints'][:valid_len, :, :] # (valid_len, 17, 2)

        torso_len = self.compute_torso_length(kp_seq)
        back_arch = self.compute_back_arch_curvature(kp_seq, torso_len)
        head_nod = self.compute_head_nodding_amplitude(kp_seq, torso_len)
        stride_len = self.compute_torso_normalized_stride_length(kp_seq, torso_len)
        stance_asym = self.compute_stance_timing_asymmetry(kp_seq)
        speed = self.compute_walking_speed(kp_seq, torso_len)
        knee_range = self.compute_knee_flexion_range(kp_seq)

        features = {
            'sample_id': sample['sample_id'],
            'animal_id': int(sample['animal_id']),
            'binary_target': int(sample['binary_target']),
            'raw_label': int(sample['raw_label']),
            'raw_sequence_length': int(valid_len),
            'torso_length': torso_len,
            'back_arch_curvature': back_arch,
            'head_nodding_amplitude': head_nod,
            'torso_normalized_stride_length': stride_len,
            'stance_timing_asymmetry': stance_asym,
            'normalized_walking_speed': speed,
            'knee_flexion_range': knee_range
        }
        return features

    def extract_dataset_features(self, samples):
        feature_records = []
        for s in samples:
            f_rec = self.extract_sample_features(s)
            feature_records.append(f_rec)
        df_features = pd.DataFrame(feature_records)
        return df_features
