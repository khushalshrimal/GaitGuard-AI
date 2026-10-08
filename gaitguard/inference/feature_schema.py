"""
GaitGuard AI - 76-Feature Schema & Model Input Validator (Phase 9)
Defines the canonical 76-dimensional feature ordering and validates input tensors before BiLSTM model inference.
"""

import numpy as np

# Canonical Keypoint Names
KP_NAMES = [
    "LFHoof", "LFAnkle", "LFKnee", "RFHoof", "RFAnkle", "RFKnee",
    "LHHoof", "LHAnkle", "LHKnee", "RHHoof", "RHAnkle", "RHKnee",
    "Nose", "HeadTop", "Spine1", "Spine2", "Spine3"
]

# Construct 76 Feature Schema List
FEATURE_SCHEMA_76 = []

# Indices 0..33: Torso-Normalized Coordinates (34)
for kp in KP_NAMES:
    FEATURE_SCHEMA_76.append(f"norm_coord_{kp}_X")
    FEATURE_SCHEMA_76.append(f"norm_coord_{kp}_Y")
    
# Indices 34..67: Keypoint Velocity Vectors (34)
for kp in KP_NAMES:
    FEATURE_SCHEMA_76.append(f"velocity_{kp}_X")
    FEATURE_SCHEMA_76.append(f"velocity_{kp}_Y")
    
# Indices 68..75: Biomechanical Gait Signals (8)
FEATURE_SCHEMA_76.extend([
    "biomech_back_arch_elevation",
    "biomech_head_vertical_elevation",
    "biomech_left_hoof_stride_sep",
    "biomech_right_hoof_stride_sep",
    "biomech_front_hoof_stance_width",
    "biomech_hind_hoof_stance_width",
    "biomech_left_knee_elevation",
    "biomech_right_knee_elevation"
])


class ModelInputValidator:
    """
    Validates sequence tensors (1, 128, 76) before passing to Phase 7 BiLSTM model.
    """

    def __init__(self, target_timesteps=128, target_features=76):
        self.target_timesteps = target_timesteps
        self.target_features = target_features

    def validate_tensor_shape(self, tensor):
        """
        Validates tensor shape.
        Expected: (1, 128, 76) or (N, 128, 76)
        """
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Model input must be numpy.ndarray, got {type(tensor)}")
            
        if tensor.ndim != 3:
            raise ValueError(f"Model input tensor must be 3D (N, 128, 76), got {tensor.ndim}D tensor of shape {tensor.shape}")
            
        N, T, F = tensor.shape
        if T != self.target_timesteps:
            raise ValueError(f"Model input timesteps invalid: expected {self.target_timesteps}, got {T}")
            
        if F != self.target_features:
            raise ValueError(f"Model input feature dimension invalid: expected {self.target_features}, got {F}")
            
        return True

    def validate_numeric_integrity(self, tensor):
        """
        Validates non-NaN, non-Inf, and finite numerical values.
        """
        if np.isnan(tensor).any():
            raise ValueError("Model input tensor contains NaN values!")
            
        if np.isinf(tensor).any():
            raise ValueError("Model input tensor contains Inf values!")
            
        return True

    def validate_full(self, tensor):
        """
        Full input validation method.
        """
        self.validate_tensor_shape(tensor)
        self.validate_numeric_integrity(tensor)
        return True
