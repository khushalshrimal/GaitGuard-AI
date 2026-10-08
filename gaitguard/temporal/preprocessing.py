"""
GaitGuard AI - Fold-Isolated Temporal Scaler (Phase 7)
Scales sequence feature tensors strictly within cross-validation training folds to prevent leakage.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class FoldTemporalScaler:
    """
    Standardizes sequence features (N, T, F) using statistics (mean, std) computed
    strictly from unpadded frames (mask > 0) of training fold sequences.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, X_train, mask_train):
        """
        Fits StandardScaler on valid unpadded frames of X_train and transforms X_train.
        Input:
            X_train: (N_train, T, F)
            mask_train: (N_train, T)
        Output:
            X_train_scaled: (N_train, T, F)
        """
        N, T, F = X_train.shape
        
        # Reshape to 2D matrix of shape (N * T, F)
        X_flat = X_train.reshape(-1, F)
        mask_flat = mask_train.reshape(-1)
        
        valid_idx = np.where(mask_flat > 0)[0]
        if len(valid_idx) == 0:
            raise ValueError("No valid unpadded frames found in training fold mask.")
            
        valid_frames = X_flat[valid_idx] # (N_valid_frames, F)
        
        # Fit scaler strictly on valid training frames
        self.scaler.fit(valid_frames)
        self.fitted = True
        
        # Transform all frames
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, T, F)
        
        # Ensure padded frames remain 0.0
        X_scaled[mask_train == 0] = 0.0
        
        return X_scaled.astype(np.float32)

    def transform(self, X_val, mask_val):
        """
        Transforms validation fold X_val using previously fitted training fold statistics.
        Input:
            X_val: (N_val, T, F)
            mask_val: (N_val, T)
        Output:
            X_val_scaled: (N_val, T, F)
        """
        if not self.fitted:
            raise RuntimeError("FoldTemporalScaler must be fit on training fold before transform.")
            
        N, T, F = X_val.shape
        X_flat = X_val.reshape(-1, F)
        
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, T, F)
        
        # Ensure padded frames remain 0.0
        X_scaled[mask_val == 0] = 0.0
        
        return X_scaled.astype(np.float32)
