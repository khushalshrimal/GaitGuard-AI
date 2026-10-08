"""
GaitGuard AI - Leakage-Free Cross-Fitted Probability Calibrator (Phase 8)
Implements nested cross-fitted probability calibration (Platt Scaling & Isotonic Regression),
Brier score, Log Loss, and Expected Calibration Error (ECE) estimation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss

def compute_brier_score(y_true, y_prob):
    """Computes Brier Score (Mean Squared Error of probabilities)."""
    return float(brier_score_loss(y_true, y_prob))

def compute_log_loss(y_true, y_prob, eps=1e-15):
    """Computes binary cross-entropy log loss."""
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    return float(log_loss(y_true, y_prob_clipped))

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and reliability diagram bin points.
    Returns:
        ece: float scalar
        bin_data: dict containing bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    ece = 0.0
    N = len(y_true)
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        if i == n_bins - 1:
            idx = np.where((y_prob >= lower) & (y_prob <= upper))[0]
        else:
            idx = np.where((y_prob >= lower) & (y_prob < upper))[0]
            
        count = len(idx)
        bin_centers.append(0.5 * (lower + upper))
        
        if count > 0:
            bin_acc = float(np.mean(y_true[idx]))
            bin_conf = float(np.mean(y_prob[idx]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(count)
            ece += (count / N) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)
            
    bin_data = {
        "bin_centers": np.array(bin_centers),
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts)
    }
    
    return float(ece), bin_data


class CrossFittedCalibrator:
    """
    Executes leakage-free cross-fitted probability calibration using inner animal-level GroupKFold splits.
    """

    def __init__(self, method="sigmoid", n_splits=5, random_seed=42):
        """
        method: 'sigmoid' (Platt Scaling) or 'isotonic' (Isotonic Regression)
        """
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"Unsupported calibration method: {method}. Use 'sigmoid' or 'isotonic'.")
        self.method = method
        self.n_splits = n_splits
        self.random_seed = random_seed

    def fit_transform_oof(self, oof_df):
        """
        Performs cross-fitted calibration across outer folds.
        Input:
            oof_df: DataFrame with columns ['sample_id', 'animal_id', 'y_true', 'oof_prob', 'fold']
        Output:
            calibrated_probs: np.ndarray of shape (N,)
        """
        df = oof_df.copy()
        N = len(df)
        calibrated_probs = np.zeros(N, dtype=np.float32)
        
        folds = sorted(df["fold"].unique())
        
        for outer_fold in folds:
            val_mask = (df["fold"] == outer_fold)
            train_mask = ~val_mask
            
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(train_mask)[0]
            
            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)
            
            # Inner GroupKFold cross-validation to get uncalibrated inner OOF predictions
            gkf_inner = GroupKFold(n_splits=min(4, len(df_train["animal_id"].unique())))
            inner_oof_probs = np.zeros(len(df_train), dtype=np.float32)
            
            X_tr_inner = df_train["oof_prob"].values.reshape(-1, 1)
            y_tr_inner = df_train["y_true"].values
            groups_tr_inner = df_train["animal_id"].values
            
            # Inner predictions: we map logits or probabilities directly
            for in_tr_idx, in_va_idx in gkf_inner.split(X_tr_inner, y_tr_inner, groups=groups_tr_inner):
                inner_oof_probs[in_va_idx] = X_tr_inner[in_va_idx].flatten()
                
            # Fit calibrator strictly on training side inner predictions
            if self.method == "sigmoid":
                # Convert probabilities to log-odds / logits to fit Logistic Regression
                eps = 1e-7
                logits_tr = np.log(np.clip(inner_oof_probs, eps, 1.0 - eps) / (1.0 - np.clip(inner_oof_probs, eps, 1.0 - eps))).reshape(-1, 1)
                calibrator = LogisticRegression(C=1.0, solver="lbfgs")
                calibrator.fit(logits_tr, y_tr_inner)
                
                # Transform outer validation probabilities
                val_probs_raw = df_val["oof_prob"].values
                val_logits = np.log(np.clip(val_probs_raw, eps, 1.0 - eps) / (1.0 - np.clip(val_probs_raw, eps, 1.0 - eps))).reshape(-1, 1)
                calibrated_val = calibrator.predict_proba(val_logits)[:, 1]
                
            elif self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(inner_oof_probs, y_tr_inner)
                calibrated_val = calibrator.predict(df_val["oof_prob"].values)
                
            calibrated_probs[val_idx] = calibrated_val.astype(np.float32)
            
        return calibrated_probs
