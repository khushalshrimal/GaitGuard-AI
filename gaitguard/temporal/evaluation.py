"""
GaitGuard AI - Temporal Model Evaluator (Phase 7)
Manages 5-Fold GroupKFold animal-level evaluation, OOF prediction generation, and performance metrics.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .preprocessing import FoldTemporalScaler
from .model import BiLSTMModelTrainer

class TemporalModelEvaluator:
    """
    Evaluates temporal sequence models under strict 5-Fold GroupKFold animal isolation.
    """

    def __init__(self, n_splits=5, random_seed=42):
        self.n_splits = n_splits
        self.random_seed = random_seed

    def evaluate_bilstm(self, X_seq, y, masks, animal_ids, sample_ids, hidden_dim=32, dropout=0.3, lr=1e-3, max_epochs=100, batch_size=16, patience=15):
        """
        Runs 5-Fold GroupKFold evaluation for BiLSTM model.
        Returns dictionary containing:
        - fold_metrics: DataFrame of fold-level metrics
        - mean_metrics: dict of mean metrics across folds
        - oof_df: DataFrame of Out-of-Fold predictions (sample_id, animal_id, y_true, oof_prob, oof_pred, fold)
        - execution_time: float (seconds)
        """
        start_time = time.time()
        
        N, T, F = X_seq.shape
        gkf = GroupKFold(n_splits=self.n_splits)
        
        oof_probs = np.zeros(N, dtype=np.float32)
        oof_preds = np.zeros(N, dtype=np.int32)
        oof_folds = np.zeros(N, dtype=np.int32)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq, y, groups=animal_ids), start=1):
            train_cows = set(animal_ids[train_idx])
            val_cows = set(animal_ids[val_idx])
            
            # Assert 100% disjoint cow sets (zero animal leakage)
            assert train_cows.isdisjoint(val_cows), f"Fold {fold} data leakage detected: train and val cows overlap!"
            
            # Fold-isolated scaling
            scaler = FoldTemporalScaler()
            X_train_scaled = scaler.fit_transform(X_seq[train_idx], masks[train_idx])
            X_val_scaled = scaler.transform(X_seq[val_idx], masks[val_idx])
            
            y_train, y_val = y[train_idx], y[val_idx]
            mask_train, mask_val = masks[train_idx], masks[val_idx]
            
            # Train BiLSTM model for current fold
            trainer = BiLSTMModelTrainer(
                input_dim=F,
                hidden_dim=hidden_dim,
                dropout=dropout,
                lr=lr,
                seed=self.random_seed + fold
            )
            
            trainer.fit(
                X_train=X_train_scaled,
                y_train=y_train,
                mask_train=mask_train,
                X_val=X_val_scaled,
                y_val=y_val,
                mask_val=mask_val,
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Predict validation probabilities
            val_probs = trainer.predict_proba(X_val_scaled, mask_val)
            val_preds = (val_probs >= 0.5).astype(int)
            
            # Store OOF predictions
            oof_probs[val_idx] = val_probs
            oof_preds[val_idx] = val_preds
            oof_folds[val_idx] = fold
            
            # Compute fold metrics
            acc = accuracy_score(y_val, val_preds)
            prec = precision_score(y_val, val_preds, zero_division=0)
            rec = recall_score(y_val, val_preds, zero_division=0)
            f1 = f1_score(y_val, val_preds, zero_division=0)
            try:
                auc = roc_auc_score(y_val, val_probs)
            except ValueError:
                auc = 0.5
                
            fold_results.append({
                "fold": fold,
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "n_train_cows": len(train_cows),
                "n_val_cows": len(val_cows),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(auc)
            })
            
        fold_df = pd.DataFrame(fold_results)
        
        # Overall Out-of-Fold metrics
        oof_acc = accuracy_score(y, oof_preds)
        oof_prec = precision_score(y, oof_preds, zero_division=0)
        oof_rec = recall_score(y, oof_preds, zero_division=0)
        oof_f1 = f1_score(y, oof_preds, zero_division=0)
        oof_auc = roc_auc_score(y, oof_probs)
        
        mean_metrics = {
            "mean_accuracy": float(fold_df["accuracy"].mean()),
            "std_accuracy": float(fold_df["accuracy"].std()),
            "mean_precision": float(fold_df["precision"].mean()),
            "std_precision": float(fold_df["precision"].std()),
            "mean_recall": float(fold_df["recall"].mean()),
            "std_recall": float(fold_df["recall"].std()),
            "mean_f1": float(fold_df["f1_score"].mean()),
            "std_f1": float(fold_df["f1_score"].std()),
            "mean_roc_auc": float(fold_df["roc_auc"].mean()),
            "std_roc_auc": float(fold_df["roc_auc"].std()),
            "oof_accuracy": float(oof_acc),
            "oof_precision": float(oof_prec),
            "oof_recall": float(oof_rec),
            "oof_f1": float(oof_f1),
            "oof_roc_auc": float(oof_auc)
        }
        
        oof_df = pd.DataFrame({
            "sample_id": sample_ids,
            "animal_id": animal_ids,
            "y_true": y,
            "oof_prob": oof_probs,
            "oof_pred": oof_preds,
            "fold": oof_folds
        })
        
        execution_time = float(time.time() - start_time)
        
        return {
            "fold_df": fold_df,
            "mean_metrics": mean_metrics,
            "oof_df": oof_df,
            "execution_time": execution_time
        }
