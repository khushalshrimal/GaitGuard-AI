"""
GaitGuard AI - Baseline Machine Learning Evaluation Module (Phase 6)
Implements leak-free 5-fold GroupKFold cross-validation, baseline models (Logistic Regression, Random Forest, XGBoost, SVM), out-of-fold prediction tracking, and feature ablation studies.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from gaitguard.config import RANDOM_SEED, N_SPLITS

def get_baseline_models(seed=RANDOM_SEED):
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, random_state=seed, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=seed))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=seed))
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=1.0, kernel='rbf', random_state=seed)) # Uses decision_function for probabilities/scores
        ])
    }
    return models

def evaluate_model_group_kfold(model_name, pipeline, X, y, animal_ids, sample_ids, n_splits=N_SPLITS):
    """
    Executes 5-fold GroupKFold evaluation grouped on animal_ids.
    Fits scaling inside each training fold to prevent preprocessing leakage.
    Returns summary metrics, fold-level DataFrame, and OOF prediction DataFrame.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_folds = np.zeros(len(y), dtype=int)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=animal_ids)):
        # Verify 0 animal leakage
        train_cows = set(animal_ids[train_idx])
        val_cows = set(animal_ids[val_idx])
        assert len(train_cows.intersection(val_cows)) == 0, f"Animal leakage detected in fold {fold_idx}!"

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train pipeline (StandardScaler is fit ONLY on X_train inside fold)
        pipeline.fit(X_train, y_train)

        val_pred = pipeline.predict(X_val)
        
        # Get decision probabilities or normalized decision function for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            try:
                val_prob = pipeline.predict_proba(X_val)[:, 1]
            except AttributeError:
                dec = pipeline.decision_function(X_val)
                val_prob = 1.0 / (1.0 + np.exp(-dec))
        else:
            dec = pipeline.decision_function(X_val)
            val_prob = 1.0 / (1.0 + np.exp(-dec))

        oof_preds[val_idx] = val_pred
        oof_probs[val_idx] = val_prob
        oof_folds[val_idx] = fold_idx

        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        auc = roc_auc_score(y_val, val_prob)
        tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()

        fold_metrics.append({
            'model': model_name,
            'fold': fold_idx,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_cows': len(train_cows),
            'val_cows': len(val_cows),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        })

    df_folds = pd.DataFrame(fold_metrics)

    # Compute overall Out-Of-Fold (OOF) metrics
    oof_acc = accuracy_score(y, oof_preds)
    oof_prec = precision_score(y, oof_preds, zero_division=0)
    oof_rec = recall_score(y, oof_preds, zero_division=0)
    oof_f1 = f1_score(y, oof_preds, zero_division=0)
    oof_auc = roc_auc_score(y, oof_probs)

    summary = {
        'model': model_name,
        'accuracy_mean': float(df_folds['accuracy'].mean()),
        'accuracy_std': float(df_folds['accuracy'].std()),
        'precision_mean': float(df_folds['precision'].mean()),
        'precision_std': float(df_folds['precision'].std()),
        'recall_mean': float(df_folds['recall'].mean()),
        'recall_std': float(df_folds['recall'].std()),
        'f1_mean': float(df_folds['f1'].mean()),
        'f1_std': float(df_folds['f1'].std()),
        'roc_auc_mean': float(df_folds['roc_auc'].mean()),
        'roc_auc_std': float(df_folds['roc_auc'].std()),
        'oof_accuracy': float(oof_acc),
        'oof_precision': float(oof_prec),
        'oof_recall': float(oof_rec),
        'oof_f1': float(oof_f1),
        'oof_roc_auc': float(oof_auc)
    }

    df_oof = pd.DataFrame({
        'sample_id': sample_ids,
        'animal_id': animal_ids,
        'actual': y,
        'predicted': oof_preds,
        'probability': oof_probs,
        'fold': oof_folds
    })

    return summary, df_folds, df_oof

def run_feature_ablation_study(df_features, feature_sets, seed=RANDOM_SEED):
    """
    Evaluates Random Forest across feature subsets (Baseline A: All, Baseline B: Primary, Baseline C: Primary+Secondary).
    """
    ablation_results = []
    y = df_features['binary_target'].values
    animal_ids = df_features['animal_id'].values
    sample_ids = df_features['sample_id'].values

    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=seed))
    ])

    for set_name, feature_list in feature_sets.items():
        X_sub = df_features[feature_list].values.astype(np.float32)
        summary, df_folds, _ = evaluate_model_group_kfold(
            f"RF ({set_name})", rf_pipeline, X_sub, y, animal_ids, sample_ids
        )
        summary['feature_set_name'] = set_name
        summary['feature_count'] = len(feature_list)
        ablation_results.append(summary)

    return pd.DataFrame(ablation_results)
