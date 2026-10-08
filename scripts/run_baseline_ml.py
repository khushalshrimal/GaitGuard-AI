"""
GaitGuard AI - Phase 6 Master Baseline Machine Learning Pipeline
Executes leak-free 5-fold GroupKFold cross-validation, feature ablation, animal-level error analysis, and QA plotting.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gaitguard.config import RANDOM_SEED, N_SPLITS
from gaitguard.models.baseline_models import (
    get_baseline_models, evaluate_model_group_kfold, run_feature_ablation_study
)

def generate_phase6_figures(all_summaries, all_folds_dict, all_oofs_dict, ablation_df, feature_names, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. Confusion Matrix Plots for Each Model
    for m_name, df_oof in all_oofs_dict.items():
        cm = confusion_matrix(df_oof['actual'], df_oof['predicted'], labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['0: Normal', '1: Lame Risk'],
                    yticklabels=['0: Normal', '1: Lame Risk'])
        plt.title(f'GaitGuard AI — {m_name} OOF Confusion Matrix', fontweight='bold', fontsize=11)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Ground Truth Class')
        plt.tight_layout()
        clean_name = m_name.lower().replace(' ', '_')
        plt.savefig(os.path.join(fig_dir, f'{clean_name}_confusion_matrix.png'), dpi=150)
        plt.close()

    # 2. Out-Of-Fold ROC Curves Comparison
    plt.figure(figsize=(7, 6))
    colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#2ca02c', 'XGBoost': '#ff7f0e', 'Support Vector Machine': '#9467bd'}
    for m_name, df_oof in all_oofs_dict.items():
        fpr, tpr, _ = roc_curve(df_oof['actual'], df_oof['probability'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{m_name} (OOF AUC = {roc_auc:.3f})', color=colors.get(m_name, 'blue'), linewidth=2.0)

    plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.500)', linewidth=1.5)
    plt.title('GaitGuard AI — Out-Of-Fold ROC Curves Comparison', fontweight='bold', fontsize=12)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'roc_curve_comparison.png'), dpi=150)
    plt.close()

    # 3. Precision-Recall Curves Comparison
    plt.figure(figsize=(7, 6))
    for m_name, df_oof in all_oofs_dict.items():
        prec, rec, _ = precision_recall_curve(df_oof['actual'], df_oof['probability'])
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f'{m_name} (PR AUC = {pr_auc:.3f})', color=colors.get(m_name, 'blue'), linewidth=2.0)

    plt.title('GaitGuard AI — Precision-Recall Curves Comparison', fontweight='bold', fontsize=12)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'precision_recall_comparison.png'), dpi=150)
    plt.close()

    # 4. Feature Ablation Study Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(ablation_df['feature_set_name'], ablation_df['recall_mean'], color='#2b5c8f', alpha=0.85)
    plt.title('Feature Ablation Study — Mean Recall Across GroupKFold Folds', fontweight='bold', fontsize=12)
    plt.ylabel('Mean GroupKFold Recall')
    plt.ylim(0.0, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_ablation_comparison.png'), dpi=150)
    plt.close()

def run_phase6_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("GAITGUARD AI — PHASE 6: BASELINE MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Load Feature Dataset
    csv_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_engineered_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Engineered feature dataset missing at {csv_path}")

    df_features = pd.read_csv(csv_path)
    sample_ids = df_features['sample_id'].values
    animal_ids = df_features['animal_id'].values
    y = df_features['binary_target'].values

    feature_cols = [c for c in df_features.columns if c not in ['sample_id', 'animal_id', 'binary_target', 'raw_label']]
    X = df_features[feature_cols].values.astype(np.float32)

    print(f"\n[STEP 1] Ingested feature table: {len(df_features)} samples, {len(feature_cols)} ML features.")
    print(f"  - Target Distribution: Normal (0) = {np.sum(y == 0)}, Lame Risk (1) = {np.sum(y == 1)}")

    # 2. Evaluate Baseline Models
    print("\n[STEP 2] Training & Evaluating Baseline Models (5-Fold GroupKFold)...")
    models = get_baseline_models(seed=RANDOM_SEED)

    all_summaries = []
    all_folds_list = []
    all_oofs_dict = {}

    for m_name, pipeline in models.items():
        summary, df_folds, df_oof = evaluate_model_group_kfold(
            m_name, pipeline, X, y, animal_ids, sample_ids, n_splits=N_SPLITS
        )
        all_summaries.append(summary)
        all_folds_list.append(df_folds)
        all_oofs_dict[m_name] = df_oof

        print(f"  - {m_name:24s} | Accuracy: {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} | Recall: {summary['recall_mean']:.4f}±{summary['recall_std']:.4f} | F1: {summary['f1_mean']:.4f}±{summary['f1_std']:.4f} | ROC-AUC: {summary['roc_auc_mean']:.4f}±{summary['roc_auc_std']:.4f}")

    df_summary = pd.DataFrame(all_summaries)
    df_all_folds = pd.concat(all_folds_list, ignore_index=True)

    # 3. Feature Ablation Study
    print("\n[STEP 3] Executing Feature Ablation Study...")
    feature_sets = {
        'Baseline A (All 7 Features)': feature_cols,
        'Baseline B (Primary 3 Candidate Features)': ['back_arch_curvature', 'torso_normalized_stride_length', 'normalized_walking_speed'],
        'Baseline C (Primary 3 + Secondary 1 Feature)': ['back_arch_curvature', 'torso_normalized_stride_length', 'normalized_walking_speed', 'knee_flexion_range']
    }
    ablation_df = run_feature_ablation_study(df_features, feature_sets, seed=RANDOM_SEED)
    for _, row in ablation_df.iterrows():
        print(f"  - {row['feature_set_name']:45s} | Recall: {row['recall_mean']:.4f} | F1: {row['f1_mean']:.4f} | ROC-AUC: {row['roc_auc_mean']:.4f}")

    # 4. Animal-Level Error Analysis (Using Best Model by Recall: Random Forest)
    print("\n[STEP 4] Performing Animal-Level Error Analysis...")
    best_model_name = 'Random Forest'
    df_oof_best = all_oofs_dict[best_model_name]
    df_oof_best['is_error'] = df_oof_best['actual'] != df_oof_best['predicted']

    cow_error_summary = df_oof_best.groupby('animal_id').agg(
        total_videos=('sample_id', 'count'),
        error_videos=('is_error', 'sum'),
        actual_class=('actual', 'first')
    ).reset_index()
    cow_error_summary['error_rate'] = cow_error_summary['error_videos'] / cow_error_summary['total_videos']

    problematic_cows = cow_error_summary[cow_error_summary['error_rate'] > 0.5]
    print(f"  - Total Unique Cows: {len(cow_error_summary)}")
    print(f"  - Cows 100% Correctly Classified: {len(cow_error_summary[cow_error_summary['error_rate'] == 0])}")
    print(f"  - Cows with >50% Misclassifications: {len(problematic_cows)}")

    # 5. Save Artifacts & Reports
    print("\n[STEP 5] Saving model comparison tables and OOF predictions...")
    docs_dir = os.path.join(ROOT_DIR, "docs")
    df_summary.to_csv(os.path.join(docs_dir, "phase6_model_comparison.csv"), index=False)
    df_all_folds.to_csv(os.path.join(docs_dir, "phase6_fold_results.csv"), index=False)

    # Export OOF predictions for Random Forest
    df_oof_best.to_csv(os.path.join(ROOT_DIR, "datasets", "processed", "phase6_oof_predictions.csv"), index=False)

    # 6. Generate Figures
    print("\n[STEP 6] Generating Phase 6 QA figures in docs/figures/phase6/...")
    fig_dir = os.path.join(docs_dir, "figures", "phase6")
    generate_phase6_figures(all_summaries, all_folds_list, all_oofs_dict, ablation_df, feature_cols, fig_dir)
    print("  - Figures generated in docs/figures/phase6/.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 6 BASELINE MACHINE LEARNING SUMMARY")
    print("=" * 70)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Model Comparison CSV: {os.path.join(docs_dir, 'phase6_model_comparison.csv')}")
    print(f"Fold Results CSV: {os.path.join(docs_dir, 'phase6_fold_results.csv')}")
    print("STATUS: SUCCESS")
    print("=" * 70)

    return df_summary, df_all_folds, all_oofs_dict, ablation_df

if __name__ == "__main__":
    run_phase6_pipeline()
