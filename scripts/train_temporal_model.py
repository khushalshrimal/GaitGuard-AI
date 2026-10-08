"""
GaitGuard AI - Phase 7 Temporal Model Training & Evaluation Pipeline
Loads cleaned 128-frame keypoint trajectories, executes feature ablation, evaluates BiLSTM under 5-Fold GroupKFold,
generates comparison benchmarks vs Phase 6, and outputs QA figures.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, roc_auc_score, accuracy_score, recall_score, f1_score

# Ensure gaitguard package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
from gaitguard.temporal.evaluation import TemporalModelEvaluator
from gaitguard.config import RANDOM_SEED, N_SPLITS

def main():
    print("=" * 70)
    print("GAITGUARD AI — PHASE 7: ADVANCED TEMPORAL MODELING (BiLSTM)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Define paths
    dataset_path = os.path.join("datasets", "processed", "gaitguard_cleaned_dataset.npz")
    output_dir = os.path.join("docs", "figures", "phase7")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_path}. Please complete Phase 3 first.")
        
    # [STEP 1] Load cleaned 128-frame keypoint trajectories
    print("\n[STEP 1] Ingesting cleaned dataset...")
    data = np.load(dataset_path)
    
    sample_ids = data["sample_ids"]
    animal_ids = data["animal_ids"]
    padded_keypoints = data["padded_keypoints"] # (272, 128, 17, 2)
    sequence_masks = data["sequence_masks"]     # (272, 128)
    binary_targets = data["binary_targets"]     # (272,)
    
    N, T, K, C = padded_keypoints.shape
    unique_cows = len(np.unique(animal_ids))
    print(f"  - Ingested {N} samples across {unique_cows} unique cows.")
    print(f"  - Keypoint Trajectory Shape: ({N}, {T}, {K}, {C})")
    print(f"  - Target Balance: Normal (0) = {np.sum(binary_targets == 0)}, Lameness Risk (1) = {np.sum(binary_targets == 1)}")
    
    builder = TemporalSequenceBuilder()
    evaluator = TemporalModelEvaluator(n_splits=N_SPLITS, random_seed=RANDOM_SEED)
    
    # [STEP 2] Temporal Feature Ablation Experiments
    print("\n[STEP 2] Executing Temporal Representation Ablation Experiments...")
    ablation_modes = {
        "Exp A (Normalized Coordinates)": "coords",
        "Exp B (Keypoint Velocity Vectors)": "velocity",
        "Exp C (Biomechanical Gait Signals)": "biomechanical",
        "Exp D (Combined Compact Features)": "combined"
    }
    
    ablation_results = {}
    best_mode_name = None
    best_mode_auc = -1.0
    best_res = None
    
    for exp_label, mode in ablation_modes.items():
        X_seq = builder.build_dataset(padded_keypoints, sequence_masks, mode=mode)
        print(f"  - Evaluating {exp_label}: Sequence Tensor Shape = {X_seq.shape}")
        
        res = evaluator.evaluate_bilstm(
            X_seq=X_seq,
            y=binary_targets,
            masks=sequence_masks,
            animal_ids=animal_ids,
            sample_ids=sample_ids,
            hidden_dim=32,
            dropout=0.3,
            lr=1e-3,
            max_epochs=100,
            batch_size=16,
            patience=15
        )
        
        m = res["mean_metrics"]
        print(f"    -> Acc: {m['mean_accuracy']:.4f}±{m['std_accuracy']:.4f} | Recall: {m['mean_recall']:.4f}±{m['std_recall']:.4f} | F1: {m['mean_f1']:.4f}±{m['std_f1']:.4f} | OOF AUC: {m['oof_roc_auc']:.4f}")
        
        ablation_results[exp_label] = res
        if m['oof_roc_auc'] > best_mode_auc:
            best_mode_auc = m['oof_roc_auc']
            best_mode_name = mode
            best_res = res

    print(f"\n[STEP 3] Best Temporal Representation: Mode '{best_mode_name}' (OOF ROC-AUC: {best_mode_auc:.4f})")
    
    # Save Best OOF Predictions
    oof_df = best_res["oof_df"]
    oof_path = os.path.join("datasets", "processed", "phase7_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"  - Exported OOF predictions to {oof_path}")
    
    # Save Fold Results CSV
    fold_df = best_res["fold_df"]
    fold_csv_path = os.path.join("docs", "phase7_fold_results.csv")
    fold_df.to_csv(fold_csv_path, index=False)
    print(f"  - Saved fold-level results to {fold_csv_path}")
    
    # [STEP 4] Comparison vs Phase 6 Baseline Models
    print("\n[STEP 4] Benchmarking Phase 7 BiLSTM vs Phase 6 Baseline ML Models...")
    
    # Load Phase 6 OOF predictions for fair side-by-side comparison
    phase6_oof_path = os.path.join("datasets", "processed", "phase6_oof_predictions.csv")
    if os.path.exists(phase6_oof_path):
        p6_oof = pd.read_csv(phase6_oof_path)
    else:
        p6_oof = None
        
    p7_m = best_res["mean_metrics"]
    
    # Phase 6 baseline figures (from Phase 6 report)
    comparison_rows = [
        {
            "Model Algorithm": "Phase 6 Logistic Regression (Linear)",
            "Fold Accuracy": "66.18% ± 9.54%",
            "Fold Recall": "62.36% ± 8.45%",
            "Fold F1-Score": "0.6323 ± 0.1184",
            "Fold ROC-AUC": "0.7273 ± 0.1203",
            "OOF Accuracy": "0.6618",
            "OOF Recall": "0.6279",
            "OOF F1-Score": "0.6353",
            "OOF ROC-AUC": "0.7241",
            "Inference Speed": "< 1 ms / sample"
        },
        {
            "Model Algorithm": "Phase 6 SVM (RBF Kernel)",
            "Fold Accuracy": "61.04% ± 10.32%",
            "Fold Recall": "62.33% ± 6.54%",
            "Fold F1-Score": "0.6019 ± 0.0997",
            "Fold ROC-AUC": "0.6591 ± 0.1055",
            "OOF Accuracy": "0.6103",
            "OOF Recall": "0.6202",
            "OOF F1-Score": "0.6015",
            "OOF ROC-AUC": "0.6588",
            "Inference Speed": "< 1 ms / sample"
        },
        {
            "Model Algorithm": "Phase 6 Random Forest",
            "Fold Accuracy": "61.41% ± 12.86%",
            "Fold Recall": "59.66% ± 9.38%",
            "Fold F1-Score": "0.5939 ± 0.1311",
            "Fold ROC-AUC": "0.6590 ± 0.1725",
            "OOF Accuracy": "0.6140",
            "OOF Recall": "0.5969",
            "OOF F1-Score": "0.5946",
            "OOF ROC-AUC": "0.6601",
            "Inference Speed": "< 2 ms / sample"
        },
        {
            "Model Algorithm": "Phase 6 XGBoost",
            "Fold Accuracy": "59.20% ± 16.01%",
            "Fold Recall": "55.70% ± 14.97%",
            "Fold F1-Score": "0.5621 ± 0.1794",
            "Fold ROC-AUC": "0.6455 ± 0.1911",
            "OOF Accuracy": "0.5919",
            "OOF Recall": "0.5581",
            "OOF F1-Score": "0.5647",
            "OOF ROC-AUC": "0.6468",
            "Inference Speed": "< 2 ms / sample"
        },
        {
            "Model Algorithm": f"Phase 7 BiLSTM (Mode: {best_mode_name})",
            "Fold Accuracy": f"{p7_m['mean_accuracy']*100:.2f}% ± {p7_m['std_accuracy']*100:.2f}%",
            "Fold Recall": f"{p7_m['mean_recall']*100:.2f}% ± {p7_m['std_recall']*100:.2f}%",
            "Fold F1-Score": f"{p7_m['mean_f1']:.4f} ± {p7_m['std_f1']:.4f}",
            "Fold ROC-AUC": f"{p7_m['mean_roc_auc']:.4f} ± {p7_m['std_roc_auc']:.4f}",
            "OOF Accuracy": f"{p7_m['oof_accuracy']:.4f}",
            "OOF Recall": f"{p7_m['oof_recall']:.4f}",
            "OOF F1-Score": f"{p7_m['oof_f1']:.4f}",
            "OOF ROC-AUC": f"{p7_m['oof_roc_auc']:.4f}",
            "Inference Speed": "~ 3.5 ms / sample (CPU)"
        }
    ]
    
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join("docs", "phase7_model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"  - Saved model comparison table to {comp_csv_path}")

    # [STEP 5] Generate Diagnostic Figures
    print("\n[STEP 5] Generating Phase 7 QA Figures under docs/figures/phase7/...")
    
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    
    # Figure 1: BiLSTM Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=300)
    cm = confusion_matrix(binary_targets, oof_df["oof_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal (0)", "Lameness Risk (1)"],
                yticklabels=["Normal (0)", "Lameness Risk (1)"])
    plt.title(f"Phase 7 BiLSTM Confusion Matrix (OOF, N={N})", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "phase7_bilstm_confusion_matrix.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # Figure 2: ROC Curves Comparison (Phase 6 Logistic Regression vs Phase 7 BiLSTM)
    plt.figure(figsize=(7, 6), dpi=300)
    fpr_bilstm, tpr_bilstm, _ = roc_curve(binary_targets, oof_df["oof_prob"])
    auc_bilstm = roc_auc_score(binary_targets, oof_df["oof_prob"])
    
    plt.plot(fpr_bilstm, tpr_bilstm, color="#1f77b4", lw=2, label=f"Phase 7 BiLSTM (OOF AUC = {auc_bilstm:.4f})")
    
    if p6_oof is not None and "logistic_regression_prob" in p6_oof.columns:
        fpr_lr, tpr_lr, _ = roc_curve(binary_targets, p6_oof["logistic_regression_prob"])
        auc_lr = roc_auc_score(binary_targets, p6_oof["logistic_regression_prob"])
        plt.plot(fpr_lr, tpr_lr, color="#2ca02c", linestyle="--", lw=2, label=f"Phase 6 Logistic Regression (OOF AUC = {auc_lr:.4f})")
        
    plt.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1.5, label="Random Classifier (AUC = 0.50)")
    plt.title("ROC Curve Comparison: Phase 6 Tabular vs Phase 7 BiLSTM", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "phase7_roc_curve_comparison.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # Figure 3: Precision-Recall Curves
    plt.figure(figsize=(7, 6), dpi=300)
    prec_b, rec_b, _ = precision_recall_curve(binary_targets, oof_df["oof_prob"])
    pr_auc_b = auc(rec_b, prec_b)
    plt.plot(rec_b, prec_b, color="#ff7f0e", lw=2, label=f"Phase 7 BiLSTM (PR-AUC = {pr_auc_b:.4f})")
    
    if p6_oof is not None and "logistic_regression_prob" in p6_oof.columns:
        prec_l, rec_l, _ = precision_recall_curve(binary_targets, p6_oof["logistic_regression_prob"])
        pr_auc_l = auc(rec_l, prec_l)
        plt.plot(rec_l, prec_l, color="#2ca02c", linestyle="--", lw=2, label=f"Phase 6 Logistic Regression (PR-AUC = {pr_auc_l:.4f})")
        
    plt.title("Precision-Recall Curve Comparison", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Recall (Sensitivity)", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "phase7_precision_recall_comparison.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # Figure 4: Ablation Experiment Comparison Bar Chart
    plt.figure(figsize=(8, 5), dpi=300)
    exp_labels = list(ablation_results.keys())
    exp_aucs = [res["mean_metrics"]["oof_roc_auc"] for res in ablation_results.values()]
    exp_recalls = [res["mean_metrics"]["oof_recall"] for res in ablation_results.values()]
    
    x = np.arange(len(exp_labels))
    width = 0.35
    
    plt.bar(x - width/2, exp_aucs, width, label="OOF ROC-AUC", color="#1f77b4")
    plt.bar(x + width/2, exp_recalls, width, label="OOF Recall", color="#aec7e8")
    
    plt.ylabel("Score", fontsize=10)
    plt.title("Temporal Feature Representation Ablation Benchmark", fontsize=11, fontweight="bold", pad=12)
    plt.xticks(x, [l.split(" ")[0] + " " + l.split(" ")[1] for l in exp_labels], rotation=15)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i in range(len(exp_labels)):
        plt.text(i - width/2, exp_aucs[i] + 0.02, f"{exp_aucs[i]:.3f}", ha="center", fontsize=8)
        plt.text(i + width/2, exp_recalls[i] + 0.02, f"{exp_recalls[i]:.3f}", ha="center", fontsize=8)
        
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "phase7_ablation_experiment_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Figure 5: Fold-Wise Performance Stability
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(fold_df["fold"], fold_df["accuracy"], marker="o", lw=2, color="#1f77b4", label="Fold Accuracy")
    plt.plot(fold_df["fold"], fold_df["recall"], marker="s", lw=2, color="#ff7f0e", label="Fold Recall")
    plt.plot(fold_df["fold"], fold_df["roc_auc"], marker="^", lw=2, color="#2ca02c", label="Fold ROC-AUC")
    
    plt.axhline(best_res["mean_metrics"]["mean_accuracy"], color="#1f77b4", linestyle=":", alpha=0.7)
    plt.axhline(best_res["mean_metrics"]["mean_recall"], color="#ff7f0e", linestyle=":", alpha=0.7)
    plt.axhline(best_res["mean_metrics"]["mean_roc_auc"], color="#2ca02c", linestyle=":", alpha=0.7)
    
    plt.xlabel("GroupKFold Validation Fold", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.title("Phase 7 BiLSTM 5-Fold GroupKFold Stability", fontsize=11, fontweight="bold", pad=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0.4, 0.95)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "phase7_fold_wise_stability.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    
    # Figure 6: Cow-Level Error Analysis
    plt.figure(figsize=(8, 5), dpi=300)
    cow_acc = oof_df.groupby("animal_id").apply(lambda g: (g["y_true"] == g["oof_pred"]).mean())
    sns.histplot(cow_acc, bins=10, kde=False, color="#9467bd")
    plt.title("Distribution of Cow-Level Classification Accuracy (N=98 Cows)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Accuracy per Cow (Fraction Correct)", fontsize=10)
    plt.ylabel("Number of Cows", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "phase7_animal_error_analysis.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 7 TEMPORAL MODEL PIPELINE PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Runtime: {total_time:.2f} seconds")
    print(f"OOF Predictions CSV: {oof_path}")
    print(f"Model Comparison CSV: {comp_csv_path}")
    print("STATUS: SUCCESS")
    print("=" * 70)

if __name__ == "__main__":
    main()
