"""
GaitGuard AI - Phase 8 Probability Calibration & Triage Master Script
Executes leakage-free cross-fitted calibration, probability quality benchmarking (Brier, Log Loss, ECE),
screening threshold analysis, 3-way triage decision logic, and outputs diagnostic QA figures.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Ensure gaitguard package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaitguard.triage.calibrator import CrossFittedCalibrator, compute_brier_score, compute_log_loss, compute_ece
from gaitguard.triage.triage_engine import ScreeningTriageEngine, ConfidenceEstimator, TriageResultContract

def main():
    print("=" * 70)
    print("GAITGUARD AI — PHASE 8: PROBABILITY CALIBRATION & TRIAGE SYSTEM")
    print("=" * 70)
    
    start_time = time.time()
    
    # Define paths
    oof_path = os.path.join("datasets", "processed", "phase7_oof_predictions.csv")
    output_dir = os.path.join("docs", "figures", "phase8")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"Phase 7 OOF predictions CSV not found at {oof_path}. Complete Phase 7 first.")
        
    # [STEP 1] Audit and load Phase 7 OOF predictions
    print("\n[STEP 1] Auditing Phase 7 Out-of-Fold Predictions...")
    oof_df = pd.read_csv(oof_path)
    
    N = len(oof_df)
    unique_cows = oof_df["animal_id"].nunique()
    print(f"  - Loaded {N} OOF sample predictions across {unique_cows} unique cows.")
    print(f"  - Target Balance: Normal (0) = {np.sum(oof_df['y_true'] == 0)}, Lameness Risk (1) = {np.sum(oof_df['y_true'] == 1)}")
    print(f"  - Raw Probability Range: [{oof_df['oof_prob'].min():.4f}, {oof_df['oof_prob'].max():.4f}]")
    
    y_true = oof_df["y_true"].values
    raw_probs = oof_df["oof_prob"].values
    
    # [STEP 2] Leakage-Free Cross-Fitted Calibration Evaluation
    print("\n[STEP 2] Executing Cross-Fitted Calibration Methods...")
    
    # 1. Raw Probabilities
    raw_brier = compute_brier_score(y_true, raw_probs)
    raw_logloss = compute_log_loss(y_true, raw_probs)
    raw_ece, raw_bin_data = compute_ece(y_true, raw_probs, n_bins=10)
    raw_auc = roc_auc_score(y_true, raw_probs)
    
    print(f"  - Raw BiLSTM: Brier = {raw_brier:.4f} | LogLoss = {raw_logloss:.4f} | ECE = {raw_ece:.4f} | ROC-AUC = {raw_auc:.4f}")
    
    # 2. Sigmoid Calibration (Platt Scaling)
    sigmoid_calibrator = CrossFittedCalibrator(method="sigmoid", n_splits=5)
    sigmoid_probs = sigmoid_calibrator.fit_transform_oof(oof_df)
    sig_brier = compute_brier_score(y_true, sigmoid_probs)
    sig_logloss = compute_log_loss(y_true, sigmoid_probs)
    sig_ece, sig_bin_data = compute_ece(y_true, sigmoid_probs, n_bins=10)
    sig_auc = roc_auc_score(y_true, sigmoid_probs)
    
    print(f"  - Sigmoid (Platt): Brier = {sig_brier:.4f} | LogLoss = {sig_logloss:.4f} | ECE = {sig_ece:.4f} | ROC-AUC = {sig_auc:.4f}")
    
    # 3. Isotonic Regression Calibration
    isotonic_calibrator = CrossFittedCalibrator(method="isotonic", n_splits=5)
    isotonic_probs = isotonic_calibrator.fit_transform_oof(oof_df)
    iso_brier = compute_brier_score(y_true, isotonic_probs)
    iso_logloss = compute_log_loss(y_true, isotonic_probs)
    iso_ece, iso_bin_data = compute_ece(y_true, isotonic_probs, n_bins=10)
    iso_auc = roc_auc_score(y_true, isotonic_probs)
    
    print(f"  - Isotonic Regression: Brier = {iso_brier:.4f} | LogLoss = {iso_logloss:.4f} | ECE = {iso_ece:.4f} | ROC-AUC = {iso_auc:.4f}")
    
    # Select best calibration method based on lowest Brier score / Log Loss
    if sig_brier <= iso_brier:
        best_method = "Sigmoid (Platt Scaling)"
        best_calibrated_probs = sigmoid_probs
        best_brier, best_logloss, best_ece = sig_brier, sig_logloss, sig_ece
        best_bin_data = sig_bin_data
    else:
        best_method = "Isotonic Regression"
        best_calibrated_probs = isotonic_probs
        best_brier, best_logloss, best_ece = iso_brier, iso_logloss, iso_ece
        best_bin_data = iso_bin_data
        
    print(f"\n[STEP 3] Best Calibration Method Selected: '{best_method}'")
    print(f"  - Brier Score Improvement: {raw_brier:.4f} -> {best_brier:.4f} ({((raw_brier - best_brier)/raw_brier)*100:+.2f}%)")
    print(f"  - Log Loss Improvement: {raw_logloss:.4f} -> {best_logloss:.4f} ({((raw_logloss - best_logloss)/raw_logloss)*100:+.2f}%)")
    print(f"  - ECE Improvement: {raw_ece:.4f} -> {best_ece:.4f} ({((raw_ece - best_ece)/raw_ece)*100:+.2f}%)")

    # Export Calibration Comparison CSV
    comp_rows = [
        {
            "Method": "Raw BiLSTM Output",
            "ROC-AUC": f"{raw_auc:.4f}",
            "Brier Score": f"{raw_brier:.4f}",
            "Log Loss": f"{raw_logloss:.4f}",
            "ECE Error": f"{raw_ece:.4f}",
            "Calibration Quality": "Uncalibrated Neural Network Output"
        },
        {
            "Method": "Sigmoid (Platt Scaling)",
            "ROC-AUC": f"{sig_auc:.4f}",
            "Brier Score": f"{sig_brier:.4f}",
            "Log Loss": f"{sig_logloss:.4f}",
            "ECE Error": f"{sig_ece:.4f}",
            "Calibration Quality": "Parametric Smooth Sigmoid Curve"
        },
        {
            "Method": "Isotonic Regression",
            "ROC-AUC": f"{iso_auc:.4f}",
            "Brier Score": f"{iso_brier:.4f}",
            "Log Loss": f"{iso_logloss:.4f}",
            "ECE Error": f"{iso_ece:.4f}",
            "Calibration Quality": "Non-parametric Piecewise Monotonic"
        }
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_csv_path = os.path.join("docs", "phase8_calibration_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"  - Saved calibration comparison table to {comp_csv_path}")

    # [STEP 4] Screening Threshold Analysis & 3-Way Triage Logic
    print("\n[STEP 4] Conducting Threshold Sweep & 3-Way Triage Logic...")
    engine = ScreeningTriageEngine(screening_threshold=0.50, margin_delta=0.10)
    
    # Sweep thresholds on calibrated probabilities
    selected_thresh, df_thresh = engine.select_optimal_screening_threshold(y_true, best_calibrated_probs, min_recall=0.80)
    thresh_csv_path = os.path.join("docs", "phase8_threshold_analysis.csv")
    df_thresh.to_csv(thresh_csv_path, index=False)
    print(f"  - Optimal Screening Threshold Selected: tau = {selected_thresh:.2f}")
    
    # Execute 3-way triage predictions
    triage_df = engine.predict_triage(oof_df["sample_id"].values, raw_probs, best_calibrated_probs)
    
    # Combine with ground truth and fold info
    calibrated_oof_df = oof_df.copy()
    calibrated_oof_df["oof_prob_calibrated"] = best_calibrated_probs
    calibrated_oof_df["confidence"] = ConfidenceEstimator.compute_confidence(best_calibrated_probs)
    calibrated_oof_df["decision"] = triage_df["decision"]
    calibrated_oof_df["inconclusive"] = triage_df["inconclusive"]
    
    calibrated_oof_path = os.path.join("datasets", "processed", "phase8_calibrated_oof_predictions.csv")
    calibrated_oof_df.to_csv(calibrated_oof_path, index=False)
    print(f"  - Exported calibrated OOF predictions to {calibrated_oof_path}")
    
    decision_counts = calibrated_oof_df["decision"].value_counts().to_dict()
    print(f"  - Triage Decisions Summary: {decision_counts}")

    # [STEP 5] Generating QA Diagnostic Figures under docs/figures/phase8/
    print("\n[STEP 5] Generating Phase 8 QA Figures under docs/figures/phase8/...")
    
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    
    # Figure 1: Raw Reliability Curve
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(raw_bin_data["bin_confidences"], raw_bin_data["bin_accuracies"], "s-", color="#d62728", label=f"Raw BiLSTM (ECE = {raw_ece:.4f})")
    plt.xlabel("Mean Predicted Probability (Confidence)", fontsize=10)
    plt.ylabel("Observed Fraction Positive (Accuracy)", fontsize=10)
    plt.title("Phase 8: Raw BiLSTM Reliability Diagram", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "phase8_raw_reliability_curve.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Calibrated Reliability Curve
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(best_bin_data["bin_confidences"], best_bin_data["bin_accuracies"], "o-", color="#2ca02c", label=f"Calibrated ({best_method}, ECE = {best_ece:.4f})")
    plt.xlabel("Mean Predicted Probability (Confidence)", fontsize=10)
    plt.ylabel("Observed Fraction Positive (Accuracy)", fontsize=10)
    plt.title("Phase 8: Calibrated Reliability Diagram", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "phase8_calibrated_reliability_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Raw vs Calibrated Probability Distribution Histograms
    plt.figure(figsize=(8, 5), dpi=300)
    sns.histplot(raw_probs[y_true == 0], color="#1f77b4", alpha=0.4, label="Normal (Raw)", kde=True, bins=20)
    sns.histplot(raw_probs[y_true == 1], color="#ff7f0e", alpha=0.4, label="Lameness Risk (Raw)", kde=True, bins=20)
    sns.histplot(best_calibrated_probs[y_true == 0], color="#2ca02c", alpha=0.6, label="Normal (Calibrated)", kde=True, bins=20)
    sns.histplot(best_calibrated_probs[y_true == 1], color="#d62728", alpha=0.6, label="Lameness Risk (Calibrated)", kde=True, bins=20)
    plt.axvline(selected_thresh, color="black", linestyle="--", lw=1.5, label=f"Threshold (τ={selected_thresh:.2f})")
    plt.title("Probability Distribution: Raw vs Calibrated", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Predicted Risk Probability", fontsize=10)
    plt.ylabel("Sample Count", fontsize=10)
    plt.legend(loc="upper center", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "phase8_probability_distributions.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Brier & Log Loss Bar Chart Comparison
    plt.figure(figsize=(7, 5), dpi=300)
    methods_label = ["Raw BiLSTM", "Sigmoid (Platt)", "Isotonic"]
    brier_vals = [raw_brier, sig_brier, iso_brier]
    logloss_vals = [raw_logloss, sig_logloss, iso_logloss]
    
    x = np.arange(len(methods_label))
    width = 0.35
    plt.bar(x - width/2, brier_vals, width, label="Brier Score (Lower Better)", color="#1f77b4")
    plt.bar(x + width/2, logloss_vals, width, label="Log Loss (Lower Better)", color="#ff7f0e")
    plt.xticks(x, methods_label)
    plt.ylabel("Error Score", fontsize=10)
    plt.title("Probability Quality Metric Comparison", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for i in range(len(methods_label)):
        plt.text(i - width/2, brier_vals[i] + 0.01, f"{brier_vals[i]:.3f}", ha="center", fontsize=8)
        plt.text(i + width/2, logloss_vals[i] + 0.01, f"{logloss_vals[i]:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "phase8_brier_logloss_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Figure 5: Threshold vs Recall, Precision, F1, Specificity Curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(df_thresh["threshold"], df_thresh["recall"], lw=2, color="#ff7f0e", label="Recall (Sensitivity)")
    plt.plot(df_thresh["threshold"], df_thresh["precision"], lw=2, color="#1f77b4", label="Precision")
    plt.plot(df_thresh["threshold"], df_thresh["f1_score"], lw=2, color="#2ca02c", label="F1-Score")
    plt.plot(df_thresh["threshold"], df_thresh["specificity"], lw=2, color="#9467bd", label="Specificity")
    plt.axvline(selected_thresh, color="red", linestyle="--", lw=1.5, label=f"Selected τ = {selected_thresh:.2f}")
    plt.xlabel("Decision Threshold (τ)", fontsize=10)
    plt.ylabel("Metric Score", fontsize=10)
    plt.title("Threshold Sweep Analysis for Screening Decision Layer", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "phase8_threshold_analysis.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # Figure 6: Triage Confusion Matrix (3-Way Decision)
    plt.figure(figsize=(6, 5), dpi=300)
    cm_triage = pd.crosstab(y_true, calibrated_oof_df["decision"], rownames=["True Class"], colnames=["Triage Decision"])
    sns.heatmap(cm_triage, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title("3-Way Triage Decision Matrix (N=272)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "phase8_triage_confusion_matrix.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    # Figure 7: Inconclusive Uncertainty Region Visualization
    plt.figure(figsize=(8, 5), dpi=300)
    sorted_idx = np.argsort(best_calibrated_probs)
    plt.plot(np.arange(N), best_calibrated_probs[sorted_idx], color="#1f77b4", lw=2, label="Calibrated Risk Probability")
    
    plt.axhline(selected_thresh - 0.10, color="orange", linestyle="--", label="Lower Boundary (τ - 0.10)")
    plt.axhline(selected_thresh + 0.10, color="orange", linestyle="--", label="Upper Boundary (τ + 0.10)")
    plt.fill_between(np.arange(N), selected_thresh - 0.10, selected_thresh + 0.10, color="yellow", alpha=0.3, label="Inconclusive Uncertainty Region")
    
    plt.xlabel("Samples Sorted by Calibrated Probability", fontsize=10)
    plt.ylabel("Calibrated Lameness Risk Probability", fontsize=10)
    plt.title("Inconclusive Uncertainty Region Boundary", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig7_path = os.path.join(output_dir, "phase8_inconclusive_region_visualization.png")
    plt.savefig(fig7_path, dpi=300)
    plt.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 8 CALIBRATION & TRIAGE PIPELINE PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Runtime: {total_time:.2f} seconds")
    print(f"Calibrated OOF CSV: {calibrated_oof_path}")
    print(f"Calibration Comparison CSV: {comp_csv_path}")
    print(f"Threshold Analysis CSV: {thresh_csv_path}")
    print("STATUS: SUCCESS")
    print("=" * 70)

if __name__ == "__main__":
    main()
