"""
GaitGuard AI - Phase 11 Explainable AI & Evidence Layer Master Script
Executes local and global SHAP explanations, evaluates stability & faithfulness,
exports CSV benchmark tables, and generates 5 diagnostic visual artifacts.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
from gaitguard.temporal.preprocessing import FoldTemporalScaler
from gaitguard.explainability.explainer import GaitGuardExplainer
from gaitguard.explainability.aggregator import AttributionAggregator
from gaitguard.explainability.stability import ExplanationStabilityEvaluator
from gaitguard.inference.feature_schema import FEATURE_SCHEMA_76

def main():
    print("=" * 80)
    print("GAITGUARD AI — PHASE 11: EXPLAINABLE AI (SHAP) & EVIDENCE LAYER EXECUTION")
    print("=" * 80)
    
    # Ensure output directories exist
    os.makedirs(os.path.join("docs", "figures", "phase11"), exist_ok=True)
    
    # 1. Ingest Cleaned Dataset
    dataset_path = os.path.join("datasets", "processed", "gaitguard_cleaned_dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset missing at {dataset_path}")
        
    data = np.load(dataset_path)
    kp = data["padded_keypoints"] # (272, 128, 17, 2)
    masks = data["sequence_masks"] # (272, 128)
    labels = data["binary_targets"] # (272,)
    
    print(f"\n[STEP 1] Ingested {len(labels)} sequence samples from dataset NPZ.")
    
    # 2. Build 76-Feature Sequence Tensors
    builder = TemporalSequenceBuilder()
    X_seq = builder.build_dataset(kp, masks, mode="combined")
    
    scaler = FoldTemporalScaler()
    X_scaled = scaler.fit_transform(X_seq, masks)
    
    # 3. Initialize Explainer
    explainer = GaitGuardExplainer(n_bg_samples=30, seed=42)
    print("[STEP 2] Initialized GaitGuardExplainer with leakage-free background subset.")

    # 4. Compute Global SHAP Attributions across Evaluation Subset (30 samples)
    eval_indices = np.linspace(0, len(labels) - 1, 30, dtype=int)
    shap_matrices = []
    latencies = []
    
    print("\n[STEP 3] Computing local & global SHAP attributions...")
    for idx in eval_indices:
        t0 = time.time()
        shap_mat = explainer.explain_sequence(X_scaled[idx:idx+1], masks[idx:idx+1])
        t_ms = (time.time() - t0) * 1000.0
        shap_matrices.append(shap_mat)
        latencies.append(t_ms)
        
    shap_stack = np.array(shap_matrices) # (30, 128, 76)
    mean_abs_shap = np.mean(np.abs(shap_stack), axis=(0, 1)) # (76,)
    mean_signed_shap = np.mean(shap_stack, axis=(0, 1)) # (76,)
    
    # 5. Export Global Feature Importance CSV
    df_global = pd.DataFrame({
        "feature_index": np.arange(76),
        "feature_name": FEATURE_SCHEMA_76,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    csv_path = os.path.join("docs", "phase11_global_attribution_summary.csv")
    df_global.to_csv(csv_path, index=False)
    print(f"[SAVED] Global attribution summary exported to {csv_path}")

    # 6. Evaluate Explanation Stability & Faithfulness
    print("\n[STEP 4] Evaluating explanation stability & perturbation faithfulness...")
    stability_res = ExplanationStabilityEvaluator.evaluate_stability(
        explainer, X_scaled[0:1], masks[0:1], n_runs=5, top_k=5
    )
    faithfulness_res = ExplanationStabilityEvaluator.evaluate_faithfulness_perturbation(
        explainer.model, X_scaled[0:1], masks[0:1], shap_stack[0], top_k=5
    )
    
    print(f"  - Top-5 Jaccard Overlap: {stability_res['jaccard_overlap']:.4f}")
    print(f"  - Sign Consistency:      {stability_res['sign_consistency']:.4f}")
    print(f"  - Faithfulness Delta P:  {faithfulness_res['prob_delta']:.4f}")
    print(f"  - Mean Explainer Latency: {np.mean(latencies):.2f} ms")

    # 7. Generate 5 QA Visual Artifacts
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_dir = os.path.join("docs", "figures", "phase11")

    # Figure 1: Global Feature Importance Top 15
    fig, ax = plt.subplots(figsize=(10, 6))
    df_top15 = df_global.head(15)
    sns.barplot(data=df_top15, x="mean_abs_shap", y="feature_name", hue="feature_name", palette="mako", legend=False, ax=ax)
    ax.set_title("GaitGuard AI — Phase 11: Top 15 Feature SHAP Attributions", fontsize=12, fontweight='bold')
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase11_global_feature_importance.png"), dpi=300)
    plt.close()

    # Figure 2: Modality & Body Region Attribution
    modality_res = AttributionAggregator.aggregate_by_modality(shap_stack[0])
    region_res = AttributionAggregator.aggregate_by_body_region(shap_stack[0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.pie(
        [modality_res["coordinates"], modality_res["velocity"], modality_res["biomechanical"]],
        labels=["Coordinates", "Velocity", "Biomechanical"],
        autopct='%1.1f%%', colors=["#4c72b0", "#55a868", "#c44e52"], startangle=90
    )
    ax1.set_title("Feature Modality Attribution Share", fontsize=11, fontweight='bold')
    
    ax2.pie(
        [region_res["Head"], region_res["Spine"], region_res["Forelimbs"], region_res["Hindlimbs"]],
        labels=["Head", "Spine", "Forelimbs", "Hindlimbs"],
        autopct='%1.1f%%', colors=["#8172b0", "#ccb974", "#64b5cd", "#818e63"], startangle=90
    )
    ax2.set_title("Anatomical Body Region Attribution Share", fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase11_modality_body_region_attribution.png"), dpi=300)
    plt.close()

    # Figure 3: Temporal Attribution Heatmap across 128 Timesteps
    fig, ax = plt.subplots(figsize=(10, 4))
    temp_profile = np.mean(np.abs(shap_stack[:, :, :]), axis=(0, 2)) # (128,)
    ax.plot(np.arange(128), temp_profile, color='#d62728', lw=2)
    ax.axvspan(0, 42, color='blue', alpha=0.1, label='Early Phase')
    ax.axvspan(43, 85, color='green', alpha=0.1, label='Middle Phase')
    ax.axvspan(86, 127, color='orange', alpha=0.1, label='Late Phase')
    ax.set_title("Temporal SHAP Attribution Profile (128 Timesteps)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Timestep (Frame Index)", fontsize=10)
    ax.set_ylabel("Mean Absolute SHAP", fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase11_temporal_attribution_heatmap.png"), dpi=300)
    plt.close()

    # Figure 4: Derived Gait Evidence Breakdown
    derived_items = AttributionAggregator.extract_derived_gait_evidence(X_scaled[0], shap_stack[0])
    df_derived = pd.DataFrame(derived_items)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_derived, x="relative_contribution", y="name", hue="name", palette="rocket", legend=False, ax=ax)
    ax.axvline(0, color='black', linestyle='--', lw=1)
    ax.set_title("Level 2 Derived Gait Evidence Contributions", fontsize=12, fontweight='bold')
    ax.set_xlabel("Relative SHAP Directional Contribution", fontsize=10)
    ax.set_ylabel("Derived Gait Concept", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase11_derived_gait_evidence_breakdown.png"), dpi=300)
    plt.close()

    # Figure 5: Stability & Faithfulness Metrics
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics_names = ["Top-5 Jaccard Overlap", "Sign Consistency", "Faithfulness Drop (Prob Delta)"]
    metrics_vals = [stability_res["jaccard_overlap"], stability_res["sign_consistency"], max(faithfulness_res["prob_delta"], 0.0)]
    sns.barplot(x=metrics_names, y=metrics_vals, hue=metrics_names, palette="viridis", legend=False, ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Explanation Stability & Faithfulness Metrics", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score / Metric Ratio", fontsize=10)
    for i, v in enumerate(metrics_vals):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase11_explanation_stability_faithfulness.png"), dpi=300)
    plt.close()

    print(f"[SAVED] 5 QA figures generated in {fig_dir}")
    print("\n" + "=" * 80)
    print("PHASE 11 EXPLAINABILITY EXECUTION COMPLETE — ALL SYSTEMS NOMINAL")
    print("=" * 80)

if __name__ == "__main__":
    main()
