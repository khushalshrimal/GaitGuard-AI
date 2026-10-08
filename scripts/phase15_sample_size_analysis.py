"""
GaitGuard AI - Phase 15 Sample Size Scenario Analysis Script
Calculates statistical sample size requirements for future independent field validation studies.
"""

import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def analyze_sample_size_scenarios():
    print("=" * 70)
    print("GAITGUARD AI — PHASE 15: SAMPLE SIZE SCENARIO ANALYSIS")
    print("=" * 70)

    # Scenarios for target sensitivity (e.g. 80%, 85%, 90%) and desired CI width (e.g. +-5%, +-10%)
    scenarios = []
    target_sensitivities = [0.75, 0.80, 0.85, 0.90]
    ci_widths = [0.05, 0.08, 0.10]
    prevalences = [0.20, 0.30, 0.40]  # Expected lameness prevalence in herd

    z = 1.96  # 95% Confidence Level

    print("\n[STEP 1] Computing Sample Size Requirements across Scenarios...")
    for sens in target_sensitivities:
        for w in ci_widths:
            for prev in prevalences:
                # n_diseased = (z^2 * sens * (1 - sens)) / w^2
                n_diseased = int(np.ceil((z**2 * sens * (1.0 - sens)) / (w**2)))
                n_total_cows = int(np.ceil(n_diseased / prev))

                scenarios.append({
                    "target_sensitivity": sens,
                    "target_ci_half_width": w,
                    "expected_prevalence": prev,
                    "required_lame_cows": n_diseased,
                    "required_total_cows": n_total_cows,
                    "recommended_videos_per_cow": 2,
                    "total_recommended_videos": n_total_cows * 2
                })

    scenarios_df = pd.DataFrame(scenarios)

    # Save CSV
    csv_out = os.path.join(repo_root, "docs", "phase15_sample_size_scenarios.csv")
    scenarios_df.to_csv(csv_out, index=False)
    print(f"  - Saved Sample Size Scenarios CSV to {csv_out}")

    # Generate Report MD
    md_out = os.path.join(repo_root, "docs", "phase15_sample_size_analysis.md")
    with open(md_out, "w") as f:
        f.write("# GaitGuard AI — Phase 15 Statistical Sample Size Analysis Report\n\n")
        f.write("## 1. Objective\n")
        f.write("This report presents statistical sample size scenario planning for future independent field validation studies based on target screening sensitivity and confidence interval precision.\n\n")
        f.write("## 2. Sample Size Scenario Table\n\n")
        f.write("| Target Sensitivity | Desired CI Half-Width | Herd Prevalence | Required Lame Cows | Total Required Cows | Total Recommended Videos |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for s in scenarios[:10]:  # Display top 10 representative scenarios
            f.write(f"| {s['target_sensitivity']*100:.0f}% | ±{s['target_ci_half_width']*100:.0f}% | {s['expected_prevalence']*100:.0f}% | {s['required_lame_cows']} | {s['required_total_cows']} | {s['total_recommended_videos']} |\n")
        f.write("\n## 3. Scientific Note\n")
        f.write("- **Animal-Level Clustering**: Sampling must collect videos from distinct individual animals rather than repeated videos of the same cow to preserve biological independence.\n")

    print(f"  - Saved Sample Size Analysis Report to {md_out}")

if __name__ == "__main__":
    analyze_sample_size_scenarios()
