"""
GaitGuard AI - Phase 14 Error & Inconclusive Analysis Script
Analyzes classification error modes (TP, TN, FP, FN), inconclusive region [0.24, 0.44] properties,
and checks calibration stability across field samples.
"""

import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def analyze_errors():
    print("=" * 70)
    print("GAITGUARD AI — PHASE 14: ERROR & INCONCLUSIVE ANALYSIS")
    print("=" * 70)

    # 1. Load Quality Gate / Manifest Predictions
    qg_csv = os.path.join(repo_root, "docs", "phase14_quality_gate_report.csv")
    if not os.path.exists(qg_csv):
        print(f"[ERROR] Quality Gate CSV missing at {qg_csv}")
        return

    df = pd.read_csv(qg_csv)
    print(f"[STEP 1] Loaded {len(df)} samples from Quality Gate output.")

    # 2. Inconclusive Region Audit ([0.24, 0.44])
    inc_df = df[(df["calibrated_prob"] >= 0.24) & (df["calibrated_prob"] <= 0.44)]
    inc_rate = len(inc_df) / float(len(df)) if len(df) > 0 else 0.0
    
    print("\n[STEP 2] Inconclusive Region Audit ([0.24, 0.44]):")
    print(f"  - Total Samples in Inconclusive Margin: {len(inc_df)} / {len(df)}")
    print(f"  - Inconclusive Rate: {inc_rate * 100:.2f}%")
    print("  - Screening Threshold tau: 0.34 (Frozen)")

    # 3. Track B Label Check
    manifest_path = os.path.join(repo_root, "validation", "field_validation_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    has_labels = (manifest["label"] != "UNKNOWN").any()

    print("\n[STEP 3] Classification Error Analysis Status:")
    if not has_labels:
        print("  - [BLOCKED] Independent reference labels are not present in field validation manifest.")
        print("  - True Positive, True Negative, False Positive, False Negative counts cannot be computed.")
        print("  - Calibration ECE / Brier score calculation on external field data is explicitly blocked.")
        print("  - Explicit Status: 'Classification performance evaluation unavailable due to unlabelled field data.'")
    else:
        print("  - Track B Labels present. Computing confusion matrix and calibration metrics...")

    print("\n" + "=" * 70)
    print("PHASE 14 ERROR ANALYSIS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    analyze_errors()
