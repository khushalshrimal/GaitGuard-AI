# GaitGuard AI — Phase 14 Frozen Evaluation Protocol

## 1. Objective & Scope
This protocol freezes all parameters of the GaitGuard AI inference stack prior to running Phase 14 scientific evaluation. No hyperparameter tuning, retraining, recalibration, threshold modification, feature schema changes, or architectural adjustments are permitted during or after Phase 14 evaluation.

## 2. Frozen Model & System Specifications
- **Input Representation**: $(128, 76)$ matrix comprising:
  - 34 normalized keypoint coordinates $(x, y)$ for 17 biomechanical landmarks.
  - 34 temporal velocity features $(\Delta x, \Delta y)$.
  - 8 gait features (torso length, back arch curvature, head nodding amplitude, torso-normalized stride length, stance timing asymmetry, walking speed, knee flexion range, sequence length).
- **Model Architecture**: Single-layer Bidirectional LSTM (BiLSTM), hidden dimension 32, dropout 0.3, dense linear layer 16, sigmoid output.
- **Probability Calibration**: Platt scaling / Sigmoid calibration.
- **Screening Decision Parameters**:
  - Screening threshold $\tau = 0.34$
  - Inconclusive region = $[0.24, 0.44]$
  - Decisions:
    - Calibrated $p < 0.24 \implies \text{NORMAL}$
    - $0.24 \le p \le 0.44 \implies \text{INCONCLUSIVE}$
    - Calibrated $p > 0.44 \implies \text{LAMENESS\_RISK}$
- **Quality Gate Thresholds**:
  - Keypoint Coverage $\ge 0.60$
  - Blur Indicator $\le 0.70$
  - Motion Quality $\ge 0.40$
  - Framing Quality $\ge 0.50$
  - Minimum Valid Frames $\ge 30$

## 3. Evaluation Tracks
- **Track A — Unlabelled Field Robustness**: Evaluates Quality Gate, pose extraction robustness, controlled perturbation stability, SHAP feature attribution stability, and runtime benchmarks on field videos. *Performance classification metrics (accuracy, recall, precision, F1, ROC-AUC) are explicitly blocked when reference labels are missing.*
- **Track B — Labelled External Validation**: Computes classification performance (sensitivity, specificity, PPV, NPV, F1, ROC-AUC, PR-AUC, ECE, Brier score) when independent reference labels are present.

## 4. Leakage Prevention Rules
1. Zero training cow IDs may exist in validation sets ($\text{ANIMAL\_OVERLAP} = 0$).
2. Preprocessing & coordinate normalizations must use frozen scalar parameters learned from training.
3. Threshold tuning or calibration curve fitting on validation data is strictly prohibited.
