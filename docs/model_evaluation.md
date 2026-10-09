# GaitGuard AI — Model Evaluation & Performance Report

## 1. Executive Summary

This document summarizes the scientific evaluation methodology, internal cross-validation performance, calibration characteristics, screening threshold selection, and explainability validation for the GaitGuard AI temporal machine learning pipeline.

All metrics reported herein represent **internal animal-level 5-fold cross-validation** on benchmark dataset sequences and do NOT constitute independently verified external clinical performance.

---

## 2. Dataset & Cross-Validation Protocol

### 2.1 Benchmark Dataset Inventory
* **Primary Source**: Russello et al. cattle keypoint dataset.
* **Sample Count**: 272 video sequence samples.
* **Animal Count**: 98 unique individual cows.
* **Sequence Length**: 128 frames (padded/resampled uniform representation).
* **Feature Schema**: 76 compact features per frame (coordinates, velocities, torso-normalized biomechanical indicators).

### 2.2 Leak-Free GroupKFold Splitting
To prevent data leakage caused by multiple video recordings of the same cow appearing in both training and test sets, all cross-validation experiments enforce **animal-level GroupKFold splitting**:
* **Folds**: 5 folds.
* **Animal Overlap**: Exactly **0 cows** shared between training and validation folds in any split.

---

## 3. Internal Model Evaluation Metrics

### 3.1 Primary Classification Performance

| Metric | Internal 5-Fold GroupKFold CV Result | Description |
| :--- | :--- | :--- |
| **Accuracy** | **81.97%** ($\pm 4.25\%$) | Overall correct classification rate |
| **Recall (Sensitivity)** | **78.96%** ($\pm 5.12\%$) | Lameness risk detection rate |
| **F1-Score** | **0.7973** ($\pm 0.0410$) | Harmonic mean of precision and recall |
| **Out-of-Fold (OOF) ROC-AUC**| **0.9016** | Area under Receiver Operating Characteristic curve |
| **Thresholded Screening Recall**| **89.15%** | Recall at optimized screening threshold ($\tau = 0.34$) |

---

## 4. Probability Calibration & 3-Way Triage

### 4.1 Platt Sigmoid Calibration
Raw predictions from neural networks often suffer from overconfidence. GaitGuard applies cross-fitted **Platt Sigmoid Calibration** to map raw BiLSTM logit outputs into calibrated probabilities:
* **Log-Loss Improvement**: Reduced out-of-fold log-loss from 0.542 to 0.418.
* **Brier Score Improvement**: Reduced Brier score from 0.182 to 0.134.

### 4.2 Frozen Threshold & Inconclusive Triage Interval
To prioritize high screening sensitivity while safely flagging ambiguous samples:
* **Screening Threshold ($\tau$)**: **0.34** (selected to guarantee $\ge 85\%$ screening recall).
* **Uncertainty Margin ($\Delta$)**: **$\pm 0.10$**.
* **Inconclusive Interval**: **$[0.24, 0.44]$**.

```
    0.00                    0.24         0.34         0.44                    1.00
      ├──────────────────────┼────────────┼────────────┼──────────────────────┤
      │     NORMAL GAIT      │    INCONCLUSIVE TRIAGE      │    LAMENESS RISK     │
      │  (Low Anomaly Risk)  │  (Uncertainty Band ±0.10)   │ (Veterinary Triage)  │
      └──────────────────────┴─────────────────────────┴──────────────────────┘
```

---

## 5. Explainable AI (SHAP) & Perturbation Stability

### 5.1 Deep SHAP Attributions
* **Level 1 Attributions**: Evaluates feature weightings (back-arch curvature, stance asymmetry, head nodding) to determine risk contribution direction (`INCREASES_RISK` vs `DECREASES_RISK`).
* **Level 2 Gait Metrics**: Extracts domain-specific kinematic parameters (stride length variance, walking speed) for veterinary interpretability.

### 5.2 Controlled Perturbation Stability (Phase 14)
* **Mean Probability Delta**: $\Delta p = 0.0001$ under controlled keypoint coordinate perturbations.
* **Top-5 SHAP Jaccard Similarity**: 100% feature ranking stability across noise iterations.

---

## 6. Scientific Limitations & External Validation Status

> [!WARNING]
> **CRITICAL SCIENTIFIC BOUNDARY STATEMENT**:
> 1. **Internal CV $\neq$ External Field Validation**: The reported 81.97% accuracy and 0.9016 ROC-AUC represent internal cross-validation performance.
> 2. **External Validation Blocked**: Independent external validation on un-seen real-world farm videos remains `BLOCKED — DATA COLLECTION REQUIRED` per Phase 15 protocol.
> 3. **Non-Diagnostic Framing**: This system is an AI-assisted screening decision support tool and must never be cited or marketed as a veterinarian-approved clinical diagnostic device.
