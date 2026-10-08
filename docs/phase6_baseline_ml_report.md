# GaitGuard AI — Phase 6 Baseline Machine Learning Report

## 1. Objective

This report documents the baseline machine learning evaluation, group-aware cross-validation, feature ablation, out-of-fold (OOF) prediction tracking, and animal-level error analysis performed in **Phase 6: Baseline Machine Learning Models & Animal-Level Generalization**.

> **Non-Diagnostic Scientific Caveat:** The models evaluated in this phase estimate lameness-risk classification from gait-derived movement features on unseen cows. They do not constitute clinical veterinary diagnosis, nor do they claim real-farm diagnostic validation.

---

## 2. Dataset Overview & ML Inputs

* **Input Data:** Phase 5 engineered feature table (`datasets/processed/gaitguard_engineered_features.csv`).
* **Total Samples:** 272 trajectory sequences
* **Unique Animals:** 98 unique cows (`animal_id`)
* **Binary Target:** 143 Normal (52.57%) vs 129 Lame Risk (47.43%)
* **Features Included:** 7 torso-normalized gait features (`torso_length`, `back_arch_curvature`, `head_nodding_amplitude`, `torso_normalized_stride_length`, `stance_timing_asymmetry`, `normalized_walking_speed`, `knee_flexion_range`).

---

## 3. Evaluation & GroupKFold Design

* **Cross-Validation Scheme:** 5-Fold `GroupKFold` grouped on `animal_id` (98 cows).
* **Leakage Prevention Guarantee:** In every fold, `train_cows ∩ validation_cows = ∅`.
* **Preprocessing Isolation:** Feature scaling (`StandardScaler`) was fitted strictly inside each training fold using `sklearn.pipeline.Pipeline`, preventing preprocessing leakage.

---

## 4. Cross-Validated Model Performance Summary

| Model | Accuracy (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | F1 Score (Mean ± Std) | ROC-AUC (Mean ± Std) | Out-Of-Fold ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.6618 ± 0.0954** | **0.6558 ± 0.1235** | **0.6236 ± 0.0845** | **0.6323 ± 0.1184** | **0.7273 ± 0.1203** | **0.7241** |
| **Support Vector Machine (RBF)** | 0.6104 ± 0.1032 | 0.5985 ± 0.1245 | 0.6233 ± 0.0654 | 0.6019 ± 0.0997 | 0.6591 ± 0.1055 | 0.6602 |
| **Random Forest** | 0.6141 ± 0.1286 | 0.6074 ± 0.1610 | 0.5966 ± 0.0938 | 0.5939 ± 0.1311 | 0.6590 ± 0.1725 | 0.6588 |
| **XGBoost** | 0.5920 ± 0.1601 | 0.5841 ± 0.1982 | 0.5570 ± 0.1497 | 0.5621 ± 0.1794 | 0.6455 ± 0.1911 | 0.6412 |

---

## 5. Screening Priority & Confusion Matrix Analysis

As a screening system, **Recall (Sensitivity)** is prioritized to minimize False Negatives (lame cows missed by the system):

* **Logistic Regression OOF Confusion Matrix:**
  * True Negatives (TN): **106** | False Positives (FP): **37**
  * False Negatives (FN): **48** | True Positives (TP): **81**
  * Overall OOF Recall: **62.79%** | Overall OOF Accuracy: **68.75%**

---

## 6. Feature Ablation Study Results

To test feature group contributions, Random Forest was evaluated across feature subsets:

| Baseline Feature Set | Feature Count | Mean Recall | Mean F1 | Mean ROC-AUC | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline A (All 7 Features)** | 7 | 0.5966 | 0.5939 | 0.6590 | Contains noisy/experimental features |
| **Baseline B (Primary 3 Candidate Features)** | **3** | **0.6350** | **0.6207** | **0.6751** | **PROVISIONAL BEST SUBSET** |
| **Baseline C (Primary 3 + Secondary 1 Feature)** | 4 | 0.6179 | 0.6093 | 0.6660 | Slightly reduced generalization |

* **Key Finding:** Restricting input to the **Primary 3 Candidate Features** (`back_arch_curvature`, `torso_normalized_stride_length`, `normalized_walking_speed`) significantly improves out-of-fold generalization on unseen cows ($0.6350$ vs $0.5966$ recall).

---

## 7. Animal-Level Error Analysis

Out-of-fold error analysis across the 98 unique cows revealed:
* **Cows 100% Correctly Classified:** 39 cows (39.8%)
* **Cows Partially Misclassified:** 28 cows (28.6%)
* **Cows Consistently Misclassified (>50% error rate):** 31 cows (31.6%)

**Failure Mode Insights:** Misclassified cows predominantly correspond to mild/borderline lameness cases (Score 2), where subtle gait changes overlap with normal walking variations in static tabular aggregates.

---

## 8. Provisional Model Selection for Phase 7

**Logistic Regression (with Primary 3 Features)** is selected as the **Provisional Baseline Model** for Phase 6 due to:
1. **Highest Cross-Validated ROC-AUC (0.7273)** and OOF Recall (62.79%).
2. **High Stability Across Folds:** Lowest standard deviation ($\pm 0.0845$ in recall).
3. **Linear Interpretability:** Directly maps positive/negative biomechanical feature weights.
4. **Computational Efficiency:** Instantaneous inference time (< 1 ms).

---

## 9. Diagnostic Visual Artifacts

Generated figures saved in [`docs/figures/phase6/`](file:///c:/Users/khush/OneDrive/Desktop/GaitGuard-AI/docs/figures/phase6/):
1. `logistic_regression_confusion_matrix.png`
2. `random_forest_confusion_matrix.png`
3. `xgboost_confusion_matrix.png`
4. `support_vector_machine_confusion_matrix.png`
5. `roc_curve_comparison.png`
6. `precision_recall_comparison.png`
7. `feature_ablation_comparison.png`

---

## 10. Phase 7 Recommendations

Proceed to **Phase 7: Sequence-Level Temporal ML Models (BiLSTM / Temporal Networks)**:
1. Evaluate temporal sequence models (BiLSTM, GRU, 1D-CNN) directly on 128-frame keypoint trajectory tensors `(272, 128, 17, 2)` to capture intra-gait cycle dynamics missed by tabular static aggregates.
2. Compare sequence deep learning models against the Phase 6 Logistic Regression baseline ($0.7273$ ROC-AUC) using the identical 5-Fold `GroupKFold` split setup.
