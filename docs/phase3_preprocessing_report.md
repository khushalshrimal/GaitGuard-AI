# GaitGuard AI — Phase 3 Data Cleaning & Quality Assurance Report

## 1. Executive Summary

This report documents the data quality investigation, keypoint trajectory smoothing, temporal windowing, class balance analysis, visual QA plotting, and regression testing performed in **Phase 3: Data Cleaning, Preprocessing & Quality Assurance**.

Key achievements:
* **Pre-Preprocessing Baseline (`docs/phase3_before_preprocessing.md`)**: Recorded exact baseline metrics across all 272 samples and 98 cows.
* **Keypoint Trajectory Smoothing**: Implemented Savitzky-Golay 1D polynomial smoothing ($W=5, p=2$) across time frames for each keypoint $(x(t), y(t))$ channel. Successfully removed high-frequency detector micro-jitter without flattening true biomechanical gait signals (e.g. hoof impact spikes, spine arching).
* **Sample Exclusion Audit**: Determined that **0 samples (0.0%) were excluded** because 100% of samples (272 / 272) passed structural and numerical validation. Unusual walking gaits in severely lame cows were correctly preserved as **legitimate biological signals**, NOT data errors.
* **Class Imbalance & SMOTE Analysis**: Verified target balance (**143 Normal [52.57%]** vs **129 Lame Risk [47.43%]**). Determined that SMOTE or synthetic oversampling is unnecessary and scientifically inappropriate for sequential pose keypoints.
* **Diagnostic QA Plots**: Generated 4 diagnostic figures saved under [`docs/figures/`](file:///c:/Users/khush/OneDrive/Desktop/GaitGuard-AI/docs/figures/):
  1. `sequence_length_distribution.png`
  2. `class_balance.png`
  3. `keypoint_trajectory_smoothing.png`
  4. `animal_sample_distribution.png`
* **Regression & Reproducibility Verification**: Executed 11 total unit and regression tests in `tests/`. Pipeline execution time is **6.36 seconds** and produces **100% bitwise identical cleaned datasets** (`datasets/processed/gaitguard_cleaned_dataset.npz`, 3.75 MB).

---

## 2. Before vs. After Preprocessing Comparison

| Metric / Attribute | Phase 2 Dataset (Before Phase 3) | Phase 3 Cleaned Dataset (After Phase 3) |
| :--- | :--- | :--- |
| **Storage Location** | `datasets/processed/gaitguard_processed_dataset.npz` | `datasets/processed/gaitguard_cleaned_dataset.npz` |
| **File Size** | 3.45 MB | 3.75 MB |
| **Total Samples** | 272 samples | 272 samples |
| **Unique Cows** | 98 unique cows | 98 unique cows |
| **Excluded Samples** | 0 samples (0.0%) | 0 samples (0.0%) |
| **Keypoint Trajectory State** | Raw frame-by-frame detector coordinates | Savitzky-Golay smoothed coordinates ($W=5, p=2$) |
| **Tracking Jitter** | Present (1–3 pixel detector noise per frame) | Removed via local polynomial smoothing |
| **Input Tensor Shape** | `(272, 128, 17, 2)` float32 | `(272, 128, 17, 2)` float32 |
| **Sequence Mask Shape** | `(272, 128)` bool | `(272, 128)` bool |
| **Target Distribution** | 143 Normal (52.6%) vs 129 Lame (47.4%) | 143 Normal (52.6%) vs 129 Lame (47.4%) |
| **Group-Aware Splits** | 5-Fold `GroupKFold` on `animal_id` | 5-Fold `GroupKFold` on `animal_id` (0 overlap verified) |
| **Diagnostic Figures** | None | 4 PNG plots generated in `docs/figures/` |

---

## 3. Cleaning & Transformation Log

| Transformation | Method / Algorithm | Reason | Affected Samples | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Trajectory Noise Removal** | Savitzky-Golay 1D Filter ($W=5, p=2$, mode=`nearest`) | Micro-jitter in raw pose estimates introduces artificial acceleration noise in gait derivative features. | All 272 samples (17 keypoints $\times 2$ axes) | Smooth trajectory curves preserving physical peak amplitudes. |
| **Coordinate Normalization** | Resolution Min-Max scaling $[0.0, 1.0]$ | Scales $1920 \times 1080$ pixel coordinates into bounded range for neural net stability. | All 272 samples | Coordinates bounded strictly in $[0.0, 1.0]$. |
| **Temporal Windowing** | Fixed 128-frame padding + boolean mask | Converts variable sequence lengths ($90 \le T \le 207$) into uniform tensors for batch ML inference. | All 272 samples | Uniform `(272, 128, 17, 2)` shape. |
| **Outlier Assessment** | Biomechanical range inspection | Severe lameness gaits contain jerky motion; must be preserved as true biological signals. | 13 severely lame cows (Score 4) | 100% samples retained (0 deleted as outliers). |
| **Imbalance Treatment** | Target distribution audit | 52.6% vs 47.4% ratio is naturally balanced; SMOTE avoided to prevent time-series distortion. | N/A | Target distribution preserved cleanly. |

---

## 4. Data Excluded & Rationale

* **Total Samples Excluded:** **0 (0.00%)**
* **Reason:** All 272 keypoint trajectories in the primary dataset contain complete 17-keypoint coverage, 0 NaNs, 0 Infs, valid coordinate ranges, and verified animal IDs. No data errors or file corruptions were detected.

---

## 5. Diagnostic Visual Quality Figures

The following diagnostic plots were generated and saved to [`docs/figures/`](file:///c:/Users/khush/OneDrive/Desktop/GaitGuard-AI/docs/figures/):

1. **`sequence_length_distribution.png`**: Visualizes histogram of raw sequence lengths (mean 134.4 frames) relative to the 128-frame target window.
2. **`class_balance.png`**: Displays binary target distribution showing balanced counts (143 Normal vs 129 Lame Risk).
3. **`keypoint_trajectory_smoothing.png`**: Demonstrates before vs after Savitzky-Golay trajectory smoothing on the `LFHoof` Y-coordinate over time.
4. **`animal_sample_distribution.png`**: Shows the distribution of video sequences per unique cow ID (98 unique cows).

---

## 6. Regression & Reproducibility Verification

Automated test runner (`tests/test_phase3_pipeline.py` & `tests/test_data_pipeline.py`) verified:
1. **Raw Validation**: 272/272 samples pass validation.
2. **Keypoint Smoothing**: Savitzky-Golay filtering preserves array shapes and coordinate bounds.
3. **Padded Tensor Shapes**: Shape `(272, 128, 17, 2)` verified.
4. **Zero Animal Leakage**: 5-Fold GroupKFold splits confirmed 0 animal ID overlap between train and val sets.
5. **Phase 2 & Phase 3 Reproducibility**: `run_phase3_pipeline()` executed twice produced **100% bitwise identical arrays**.
6. **Regression Tests**: All 11 unit tests passed cleanly in **21.82 seconds**.
