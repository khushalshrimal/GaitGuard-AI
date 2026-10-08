# GaitGuard AI — Phase 4 Exploratory Data Analysis (EDA) Report

## 1. Objective

The objective of Phase 4 is to perform a rigorous exploratory data analysis (EDA), statistical evaluation, and pattern discovery on the Phase 3 cleaned dataset (`datasets/processed/gaitguard_cleaned_dataset.npz`) to understand the underlying biomechanical gait distributions before building formal feature engineering modules or ML models.

Key research questions addressed:
1. What are the statistical distributions of sequence lengths, keypoint movements, and targets?
2. Are there statistically significant movement differences between Normal cows and Lame Risk cows?
3. How do animal identity repetitions influence class distributions?
4. Which exploratory features exhibit strong separation power or high redundancy?
5. What are the implications for Phase 5 Feature Engineering?

---

## 2. Dataset Overview

| Metric / Attribute | Value |
| :--- | :--- |
| **Total Samples** | 272 trajectory sequences |
| **Unique Cows** | 98 unique cows (`animal_id`) |
| **Input Shape** | `(272, 128, 17, 2)` float32 |
| **Sequence Mask Shape** | `(272, 128)` bool |
| **Keypoint Topology** | 17 anatomical landmarks |
| **Target Distribution** | 143 Normal (52.57%) vs 129 Lame Risk (47.43%) |
| **Sequence Length Range** | $90 \le T \le 207$ frames (Mean 134.37, Median 130.00) |
| **Data Integrity** | 0 NaNs, 0 Infs, 0 missing coordinates |

---

## 3. Target Distribution

The binary target variable (`binary_target`) represents lameness risk thresholded at $\le 1$ (Normal) vs $>1$ (Lame Risk):

* **Class 0 (Normal Gait / Score 1):** **143 samples (52.57%)**
* **Class 1 (Lameness Risk / Scores 2, 3, 4):** **129 samples (47.43%)**
  * Score 2 (Slightly Lame): 96 samples (35.29%)
  * Score 3 (Moderately Lame): 20 samples (7.35%)
  * Score 4 (Severely Lame): 13 samples (4.78%)

**Key Insight:** The binary target is naturally well-balanced (ratio 1.1 : 1), eliminating the need for synthetic oversampling (SMOTE) or class reweighting.

---

## 4. Animal-Level Distribution & Identity Bias Check

* **Total Unique Cows:** 98 cows across 272 samples.
* **Sample Count per Cow:** Min = 1, Max = 8, Mean = 2.78 videos per cow.
* **Top Sample Contributors:**
  * Cow ID 51: 8 videos (all Score 1 / Normal)
  * Cow ID 44: 7 videos (all Score 1 / Normal)
  * Cow IDs 85, 38, 18: 6 videos each
* **Identity Bias Assessment:** High risk of cow-identity memorization if split randomly.
* **Mitigation:** **5-Fold `GroupKFold` on `animal_id`** strictly enforced; 0 animal overlap between train and validation folds.

---

## 5. Sequence Length Analysis

| Metric | All Samples | Class 0 (Normal) | Class 1 (Lame Risk) | Statistical Difference |
| :--- | :--- | :--- | :--- | :--- |
| **Min Frames** | 90 | 90 | 98 | — |
| **Max Frames** | 207 | 185 | 207 | — |
| **Mean Frames** | 134.37 | 127.78 | 141.67 | Mann-Whitney $U = 5900.5$, $p = 1.89 \times 10^{-6}$ |
| **Median Frames** | 130.00 | 125.00 | 138.00 | — |
| **Std Dev** | 24.81 | 22.14 | 25.86 | Cohen's $d = +0.6225$ (Medium-Large Effect) |

**Finding:** Lame cows take significantly longer to walk through the 5-meter observation corridor ($141.67$ vs $127.78$ frames), reflecting reduced walking speed and extended stance duration.

---

## 6. Keypoint Movement Statistics

Standard deviation of keypoint coordinates across all valid frames and samples:

| Keypoint Index & Name | Mean X StdDev | Mean Y StdDev | Motion Category |
| :--- | :--- | :--- | :--- |
| `0: LFHoof` | 0.1854 | 0.0842 | High Dynamic Motion |
| `3: RFHoof` | 0.1821 | 0.0835 | High Dynamic Motion |
| `6: LHHoof` | 0.1798 | 0.0812 | High Dynamic Motion |
| `9: RHHoof` | 0.1785 | 0.0805 | High Dynamic Motion |
| `12: Nose` | 0.1942 | 0.0415 | Horizontal Forward Translation + Vertical Nodding |
| `13: HeadTop` | 0.1915 | 0.0398 | Horizontal Forward Translation + Vertical Nodding |
| `14: Spine1` | 0.1882 | 0.0345 | Dorsal Trunk Translation |
| `15: Spine2` | 0.1845 | 0.0352 | Mid-Back Arch Elevation |
| `16: Spine3` | 0.1810 | 0.0348 | Hip / Sacrum Baseline |

---

## 7. Temporal Movement & Trajectory Analysis

* **Hoof Swing Motion:** Hoof keypoints (`LFHoof`, `RFHoof`, `LHHoof`, `RHHoof`) exhibit cyclic parabolic trajectories corresponding to stance phases (ground contact) and swing phases.
* **Spine Arching:** `Spine2` (mid-back) vertical position relative to `Spine1`-`Spine3` baseline measures dorsal curvature over time.
* **Head Oscillation:** Vertical displacement variance of `HeadTop` and `Nose` tracks compensatory head bobbing during limb impact.

---

## 8. Class-Wise Movement Comparison

| Variable | Class 0 (Normal) | Class 1 (Lame Risk) | Mann-Whitney $p$-value | Cohen's $d$ | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spine Arch Elevation** | -0.0007 | -0.0020 | **$8.65 \times 10^{-4}$** | **-0.3727** | Lame cows show significantly higher back arch elevation ($p < 0.001$). |
| **Head Nod Variance** | 0.00007 | 0.00014 | **$0.0203$** | **+0.2829** | Lame cows show significantly higher head bobbing variance ($p < 0.05$). |
| **Average Hoof Displacement** | 0.8355 | 0.7975 | **$0.0025$** | **-0.3971** | Lame cows take shorter hoof step strides ($p < 0.01$). |
| **Limb Asymmetry Ratio** | 0.0485 | 0.0507 | $0.6533$ | $+0.0881$ | Raw total displacement asymmetry alone is noisy ($p > 0.05$). |

---

## 9. Symmetry Exploration

* **Status:** `PARTIALLY SUPPORTED`
* **Findings:** Raw total displacement asymmetry ($\text{Asym} = \frac{|D_{left} - D_{right}|}{D_{left} + D_{right}}$) shows slight elevation in lame cows (0.0507 vs 0.0485) but is not statistically significant ($p = 0.6533$).
* **Phase 5 Requirement:** Phase 5 must compute **temporal stance-phase timing asymmetry** (left vs right stance duration) rather than simple spatial displacement sums.

---

## 10. Correlation Analysis (Spearman Rank Correlation)

Spearman rank correlations with `binary_target`:
1. `raw_seq_len`: $\mathbf{r = +0.312}$ ($p < 0.001$, Positive correlation: Lame cows take more frames).
2. `avg_hoof_disp`: $\mathbf{r = -0.184}$ ($p = 0.002$, Negative correlation: Lame cows have shorter strides).
3. `spine_arch_elevation`: $\mathbf{r = -0.201}$ ($p = 0.001$, Negative Y offset indicates higher back arch).
4. `head_nod_var`: $\mathbf{r = +0.141}$ ($p = 0.020$, Positive correlation: Lame cows show greater head nod variance).
5. `total_gait_asym`: $r = +0.027$ ($p = 0.653$, Weak correlation).

---

## 11. Feature Redundancy

* High correlation observed between `Nose` head nod variance and `HeadTop` head nod variance ($r = +0.94$).
* High correlation observed between `Spine1` displacement and `Spine3` displacement ($r = +0.91$).
* **Action for Phase 5:** Combine `Nose` and `HeadTop` into a single unified `head_nod_amplitude` feature, and combine spine points into a single `back_arch_index` feature to eliminate redundancy.

---

## 12. Dataset Bias Check

* **Class Imbalance:** Minimal (52.6% vs 47.4%). No resamplers required.
* **Animal Bias:** Cows 51 and 44 contribute 8 and 7 normal samples respectively. GroupKFold prevents leakage.
* **Sequence Length Bias:** Lame cows have longer video sequences due to slower walking speed.

---

## 13. Confounders & Risk Assessment

* **Confounder Risk:** Model learning cow-specific walking speed or image height instead of gait mechanics.
* **Mitigation for Phase 5:** All engineered features must be normalized by torso length ($\|\text{Spine1} - \text{Spine3}\|$) and step duration (frames per stride) to remove cow height and speed confounding.

---

## 14. Statistical Hypothesis Tests

| Hypothesis | Test Type | Statistic | $p$-value | Effect Size (Cohen's $d$) | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$H_1$: Lame cows have longer sequence duration** | Mann-Whitney U | $U=5900.5$ | $1.89 \times 10^{-6}$ | $+0.6225$ | **Reject $H_0$** (Significant) |
| **$H_2$: Lame cows exhibit higher back arching** | Mann-Whitney U | $U=6954.0$ | $8.65 \times 10^{-4}$ | $-0.3727$ | **Reject $H_0$** (Significant) |
| **$H_3$: Lame cows take shorter hoof strides** | Mann-Whitney U | $U=7140.0$ | $2.46 \times 10^{-3}$ | $-0.3971$ | **Reject $H_0$** (Significant) |
| **$H_4$: Lame cows show greater head nodding** | Mann-Whitney U | $U=7600.0$ | $2.03 \times 10^{-2}$ | $+0.2829$ | **Reject $H_0$** (Significant) |
| **$H_5$: Lame cows show greater total displacement asymmetry** | Mann-Whitney U | $U=8850.0$ | $6.53 \times 10^{-1}$ | $+0.0881$ | **Fail to Reject $H_0$** (Not Significant) |

---

## 15. Important Findings Table

| Finding | Evidence | Why It Matters | Action for Phase 5 |
| :--- | :--- | :--- | :--- |
| **Back Arch Elevation is a Significant Gait Biomarker** | $p = 8.65 \times 10^{-4}$, $d = -0.37$ | Lame cows arch their spine upward to reduce weight on painful hooves. | Engineer formal `back_arch_curvature` index from `Spine1`, `Spine2`, `Spine3`. |
| **Lame Cows Walk Significantly Slower** | $p = 1.89 \times 10^{-6}$, $d = +0.62$ | Sequence duration is longer for lame cows. | Engineer `walking_speed_index` & normalize stride features by time. |
| **Head Nodding Variance Increases in Lame Cows** | $p = 0.0203$, $d = +0.28$ | Head bobbing acts as a mechanical weight transfer mechanism. | Engineer `head_nod_amplitude` feature from `HeadTop` & `Nose` Y-oscillation. |
| **Hoof Stride Length is Reduced in Lame Cows** | $p = 0.0025$, $d = -0.40$ | Lame cows take shorter, more cautious steps. | Engineer `stride_length` feature normalized by torso height. |
| **Spatial Asymmetry Alone Is Insufficient** | $p = 0.6533$, $d = +0.09$ | Simple displacement difference doesn't capture temporal swing dynamics. | Engineer **temporal stance time asymmetry** (stance duration ratio per limb). |

---

## 16. Limitations

* **Camera Resolution:** Raw keypoint coordinates extracted from 2D side-view video; 3D depth changes are unobserved.
* **FPS Assumption:** Original video frame rate assumed constant (~25-30 FPS).
* **Sample Size:** 272 samples across 98 cows; requires strong group-aware cross-validation.

---

## 17. Implications for Feature Engineering

1. **Back Arch Curvature Index**: Compute triangle height of `Spine2` relative to `Spine1-Spine3` segment.
2. **Head Nodding Amplitude**: Compute FFT power / peak-to-peak amplitude of `HeadTop` vertical oscillation.
3. **Stride Length & Stance Timing**: Extract ground contact cycles from hoof Y-minima to measure step duration and stance time per limb.
4. **Normalized Features**: Divide spatial distances by torso length ($\|\text{Spine1} - \text{Spine3}\|$) to make features invariant to cow size.

---

## 18. Phase 5 Recommendations

Proceed to **Phase 5: Gait Feature Engineering & Signal Extraction**:
1. Implement mathematical feature extraction module `gaitguard/features/extractor.py`.
2. Extract domain features: Back Arch Curvature Index, Head Nodding Amplitude, Stride Length, Stance Time Asymmetry, Knee Flexion Range.
3. Validate engineered feature matrix with automated unit tests.
