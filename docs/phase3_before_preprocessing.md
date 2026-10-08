# GaitGuard AI — Phase 3 Pre-Preprocessing Baseline Report

## 1. Baseline Summary & Inventory

This baseline report captures the exact state of the processed dataset prior to applying Phase 3 data cleaning, trajectory smoothing, noise reduction, and quality assurance transformations.

* **Total Samples:** 272 trajectory sequences
* **Total Unique Animals:** 98 unique cows (`animal_id`)
* **Total Video Sequences:** 272 videos (`video_id` matching `sample_id` `"001"` through `"272"`)
* **Target Label Distribution:**
  * Raw Score 1 (Normal Gait): **143 samples (52.57%)**
  * Raw Score 2 (Slightly Lame): **96 samples (35.29%)**
  * Raw Score 3 (Moderately Lame): **20 samples (7.35%)**
  * Raw Score 4 (Severely Lame): **13 samples (4.78%)**
  * Binary Target 0 (Normal): **143 samples (52.57%)**
  * Binary Target 1 (Lameness Risk): **129 samples (47.43%)**
* **Sequence Length Statistics:**
  * Minimum: 90 frames
  * Maximum: 207 frames
  * Mean: 134.37 frames
  * Median: 130.00 frames
  * Uniform Padded Length: 128 frames
* **Keypoint Topology:** 17 anatomical keypoints
* **Current Data Shapes:**
  * `padded_keypoints`: `(272, 128, 17, 2)` float32
  * `sequence_masks`: `(272, 128)` bool
  * `raw_labels`: `(272,)` int32
  * `binary_targets`: `(272,)` int32
  * `animal_ids`: `(272,)` int32
* **Current Processed File Size:** 3.45 MB (`datasets/processed/gaitguard_processed_dataset.npz`)
* **Data Quality Baseline:**
  * Missing Values (NaNs): **0 (0.00%)**
  * Infinite Values (Infs): **0 (0.00%)**
  * Invalid Coordinates ($<0$ or NaN): **0 (0.00%)**
  * Corrupted Files: **0 (0.00%)**
  * Exact Duplicate Trajectories: **0 (0.00%)**

---

## 2. Baseline Keypoint Quality Metrics

| Index | Keypoint Name | Raw Coords | Missing % | NaN % | Coordinate Range (Min - Max) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `LFHoof` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 1 | `LFAnkle` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 2 | `LFKnee` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 3 | `RFHoof` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 4 | `RFAnkle` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 5 | `RFKnee` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 6 | `LHHoof` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 7 | `LHAnkle` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 8 | `LHKnee` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 9 | `RHHoof` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 10 | `RHAnkle` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 11 | `RHKnee` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 12 | `Nose` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 13 | `HeadTop` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 14 | `Spine1` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 15 | `Spine2` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |
| 16 | `Spine3` | $(X, Y)$ | 0.0% | 0.0% | $0.0000 - 0.9891$ | Valid |

---

## 3. Investigation Targets for Phase 3

While raw data contains 0 NaNs, keypoint trajectories extracted from video tracking naturally contain minor **high-frequency tracking jitter** (frame-to-frame pixel noise caused by pose detector variance).

Phase 3 will evaluate and implement:
1. **Savitzky-Golay / Moving-Average Trajectory Smoothing**: Removing high-frequency noise from keypoint trajectories without flattening true physical motion peaks (e.g. hoof strikes, back arch elevation).
2. **Duplicate & Leakage Audit**: Verifying zero duplicate sequences across animal IDs.
3. **Anatomical Scale & Torso Normalization Analysis**: Comparing resolution Min-Max scaling vs reference bone length scaling (e.g., `Spine1` to `Spine3` distance).
4. **Diagnostic Plotting**: Creating visual QA figures (sequence length distribution, target balance, smoothed vs raw keypoint trajectories) saved to `docs/figures/`.
