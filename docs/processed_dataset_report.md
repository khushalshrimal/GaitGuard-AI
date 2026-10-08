# GaitGuard AI — Processed Dataset & Data Pipeline Report (Phase 2)

## 1. Executive Summary

This report documents the dataset loading, validation, cleaning, standardization, coordinate normalization, temporal windowing, and reproducibility verification performed in **Phase 2: Dataset Loading, Standardization & Reproducible Data Pipeline**.

Key achievements:
* **Raw Dataset Ingestion**: Successfully ingested all **272 raw keypoint trajectory samples** and ground-truth mobility scores from the Russello et al. primary dataset without modifying raw files.
* **Automated Data Validation**: Programmatically validated structural, numerical, metadata, and label integrity across all samples. **100% of samples (272 / 272) passed validation** with 0 NaNs, 0 Infs, and 0 corrupted sequences.
* **Canonical Standardization**: Standardized keypoint topology to 17 anatomical landmarks `(T, 17, 2)` and mapped raw mobility scores (`hard_vote` 1..4) to a balanced **Binary Target** (0: Normal [143 samples / 52.6%], 1: Lame Risk [129 samples / 47.4%]).
* **Spatial Coordinate Normalization**: Normalized pixel coordinates ($1920 \times 1080$) into relative bounding scale $[0.0, 1.0]$.
* **Temporal Window Standardization**: Standardized sequence length to a uniform 128-frame window `(272, 128, 17, 2)` accompanied by explicit boolean sequence masks `(272, 128)`.
* **Group-Aware Splitting**: Computed leak-free 5-fold `GroupKFold` splits grouped by `animal_id` (98 unique cows) guaranteeing 0 animal overlap between training and validation/testing folds.
* **Reproducibility & Performance**: Exported binary processed dataset to `datasets/processed/gaitguard_processed_dataset.npz` (3.45 MB) and `datasets/processed/dataset_metadata.json`. Pipeline executes in **2.42 seconds** and achieves **100% bitwise identical output** across repeated runs.

---

## 2. Raw Dataset vs. Processed Dataset Inventory

| Attribute | Raw Primary Dataset | Processed GaitGuard Dataset |
| :--- | :--- | :--- |
| **Storage Location** | `Datasets/lstm-lameness-detection-main/.../` | `datasets/processed/gaitguard_processed_dataset.npz` |
| **File Format** | 272 individual `.csv` files + 1 score `.csv` | 1 compressed `.npz` archive + 1 metadata `.json` |
| **File Size** | ~9.2 MB (raw CSV files) | 3.45 MB (compressed NumPy binary) |
| **Total Samples** | 272 samples | 272 samples (0 excluded) |
| **Unique Cows** | 98 unique cows (`ID`) | 98 unique cows (`animal_id`) |
| **Keypoints per Frame** | 17 keypoints $\times$ 3 columns $(x, y, \text{likelihood})$ | 17 keypoints $\times$ 2 coordinates $(x, y)$ |
| **Coordinate Range** | Raw pixels ($x \in [0, 1920]$, $y \in [0, 1080]$) | Spatially normalized floats ($x \in [0.0, 1.0]$, $y \in [0.0, 1.0]$) |
| **Sequence Length** | Variable ($90 \le T \le 207$ frames) | Fixed uniform window (128 frames) + boolean mask |
| **Target Representation** | Raw integer scores `1, 2, 3, 4` | Binary integer `0` (Normal) vs `1` (Lame Risk) |
| **Splitting Strategy** | Unspecified | 5-Fold Leak-Free `GroupKFold` on `animal_id` |

---

## 3. Cleaning & Decision Log

| Problem | Detection Method | Decision | Rationale | Affected Count |
| :--- | :--- | :--- | :--- | :--- |
| **Missing Values (NaN)** | `DataValidator` scan across all 272 CSVs | No repair / No deletion needed | Dataset contains 0 NaNs across all keypoints. | 0 samples |
| **Infinite Values (Inf)** | `DataValidator` numeric check | No repair / No deletion needed | Dataset contains 0 Infs. | 0 samples |
| **Corrupted CSV Files** | Pandas CSV parse test | Retain all files | 100% of CSV files parse cleanly with 53 columns. | 0 samples |
| **Likelihood Column** | Column schema check | Drop 3rd likelihood column | Likelihood values are uniformly `1.0` in post-processed CSVs; dropping reduces array memory without losing information. | All 272 samples |
| **Variable Sequence Length** | Length distribution check | Truncate / Zero-pad to 128 frames + mask | ML sequence models (LSTMs / Transformers / CNNs) require uniform tensor shapes or explicit sequence masks. | All 272 samples |

---

## 4. Preprocessing Transformations

### A. Coordinate Normalization
$$\hat{x}_{t, k} = \frac{x_{t, k}}{1920.0}, \quad \hat{y}_{t, k} = \frac{y_{t, k}}{1080.0}$$
* Preserves spatial aspect ratio.
* Constrains coordinates strictly within $[0.0, 1.0]$.
* Prevents scale distortion across different video resolutions.

### B. Temporal Padding & Masking
For a sample with raw length $T$:
* If $T \ge 128$: Keep first 128 frames ($\text{mask}[0..127] = \text{True}$).
* If $T < 128$: Copy $T$ frames, fill remaining $128 - T$ frames with $0.0$ ($\text{mask}[0..T-1] = \text{True}, \text{mask}[T..127] = \text{False}$).

### C. Target Label Mapping
$$\text{Binary Target} = \begin{cases} 0 & \text{if } \text{hard\_vote} = 1 \text{ (Normal Gait)} \\ 1 & \text{if } \text{hard\_vote} \in \{2, 3, 4\} \text{ (Lameness Risk)} \end{cases}$$

---

## 5. Group-Aware Split Validation (GroupKFold)

To prevent animal-level data leakage, a 5-Fold `GroupKFold` split was generated on `animal_id`:

```text
Fold 0: 78 Train Cows (217 samples) | 20 Val Cows (55 samples) | Overlap = 0 cows
Fold 1: 78 Train Cows (217 samples) | 20 Val Cows (55 samples) | Overlap = 0 cows
Fold 2: 79 Train Cows (218 samples) | 19 Val Cows (54 samples) | Overlap = 0 cows
Fold 3: 79 Train Cows (218 samples) | 19 Val Cows (54 samples) | Overlap = 0 cows
Fold 4: 78 Train Cows (218 samples) | 20 Val Cows (54 samples) | Overlap = 0 cows
```

---

## 6. Verification & Reproducibility Results

Automated test suite `tests/test_data_pipeline.py` verified:
1. **Raw DataLoader**: 272/272 samples loaded.
2. **Data Validator**: 272/272 valid samples, 0 invalid.
3. **Shape & Bounds**: Padded shape `(272, 128, 17, 2)`, coordinate bounds $0.0000 \le \hat{x}, \hat{y} \le 0.9891$.
4. **Label Mapping**: Score 1 mapped to 0, Scores 2-4 mapped to 1.
5. **Leakage Prevention**: 0 animal overlap across all 5 cross-validation folds.
6. **Reproducibility**: `run_pipeline()` executed twice produced **100% bitwise identical `.npz` arrays**.
