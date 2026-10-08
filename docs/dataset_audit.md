# GaitGuard AI — Phase 1 Dataset Audit Report

## 1. Executive Summary

This report documents the dataset and project audit performed for **GaitGuard AI** (Phase 1). The objective of Phase 1 is to evaluate existing codebase artifacts, discover locally available datasets, assess ML feasibility for cattle lameness risk screening, analyze data quality, determine animal identity leakage risks, and establish video pipeline compatibility before training any models or writing application backends.

Key audit findings:
* **Primary Training Dataset:** **Russello et al. Dataset** (from `lstm-lameness-detection-main`), consisting of **272 keypoint trajectories** extracted via T-LEAP and matched with expert mobility/lameness scores.
* **Primary Dataset Quality:** Clean and high quality. 0 missing values (NaNs), 0 infinite values, uniform 53-column structure (17 keypoints $\times$ 3 coordinates $(x, y, \text{likelihood})$), sequence length ranging from 90 to 207 frames (mean 134.37 frames).
* **Target Labeling:** Lameness scores range from 1 to 4 (`hard_vote`). Binary classification thresholding ($\le 1$ as Normal, $>1$ as Lame) yields **143 Normal (52.57%)** vs **129 Lame (47.43%)** samples, providing a balanced target distribution.
* **Animal Identity & Data Leakage Risk:** **CRITICAL**. The 272 video samples originate from only **98 unique cows**. 74 cows appear in multiple videos (up to 8 videos for Cow ID 51). A random train/test split will cause severe data leakage. **Group-aware splitting (`GroupKFold` or `GroupShuffleSplit` on `ID`) is strictly required.**
* **Secondary Datasets Evaluated:**
  * **CowScreeningDB:** Contains leg-mounted IMU sensor data (accelerometers, gyroscopes, heart rate), NOT video keypoints. The dataset `.zip` file is encrypted and password-protected via a formal license agreement. Classified as **AUXILIARY DATA / INACCESSIBLE**.
  * **CVB Dataset (Cattle Visual Behaviors):** Contains 502 15-second videos annotated with 11 visual behaviors (e.g., grazing, walking, standing), but **lacks veterinary lameness labels**. Classified as **NOT SUITABLE FOR SUPERVISED TRAINING**.

---

## 2. Project Structure

The project root directory `c:\Users\khush\OneDrive\Desktop\GaitGuard-AI` contains the following layout:

```text
GaitGuard-AI/
├── Datasets/
│   ├── lstm-lameness-detection-main/          # Russello et al. repository & dataset
│   │   └── lstm-lameness-detection-main/
│   │       ├── cfg/config.yml                 # Model configuration file
│   │       ├── data/
│   │       │   ├── videos_keypoints/          # 272 keypoint CSV files (001.csv - 272.csv)
│   │       │   └── videos_lameness_scores.csv # CSV file mapping Video, ID, hard_vote
│   │       ├── datasets/tsKeypointDataset.py  # PyTorch Dataset implementation
│   │       ├── models/KPLSTM.py               # Bidirectional LSTM PyTorch model definition
│   │       ├── main.py                        # Model training and evaluation script
│   │       └── requirements.txt               # PyTorch / Scikit-learn dependencies
│   ├── CowScreeningDB-A-public-database-for-lameness-detection-main/
│   │   └── CowScreeningDB-A-public-database-for-lameness-detection-main/
│   │       ├── CowScreeningDB-Dataset v1.0.zip # Password-protected dataset archive
│   │       ├── CowScreeningDB_License_Agreement.pdf # License agreement form
│   │       └── Cow_Data_Channel_Defintion.txt # IMU sensor channel layout
│   └── CVB_dataset-main/
│       └── CVB_dataset-main/
│           └── README.md                      # CVB dataset overview (CSIRO action dataset)
├── docs/
│   └── dataset_audit.md                       # Comprehensive Phase 1 Audit Report
└── scripts/
    └── audit_datasets.py                      # Automated and repeatable audit script
```

---

## 3. Dataset Inventory

| Dataset Name | Folder Location | File Types | Count | Data Modality | Labels Present | Classification | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Russello et al. Dataset** | `Datasets/lstm-lameness-detection-main/` | `.csv`, `.yml`, `.py` | 272 CSV keypoint files + 1 master score CSV | Pose keypoint trajectories ($N$ frames $\times 17$ keypoints $\times 3$) | Lameness scores (1, 2, 3, 4) & Animal IDs | **PRIMARY TRAINING DATA** | Contains high-quality side-view 2D pose trajectories with verified lameness ground truth labels and cow IDs. |
| **CowScreeningDB** | `Datasets/CowScreeningDB-.../` | `.zip`, `.rar`, `.pdf`, `.txt` | Archive files + PDF | Leg-mounted IMU sensor readings | Leg illness degree (1 to 4) | **AUXILIARY / INACCESSIBLE DATA** | Uses wearable sensor hardware (accelerometer/gyroscope) rather than computer vision pose keypoints. Raw zip is password-protected. |
| **CVB Dataset** | `Datasets/CVB_dataset-main/` | `.md` | 1 README doc (external download required) | 502 15-second video clips | 11 visual behaviors (grazing, walking, etc.) | **NOT SUITABLE FOR TRAINING** | Focuses on general behavior recognition and completely lacks veterinary lameness scoring labels. |

---

## 4. Primary Dataset Details (Russello et al.)

* **Total Samples:** 272 keypoint trajectory sequences (CSV files `001.csv` through `272.csv`).
* **Ground Truth File:** `videos_lameness_scores.csv` (Columns: `Video`, `ID`, `hard_vote`).
* **Data Origin:** Pose keypoints extracted using T-LEAP (Tracking-based Locomotion Estimation via Animal Pose) from side-view videos of walking dairy cows.

---

## 5. Keypoint Format

Each sample file contains 53 columns structured as: `video`, `frame`, followed by 17 keypoints with 3 coordinates each: `(keypoint_x, keypoint_y, keypoint_likelihood)`.

### List of 17 Keypoints:
1. `LFHoof` (Left Front Hoof)
2. `LFAnkle` (Left Front Ankle)
3. `LFKnee` (Left Front Knee)
4. `RFHoof` (Right Front Hoof)
5. `RFAnkle` (Right Front Ankle)
6. `RFKnee` (Right Front Knee)
7. `LHHoof` (Left Hind Hoof)
8. `LHAnkle` (Left Hind Ankle)
9. `LHKnee` (Left Hind Knee)
10. `RHHoof` (Right Hind Hoof)
11. `RHAnkle` (Right Hind Ankle)
12. `RHKnee` (Right Hind Knee)
13. `Nose` (Nose tip)
14. `HeadTop` (Top of head / Crown)
15. `Spine1` (Withers / Front Spine)
16. `Spine2` (Mid-back / Center Spine)
17. `Spine3` (Hook bone / Rear Spine / Sacrum)

### Keypoint Data Properties:
* **Coordinate Dimensions:** Unnormalized pixel coordinates ($X \in [0.0, 1907.24]$, $Y \in [0.0, 1000.70]$).
* **Confidence Values:** `likelihood` column is uniformly `1.0` in the post-processed CSV files.
* **Temporal Structure:** Min sequence length = 90 frames, Max = 207 frames, Mean = 134.37 frames, Median = 130 frames.

---

## 6. Label Analysis

Ground truth label is stored in the `hard_vote` column of `videos_lameness_scores.csv`.

* **Score Range:** Integers 1, 2, 3, and 4 (representing mobility scores).
* **Score Distribution:**
  * Score 1 (Non-Lame / Sound Gait): **143 samples (52.57%)**
  * Score 2 (Slightly Lame): **96 samples (35.29%)**
  * Score 3 (Moderately Lame): **20 samples (7.35%)**
  * Score 4 (Severely Lame): **13 samples (4.78%)**
* **Task Formulation:**
  * **Binary Classification (Standard Setup):** Threshold $\le 1$ vs $>1$.
    * Non-Lame (Class 0): **143 (52.57%)**
    * Lame (Class 1): **129 (47.43%)**
  * **Multiclass Classification:** 4 classes (Severe class imbalance present for scores 3 and 4).
* **Severity Note:** While scores 1–4 indicate mobility degrees, severe class imbalance (only 13 severe cases) makes 4-class regression/classification harder without reweighting or binary grouping.

---

## 7. Animal Identity & Data Leakage Risk

* **Animal Count:** The 272 videos belong to **98 unique cow IDs** (`ID` column).
* **Multi-Sample Distribution:** 74 cows have multiple videos recorded across different sessions/days.
  * Top cow sample counts: Cow ID 51 (8 videos), Cow ID 44 (7 videos), Cow IDs 85, 38, 18 (6 videos each).
* **Data Leakage Hazard:** A simple `train_test_split(shuffle=True)` will randomly place different videos of the same cow into both training and validation/test sets. The ML model would memorize cow-specific visual geometry/background features rather than learning general lameness gait mechanics, leading to artificially high validation scores and poor real-world generalization.
* **Required Mitigation:** **Group-Aware Splitting** (`GroupKFold` or `GroupShuffleSplit` grouped by `ID`).

---

## 8. Data Quality Audit

PROGRAMMATIC AUDIT RESULTS (`scripts/audit_datasets.py`):
* **Missing Values (NaNs):** **0 NaNs** across all 272 files.
* **Infinite Values (Infs):** **0 Infs** across all 272 files.
* **Corrupted / Malformed Files:** **0 files**.
* **Column Consistency:** 100% of files have exactly 53 columns matching the schema.
* **Duplicate Trajectories:** 0 duplicate keypoint sequences found.

---

## 9. License Audit

* **Code License:** `UNSTATED / NOT VERIFIED`. (Russello et al. GitHub repository does not contain an explicit open-source license file like MIT or Apache 2.0).
* **Dataset License:** `DATA LICENSE: NOT VERIFIED`. Academic benchmark data linked to published research (*Smart Agricultural Technology*, 2026).
* **CowScreeningDB License:** Proprietary / Restricted academic license (`CowScreeningDB_License_Agreement.pdf` requiring email permission).

---

## 10. ML Feasibility

* **Model Input:** Temporal sequence of 17 keypoint coordinates over $T$ frames: shape $(T, 17, 2)$ or flattened $(T, 34)$.
* **Target Output:** Binary Lameness Risk (0: Non-lame, 1: Lame Risk) or ordinal score (1..4).
* **Machine Learning Task:** Supervised Sequence Classification / Binary Classification.
* **Candidate Gait Features (Phase 2/3 Feature Engineering):**
  1. **Spine Arch / Back Curvature:** Height / curvature of `Spine1`, `Spine2`, `Spine3` relative to hip line (lame cows arch their back upward).
  2. **Head Bobbing / Asymmetry:** Vertical displacement of `HeadTop` and `Nose` synchronized with hoof strikes.
  3. **Stride Length & Timing:** Distance between successive hoof ground contacts (`LFHoof`, `RFHoof`, `LHHoof`, `RHHoof`).
  4. **Step Symmetry & Stance Duration:** Time ratio of left vs. right limb ground contact.
  5. **Joint Flexion / Range of Motion:** Angle trajectories at knees (`LFKnee`, `RFKnee`, `LHKnee`, `RHKnee`) and ankles.

---

## 11. Real Video Compatibility

Final Target Pipeline:
$$\text{Input Video} \longrightarrow \text{Pose Estimator} \longrightarrow \text{17 Keypoints} \longrightarrow \text{Preprocessing} \longrightarrow \text{Gait Feature Extraction} \longrightarrow \text{ML Classifier} \longrightarrow \text{Lameness Risk Score}$$

Compatibility Requirements:
1. **Keypoint Topology Match:** Real-time video pose estimator (e.g. YOLOv8-pose / DeepLabCut / Animal-Pose) MUST output or be mapped to the exact 17 anatomical keypoints used by Russello et al.
2. **Side-View Camera Placement:** Videos must capture the lateral profile of walking cattle so that back curvature, head movements, and all 4 limbs are clearly visible without heavy frontal occlusion.
3. **Normalization Requirements:** Raw pixel coordinates must be normalized for scale, resolution, and walking speed relative to cow height (e.g. distance between head and spine points).

---

## 12. Limitations & Unknowns

* `UNKNOWN — NOT VERIFIED`: Real-world FPS of original raw videos from which keypoints were extracted (assumed 25–30 FPS).
* `UNKNOWN — NOT VERIFIED`: Camera distance and exact focal length calibration in original recording setup.
* **Dataset Size Limitation:** 272 sequence samples is relatively small for deep end-to-end architectures without domain-specific gait feature extraction or strong regularization.

---

## 13. Phase 2 Requirements

Before moving to Phase 2 (Data Preprocessing & Feature Engineering):
1. Implement sequence trimming / padding or fixed frame sampling.
2. Implement spatial coordinate normalization (e.g., bounding-box scale normalization or reference-bone length scaling).
3. Implement `GroupKFold` split generator on cow `ID`.
4. Implement mathematical gait feature extraction functions (back arch height, head nod amplitude, stride length, limb phase asymmetry).
