# GaitGuard AI — Canonical Data Schema Specification (Phase 2)

## 1. Overview

This document specifies the official internal data schema for **GaitGuard AI**. All raw 2D keypoint trajectories and ground-truth lameness annotations extracted from the primary dataset (Russello et al.) are parsed, validated, and transformed into this canonical format for downstream machine learning and gait feature extraction.

---

## 2. Canonical Sample Schema

Each sample in the GaitGuard AI processed dataset corresponds to a single continuous walking sequence of a cow and contains the following fields:

| Field Name | Datatype | Shape | Description | Source Field | Allowed Values / Range | Missing-Value Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sample_id` | `str` | Scalar | Unique string identifier for the sequence sample. | `Video` column (e.g., `"001"`) | `"001"` to `"272"` | Required; must not be null/empty. |
| `animal_id` | `int32` | Scalar | Unique numerical identifier for the cow. | `ID` column in scores CSV | Positive integers (e.g., `87`, `51`) | Required; used for group-aware splitting. |
| `video_id` | `str` | Scalar | Video sequence name. | `Video` column | String matching sample ID | Required. |
| `raw_sequence_length` | `int32` | Scalar | Original number of frames in the raw keypoint trajectory. | Row count of keypoint CSV | $90 \le T \le 207$ (frames) | Required; must be $\ge 1$. |
| `raw_keypoints` | `float64` | $(T, 17, 3)$ | Raw pixel coordinates and likelihood per frame: $(x, y, \text{likelihood})$. | 51 CSV columns in keypoint file | $x \in [0, 1920]$, $y \in [0, 1080]$, likelihood $\in [0, 1]$ | Must contain 0 NaNs / 0 Infs. |
| `standard_keypoints` | `float32` | $(T, 17, 2)$ | Canonical $(x, y)$ keypoint coordinates with likelihood column dropped. | Extracted from `raw_keypoints` | Raw pixel coordinates | Must match 17 canonical keypoint topology. |
| `normalized_keypoints`| `float32` | $(T, 17, 2)$ | Spatially normalized coordinates scaled to frame dimensions $[0, 1]$. | $(x / 1920.0, y / 1080.0)$ | $x \in [0.0, 1.0]$, $y \in [0.0, 1.0]$ | Checked for bounds. |
| `padded_keypoints` | `float32` | $(128, 17, 2)$ | Fixed temporal window representation (clipped at 128 frames or zero-padded). | Transformed from `normalized_keypoints` | Float32 normalized coordinates | Padded frames set to `0.0`. |
| `sequence_mask` | `bool` | $(128,)$ | Boolean mask indicating valid frames (`True`) vs padded frames (`False`). | Derived during temporal padding | `True` or `False` | Length strictly equals 128. |
| `raw_label` | `int32` | Scalar | Original expert mobility/lameness score. | `hard_vote` column | Integers `1, 2, 3, 4` | Required; valid integer 1 to 4. |
| `binary_target` | `int32` | Scalar | Canonical binary target: `0` (Normal) vs `1` (Lame Risk). | Mapped from `raw_label` ($\le 1 \to 0$, $>1 \to 1$) | `0` or `1` | Strictly binary. |
| `split_group` | `int32` | Scalar | Group identifier for GroupKFold validation splits. | Equal to `animal_id` | `1` to `98` | Required for leakage prevention. |

---

## 3. Canonical 17-Keypoint Index Mapping

Keypoints are indexed strictly from `0` to `16` as follows:

```text
Index 0  : LFHoof    (Left Front Hoof)
Index 1  : LFAnkle   (Left Front Ankle)
Index 2  : LFKnee    (Left Front Knee)
Index 3  : RFHoof    (Right Front Hoof)
Index 4  : RFAnkle   (Right Front Ankle)
Index 5  : RFKnee    (Right Front Knee)
Index 6  : LHHoof    (Left Hind Hoof)
Index 7  : LHAnkle   (Left Hind Ankle)
Index 8  : LHKnee    (Left Hind Knee)
Index 9  : RHHoof    (Right Hind Hoof)
Index 10 : RHAnkle   (Right Hind Ankle)
Index 11 : RHKnee    (Right Hind Knee)
Index 12 : Nose      (Nose tip)
Index 13 : HeadTop   (Head crown)
Index 14 : Spine1    (Withers / Front spine)
Index 15 : Spine2    (Mid-back spine)
Index 16 : Spine3    (Rear spine / Hip sacrum)
```

---

## 4. Binary Target Mapping

```text
Raw Score (hard_vote)     Canonical Target     Semantic Meaning     Sample Count (%)
--------------------------------------------------------------------------------------
        1                        0               Normal Gait           143 (52.57%)
        2                        1               Slightly Lame           96 (35.29%)
        3                        1              Moderately Lame          20  (7.35%)
        4                        1              Severely Lame            13  (4.78%)
--------------------------------------------------------------------------------------
Total                                                                  272 (100.0%)
```

---

## 5. Processed Dataset File Format

The processed dataset is saved in binary NumPy compressed archive format:
`datasets/processed/gaitguard_processed_dataset.npz`

Arrays contained in `.npz`:
* `sample_ids`: String array of shape `(272,)`
* `animal_ids`: Int32 array of shape `(272,)`
* `raw_sequence_lengths`: Int32 array of shape `(272,)`
* `normalized_keypoints`: Object array of shape `(272,)` storing variable length arrays `(T_i, 17, 2)`
* `padded_keypoints`: Float32 array of shape `(272, 128, 17, 2)`
* `sequence_masks`: Bool array of shape `(272, 128)`
* `raw_labels`: Int32 array of shape `(272,)`
* `binary_targets`: Int32 array of shape `(272,)`
* `group_ids`: Int32 array of shape `(272,)`
