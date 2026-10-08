# GaitGuard AI — Phase 9 Keypoint Compatibility & Feature Schema Specification

## 1. Executive Summary & Compatibility Audit

Phase 9 establishes the **Real Video Inference Pipeline** connecting raw video input (MP4/AVI/MOV) to Phase 7 BiLSTM model inference and Phase 8 probability calibration triage.

To guarantee scientific validity, real-video inference keypoints MUST strictly match the canonical 17-keypoint schema and 76-feature vector ordering established during Phase 7 model training.

---

## 2. Canonical 17-Keypoint Landmark Schema

The Phase 7 BiLSTM model was trained on 17 keypoint trajectories per frame:

| Index | Keypoint Landmark Name | Anatomical Description | Body Component |
| :---: | :--- | :--- | :--- |
| **0** | `LFHoof` | Left Front Hoof | Front Left Limb |
| **1** | `LFAnkle` | Left Front Ankle (Fetlock) | Front Left Limb |
| **2** | `LFKnee` | Left Front Knee (Carpus) | Front Left Limb |
| **3** | `RFHoof` | Right Front Hoof | Front Right Limb |
| **4** | `RFAnkle` | Right Front Ankle (Fetlock) | Front Right Limb |
| **5** | `RFKnee` | Right Front Knee (Carpus) | Front Right Limb |
| **6** | `LHHoof` | Left Hind Hoof | Hind Left Limb |
| **7** | `LHAnkle` | Left Hind Ankle (Hock) | Hind Left Limb |
| **8** | `LHKnee` | Left Hind Knee (Stifle) | Hind Left Limb |
| **9** | `RHHoof` | Right Hind Hoof | Hind Right Limb |
| **10** | `RHAnkle` | Right Hind Ankle (Hock) | Hind Right Limb |
| **11** | `RHKnee` | Right Hind Knee (Stifle) | Hind Right Limb |
| **12** | `Nose` | Muzzle / Nose Tip | Head |
| **13** | `HeadTop` | Crown / Poll between ears | Head |
| **14** | `Spine1` | Withers / Shoulder Spine Landmark | Torso / Spine |
| **15** | `Spine2` | Mid-back Spine Elevation Peak | Torso / Spine |
| **16** | `Spine3` | Hip / Rump Spine Landmark | Torso / Spine |

---

## 3. 76-Feature Vector Layout & Order

Every single frame $t \in [1, 128]$ produces a feature vector of length **76**:

### Part A: Torso-Normalized Keypoint Coordinates ($F = 34$, Indices 0..33)
For each keypoint $k \in [0, 16]$, the coordinates $(X_k, Y_k)$ are centered relative to the frame torso center ($0.5 \times (\text{Spine1} + \text{Spine3})$) and normalized by sequence median torso length ($D_{\text{torso}} = \|\text{Spine1} - \text{Spine3}\|_2$):
- Indices `0..1`: Normalized `LFHoof` $(X, Y)$
- Indices `2..3`: Normalized `LFAnkle` $(X, Y)$
- ...
- Indices `28..29`: Normalized `Spine1` $(X, Y)$
- Indices `30..31`: Normalized `Spine2` $(X, Y)$
- Indices `32..33`: Normalized `Spine3` $(X, Y)$

### Part B: Keypoint Velocity Vectors ($F = 34$, Indices 34..67)
First-order temporal difference normalized by torso length: $V_t = (KP_t - KP_{t-1}) / D_{\text{torso}}$ (with $V_0 = 0$):
- Indices `34..35`: Velocity `LFHoof` $(\Delta X, \Delta Y)$
- ...
- Indices `66..67`: Velocity `Spine3` $(\Delta X, \Delta Y)$

### Part C: Biomechanical Gait Signals ($F = 8$, Indices 68..75)
- **Index 68:** Back Arch Elevation Index $(0.5 \times (Y_{\text{Spine1}} + Y_{\text{Spine3}}) - Y_{\text{Spine2}}) / D_{\text{torso}}$
- **Index 69:** Head Vertical Elevation $0.5 \times (Y_{\text{Nose}} + Y_{\text{HeadTop}}) / D_{\text{torso}}$
- **Index 70:** Left Hoof Stride Separation $\|\text{LFHoof} - \text{LHHoof}\|_2 / D_{\text{torso}}$
- **Index 71:** Right Hoof Stride Separation $\|\text{RFHoof} - \text{RHHoof}\|_2 / D_{\text{torso}}$
- **Index 72:** Front Hoof Stance Width $\|\text{LFHoof} - \text{RFHoof}\|_2 / D_{\text{torso}}$
- **Index 73:** Hind Hoof Stance Width $\|\text{LHHoof} - \text{RHHoof}\|_2 / D_{\text{torso}}$
- **Index 74:** Left Knee Elevation $Y_{\text{LFKnee}} / D_{\text{torso}}$
- **Index 75:** Right Knee Elevation $Y_{\text{RFKnee}} / D_{\text{torso}}$

---

## 4. End-to-End Pipeline Data Flow & Shape Validation

```text
Raw Video File (.mp4, .avi, .mov)
       │
       ▼ VideoReader (OpenCV)
Frames Stream (FPS, Resolution, Frame Count)
       │
       ▼ FrameSampler
Fixed 128 Sampled Frames
       │
       ▼ QuadrupedPoseEstimator
Raw Keypoint Array (128, 17, 2) & Confidence Array (128, 17)
       │
       ▼ KeypointCleaner (Linear Interpolation + SavGol Smoothing W=5, P=2)
Cleaned Keypoints (128, 17, 2)
       │
       ▼ KeypointNormalizer (Torso Center Translation + Torso Length Scaling)
Normalized Keypoints & Sequence Mask (128,)
       │
       ▼ TemporalSequenceBuilder
76-Feature Sequence Matrix (128, 76)
       │
       ▼ ModelInputValidator
Validated Tensor Shape (1, 128, 76), Zero NaNs/Infs
       │
       ▼ Phase 7 BiLSTM Model
Raw Risk Probability
       │
       ▼ Phase 8 Calibrator & Screening Triage Engine (τ = 0.34)
Triage Result JSON (NORMAL, LAMENESS_RISK, INCONCLUSIVE)
```
