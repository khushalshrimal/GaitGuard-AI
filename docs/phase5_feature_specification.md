# GaitGuard AI — Phase 5 Gait Feature Specification & Schema

## 1. Overview

This document specifies the formal mathematical definitions, normalization strategies, temporal aggregation rules, and real-video compatibility ratings for the domain-specific gait features engineered in **Phase 5: Domain-Specific Gait Feature Engineering & Validation**.

All feature calculations receive the Phase 3 cleaned keypoint trajectories (`datasets/processed/gaitguard_cleaned_dataset.npz`) of shape `(272, 128, 17, 2)` float32 and produce a tabular feature matrix of shape `(272, 8)`.

---

## 2. Anatomical Reference & Torso Normalization

### A. Reference Keypoints
* `Spine1` (Index 14): Front spine / Withers
* `Spine2` (Index 15): Mid-back spine
* `Spine3` (Index 16): Rear spine / Sacrum (Hip)

### B. Robust Torso Length Definition
To remove cow body size confounding, a sequence-level robust torso length $D_{\text{torso}}$ is defined as:
$$D_{\text{torso}} = \text{median}_{t \in \text{ValidFrames}} \sqrt{(X_{\text{Spine1}}(t) - X_{\text{Spine3}}(t))^2 + (Y_{\text{Spine1}}(t) - Y_{\text{Spine3}}(t))^2}$$
All spatial distance metrics (stride length, elevation offsets) are divided by $D_{\text{torso}}$ to render them scale-invariant.

---

## 3. Formal Feature Contract

### 1. `torso_length`
* **Mathematical Definition:** $D_{\text{torso}} = \text{median}_t \|\text{Spine1}(t) - \text{Spine3}(t)\|_2$.
* **Units:** Normalized spatial resolution units.
* **Purpose:** Reference anatomical scale normalization factor.
* **Real-Video Compatibility:** `HIGH` (Spine points are reliably visible in lateral side-view videos).

---

### 2. `back_arch_curvature` (Back Arch Curvature Index)
* **Mathematical Definition:** Per-frame vertical elevation of `Spine2` relative to the straight line segment between `Spine1` and `Spine3`, normalized by $D_{\text{torso}}$ and aggregated over valid sequence frames:
  $$Y_{\text{baseline}}(t) = 0.5 \cdot (Y_{\text{Spine1}}(t) + Y_{\text{Spine3}}(t))$$
  $$\text{Elevation}(t) = \frac{Y_{\text{baseline}}(t) - Y_{\text{Spine2}}(t)}{D_{\text{torso}}}$$
  $$\text{back\_arch\_curvature} = \text{mean}_{t \in \text{ValidFrames}} (\text{Elevation}(t))$$
* **Biological Context:** Lame cows arch their spine upward ($Y_{\text{Spine2}}$ moves up, decreasing Y in pixel space), resulting in a positive elevation index.
* **Expected Trend:** Lame Risk cows > Normal cows.
* **Real-Video Compatibility:** `HIGH` (Side-view back arch profile is a primary veterinary diagnostic trait).

---

### 3. `head_nodding_amplitude` (Head Nodding Amplitude Index)
* **Mathematical Definition:** Peak-to-peak vertical oscillation amplitude of the combined head centroid ($0.5 \cdot (\text{HeadTop} + \text{Nose})$), normalized by $D_{\text{torso}}$:
  $$Y_{\text{head}}(t) = 0.5 \cdot (Y_{\text{HeadTop}}(t) + Y_{\text{Nose}}(t))$$
  $$\text{head\_nodding\_amplitude} = \frac{\max_t(Y_{\text{head}}(t)) - \min_t(Y_{\text{head}}(t))}{D_{\text{torso}}}$$
* **Consolidation Note:** Combines `HeadTop` and `Nose` to eliminate the $r=0.94$ redundancy identified in Phase 4.
* **Expected Trend:** Lame Risk cows > Normal cows (compensatory weight transfer during painful stance).
* **Real-Video Compatibility:** `HIGH`.

---

### 4. `torso_normalized_stride_length` (Torso-Normalized Hoof Stride Length)
* **Mathematical Definition:** Mean horizontal forward stride distance covered by hooves (`LFHoof`, `RFHoof`, `LHHoof`, `RHHoof`) during stance-to-swing cycles, normalized by $D_{\text{torso}}$:
  $$\text{Stride}_k = \frac{\text{sum}_t \sqrt{\Delta X_{k}(t)^2 + \Delta Y_{k}(t)^2}}{D_{\text{torso}} \cdot N_{\text{cycles}}}$$
  $$\text{torso\_normalized\_stride\_length} = \text{mean}_{k \in \{\text{hooves}\}} (\text{Stride}_k)$$
* **Expected Trend:** Normal cows > Lame Risk cows (lame cows take shorter, more cautious steps).
* **Real-Video Compatibility:** `HIGH`.

---

### 5. `stance_timing_asymmetry` (Stance-Phase Timing Asymmetry Ratio)
* **Mathematical Definition:** Temporal stance duration ratio between left and right limb pairs. Low frame-to-frame hoof velocity ($\|v(t)\|_2 < \tau$) identifies ground contact / stance frames $T_{\text{stance}}$:
  $$\text{Asym}_{\text{front}} = \frac{|T_{\text{stance, LF}} - T_{\text{stance, RF}}|}{T_{\text{stance, LF}} + T_{\text{stance, RF}} + \epsilon}$$
  $$\text{Asym}_{\text{hind}} = \frac{|T_{\text{stance, LH}} - T_{\text{stance, RH}}|}{T_{\text{stance, LH}} + T_{\text{stance, RH}} + \epsilon}$$
  $$\text{stance\_timing\_asymmetry} = 0.5 \cdot (\text{Asym}_{\text{front}} + \text{Asym}_{\text{hind}})$$
* **Expected Trend:** Lame Risk cows > Normal cows (lame cows minimize stance time on painful limbs).
* **Real-Video Compatibility:** `MEDIUM` (Requires sufficient video FPS to accurately measure stance frames).

---

### 6. `normalized_walking_speed` (Torso-Normalized Walking Velocity)
* **Mathematical Definition:** Total forward displacement of trunk centroid per valid frame, normalized by $D_{\text{torso}}$:
  $$\text{Speed} = \frac{\text{sum}_{t} \sqrt{\Delta X_{\text{trunk}}(t)^2 + \Delta Y_{\text{trunk}}(t)^2}}{T_{\text{frames}} \cdot D_{\text{torso}}}$$
* **Expected Trend:** Normal cows > Lame Risk cows (lame cows walk significantly slower).
* **Real-Video Compatibility:** `HIGH`.

---

### 7. `knee_flexion_range` (Knee Flexion Range of Motion)
* **Mathematical Definition:** Range of motion of knee joint angles ($\angle \text{Hoof-Ankle-Knee}$):
  $$\text{knee\_flexion\_range} = \text{mean}_{k \in \{\text{knees}\}} (\max_t \theta_k(t) - \min_t \theta_k(t))$$
* **Expected Trend:** Normal cows > Lame Risk cows (joint stiffness reduces range of motion).
* **Real-Video Compatibility:** `MEDIUM`.

---

## 4. Summary Schema Table

| Feature Name | Type | Allowed Bounds | Missing Policy | Target Leakage Risk | Real-Video Rating |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `torso_length` | `float32` | $(0.05, 0.50)$ | None (Must be $>0$) | Zero | `HIGH` |
| `back_arch_curvature` | `float32` | $(-0.05, +0.05)$ | None | Zero | `HIGH` (Primary Candidate) |
| `head_nodding_amplitude` | `float32` | $(0.01, 0.30)$ | None | Zero | `HIGH` (Primary Candidate) |
| `torso_normalized_stride_length` | `float32` | $(0.50, 5.00)$ | None | Zero | `HIGH` (Primary Candidate) |
| `stance_timing_asymmetry` | `float32` | $(0.00, 0.50)$ | None | Zero | `MEDIUM` (Primary Candidate) |
| `normalized_walking_speed` | `float32` | $(0.001, 0.050)$ | None | Zero | `HIGH` (Secondary Candidate) |
| `knee_flexion_range` | `float32` | $(5.0^\circ, 90.0^\circ)$ | None | Zero | `MEDIUM` (Secondary Candidate) |
