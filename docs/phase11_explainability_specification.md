# GaitGuard AI — Phase 11 Technical Specification: Explainable AI & Evidence Layer

## 1. Executive Overview & Purpose
The Explainable AI (XAI) layer of GaitGuard AI provides transparent, scientifically defensible explanations for the Phase 7 BiLSTM temporal neural network predictions without altering model weights, retraining, or modifying inference contracts. 

Rather than attempting to present veterinary diagnostic claims, the XAI layer delivers **supportive model-behavior evidence**, explaining *which input sequence dimensions and derived gait patterns contributed most to the model's lameness-risk screening score*.

---

## 2. Model Input & Explainer Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │    Sequence Feature Matrix X (128, 76)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   PyTorch BiLSTMGaitClassifier (Mode D)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │  PyTorch GradientExplainer (SHAP v0.52.0)    │
                  └──────────────────────┬───────────────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             │                                                       │
             ▼                                                       ▼
   [ LEVEL 1 ATTRIBUTION ]                                 [ LEVEL 2 GAIT EVIDENCE ]
Direct SHAP Attributions (128, 76)                     Derived Interpretable Gait Concepts
 - Top-K Feature Contributors                          - Normalized Walking Speed
 - Modality Share (Coord vs Velo vs Bio)               - Stride Length Separation
 - Body Region Share (Head, Spine, Limbs)              - Back Arch Curvature
 - Temporal Phase (Early, Mid, Late)                   - Knee Flexion / Elevation
```

### Model Architecture Parameters:
- **Model Class**: `BiLSTMGaitClassifier` (1-Layer BiLSTM, hidden dimension 32, dropout 0.3, Dense 16, Sigmoid output).
- **Input Dimension**: $T = 128$ timesteps, $F = 76$ features.
- **76-Feature Schema**:
  - `0..33`: 34 Torso-Normalized Coordinates ($17\times 2$)
  - `34..67`: 34 Keypoint Velocities ($17\times 2$)
  - `68..75`: 8 Biomechanical Gait Signals (Spine arching, head nodding, hoof separation, stance width, knee elevation)

---

## 3. Leakage-Free Background Dataset Strategy

- **Sampling Source**: Selected strictly from the Phase 3/7 training sequences (`datasets/processed/gaitguard_cleaned_dataset.npz`).
- **Animal Group Isolation**: Samples are grouped by `animal_ids` to ensure no held-out evaluation animals leak into the background distribution.
- **Sample Size**: $N_{\text{bg}} = 30$ training sequences, deterministically sampled with fixed seed (`seed=42`).

---

## 4. Two Explanation Levels

### LEVEL 1 — Model Input Attribution
Direct SHAP feature attributions on the $(128, 76)$ input tensor:
- **Modality Attribution**:
  - `Coordinates` (Indices 0..33)
  - `Velocity` (Indices 34..67)
  - `Biomechanical` (Indices 68..75)
- **Anatomical Body Region Attribution**:
  - `Head`: Nose (12), HeadTop (13)
  - `Spine`: Spine1 (14), Spine2 (15), Spine3 (16)
  - `Forelimbs`: LFHoof (0), LFAnkle (1), LFKnee (2), RFHoof (3), RFAnkle (4), RFKnee (5)
  - `Hindlimbs`: LHHoof (6), LHAnkle (7), LHKnee (8), RHHoof (9), RHAnkle (10), RHKnee (11)
- **Temporal Phase Attribution**:
  - `Early Phase`: Frames 0..42
  - `Middle Phase`: Frames 43..85
  - `Late Phase`: Frames 86..127

### LEVEL 2 — Derived Gait Evidence
Aggregates Level 1 SHAP attributions into domain gait concepts, explicitly labeled as `"derived_gait_evidence"`:
1. `normalized_walking_speed`
2. `torso_normalized_stride_length`
3. `back_arch_curvature`
4. `knee_flexion_elevation`
5. `head_nodding_movement`

---

## 5. Directionality & Triage Language Matrix

- **`INCREASES_RISK`**: Positive SHAP value ($\phi_i > +10^{-5}$), pushing prediction towards Class 1.
- **`DECREASES_RISK`**: Negative SHAP value ($\phi_i < -10^{-5}$), pulling prediction towards Class 0.
- **`LOW_CONTRIBUTION`**: Near zero SHAP value ($|\phi_i| \le 10^{-5}$).

### Decision Result Language:
- **`NORMAL`**: `"No elevated lameness risk detected by this screening model."`
- **`LAMENESS_RISK`**: `"Elevated lameness risk indicated by the screening model."`
- **`INCONCLUSIVE`**: `"Screening result is inconclusive. Model evidence was insufficient for a reliable screening decision."` (Suppresses confident local explanations).

---

## 6. Stability & Perturbation Faithfulness

- **Top-K Jaccard Overlap**: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$ across 5 repeated explanation runs ($0.7333$).
- **Sign Consistency**: Percentage of feature directions remaining identical across runs ($82.89\%$).
- **Perturbation Faithfulness ($\Delta P$)**: Zeroing top positive contributors drops prediction probability ($\Delta P = +0.0079$), verifying explanation faithfulness.
