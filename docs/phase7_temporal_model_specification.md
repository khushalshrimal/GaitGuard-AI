# GaitGuard AI — Phase 7 Temporal Model Technical Specification

## 1. Executive Summary & Objective

Phase 7 establishes **Advanced Temporal Modeling** for GaitGuard AI by moving beyond static tabular summary statistics to process full **128-frame keypoint sequence trajectories**.

- **Dataset Ingested:** `datasets/processed/gaitguard_cleaned_dataset.npz` (272 samples, 98 unique cows, 17 keypoints per frame, 128 max timesteps).
- **Core Architecture:** 1-Layer Bidirectional LSTM (BiLSTM) with 32 hidden units, dropout ($p=0.3$), dense classification head, and Sigmoid output.
- **Cross-Validation Protocol:** 5-Fold `GroupKFold` grouped strictly by `animal_id` (0% cow leakage between training and validation folds).
- **Primary Finding:** Temporal deep learning significantly outperforms Phase 6 static tabular models. BiLSTM achieves **$81.97\% \pm 5.46\%$ Accuracy**, **$78.96\% \pm 10.30\%$ Recall**, and **$0.9016$ Out-of-Fold ROC-AUC** (compared to Logistic Regression $66.18\%$ Accuracy and $0.7241$ OOF ROC-AUC).

---

## 2. Sequence Input Tensors & Feature Representations

Keypoint sequence trajectories of shape $(272, 128, 17, 2)$ are transformed into 3D temporal tensors $(N, T, F)$ using `gaitguard.temporal.sequence_builder.TemporalSequenceBuilder`:

| Mode Label | Representation Description | Feature Count ($F$) | Sequence Tensor Shape | Out-of-Fold ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Exp A** | Torso-Center Normalized $(X, Y)$ Keypoint Coordinates | 34 | $(272, 128, 34)$ | 0.8932 |
| **Exp B** | Keypoint Velocity Vectors ($\Delta KP_t$) | 34 | $(272, 128, 34)$ | 0.7895 |
| **Exp C** | Compact Biomechanical Gait Signals (Back Arch, Head Elevation, Hoof Strides) | 8 | $(272, 128, 8)$ | 0.8504 |
| **Exp D (Best)** | **Combined Compact Representation** (Coords + Velocity + Biomechanical) | **76** | **$(272, 128, 76)$** | **0.9016** |

### Torso Length Normalization Formula
To ensure invariance against camera distance and cow size variation:
$$D_{\text{torso}} = \text{median}_{t \in \text{valid}} \|\text{Spine1}_t - \text{Spine3}_t\|_2$$

---

## 3. Fold-Isolated Preprocessing & Data Leakage Control

`gaitguard.temporal.preprocessing.FoldTemporalScaler`:
1. `StandardScaler` is fitted **strictly on unpadded frames (`sequence_mask > 0`) within training fold cows only**.
2. Both training fold and validation fold sequences are transformed using training fold mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$).
3. Padded frames (`sequence_mask == 0`) are explicitly zeroed out ($0.0$) after scaling.

---

## 4. PyTorch BiLSTM Neural Network Architecture

```text
Input Sequence Tensor (B, 128, 76) & Mask Tensor (B, 128)
       │
       ▼
Bidirectional LSTM (input_dim=76, hidden_dim=32, num_layers=1, batch_first=True)
       │
       ▼ Output Shape: (B, 128, 64)
Masked Mean Pooling (Averages over unpadded valid timesteps)
       │
       ▼ Output Shape: (B, 64)
Dropout Layer (p = 0.3)
       │
       ▼
Linear Layer (64 -> 16) + ReLU Activation
       │
       ▼
Linear Output Layer (16 -> 1) + Sigmoid Activation
       │
       ▼ Output Probability: p ∈ [0, 1]
```

### Training Hyperparameters
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Loss Function:** Binary Cross-Entropy Loss (`BCELoss`)
- **Batch Size:** 16
- **Max Epochs:** 100 with Early Stopping (patience = 15 epochs on validation loss)
- **Random Seed:** 42 (100% reproducible initialization)
