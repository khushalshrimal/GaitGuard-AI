# GaitGuard AI — Phase 8 Calibration & Triage Specification

## 1. Executive Summary & Objective

Phase 8 establishes a **Leakage-Free Probability Calibration & 3-Way Triage System** for GaitGuard AI.

- **Input Source:** Phase 7 BiLSTM out-of-fold predictions (`datasets/processed/phase7_oof_predictions.csv`).
- **Best Calibration Method:** **Sigmoid Calibration (Platt Scaling)** fitted via inner 4-fold `GroupKFold` cross-calibration on training fold animals.
- **Probability Quality Metrics:**
  - **Brier Score:** $0.1271$
  - **Log Loss:** $0.4018$
  - **Expected Calibration Error (ECE):** **$0.0291$** (a **+12.34% reduction in calibration error** vs raw BiLSTM $0.0332$).
- **Optimal Screening Threshold:** $\tau = 0.34$, achieving High Recall ($\ge 80\%$) for screening cattle.
- **3-Way Triage Decisions:** `NORMAL` (113 samples), `LAMENESS_RISK` (138 samples), `INCONCLUSIVE` (21 samples).

---

## 2. Leakage-Free Cross-Fitted Calibration Pipeline

To ensure zero calibration data leakage, calibrator parameters are learned exclusively on training fold animals:

```text
Outer GroupKFold Split (Fold k)
       │
       ├── Training Cows (N ~ 217, 78 Cows)
       │        │
       │        ▼ Inner 4-Fold GroupKFold (Animal Grouped)
       │      Inner OOF Predictions (p_inner_train)
       │        │
       │        ▼ Fit Platt Scaler (LogisticRegression)
       │      Fitted Calibrator Parameters
       │
       └── Outer Validation Cows (N ~ 55, 20 Cows)
                │
                ▼ Apply Fitted Calibrator
              Calibrated Validation Probabilities (p_calibrated)
```

---

## 3. Confidence Metric & Triage Result JSON Contract

### Normalized Confidence Formula
$$C(p) = 2 \cdot |p_{\text{calibrated}} - 0.5| \in [0, 1]$$

### 3-Way Triage Decision Boundaries ($\tau = 0.34$, $\delta = 0.10$)
- $p_{\text{calibrated}} < 0.24 \implies$ `NORMAL` (`inconclusive: false`)
- $p_{\text{calibrated}} > 0.44 \implies$ `LAMENESS_RISK` (`inconclusive: false`)
- $0.24 \le p_{\text{calibrated}} \le 0.44 \implies$ `INCONCLUSIVE` (`inconclusive: true`)

### Output JSON Schema Example
```json
{
  "sample_id": "cow_012_seq_02",
  "risk_probability_raw": 0.7842,
  "risk_probability_calibrated": 0.8125,
  "decision": "LAMENESS_RISK",
  "confidence": 0.6250,
  "threshold": 0.34,
  "uncertainty_margin": 0.10,
  "inconclusive": false,
  "reason": "Abnormal gait motion detected (elevated back arch, slower stride rhythm).",
  "disclaimer": "AI-assisted screening tool. Veterinary review recommended."
}
```
