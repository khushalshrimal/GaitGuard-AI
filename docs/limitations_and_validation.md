# GaitGuard AI — Responsible Use, Limitations & External Field Validation Policy

## 1. Non-Diagnostic Medical Boundary Statement

GaitGuard AI is an **AI-assisted cattle gait screening and decision-support tool**. It is **NOT a veterinary diagnostic system** and must never be marketed, cited, or deployed as a substitute for clinical examination by a qualified veterinarian or professional hoof trimmer.

### 1.1 Screening vs Diagnosis Scope

```
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│           GAITGUARD AI SCOPE            │     │            VETERINARY SCOPE             │
│  - Non-invasive video gait screening    │     │  - Physical hoof palpation & trimming   │
│  - Movement anomaly probability estimation│    │  - Pathological lesion diagnosis        │
│  - Automated Quality Gate validation    │  ≠  │  - Medical/surgical treatment plan      │
│  - 3-Way Triage (Normal/Risk/Inconclusive)│   │  - Individual animal prognosis          │
│  - SHAP feature attributions            │     │  - Clinical diagnostic certification    │
└─────────────────────────────────────────┘     └─────────────────────────────────────────┘
```

---

## 2. Internal Cross-Validation vs External Field Validation

### 2.1 Internal Cross-Validation Summary
* **Dataset**: Russello et al. benchmark dataset (272 video sequence samples, 98 unique cows).
* **Protocol**: Animal-level 5-fold `GroupKFold` (0 cow overlap).
* **Results**: Accuracy ~81.97%, F1-Score ~0.7973, OOF ROC-AUC 0.9016.

### 2.2 External Validation Status
> [!WARNING]
> **EXTERNAL FIELD VALIDATION BLOCKED**:
> Independent external field validation on unseen farm video recordings remains **`BLOCKED — DATA COLLECTION REQUIRED`** per Phase 15 protocol. Reported internal cross-validation performance must not be cited as real-world field sensitivity/specificity.

---

## 3. Potential Sources of Domain Shift & Operational Limitations

1. **Camera Angle & Framing Variance**: GaitGuard requires a parallel side-profile view of the walking cow. Diagonal, rear, or overhead camera angles degrade keypoint extraction accuracy.
2. **Surface & Terrain Differences**: Models trained on concrete alleyways may exhibit performance degradation when evaluating cattle walking on muddy pastures or deep straw bedding.
3. **Lighting & Occlusion**: Severe shadow cast, low indoor barn lighting, or partial railing occlusion reduce anatomical keypoint confidence scores.
4. **Breed & Size Differences**: Physical proportions differ across dairy (Holstein-Friesian, Jersey) and beef breeds. While torso length normalization reduces scale variance, extreme anatomical variations require further field dataset representation.

---

## 4. Preregistered 6-Step Protocol for External Field Validation

When independent field video data collection commences, validation must strictly follow this protocol:

1. **Independent Field Video Collection**: Collect dual-view high-definition cattle walking videos across multiple commercial dairy and beef farms.
2. **Blinded Ground-Truth Annotation**: Obtain independent locomotion scoring (Sprecher 1–5 scale) from 3 certified veterinarians blinded to model predictions.
3. **Preserve Grouping Integrity**: Enforce strict animal-level GroupKFold splitting (0 cow overlap between training/calibration and external evaluation folds).
4. **Evaluate Frozen Model**: Benchmark frozen model ($\tau = 0.34$, inconclusive interval $[0.24, 0.44]$) against external ground-truth labels without retraining weights or retuning thresholds.
5. **Report Subgroup Performance**: Analyze sensitivity, specificity, and inconclusive rates stratified by farm site, camera type, lighting condition, and walking surface.
6. **Veterinary Advisory Review**: Review external evaluation results with veterinary expert panel before commercial deployment.
