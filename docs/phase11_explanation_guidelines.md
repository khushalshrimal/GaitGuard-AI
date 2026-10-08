# GaitGuard AI — User-Facing Explanation & Non-Diagnostic Guidelines

## 1. Purpose & Target Audience
This guideline defines user presentation rules for agricultural technology developers, veterinary technicians, and dairy farm software interfaces integrating GaitGuard AI's Explainable AI layer.

---

## 2. Core Operational Rules & Terminology

### Mandatory Language Guidelines:
- **Do NOT claim veterinary diagnosis**: Never present explanations as "proof of lameness" or "medical diagnosis".
- **Use model-centric attribution language**: Present findings as "features contributing to the AI model's screening score".
- **Separate Level 1 from Level 2**:
  - Level 1: "Model Input Attribution" (technical feature attributions).
  - Level 2: "Derived Gait Evidence" (interpretable concepts like walking speed or back arching).

### Standardized Result Strings:
- **`NORMAL`**: `"No elevated lameness risk detected by this screening model."`
- **`LAMENESS_RISK`**: `"Elevated lameness risk indicated by the screening model."`
- **`INCONCLUSIVE`**: `"Screening result is inconclusive. Model evidence was insufficient for a reliable screening decision."`

---

## 3. Forbidden vs Approved User UI Statements

| Forbidden UI Statement | Approved UI Statement | Rationale |
| :--- | :--- | :--- |
| "This cow is 100% lame." | "Elevated lameness risk indicated by the screening model." | Avoids illegal veterinary diagnostic claims. |
| "This cow is healthy." | "No elevated lameness risk detected by this screening model." | Prevents false reassurance on non-screened ailments. |
| "Slow walking causes lameness." | "Lower normalized walking speed contributed to the model's risk estimate." | Avoids inferring false causality from correlation. |
| "SHAP proved hindleg injury." | "Model attributions were highest in hindlimb velocity channels." | Preserves statistical explanation boundaries. |

---

## 4. Structured UI Contract Payload Example

```json
{
  "explanation_available": true,
  "prediction": {
    "raw_probability": 0.72,
    "calibrated_probability": 0.78,
    "decision": "LAMENESS_RISK",
    "confidence": "HIGH"
  },
  "result_summary": "Elevated lameness risk indicated by the screening model.",
  "top_contributors": [
    {
      "feature_name": "velocity_LHHoof_X",
      "attribution": 0.0142,
      "direction": "INCREASES_RISK",
      "modality": "velocity",
      "body_region": "Hindlimbs"
    }
  ],
  "derived_gait_evidence": [
    {
      "name": "normalized_walking_speed",
      "value": 0.089,
      "relative_contribution": -0.0125,
      "direction": "INCREASES_RISK",
      "interpretation": "Slower walking speed observed in video sequence."
    }
  ],
  "disclaimer": "AI-assisted screening tool. Model explanations describe AI pattern behavior, not veterinary diagnosis."
}
```
