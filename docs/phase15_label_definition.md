# GaitGuard AI — Phase 15 Locomotion Reference Label Specification

## 1. Executive Summary
This document specifies the scientific protocol for defining, scoring, and binary-mapping cattle locomotion reference labels for external validation. Reference labels represent expert human locomotion assessments, **never** AI predictions or model probability outputs.

## 2. Reference Locomotion Scoring System
External locomotion scoring utilizes the validated **Sprecher 5-Point Locomotion Scale** (Sprecher et al., 1997) evaluated by qualified livestock veterinarians or trained locomotion scorers.

### Ordinal Locomotion Scale Definitions
| Score | Clinical Description | Biomechanical Characteristics | Binary Screening Mapping |
| :--- | :--- | :--- | :--- |
| **1 (Normal)** | Normal walking posture | Flat spine posture while standing and walking; symmetrical stride length. | `NORMAL` |
| **2 (Slightly Lame)** | Mild gait asymmetry | Flat spine standing, arched spine while walking; slight shortening of stride. | `NORMAL` (or `LAMENESS_RISK` per protocol) |
| **3 (Moderately Lame)** | Moderate lameness | Arched spine standing and walking; short gait, asymmetric weight bearing. | `LAMENESS_RISK` |
| **4 (Severely Lame)** | Severe lameness | Arched spine standing and walking; favors one or more limbs heavily. | `LAMENESS_RISK` |
| **5 (Extremely Lame)** | Inability / extreme difficulty walking | Arched spine, extreme reluctance to move, severe weight-bearing asymmetry. | `LAMENESS_RISK` |

## 3. Binary Screening Label Mapping Rule
- **`NORMAL`**: Sprecher Score 1 (Normal gait locomotion).
- **`LAMENESS_RISK`**: Sprecher Score $\ge 3$ (Moderate to severe locomotion asymmetry).
- *Score 2 Mapping Rule*: If Sprecher Score 2 is evaluated, it is recorded as `NORMAL` for primary binary screening cutoff, or tracked as a secondary borderline subgroup.

## 4. Evaluator Qualification & Observation Procedure
- Evaluators must possess a veterinary degree or $\ge 2$ years certified dairy herd locomotion scoring experience.
- Assessments must observe lateral walking on flat, non-slippery surfaces for at least 3–5 full stride cycles.
- Evaluators work independently without access to other evaluators' scores or GaitGuard model outputs.
