# GaitGuard AI — Phase 15 Technical Specification for External Validation

## 1. Data Schema Hierarchy
External validation data is organized hierarchically to enforce animal-level grouping:

```text
Site ID (e.g. FARM_ALPHA)
 └── Animal ID (e.g. EXT_COW_201)
      └── Session ID (e.g. EXT_SESS_01)
           └── Video ID (e.g. EXT_VID_001.mp4)
```

## 2. Animal-Level Clustering Controls
- Multiple video recordings from the same cow ID are grouped at the animal level.
- Confidence intervals for sensitivity and specificity utilize animal-level bootstrap resampling to prevent underestimating variance from repeated cow videos.

## 3. Pre-Analysis Exclusion Protocol
All video eligibility checks (keypoint coverage $\ge 0.60$, blur $\le 0.70$, duration $\ge 2.0\text{ s}$) are evaluated prior to inference and logged in `validation/results/exclusion_log.csv`.
