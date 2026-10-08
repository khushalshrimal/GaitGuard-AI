# GaitGuard AI — Phase 14 Domain Shift Analysis Report

## 1. Executive Summary
This report evaluates distribution stability between Phase 5 training feature distributions (272 samples) and external field validation samples using Kolmogorov-Smirnov 2-sample tests, Wasserstein distance, and Population Stability Index (PSI).

## 2. Statistical Metrics Table

| Feature | Train Mean | Field Mean | KS Statistic | KS p-value | Wasserstein Dist | PSI | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| animal_id | 52.3272 | 48.0580 | 0.1807 | 0.1102 | 6.0289 | 1.0701 | SIGNIFICANT_SHIFT |
| binary_target | 0.4743 | 0.5078 | 0.4257 | 0.0000 | 0.2716 | 4.0643 | SIGNIFICANT_SHIFT |
| raw_label | 1.6434 | 1.6903 | 0.2657 | 0.0040 | 0.3288 | 0.0953 | STABLE |
| raw_sequence_length | 134.3676 | 143.2406 | 0.2781 | 0.0022 | 9.0297 | 0.5439 | SIGNIFICANT_SHIFT |
| torso_length | 0.0563 | 0.0600 | 0.3410 | 0.0001 | 0.0038 | 0.4528 | SIGNIFICANT_SHIFT |
| back_arch_curvature | -0.0237 | -0.0264 | 0.0938 | 0.8129 | 0.0127 | 0.0773 | STABLE |
| head_nodding_amplitude | 0.6993 | 0.7445 | 0.1932 | 0.0733 | 0.0810 | 0.2926 | SIGNIFICANT_SHIFT |
| torso_normalized_stride_length | 14.7059 | 15.9485 | 0.2599 | 0.0052 | 1.3779 | 0.4744 | SIGNIFICANT_SHIFT |
| stance_timing_asymmetry | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | STABLE |
| normalized_walking_speed | 0.0978 | 0.1012 | 0.1456 | 0.2990 | 0.0056 | 0.1409 | MODERATE_SHIFT |
| knee_flexion_range | 38.8453 | 37.7311 | 0.2101 | 0.0404 | 4.5517 | 0.5477 | SIGNIFICANT_SHIFT |

## 3. Scientific Interpretation
- **PSI Interpretation**: PSI < 0.1 indicates distribution stability; 0.1 <= PSI < 0.25 indicates moderate shift; PSI >= 0.25 indicates significant domain shift.
- **Conclusion**: Domain shift represents a shift in data distribution due to differing field video capture environments and does not automatically imply model failure.
