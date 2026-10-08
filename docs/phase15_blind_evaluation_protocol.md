# GaitGuard AI — Phase 15 Blind Evaluation Protocol

## 1. 10-Step Blind Evaluation Protocol
To prevent post-hoc parameter adjustments, cherry-picking, or threshold overfitting, external validation executes according to the following strict 10-step protocol:

```text
STEP 1: Freeze BiLSTM Model Weights & Architecture.
STEP 2: Freeze Keypoint Cleaning & Coordinate Preprocessing.
STEP 3: Freeze Screening Decision Threshold (tau = 0.34).
STEP 4: Freeze Platt Scaling Sigmoid Calibration Parameters.
STEP 5: Freeze Evaluation Endpoints (Primary: Sensitivity for LAMENESS_RISK).
STEP 6: Finalize Independent Reference Locomotion Labels.
STEP 7: Execute GaitGuard Real-Video Inference Stack.
STEP 8: Compare Predicted Triage Outcomes Against Reference Labels.
STEP 9: Compute Classification Metrics, Calibration ECE, and Confidence Intervals.
STEP 10: Perform Error Analysis & Report Limitations.
```

## 2. Integrity Rule
Steps 1–5 must be locked prior to executing Step 6 (Finalizing external reference labels). Zero threshold adjustments or recalibration steps are allowed between Steps 6 and 10.
