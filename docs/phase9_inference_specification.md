# GaitGuard AI — Phase 9 Inference Specification

## 1. Executive Summary & Objective

Phase 9 integrates the real-video inference pipeline for GaitGuard AI:

$$\text{Real Video (MP4/AVI/MOV)} \to \text{VideoReader} \to \text{FrameSampler} \to \text{PoseEstimator} \to \text{KeypointCleaner} \to \text{Normalizer} \to \text{76-Feature Builder} \to \text{Validator} \to \text{BiLSTM} \to \text{Calibrated Triage}$$

- **Target Model Input Shape:** `(1, 128, 76)` matching Phase 7 BiLSTM model expectations.
- **Canonical Landmark Count:** 17 Keypoints.
- **Numerical Consistency:** Verified against Phase 7 training tensors within $10^{-5}$ absolute tolerance ($0.00000000$ max difference).
- **Mean Pipeline Latency:** ~22.8 ms / sequence sample on CPU.

---

## 2. Component Pipeline Architecture

```text
Video File (.mp4, .avi, .mov)
       │
       ▼ VideoReader (gaitguard.video.reader)
VideoMetadata (FPS, Resolution, Duration, Valid Status)
       │
       ▼ FrameSampler (gaitguard.video.sampler)
Uniform 128 Sampled Frames
       │
       ▼ QuadrupedPoseEstimator (gaitguard.pose.estimator)
Keypoints (128, 17, 2) & Confidence Array (128, 17)
       │
       ▼ KeypointCleaner (gaitguard.pose.cleaner)
Short-Gap Linear Interpolation + Savitzky-Golay Trajectory Smoothing (W=5, P=2)
       │
       ▼ KeypointNormalizer (gaitguard.pose.normalizer)
Torso Translation Centering & Torso Length Scaling (D_torso = ||Spine1 - Spine3||_2)
       │
       ▼ TemporalSequenceBuilder (gaitguard.temporal.sequence_builder)
76-Feature Sequence Matrix (128, 76)
       │
       ▼ ModelInputValidator (gaitguard.inference.feature_schema)
Validates Shape (1, 128, 76), Non-NaN, Non-Inf
       │
       ▼ BiLSTMGaitClassifier (gaitguard.temporal.model)
Raw Risk Probability
       │
       ▼ CrossFittedCalibrator & ScreeningTriageEngine (gaitguard.triage)
Calibrated Probability & 3-Way Triage Result JSON
```

---

## 3. Input Validation & Error Handling Rules

1. **Min Frame Count:** Videos with $< 30$ frames are rejected by the quality gate (`INSUFFICIENT_VIDEO_QUALITY`).
2. **Keypoint Confidence:** Frames with low confidence ($conf < 0.20$) are interpolated across short gaps ($\le 10$ frames).
3. **Tensor Validation:** Input tensors are validated before model execution:
   - Shape must be `(1, 128, 76)` or `(N, 128, 76)`.
   - Must contain zero NaNs and zero Infs.
