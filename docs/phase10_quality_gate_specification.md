# GaitGuard AI — Phase 10: Video Quality Gate Technical Specification

## 1. Overview & Purpose
The GaitGuard AI Video Quality Gate evaluates cattle walking videos to determine if they contain sufficient signal and video quality for deep temporal neural network screening (BiLSTM). 

The primary objective of the Video Quality Gate is to **prevent invalid, corrupted, occluded, stationary, or highly blurry videos from triggering false ML predictions**. Instead of outputting potentially misleading screening results on corrupt inputs, the system gracefully intercepts low-quality input and returns actionable user feedback via the **Capture Coach**.

---

## 2. Architecture & Control Flow

```
                  ┌───────────────────────────────┐
                  │      Cattle Walking Video     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   OpenCV Video Reader & Meta  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │      Video Quality Analyzer   │
                  └───────────────┬───────────────┘
                                  │
             ┌────────────────────┴────────────────────┐
             │                                         │
             ▼                                         ▼
   [ Actionable Defect ]                     [ Satisfactory Quality ]
(Blur, Occlusion, Static)                   (Coverage >= 65%, Displ >= 0.05)
             │                                         │
             ▼                                         ▼
   ┌───────────────────┐                     ┌───────────────────┐
   │    STATUS: RETRY  │                     │   STATUS: READY   │
   └─────────┬─────────┘                     └─────────┬─────────┘
             │                                         │
             ▼                                         ▼
   ┌───────────────────┐                     ┌───────────────────┐
   │   Capture Coach   │                     │  Phase 9 Inference│
   │ Guidance Output   │                     │  (BiLSTM Model)   │
   └───────────────────┘                     └───────────────────┘
```

---

## 3. Measurable Quality Metrics

| Metric | Code Symbol | Quantitative Formula / Method | Threshold | Impact on Failure |
| :--- | :--- | :--- | :--- | :--- |
| **Video Metadata** | `meta_valid` | OpenCV `isOpened()` & `frame_count >= 30` | Valid decoding | `VIDEO_UNREADABLE` / `VIDEO_TOO_SHORT` |
| **Resolution Height**| `height` | Frame spatial pixel height | $\ge 360\text{p}$ | `LOW_RESOLUTION` |
| **Keypoint Coverage**| `coverage` | $\frac{1}{T \cdot K} \sum_{t,k} \mathbb{I}(c_{t,k} \ge 0.20)$ | $\ge 0.65$ (65%) | `LOW_KEYPOINT_COVERAGE` |
| **Walking Motion** | `motion_disp` | $\frac{\max(x_{\text{Spine1}}) - \min(x_{\text{Spine1}})}{L_{\text{torso}}}$ | $\ge 0.05$ torso length | `INSUFFICIENT_WALKING` |
| **Laplacian Blur** | `blur_var` | $\text{Var}\left(\nabla^2 I_{\text{gray}}\right)$ | $\ge 35.0$ | `EXCESSIVE_BLUR` |
| **Framing Area** | `framing_area` | $\frac{\text{Area}(\text{BoundingBox})}{\text{Area}(\text{Frame})}$ | $0.08 \le \text{Area} \le 0.95$ | `POOR_FRAMING` |

---

## 4. Machine-Readable Quality Issue Codes

| Issue Code | User Description | Severity | Trigger Threshold |
| :--- | :--- | :--- | :--- |
| `VIDEO_UNREADABLE` | Video codec unreadable or corrupt file | CRITICAL | OpenCV decode failure |
| `VIDEO_TOO_SHORT` | Video duration under 1.0 seconds | HIGH | `frame_count < 30` |
| `LOW_RESOLUTION` | Frame height under 360p | MEDIUM | `height < 360` |
| `LOW_KEYPOINT_COVERAGE`| Extreme body occlusion or poor keypoint detection | HIGH | `coverage < 0.65` |
| `INSUFFICIENT_WALKING` | Stationary or standing cow | HIGH | `motion_disp < 0.05` |
| `EXCESSIVE_BLUR` | Fast camera motion or lens blur | HIGH | `blur_var < 35.0` |
| `POOR_FRAMING` | Cow too far or cropped out of frame | MEDIUM | `framing_area < 0.08` or $> 0.95$ |

---

## 5. Triage Status & Capture Coach Matrix

- **`READY`**: Video meets all minimum thresholds. Inference proceeds directly to Phase 7 BiLSTM model.
- **`RETRY`**: Actionable video quality defect detected. ML prediction is bypassed to protect validity. Non-technical instructions are provided to the user.
- **`INCONCLUSIVE`**: Marginal video quality or sub-optimal framing where inference is allowed but flagged as uncertain.

---

## 6. System Integration Contract

The Quality Gate attaches a structured `video_quality` dictionary to all screening responses:

```json
{
  "status": "RETRY",
  "quality_score": 55.0,
  "usable_frame_ratio": 0.50,
  "keypoint_coverage": 0.50,
  "motion_quality": 0.20,
  "framing_quality": 0.35,
  "blur_indicator": 100.0,
  "issues": ["LOW_KEYPOINT_COVERAGE"],
  "user_guidance": [
    "Cow keypoints are partially hidden. Ensure full side view of cow without barriers or fences blocking the view."
  ],
  "disclaimer": "AI-assisted screening tool. Video quality insufficient for ML prediction."
}
```
