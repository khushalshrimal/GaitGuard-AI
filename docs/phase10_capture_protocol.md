# GaitGuard AI — Farm Video Capture Protocol & Operational Guidelines

## 1. Objective & Target Audience
This protocol provides practical, field-tested video capture guidelines for farm handlers, veterinary technicians, and dairy farm staff using **GaitGuard AI**. Following this protocol ensures high keypoint coverage, clear gait trajectory extraction, and reliable ML screening.

---

## 2. Field Recording Checklist

### Camera Positioning & Distance
- **Angle**: Position camera perpendicular ($90^\circ$ side view / lateral profile) to the cow's walking path.
- **Distance**: Stand $4\text{ to }6\text{ meters}$ (12 - 20 feet) away from the cow to capture full body framing.
- **Height**: Hold phone/camera at chest level (approx. $1.2\text{ meters}$ above ground). Avoid filming from extreme top or bottom angles.

### Walking Surface & Environment
- **Surface**: Concrete alley, flat gravel, or firm dirt walkway. Avoid deep mud or tall grass that hides hooves.
- **Pacing**: Cow must walk at a natural, uninterrupted pace for at least 5 to 10 consecutive strides ($3\text{ to }6\text{ seconds}$).
- **Lighting**: Film in well-lit daylight or well-lit indoor alleyways. Avoid extreme backlighting (e.g. filming directly into bright sunlight).

---

## 3. Recommended vs Unrecommended Field Setup

| Recommended Setup | Unrecommended Setup | Reason for Interception |
| :--- | :--- | :--- |
| Perpendicular side view ($90^\circ$) | Frontal head-on camera angle | High perspective distortion; hooves occluded |
| Distance: 4 - 6 meters | Close-up on single leg | Insufficient body framing (`POOR_FRAMING`) |
| Flat concrete or firm dirt | Deep mud or knee-high pasture grass | Hooves hidden from keypoint estimator |
| Continuous walk (3-5 sec) | Cow standing still or eating | Zero movement signal (`INSUFFICIENT_WALKING`) |
| Steady handheld or tripod | Walking/running alongside cow | High camera shake (`EXCESSIVE_BLUR`) |

---

## 4. Capture Coach User Guidance Reference

When the Video Quality Gate detects a video issue, it displays one or more of the following guidance messages:

- **Corrupt File**: *"Video file could not be read. Please record a new video and upload again."*
- **Too Short**: *"Video is too short. Record a slightly longer walking sequence (at least 3-5 seconds)."*
- **Low Resolution**: *"Video resolution is too low. Please set camera recording quality to 720p or 1080p."*
- **Occlusion**: *"Cow keypoints are partially hidden. Ensure full side view of cow without barriers or fences blocking the view."*
- **Stationary**: *"Cow is standing still or moving too slowly. Ensure the cow is walking continuously in a straight line."*
- **Blur**: *"Hold the camera steady and ensure adequate lighting to reduce motion blur."*
- **Framing**: *"Cow framing is too close or too far away. Position camera 4-6 meters away to capture full side profile."*
