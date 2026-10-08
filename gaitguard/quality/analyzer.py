"""
GaitGuard AI - Video Quality Analyzer Module (Phase 10)
Integrates metadata quality, keypoint coverage, walking motion detection, motion blur, framing, and capture coach logic.
"""

from dataclasses import dataclass, asdict
import numpy as np

from .metrics import QualityMetrics
from .rules import QualityRules, QualityStatus
from .capture_coach import CaptureCoach

@dataclass
class QualityResult:
    status: str
    quality_score: float
    usable_frame_ratio: float
    keypoint_coverage: float
    motion_quality: float
    framing_quality: float
    blur_indicator: float
    issues: list
    user_guidance: list

    def to_dict(self):
        return asdict(self)


class VideoQualityAnalyzer:
    """
    Analyzes cattle video streams and produces structured quality evaluation reports.
    """

    def __init__(self):
        self.metrics = QualityMetrics()

    def analyze_quality(self, video_metadata, frames=None, keypoints=None, confidences=None):
        """
        Executes full video quality evaluation.
        Inputs:
            video_metadata: VideoMetadata dataclass
            frames: list of BGR numpy frames (optional)
            keypoints: numpy.ndarray of shape (T, 17, 2) (optional)
            confidences: numpy.ndarray of shape (T, 17) (optional)
        """
        meta_valid = video_metadata.is_valid if video_metadata else False
        frame_count = video_metadata.frame_count if video_metadata else 0
        height = video_metadata.height if video_metadata else 0
        fps = video_metadata.fps if video_metadata else 0.0
        
        # 1. Keypoint Coverage
        if confidences is not None and confidences.size > 0:
            coverage = QualityMetrics.compute_keypoint_coverage(confidences)
        elif keypoints is not None and keypoints.size > 0:
            coverage = 0.95 # Default valid keypoints assumption
        else:
            coverage = 0.0
            
        # 2. Walking Motion Quality
        if keypoints is not None and keypoints.size > 0:
            motion_disp = QualityMetrics.compute_walking_motion_quality(keypoints)
            framing_area, _ = QualityMetrics.compute_framing_quality(keypoints, width=video_metadata.width, height=video_metadata.height)
        else:
            motion_disp = 0.0
            framing_area = 0.50
            
        # 3. Blur Indicator
        if frames is not None and len(frames) > 0:
            blur_var = QualityMetrics.compute_blur_indicator(frames)
        else:
            blur_var = 100.0 # Default clear fallback
            
        # 4. Evaluate Rules & Status
        status, issues = QualityRules.evaluate(
            meta_valid=meta_valid,
            frame_count=frame_count,
            height=height,
            fps=fps,
            coverage=coverage,
            motion_disp=motion_disp,
            blur_var=blur_var,
            framing_area=framing_area
        )
        
        # 5. Composite Quality Score (0.0 to 100.0)
        score_comp = (
            0.40 * (coverage * 100.0) +
            0.30 * min(motion_disp * 100.0, 100.0) +
            0.20 * min(blur_var, 100.0) +
            0.10 * (100.0 if 0.15 <= framing_area <= 0.85 else 50.0)
        )
        quality_score = float(np.clip(score_comp, 0.0, 100.0))
        
        # 6. Capture Coach Guidance
        guidance = CaptureCoach.get_user_guidance(issues)
        
        return QualityResult(
            status=status.value,
            quality_score=round(quality_score, 2),
            usable_frame_ratio=round(float(coverage), 4),
            keypoint_coverage=round(float(coverage), 4),
            motion_quality=round(float(motion_disp), 4),
            framing_quality=round(float(framing_area), 4),
            blur_indicator=round(float(blur_var), 2),
            issues=issues,
            user_guidance=guidance
        )
