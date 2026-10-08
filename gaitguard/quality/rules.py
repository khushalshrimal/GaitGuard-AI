"""
GaitGuard AI - Machine-Readable Quality Issue Codes & Rule Engine (Phase 10)
Defines quality status categories, issue codes, and deterministic threshold evaluation rules.
"""

from enum import Enum

class QualityStatus(str, Enum):
    READY = "READY"
    RETRY = "RETRY"
    INCONCLUSIVE = "INCONCLUSIVE"


class QualityIssueCode(str, Enum):
    VIDEO_UNREADABLE = "VIDEO_UNREADABLE"
    VIDEO_TOO_SHORT = "VIDEO_TOO_SHORT"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    LOW_KEYPOINT_COVERAGE = "LOW_KEYPOINT_COVERAGE"
    LONG_KEYPOINT_GAPS = "LONG_KEYPOINT_GAPS"
    INSUFFICIENT_WALKING = "INSUFFICIENT_WALKING"
    EXCESSIVE_BLUR = "EXCESSIVE_BLUR"
    POOR_FRAMING = "POOR_FRAMING"
    EXCESSIVE_CAMERA_MOTION = "EXCESSIVE_CAMERA_MOTION"


class QualityRules:
    """
    Evaluates measurable quality signals against documented thresholds.
    """

    MIN_FRAME_COUNT = 30
    MIN_RESOLUTION_HEIGHT = 360
    MIN_FPS = 15.0
    MIN_KEYPOINT_COVERAGE = 0.65
    MIN_WALKING_MOTION = 0.05
    MIN_LAPLACIAN_VAR = 35.0
    MIN_FRAMING_AREA = 0.08
    MAX_FRAMING_AREA = 0.95

    @classmethod
    def evaluate(cls, meta_valid, frame_count, height, fps, coverage, motion_disp, blur_var, framing_area):
        """
        Evaluates metrics against rules.
        Returns:
            status: QualityStatus (READY, RETRY, INCONCLUSIVE)
            issues: list of QualityIssueCode strings
        """
        issues = []
        
        if not meta_valid:
            issues.append(QualityIssueCode.VIDEO_UNREADABLE.value)
            return QualityStatus.RETRY, issues
            
        if frame_count < cls.MIN_FRAME_COUNT:
            issues.append(QualityIssueCode.VIDEO_TOO_SHORT.value)
            
        if height > 0 and height < cls.MIN_RESOLUTION_HEIGHT:
            issues.append(QualityIssueCode.LOW_RESOLUTION.value)
            
        if coverage < cls.MIN_KEYPOINT_COVERAGE:
            issues.append(QualityIssueCode.LOW_KEYPOINT_COVERAGE.value)
            
        if motion_disp < cls.MIN_WALKING_MOTION:
            issues.append(QualityIssueCode.INSUFFICIENT_WALKING.value)
            
        if blur_var < cls.MIN_LAPLACIAN_VAR:
            issues.append(QualityIssueCode.EXCESSIVE_BLUR.value)
            
        if framing_area < cls.MIN_FRAMING_AREA or framing_area > cls.MAX_FRAMING_AREA:
            issues.append(QualityIssueCode.POOR_FRAMING.value)
            
        retry_triggers = {
            QualityIssueCode.VIDEO_UNREADABLE.value,
            QualityIssueCode.VIDEO_TOO_SHORT.value,
            QualityIssueCode.LOW_RESOLUTION.value,
            QualityIssueCode.LOW_KEYPOINT_COVERAGE.value,
            QualityIssueCode.EXCESSIVE_BLUR.value,
            QualityIssueCode.INSUFFICIENT_WALKING.value
        }
        
        if len(issues) == 0:
            status = QualityStatus.READY
        elif any(issue in retry_triggers for issue in issues):
            status = QualityStatus.RETRY
        else:
            status = QualityStatus.INCONCLUSIVE
            
        return status, issues
