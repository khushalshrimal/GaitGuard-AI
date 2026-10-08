"""
GaitGuard AI - Capture Coach & User Guidance Module (Phase 10)
Translates technical quality failure issue codes into non-technical actionable guidance for farmers and field operators.
"""

from .rules import QualityIssueCode

class CaptureCoach:
    """
    Translates technical quality issue codes into simple user capture advice.
    """

    GUIDANCE_MAP = {
        QualityIssueCode.VIDEO_UNREADABLE.value: "Video file is unreadable or corrupted. Please record a new video file.",
        QualityIssueCode.VIDEO_TOO_SHORT.value: "Video is too short. Record a slightly longer walking sequence (at least 3-5 seconds).",
        QualityIssueCode.LOW_RESOLUTION.value: "Video resolution is low. Increase camera resolution (at least 480p).",
        QualityIssueCode.LOW_KEYPOINT_COVERAGE.value: "Move to a clearer side-profile view and ensure the cow's body is fully visible without obstructions.",
        QualityIssueCode.LONG_KEYPOINT_GAPS.value: "Avoid blocking the camera view and keep limbs clearly visible.",
        QualityIssueCode.INSUFFICIENT_WALKING.value: "Record the cow while it walks naturally in a straight path.",
        QualityIssueCode.EXCESSIVE_BLUR.value: "Hold the camera steady and ensure adequate lighting to reduce motion blur.",
        QualityIssueCode.POOR_FRAMING.value: "Keep the entire cow centered in the frame, avoiding extreme zoom or cropping.",
        QualityIssueCode.EXCESSIVE_CAMERA_MOTION.value: "Keep the camera stationary or pan smoothly alongside the cow."
    }

    @classmethod
    def get_user_guidance(cls, issue_codes):
        """
        Returns list of user-facing actionable guidance strings.
        """
        guidance_messages = []
        for code in issue_codes:
            msg = cls.GUIDANCE_MAP.get(code, f"Quality issue detected: {code}. Please re-record video.")
            guidance_messages.append(msg)
            
        if len(guidance_messages) == 0:
            guidance_messages.append("Video capture quality is excellent. Proceeding to screening analysis.")
            
        return guidance_messages

    @classmethod
    def get_capture_protocol_summary(cls):
        """
        Returns recommended capture protocol for field users.
        """
        return {
            "viewpoint": "Side profile (90 degree lateral view)",
            "path": "Straight, level walking path on firm ground",
            "duration": "3 to 8 seconds of continuous walking",
            "framing": "Full cow visible in frame (15% to 85% area occupancy)",
            "lighting": "Clear daylight or well-lit indoor alleyway",
            "stability": "Camera held steady at cow mid-body height"
        }
