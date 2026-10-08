"""
GaitGuard AI - Video Quality & Capture Guidance Package (Phase 10)
Provides video metadata quality analysis, keypoint visibility coverage metrics, walking motion detection,
motion blur calculation, capture coach guidance, and system quality gate integration.
"""

from .metrics import QualityMetrics
from .rules import QualityRules, QualityIssueCode, QualityStatus
from .capture_coach import CaptureCoach
from .analyzer import VideoQualityAnalyzer, QualityResult

__all__ = [
    "QualityMetrics",
    "QualityRules",
    "QualityIssueCode",
    "QualityStatus",
    "CaptureCoach",
    "VideoQualityAnalyzer",
    "QualityResult",
]
