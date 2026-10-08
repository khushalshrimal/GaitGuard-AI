"""
GaitGuard AI - Pose Processing Package (Phase 9)
Provides quadruped pose estimation, short-gap keypoint cleaning, trajectory smoothing, and torso normalization.
"""

from .estimator import QuadrupedPoseEstimator
from .cleaner import KeypointCleaner
from .normalizer import KeypointNormalizer

__all__ = ["QuadrupedPoseEstimator", "KeypointCleaner", "KeypointNormalizer"]
