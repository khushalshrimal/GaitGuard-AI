"""
GaitGuard AI - Probability Calibration & Triage System (Phase 8)
Contains cross-fitted probability calibrators, confidence estimators, threshold selectors, and 3-way triage engines.
"""

from .calibrator import CrossFittedCalibrator, compute_brier_score, compute_log_loss, compute_ece
from .triage_engine import ScreeningTriageEngine, ConfidenceEstimator, TriageResultContract

__all__ = [
    "CrossFittedCalibrator",
    "compute_brier_score",
    "compute_log_loss",
    "compute_ece",
    "ScreeningTriageEngine",
    "ConfidenceEstimator",
    "TriageResultContract",
]
