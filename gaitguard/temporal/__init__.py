"""
GaitGuard AI - Advanced Temporal Modeling Package (Phase 7)
Contains sequence dataset builders, fold-isolated temporal scalers, BiLSTM architecture, and evaluation utilities.
"""

from .sequence_builder import TemporalSequenceBuilder
from .preprocessing import FoldTemporalScaler
from .model import BiLSTMGaitClassifier
from .evaluation import TemporalModelEvaluator

__all__ = [
    "TemporalSequenceBuilder",
    "FoldTemporalScaler",
    "BiLSTMGaitClassifier",
    "TemporalModelEvaluator",
]
