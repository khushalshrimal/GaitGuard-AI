"""
GaitGuard AI - Video Ingestion & Temporal Sampling Package (Phase 9)
Provides video metadata reading, frame extraction, and temporal resampling utilities.
"""

from .reader import VideoReader, VideoMetadata
from .sampler import FrameSampler

__all__ = ["VideoReader", "VideoMetadata", "FrameSampler"]
