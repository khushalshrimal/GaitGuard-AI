"""
GaitGuard AI - Video Reader Module (Phase 9)
Extracts video metadata and frame streams using OpenCV.
"""

import os
from dataclasses import dataclass
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

@dataclass
class VideoMetadata:
    file_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float
    is_valid: bool
    error_msg: str = ""


class VideoReader:
    """
    Reads video metadata and frame streams from MP4, AVI, and MOV files.
    """

    def __init__(self, min_frames=15):
        self.min_frames = min_frames

    def get_metadata(self, video_path):
        """
        Extracts metadata from video file.
        """
        if not os.path.exists(video_path):
            return VideoMetadata(video_path, 0.0, 0, 0, 0, 0.0, False, "File not found.")
            
        if cv2 is None:
            return VideoMetadata(video_path, 0.0, 0, 0, 0, 0.0, False, "OpenCV (cv2) library not installed.")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoMetadata(video_path, 0.0, 0, 0, 0, 0.0, False, "Failed to open video file.")
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if frame_count <= 0 or width <= 0 or height <= 0:
            return VideoMetadata(video_path, fps, width, height, frame_count, 0.0, False, "Corrupted video dimensions or frame count.")
            
        duration = float(frame_count / fps) if fps > 0 else 0.0
        
        if frame_count < self.min_frames:
            return VideoMetadata(video_path, fps, width, height, frame_count, duration, False, f"Insufficient frames ({frame_count} < min {self.min_frames}).")
            
        return VideoMetadata(video_path, fps, width, height, frame_count, duration, True, "")

    def read_all_frames(self, video_path):
        """
        Reads all frames as list of BGR uint8 numpy arrays.
        Handles decode errors gracefully without dropping valid clips.
        """
        meta = self.get_metadata(video_path)
        if not meta.is_valid:
            raise ValueError(f"Cannot read invalid video '{video_path}': {meta.error_msg}")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            frames.append(frame)
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames read from video '{video_path}'.")
            
        return frames, meta
