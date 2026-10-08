"""
GaitGuard AI - Central Configuration & Threshold Registry (Phase 3)
Contains validated system constants, preprocessing thresholds, and pipeline settings.
"""

# Video & Coordinate Parameters
FRAME_WIDTH = 1920.0
FRAME_HEIGHT = 1080.0
KEYPOINT_COUNT = 17

# Temporal Parameters
TARGET_SEQUENCE_LENGTH = 128
MIN_SEQUENCE_LENGTH = 90      # PROVISIONAL — REQUIRES VALIDATION
MAX_SEQUENCE_LENGTH = 207     # PROVISIONAL — REQUIRES VALIDATION

# Trajectory Smoothing Parameters
SAVGOL_WINDOW_LENGTH = 5      # Must be odd integer; smooths 5-frame local window
SAVGOL_POLYORDER = 2          # Quadratic polynomial fitting for trajectory smoothing

# ML & Cross-Validation Parameters
N_SPLITS = 5
RANDOM_SEED = 42

# Threshold Rationale & Documentation:
# - FRAME_WIDTH/HEIGHT (1920x1080): Matches raw video aspect ratio from Russello et al.
# - TARGET_SEQUENCE_LENGTH (128): Chosen near dataset median (130 frames) to minimize padding waste.
# - SAVGOL_WINDOW_LENGTH (5): Removes pose detector jitter over 5 frames (~0.16s) without flattening motion peaks.
# - SAVGOL_POLYORDER (2): Preserves acceleration and curvature in joint trajectories.
# - RANDOM_SEED (42): Fixed seed ensuring 100% reproducible splits.
