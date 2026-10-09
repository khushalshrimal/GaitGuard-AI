"""
GaitGuard AI - Measurable Quality Metrics Module (Phase 10)
Computes keypoint coverage ratios, walking motion quality, Laplacian variance blur, and framing occupancy.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

class QualityMetrics:
    """
    Computes measurable computer vision and keypoint trajectory quality metrics.
    """

    @staticmethod
    def compute_keypoint_coverage(confidences, min_conf=0.20):
        """
        Calculates ratio of valid keypoint detections across sequence (0.0 to 1.0).
        """
        conf = np.asarray(confidences, dtype=np.float32)
        if conf.size == 0:
            return 0.0
        valid_ratio = float(np.mean(conf >= min_conf))
        return valid_ratio

    @staticmethod
    def compute_walking_motion_quality(keypoints, torso_len=0.1):
        """
        Calculates normalized horizontal displacement of Spine1 (14) and hooves across timesteps.
        Returns max_disp_torso_scale (scalar float).
        """
        kp = np.asarray(keypoints, dtype=np.float32)
        T = kp.shape[0]
        if T < 2:
            return 0.0
            
        spine1_x = kp[:, 14, 0]
        disp_x = float(np.max(spine1_x) - np.min(spine1_x))
        if float(np.max(kp)) > 1.5:
            # Pixel coordinate space: normalize by max x dimension
            max_x = max(float(np.max(kp[:, :, 0])), 1.0)
            norm_disp = float((disp_x / max_x) / max(torso_len, 1e-4))
        else:
            norm_disp = float(disp_x / max(torso_len, 1e-4))
        return norm_disp

    @staticmethod
    def compute_blur_indicator(frames):
        """
        Computes mean Laplacian variance across sampled BGR frames.
        Higher value indicates sharp focus; lower value (< 50.0) indicates motion blur.
        """
        if cv2 is None or len(frames) == 0:
            return 100.0 # Default fallback if cv2 uninitialized
            
        variances = []
        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            variances.append(var)
            
        if len(variances) == 0:
            return 100.0
            
        return float(np.mean(variances))

    @staticmethod
    def compute_framing_quality(keypoints, width=1920, height=1080):
        """
        Calculates bounding box area ratio occupied by cow keypoints relative to frame area.
        Returns:
            area_ratio: float (0.0 to 1.0)
            in_bounds_ratio: float (ratio of keypoints inside frame bounds)
        """
        kp = np.asarray(keypoints, dtype=np.float32)
        T = kp.shape[0]
        
        # Min/max bounds
        x_min, x_max = float(np.min(kp[:, :, 0])), float(np.max(kp[:, :, 0]))
        y_min, y_max = float(np.min(kp[:, :, 1])), float(np.max(kp[:, :, 1]))
        
        # Bounding box width and height
        bbox_w = max(x_max - x_min, 1e-4)
        bbox_h = max(y_max - y_min, 1e-4)
        
        # Check if coordinates are normalized [0, 1] vs pixel
        if x_max <= 1.5 and y_max <= 1.5:
            area_ratio = float(bbox_w * bbox_h)
            in_bounds_ratio = float(np.mean((kp[:, :, 0] >= 0.0) & (kp[:, :, 0] <= 1.0) & (kp[:, :, 1] >= 0.0) & (kp[:, :, 1] <= 1.0)))
        else:
            w_frame = width if width > 0 else 1920
            h_frame = height if height > 0 else 1080
            area_ratio = float((bbox_w * bbox_h) / (w_frame * h_frame))
            in_bounds_ratio = float(np.mean((kp[:, :, 0] >= 0) & (kp[:, :, 0] <= w_frame) & (kp[:, :, 1] >= 0) & (kp[:, :, 1] <= h_frame)))
            
        return area_ratio, in_bounds_ratio
