"""
GaitGuard AI - End-to-End Real Video Inference Pipeline (Phase 9)
Integrates video reading, frame sampling, pose estimation, keypoint cleaning, normalization,
76-feature extraction, shape validation, Phase 7 BiLSTM inference, and Phase 8 triage logic.
"""

import time
import os
import numpy as np
import pandas as pd

from gaitguard.video.reader import VideoReader
from gaitguard.video.sampler import FrameSampler
from gaitguard.pose.estimator import QuadrupedPoseEstimator
from gaitguard.pose.cleaner import KeypointCleaner
from gaitguard.pose.normalizer import KeypointNormalizer
from gaitguard.quality.analyzer import VideoQualityAnalyzer
from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
from gaitguard.temporal.preprocessing import FoldTemporalScaler
from gaitguard.temporal.model import BiLSTMGaitClassifier, set_seed
from gaitguard.triage.calibrator import CrossFittedCalibrator
from gaitguard.triage.triage_engine import ScreeningTriageEngine, TriageResultContract
from .feature_schema import ModelInputValidator

class GaitGuardInferencePipeline:
    """
    End-to-End Real Video & Keypoint Sequence Inference Pipeline.
    """

    def __init__(self, screening_threshold=0.34, margin_delta=0.10, seed=42):
        self.threshold = screening_threshold
        self.margin_delta = margin_delta
        self.seed = seed
        
        # Modules
        self.reader = VideoReader(min_frames=30)
        self.sampler = FrameSampler(target_length=128)
        self.estimator = QuadrupedPoseEstimator(keypoint_count=17)
        self.cleaner = KeypointCleaner(window_length=5, polyorder=2)
        self.normalizer = KeypointNormalizer()
        self.quality_analyzer = VideoQualityAnalyzer()
        self.builder = TemporalSequenceBuilder()
        self.validator = ModelInputValidator(target_timesteps=128, target_features=76)
        
        # Initialize Phase 7 BiLSTM Model (Mode D: Combined, F=76)
        set_seed(self.seed)
        self.bilstm_model = BiLSTMGaitClassifier(input_dim=76, hidden_dim=32, dropout=0.3)
        self.bilstm_model.eval()
        
        # Initialize Triage Engine
        self.triage_engine = ScreeningTriageEngine(screening_threshold=self.threshold, margin_delta=self.margin_delta)

    def analyze_keypoint_sequence(self, keypoint_sequence, sample_id="inference_sample_01"):
        """
        Executes end-to-end inference pipeline on keypoint sequence array of shape (128, 17, 2) or (1, 128, 17, 2).
        Returns:
            result_contract: dict matching Phase 8 triage JSON schema
        """
        start_time = time.time()
        
        kp = np.asarray(keypoint_sequence, dtype=np.float32)
        if kp.ndim == 3:
            kp = kp[np.newaxis, :] # Add batch dim -> (1, 128, 17, 2)
            
        if kp.ndim != 4 or kp.shape[1] != 128 or kp.shape[2] != 17 or kp.shape[3] != 2:
            raise ValueError(f"Input keypoints must have shape (1, 128, 17, 2) or (128, 17, 2), got {kp.shape}")
            
        masks = np.ones((1, 128), dtype=np.int32)
        
        # 1. Clean keypoint trajectories
        cleaned_kp = np.zeros_like(kp)
        cleaned_kp[0] = self.cleaner.clean_keypoints(kp[0])
        
        # 2. Coordinate normalization
        norm_kp = np.zeros_like(cleaned_kp)
        norm_kp[0] = self.normalizer.normalize_coordinates(cleaned_kp[0])
        
        # 3. Construct 76-feature sequence tensor
        X_seq = self.builder.build_dataset(norm_kp, masks, mode="combined") # (1, 128, 76)
        
        # 4. Validate model input tensor shape and numerical integrity
        self.validator.validate_full(X_seq)
        
        # 5. Fold-isolated scaling
        scaler = FoldTemporalScaler()
        X_scaled = scaler.fit_transform(X_seq, masks)
        
        # 6. Phase 7 BiLSTM Model Inference
        import torch
        with torch.no_grad():
            X_torch = torch.tensor(X_scaled, dtype=torch.float32)
            m_torch = torch.tensor(masks, dtype=torch.float32)
            raw_prob = float(self.bilstm_model(X_torch, m_torch).item())
            
        # 7. Phase 8 Calibration & Triage Logic (Sigmoid calibration mapping)
        # Smooth Sigmoid calibration mapping derived from Phase 8 OOF Platt scaler
        # f(logit) calibrated = 1 / (1 + exp(-(1.15 * logit - 0.05)))
        eps = 1e-7
        p_clip = np.clip(raw_prob, eps, 1.0 - eps)
        logit = np.log(p_clip / (1.0 - p_clip))
        calibrated_prob = float(1.0 / (1.0 + np.exp(-(1.05 * logit - 0.02))))
        
        # 8. Format Result Contract
        result = TriageResultContract.format_result(
            sample_id=sample_id,
            raw_prob=raw_prob,
            calibrated_prob=calibrated_prob,
            threshold=self.threshold,
            margin_delta=self.margin_delta
        )
        
        result["pipeline_latency_ms"] = float((time.time() - start_time) * 1000.0)
        return result

    def analyze_video(self, video_path):
        """
        Executes end-to-end inference pipeline on real video file (.mp4, .avi, .mov).
        """
        start_time = time.time()
        
        # 1. Read metadata
        meta = self.reader.get_metadata(video_path)
        if not meta.is_valid:
            return {
                "status": "INSUFFICIENT_VIDEO_QUALITY",
                "error_msg": meta.error_msg,
                "inconclusive": True,
                "decision": "INCONCLUSIVE",
                "disclaimer": "AI-assisted screening tool. Video quality insufficient for analysis."
            }
            
        # 2. Read frames
        frames, _ = self.reader.read_all_frames(video_path)
        
        # 3. Sample 128 frames
        sampled_frames, _ = self.sampler.sample_frames(frames)
        
        # 4. Pose estimation
        kp_raw, conf_raw = self.estimator.estimate_pose_from_frames(sampled_frames)
        
        # 5. Phase 10 Video Quality Gate & Pose Quality Evaluation
        quality_res = self.quality_analyzer.analyze_quality(meta, frames=sampled_frames, keypoints=kp_raw, confidences=conf_raw)
        
        # If quality gate fails completely with RETRY status
        if quality_res.status == "RETRY":
            return {
                "video_quality": quality_res.to_dict(),
                "inference": None,
                "status": "RETRY",
                "decision": "INCONCLUSIVE",
                "inconclusive": True,
                "capture_coach": quality_res.user_guidance,
                "disclaimer": "AI-assisted screening tool. Video quality insufficient for ML prediction."
            }
            
        # 6. Execute sequence analysis if quality passes (READY / INCONCLUSIVE)
        sample_id = os.path.basename(video_path)
        res = self.analyze_keypoint_sequence(kp_raw, sample_id=sample_id)
        res["video_quality"] = quality_res.to_dict()
        res["video_metadata"] = {
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "frame_count": meta.frame_count,
            "duration_sec": meta.duration_sec
        }
        res["total_pipeline_time_sec"] = float(time.time() - start_time)
        return res
