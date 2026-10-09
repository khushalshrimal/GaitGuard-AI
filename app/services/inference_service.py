"""
GaitGuard AI - End-to-End Inference & Explainability Service (Phase 12)
Connects Phase 10 Quality Gate -> Phase 9 Pose Pipeline -> Phase 7 BiLSTM -> Phase 8 Triage -> Phase 11 SHAP Explainer.
"""

import os
import time
import logging
import numpy as np

from gaitguard.inference.pipeline import GaitGuardInferencePipeline
from gaitguard.explainability.explainer import GaitGuardExplainer
from app.config import settings

logger = logging.getLogger("gaitguard.inference_service")

class InferenceService:
    """
    Main backend service orchestrating end-to-end video screening and explainability.
    Reuses canonical GaitGuardInferencePipeline and GaitGuardExplainer.
    """

    def __init__(self, pipeline: GaitGuardInferencePipeline = None, explainer: GaitGuardExplainer = None):
        self.pipeline = pipeline or GaitGuardInferencePipeline(
            screening_threshold=settings.SCREENING_THRESHOLD,
            margin_delta=settings.MARGIN_DELTA
        )
        self.explainer = explainer or GaitGuardExplainer(n_bg_samples=5, seed=42)

    def process_screening(self, video_path: str, animal_id: str = None, session_id: str = None, request_id: str = "req_default") -> dict:
        """
        Executes end-to-end screening workflow.
        """
        t0 = time.time()
        logger.info(f"[{request_id}] Starting video screening processing for '{video_path}'")
        
        # 1. Read Video Metadata
        meta = self.pipeline.reader.get_metadata(video_path)
        if not meta.is_valid:
            logger.warning(f"[{request_id}] Video metadata invalid: {meta.error_msg}")
            return {
                "status": "retry",
                "request_id": request_id,
                "video_quality": {
                    "status": "RETRY",
                    "quality_score": 0.0,
                    "keypoint_coverage": 0.0,
                    "motion_quality": 0.0,
                    "blur_indicator": 0.0,
                    "framing_quality": 0.0,
                    "issues": ["VIDEO_UNREADABLE"],
                    "user_guidance": ["Video file could not be read. Please record a new video and upload again."]
                },
                "inference": None,
                "explanation": None,
                "metadata": {
                    "request_id": request_id,
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "pipeline_version": settings.PIPELINE_VERSION,
                    "total_processing_time_sec": round(time.time() - t0, 4)
                },
                "result_summary": "Video quality insufficient for ML prediction.",
                "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
            }

        # 2. Read frames & Pose estimation
        try:
            frames, _ = self.pipeline.reader.read_all_frames(video_path)
            sampled_frames, _ = self.pipeline.sampler.sample_frames(frames)
            kp_raw, conf_raw = self.pipeline.estimator.estimate_pose_from_frames(sampled_frames)
        except Exception as e:
            logger.error(f"[{request_id}] Error during frame sampling or pose estimation: {e}")
            return {
                "status": "retry",
                "request_id": request_id,
                "video_quality": {
                    "status": "RETRY",
                    "quality_score": 30.0,
                    "keypoint_coverage": 0.0,
                    "motion_quality": 0.0,
                    "blur_indicator": 0.0,
                    "framing_quality": 0.0,
                    "issues": ["LOW_KEYPOINT_COVERAGE"],
                    "user_guidance": ["Failed to extract valid quadruped pose keypoints. Ensure cow is clearly visible."]
                },
                "inference": None,
                "explanation": None,
                "metadata": {
                    "request_id": request_id,
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "pipeline_version": settings.PIPELINE_VERSION,
                    "total_processing_time_sec": round(time.time() - t0, 4)
                },
                "result_summary": "Video quality insufficient for ML prediction.",
                "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
            }

        # 3. Phase 10 Video Quality Gate Evaluation
        quality_res = self.pipeline.quality_analyzer.analyze_quality(meta, frames=sampled_frames, keypoints=kp_raw, confidences=conf_raw)
        quality_dict = quality_res.to_dict()

        logger.info(
            f"[{request_id}] Quality Gate evaluation: status={quality_res.status}, "
            f"score={quality_res.quality_score}, coverage={quality_res.keypoint_coverage}, "
            f"motion={quality_res.motion_quality}, blur={quality_res.blur_indicator}, issues={quality_res.issues}"
        )

        # If Quality Gate returns RETRY: Intercept ML prediction entirely due to unrecoverable evidence failure
        if quality_res.status == "RETRY":
            logger.warning(f"[{request_id}] Quality Gate intercepted video with status RETRY. Bypassing ML inference. Issues: {quality_res.issues}")
            summary = "Insufficient visual evidence for ML prediction." if "INSUFFICIENT_VISUAL_EVIDENCE" in quality_res.issues else "Video quality insufficient for ML prediction."
            return {
                "status": "retry",
                "request_id": request_id,
                "video_quality": quality_dict,
                "inference": None,
                "explanation": None,
                "metadata": {
                    "request_id": request_id,
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "pipeline_version": settings.PIPELINE_VERSION,
                    "total_processing_time_sec": round(time.time() - t0, 4)
                },
                "result_summary": summary,
                "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
            }

        # 4. Phase 9 Pose Pipeline -> Phase 7 BiLSTM -> Phase 8 Calibration & Triage
        triage_res = self.pipeline.analyze_keypoint_sequence(kp_raw, sample_id=os.path.basename(video_path))
        
        inference_details = {
            "raw_probability": round(triage_res["risk_probability_raw"], 4),
            "calibrated_probability": round(triage_res["risk_probability_calibrated"], 4),
            "decision": triage_res["decision"],
            "confidence": triage_result_conf(triage_res),
            "threshold": triage_res.get("threshold", settings.SCREENING_THRESHOLD),
            "uncertainty_margin": triage_res.get("uncertainty_margin", settings.MARGIN_DELTA)
        }

        # 5. Phase 11 Explainability Layer (Safe Execution)
        try:
            exp_res = self.explainer.explain_triage_contract(triage_res, kp_raw)
            explanation_details = {
                "available": exp_res.get("explanation_available", True),
                "top_contributors": exp_res.get("top_contributors", []),
                "modality_attribution": exp_res.get("modality_attribution"),
                "body_region_attribution": exp_res.get("body_region_attribution"),
                "temporal_attribution": exp_res.get("temporal_attribution"),
                "derived_gait_evidence": exp_res.get("derived_gait_evidence", []),
                "reason": None
            }
        except Exception as e:
            logger.error(f"[{request_id}] Explainability calculation failed gracefully: {e}")
            explanation_details = {
                "available": False,
                "top_contributors": [],
                "modality_attribution": None,
                "body_region_attribution": None,
                "temporal_attribution": None,
                "derived_gait_evidence": [],
                "reason": "Explanation calculation temporarily unavailable"
            }

        # 6. Non-Diagnostic User Summary Assignment
        decision = triage_res["decision"]
        if decision == "NORMAL":
            summary = "No elevated lameness risk detected by this screening model."
        elif decision == "LAMENESS_RISK":
            summary = "Elevated lameness risk indicated by the screening model."
        else:
            summary = "Screening result is inconclusive."

        total_time = round(time.time() - t0, 4)
        logger.info(f"[{request_id}] Completed screening in {total_time}s | Decision: {decision}")

        return {
            "status": "success",
            "request_id": request_id,
            "video_quality": quality_dict,
            "inference": inference_details,
            "explanation": explanation_details,
            "metadata": {
                "request_id": request_id,
                "animal_id": animal_id,
                "session_id": session_id,
                "pipeline_version": settings.PIPELINE_VERSION,
                "total_processing_time_sec": total_time
            },
            "result_summary": summary,
            "disclaimer": "AI-assisted screening tool. This output is not a veterinary diagnosis."
        }


def triage_result_conf(triage_res):
    """Helper to extract confidence string safely."""
    conf = triage_res.get("confidence", "HIGH")
    if isinstance(conf, str):
        return conf
    elif isinstance(conf, (float, int)):
        return "HIGH" if conf > 0.8 else ("MEDIUM" if conf > 0.5 else "LOW")
    return "HIGH"
