"""
GaitGuard AI - Attribution & Gait Evidence Aggregator (Phase 11)
Aggregates raw 76-dimensional temporal SHAP attributions into modality,
anatomical body regions, temporal phases, and Level 2 derived gait evidence.
"""

import numpy as np
from gaitguard.inference.feature_schema import FEATURE_SCHEMA_76, KP_NAMES

# Canonical Keypoint Index Mapping
KP_INDICES = {kp: i for i, kp in enumerate(KP_NAMES)}

BODY_REGION_MAP = {
    "Head": [12, 13],                             # Nose, HeadTop
    "Spine": [14, 15, 16],                        # Spine1, Spine2, Spine3
    "Forelimbs": [0, 1, 2, 3, 4, 5],              # LFHoof, LFAnkle, LFKnee, RFHoof, RFAnkle, RFKnee
    "Hindlimbs": [6, 7, 8, 9, 10, 11]             # LHHoof, LHAnkle, LHKnee, RHHoof, RHAnkle, RHKnee
}

class AttributionAggregator:
    """
    Aggregates (128, 76) or (76,) SHAP attributions into structured interpretable evidence.
    """

    @staticmethod
    def compute_feature_attributions(shap_matrix):
        """
        Computes 1D mean SHAP attribution vector of shape (76,) from (128, 76) sequence matrix.
        Input:
            shap_matrix: numpy array of shape (128, 76) or (76,)
        Returns:
            attr_1d: numpy array of shape (76,)
        """
        arr = np.asarray(shap_matrix, dtype=np.float32)
        if arr.ndim == 2:
            return np.mean(arr, axis=0) # Mean across timesteps
        elif arr.ndim == 1:
            return arr
        else:
            raise ValueError(f"Expected SHAP array of dim 1 or 2, got shape {arr.shape}")

    @staticmethod
    def aggregate_by_modality(shap_matrix):
        """
        Aggregates attribution across input feature modalities:
            - Coordinates: indices 0..33
            - Velocity: indices 34..67
            - Biomechanical: indices 68..75
        """
        attr = AttributionAggregator.compute_feature_attributions(shap_matrix)
        
        coord_attr = float(np.mean(np.abs(attr[0:34])))
        velo_attr = float(np.mean(np.abs(attr[34:68])))
        bio_attr = float(np.mean(np.abs(attr[68:76])))
        
        total = max(coord_attr + velo_attr + bio_attr, 1e-8)
        
        return {
            "coordinates": round(coord_attr / total, 4),
            "velocity": round(velo_attr / total, 4),
            "biomechanical": round(bio_attr / total, 4),
            "raw_abs_means": {
                "coordinates": round(coord_attr, 6),
                "velocity": round(velo_attr, 6),
                "biomechanical": round(bio_attr, 6)
            }
        }

    @staticmethod
    def aggregate_by_body_region(shap_matrix):
        """
        Aggregates attribution by anatomical region (Head, Spine, Forelimbs, Hindlimbs).
        """
        attr = AttributionAggregator.compute_feature_attributions(shap_matrix)
        region_scores = {}
        
        for region, kp_indices in BODY_REGION_MAP.items():
            feat_indices = []
            for kp_idx in kp_indices:
                feat_indices.append(kp_idx * 2)       # X coord
                feat_indices.append(kp_idx * 2 + 1)   # Y coord
                feat_indices.append(34 + kp_idx * 2)   # X velocity
                feat_indices.append(34 + kp_idx * 2 + 1) # Y velocity
                
            region_attr = float(np.mean(np.abs(attr[feat_indices])))
            region_scores[region] = region_attr
            
        total = max(sum(region_scores.values()), 1e-8)
        
        return {region: round(score / total, 4) for region, score in region_scores.items()}

    @staticmethod
    def aggregate_by_temporal_phase(shap_matrix):
        """
        Aggregates attribution across sequence time phases:
            - early_sequence: frames 0..42
            - middle_sequence: frames 43..85
            - late_sequence: frames 86..127
        """
        arr = np.asarray(shap_matrix, dtype=np.float32)
        if arr.ndim != 2:
            return {"early_sequence": 0.333, "middle_sequence": 0.333, "late_sequence": 0.334}
            
        early = float(np.mean(np.abs(arr[0:43, :])))
        middle = float(np.mean(np.abs(arr[43:86, :])))
        late = float(np.mean(np.abs(arr[86:128, :])))
        
        total = max(early + middle + late, 1e-8)
        
        return {
            "early_sequence": round(early / total, 4),
            "middle_sequence": round(middle / total, 4),
            "late_sequence": round(late / total, 4)
        }

    @staticmethod
    def get_top_contributors(shap_matrix, top_k=5):
        """
        Extracts top-K features ranked by absolute SHAP contribution with directionality.
        """
        attr = AttributionAggregator.compute_feature_attributions(shap_matrix)
        abs_attr = np.abs(attr)
        top_indices = np.argsort(abs_attr)[::-1][:top_k]
        
        contributors = []
        for idx in top_indices:
            feat_name = FEATURE_SCHEMA_76[idx]
            val = float(attr[idx])
            
            if val > 1e-5:
                direction = "INCREASES_RISK"
            elif val < -1e-5:
                direction = "DECREASES_RISK"
            else:
                direction = "LOW_CONTRIBUTION"
                
            # Classify modality and region
            if idx < 34:
                modality = "coordinates"
                kp_idx = idx // 2
                region = next((r for r, kps in BODY_REGION_MAP.items() if kp_idx in kps), "Other")
            elif idx < 68:
                modality = "velocity"
                kp_idx = (idx - 34) // 2
                region = next((r for r, kps in BODY_REGION_MAP.items() if kp_idx in kps), "Other")
            else:
                modality = "biomechanical"
                region = "Spine" if "arch" in feat_name or "head" in feat_name else "Limbs"

            contributors.append({
                "feature_name": feat_name,
                "feature_index": int(idx),
                "attribution": round(val, 6),
                "abs_attribution": round(float(abs_attr[idx]), 6),
                "direction": direction,
                "modality": modality,
                "body_region": region
            })
            
        return contributors

    @staticmethod
    def extract_derived_gait_evidence(sequence_tensor, shap_matrix):
        """
        Maps model attributions to Level 2 human-interpretable derived gait concepts.
        Labels concepts strictly as 'derived_gait_evidence' (NOT direct SHAP features).
        """
        from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
        seq = np.asarray(sequence_tensor, dtype=np.float32)
        
        if seq.ndim == 3 and seq.shape[1] == 17 and seq.shape[2] == 2:
            builder = TemporalSequenceBuilder()
            seq = builder.build_dataset(seq[np.newaxis, :], np.ones((1, 128), dtype=np.int32), mode="combined")[0]
        elif seq.ndim == 4 and seq.shape[2] == 17 and seq.shape[3] == 2:
            builder = TemporalSequenceBuilder()
            seq = builder.build_dataset(seq, np.ones((1, 128), dtype=np.int32), mode="combined")[0]
        elif seq.ndim == 3 and seq.shape[2] == 76:
            seq = seq[0]
            
        attr = AttributionAggregator.compute_feature_attributions(shap_matrix)
        
        # 1. Walking Speed (Velocity of Spine1 & Spine3)
        spine_vel_idx = [34 + 14*2, 34 + 14*2 + 1, 34 + 16*2, 34 + 16*2 + 1]
        speed_attr = float(np.mean(attr[spine_vel_idx]))
        speed_val = float(np.mean(np.sqrt(seq[:, 34+14*2]**2 + seq[:, 34+14*2+1]**2)))
        
        # 2. Stride Length (Hoof X separation)
        hoof_sep_attr = float(attr[70] + attr[71]) / 2.0
        stride_val = float(np.mean(seq[:, 70] + seq[:, 71]))
        
        # 3. Back Arch Curvature
        back_arch_attr = float(attr[68])
        back_arch_val = float(np.mean(seq[:, 68]))
        
        # 4. Knee Flexion / Elevation
        knee_attr = float(attr[74] + attr[75]) / 2.0
        knee_val = float(np.mean(seq[:, 74] + seq[:, 75]))

        # 5. Head Movement
        head_attr = float(attr[69])
        head_val = float(np.mean(seq[:, 69]))

        evidence_items = [
            {
                "name": "normalized_walking_speed",
                "value": round(speed_val, 4),
                "relative_contribution": round(speed_attr, 6),
                "direction": "INCREASES_RISK" if speed_attr > 1e-5 else ("DECREASES_RISK" if speed_attr < -1e-5 else "LOW_CONTRIBUTION"),
                "interpretation": "Variation in forward walking speed across movement sequence."
            },
            {
                "name": "torso_normalized_stride_length",
                "value": round(stride_val, 4),
                "relative_contribution": round(hoof_sep_attr, 6),
                "direction": "INCREASES_RISK" if hoof_sep_attr > 1e-5 else ("DECREASES_RISK" if hoof_sep_attr < -1e-5 else "LOW_CONTRIBUTION"),
                "interpretation": "Spatial stride separation between left and right hooves."
            },
            {
                "name": "back_arch_curvature",
                "value": round(back_arch_val, 4),
                "relative_contribution": round(back_arch_attr, 6),
                "direction": "INCREASES_RISK" if back_arch_attr > 1e-5 else ("DECREASES_RISK" if back_arch_attr < -1e-5 else "LOW_CONTRIBUTION"),
                "interpretation": "Dorsal spinal curvature elevation observed during gait sequence."
            },
            {
                "name": "knee_flexion_elevation",
                "value": round(knee_val, 4),
                "relative_contribution": round(knee_attr, 6),
                "direction": "INCREASES_RISK" if knee_attr > 1e-5 else ("DECREASES_RISK" if knee_attr < -1e-5 else "LOW_CONTRIBUTION"),
                "interpretation": "Vertical knee joint elevation during movement cycle."
            },
            {
                "name": "head_nodding_movement",
                "value": round(head_val, 4),
                "relative_contribution": round(head_attr, 6),
                "direction": "INCREASES_RISK" if head_attr > 1e-5 else ("DECREASES_RISK" if head_attr < -1e-5 else "LOW_CONTRIBUTION"),
                "interpretation": "Head vertical displacement amplitude during gait sequence."
            }
        ]
        
        return evidence_items
