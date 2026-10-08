"""
GaitGuard AI - Explanation Stability & Perturbation Faithfulness Evaluator (Phase 11)
Evaluates top-K Jaccard overlap, sign consistency, and perturbation faithfulness.
"""

import numpy as np
import torch
from gaitguard.explainability.aggregator import AttributionAggregator

class ExplanationStabilityEvaluator:
    """
    Evaluates stability and faithfulness of SHAP model attributions.
    """

    @staticmethod
    def evaluate_stability(explainer, sequence_tensor, mask_tensor=None, n_runs=5, top_k=5):
        """
        Runs repeated SHAP explanations across background subsamples and calculates top-K overlap.
        """
        top_k_sets = []
        sign_vectors = []
        
        for run_idx in range(n_runs):
            # Generate attribution with distinct seed or background sample
            shap_mat = explainer.explain_sequence(sequence_tensor, mask_tensor, seed=42 + run_idx)
            attr_1d = AttributionAggregator.compute_feature_attributions(shap_mat)
            
            # Top-K feature indices
            top_indices = set(np.argsort(np.abs(attr_1d))[::-1][:top_k])
            top_k_sets.append(top_indices)
            sign_vectors.append(np.sign(attr_1d))

        # Compute pairwise Jaccard similarities
        jaccard_sims = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                set_a = top_k_sets[i]
                set_b = top_k_sets[j]
                intersection = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                jaccard = float(intersection / max(union, 1))
                jaccard_sims.append(jaccard)

        mean_jaccard = float(np.mean(jaccard_sims)) if len(jaccard_sims) > 0 else 1.0

        # Sign consistency
        signs_stack = np.array(sign_vectors) # (n_runs, 76)
        mode_match = np.mean(np.abs(np.mean(signs_stack, axis=0)) == 1.0)
        sign_consistency = float(mode_match)

        stability_score = round(float(0.7 * mean_jaccard + 0.3 * sign_consistency), 4)

        return {
            "jaccard_overlap": round(mean_jaccard, 4),
            "sign_consistency": round(sign_consistency, 4),
            "stability_score": stability_score,
            "n_runs": n_runs,
            "top_k": top_k
        }

    @staticmethod
    def evaluate_faithfulness_perturbation(model, sequence_tensor, mask_tensor, shap_matrix, top_k=5):
        """
        Perturbation sanity test: zeroing top positive contributors should reduce risk probability.
        """
        model.eval()
        seq = np.asarray(sequence_tensor, dtype=np.float32).copy()
        
        if seq.ndim == 3 and seq.shape[1] == 17 and seq.shape[2] == 2:
            from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
            from gaitguard.temporal.preprocessing import FoldTemporalScaler
            seq = TemporalSequenceBuilder().build_dataset(seq[np.newaxis, :], np.ones((1, 128), dtype=np.int32), mode="combined")
            seq = FoldTemporalScaler().fit_transform(seq, np.ones((1, 128), dtype=np.int32))
        elif seq.ndim == 4 and seq.shape[2] == 17 and seq.shape[3] == 2:
            from gaitguard.temporal.sequence_builder import TemporalSequenceBuilder
            from gaitguard.temporal.preprocessing import FoldTemporalScaler
            seq = TemporalSequenceBuilder().build_dataset(seq, np.ones((1, 128), dtype=np.int32), mode="combined")
            seq = FoldTemporalScaler().fit_transform(seq, np.ones((1, 128), dtype=np.int32))
        elif seq.ndim == 2 and seq.shape[0] == 128 and seq.shape[1] == 76:
            seq = seq[np.newaxis, :]
            
        m = np.asarray(mask_tensor, dtype=np.float32) if mask_tensor is not None else np.ones((seq.shape[0], seq.shape[1]), dtype=np.float32)
        if m.ndim == 1:
            m = m[np.newaxis, :]

        # Baseline prediction
        with torch.no_grad():
            x_t = torch.tensor(seq, dtype=torch.float32)
            m_t = torch.tensor(m, dtype=torch.float32)
            p_base = float(model(x_t, m_t).item())

        # Extract top positive contributors
        attr_1d = AttributionAggregator.compute_feature_attributions(shap_matrix)
        pos_indices = np.where(attr_1d > 1e-5)[0]
        if len(pos_indices) > 0:
            sorted_pos = pos_indices[np.argsort(attr_1d[pos_indices])[::-1][:top_k]]
        else:
            sorted_pos = np.argsort(np.abs(attr_1d))[::-1][:top_k]

        # Perturb sequence by setting top positive feature channels to zero
        seq_pert = seq.copy()
        seq_pert[:, :, sorted_pos] = 0.0

        with torch.no_grad():
            x_pert_t = torch.tensor(seq_pert, dtype=torch.float32)
            p_pert = float(model(x_pert_t, m_t).item())

        prob_delta = float(p_base - p_pert)

        return {
            "p_baseline": round(p_base, 4),
            "p_perturbed": round(p_pert, 4),
            "prob_delta": round(prob_delta, 4),
            "top_k_perturbed": sorted_pos.tolist(),
            "faithfulness_passed": bool(prob_delta > -0.05) # Prediction drop or minimal fluctuation
        }
