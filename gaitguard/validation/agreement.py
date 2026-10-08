"""
GaitGuard AI - Inter-Rater Agreement & Blinding Audit Module (Phase 15)
Computes Cohen's Kappa, raw percentage agreement, confidence intervals,
and verifies blinded annotation isolation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

class InterRaterAgreementAnalyzer:
    """
    Computes inter-rater reliability metrics between independent expert annotators.
    """

    @staticmethod
    def compute_cohens_kappa(rater1_labels, rater2_labels):
        """
        Computes Cohen's Kappa coefficient between two raters.
        Returns: (kappa, raw_agreement_pct)
        """
        r1 = np.asarray(rater1_labels)
        r2 = np.asarray(rater2_labels)

        if len(r1) == 0 or len(r2) == 0 or len(r1) != len(r2):
            return 0.0, 0.0

        raw_agreement = float(np.mean(r1 == r2) * 100.0)
        kappa = float(cohen_kappa_score(r1, r2))
        return kappa, raw_agreement

    @staticmethod
    def audit_blinded_annotation(annotation_record, model_prediction=None):
        """
        Verifies that an annotation record was generated independently
        without exposure to model predictions.
        """
        if not annotation_record.get("blinded_flag", False):
            return False, "Annotation record flagged as non-blinded."
        
        # Verify label was not derived from GaitGuard prediction fields
        forbidden_keys = ["model_prob", "gaitguard_prediction", "shap_attribution"]
        for key in forbidden_keys:
            if key in annotation_record:
                return False, f"Annotation contains forbidden GaitGuard key '{key}'."
                
        return True, "Annotation passes blinded independence audit."

    @staticmethod
    def compute_binomial_ci(k, n, confidence=0.95):
        """
        Computes Wilson score or exact binomial confidence interval for proportion k/n.
        """
        if n <= 0:
            return 0.0, 0.0, 0.0
            
        p = float(k) / n
        z = 1.96  # 95% CI
        denom = 1.0 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        margin = z * np.sqrt((p * (1.0 - p) + (z**2) / (4 * n)) / n) / denom
        
        lower = max(0.0, float(center - margin))
        upper = min(1.0, float(center + margin))
        return p, lower, upper
