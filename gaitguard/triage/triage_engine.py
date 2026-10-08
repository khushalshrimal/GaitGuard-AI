"""
GaitGuard AI - Screening Triage Engine & Decision Contract (Phase 8)
Implements threshold selection, confidence metrics, and 3-way triage logic (NORMAL, LAMENESS_RISK, INCONCLUSIVE).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

class ConfidenceEstimator:
    """
    Computes probability-based confidence and margin metrics.
    """

    @staticmethod
    def compute_confidence(calibrated_prob):
        """
        Computes normalized confidence C(p) = 2 * |p - 0.5| in range [0, 1].
        """
        prob = np.asarray(calibrated_prob, dtype=np.float32)
        conf = 2.0 * np.abs(prob - 0.5)
        return conf.astype(np.float32)

    @staticmethod
    def compute_margin(calibrated_prob, threshold=0.50):
        """
        Computes signed distance from decision threshold.
        """
        prob = np.asarray(calibrated_prob, dtype=np.float32)
        margin = prob - threshold
        return margin.astype(np.float32)


class TriageResultContract:
    """
    Standardized JSON contract formatter for GaitGuard triage output.
    """

    @staticmethod
    def format_result(sample_id, raw_prob, calibrated_prob, threshold, margin_delta=0.10):
        """
        Formats single sample decision into standardized dictionary.
        """
        conf = float(ConfidenceEstimator.compute_confidence(calibrated_prob))
        lower_bound = threshold - margin_delta
        upper_bound = threshold + margin_delta
        
        if calibrated_prob < lower_bound:
            decision = "NORMAL"
            inconclusive = False
            reason = "Gait movement trajectory falls within normal parameters."
        elif calibrated_prob > upper_bound:
            decision = "LAMENESS_RISK"
            inconclusive = False
            reason = "Abnormal gait motion detected (elevated back arch, slower stride rhythm)."
        else:
            decision = "INCONCLUSIVE"
            inconclusive = True
            reason = "Calibrated probability falls within ambiguous uncertainty band around decision threshold."
            
        return {
            "sample_id": str(sample_id),
            "risk_probability_raw": float(raw_prob),
            "risk_probability_calibrated": float(calibrated_prob),
            "decision": decision,
            "confidence": conf,
            "threshold": float(threshold),
            "uncertainty_margin": float(margin_delta),
            "inconclusive": inconclusive,
            "reason": reason,
            "disclaimer": "AI-assisted screening tool. Veterinary review recommended."
        }


class ScreeningTriageEngine:
    """
    Evaluates threshold analysis grid search and executes 3-way triage classification.
    """

    def __init__(self, screening_threshold=0.50, margin_delta=0.10):
        self.threshold = screening_threshold
        self.margin_delta = margin_delta

    def analyze_thresholds(self, y_true, y_prob, threshold_range=None):
        """
        Sweeps decision thresholds from 0.10 to 0.90 and computes detailed metrics table.
        """
        if threshold_range is None:
            threshold_range = np.arange(0.10, 0.91, 0.02)
            
        records = []
        for t in threshold_range:
            preds = (y_prob >= t).astype(int)
            tp = int(np.sum((y_true == 1) & (preds == 1)))
            tn = int(np.sum((y_true == 0) & (preds == 0)))
            fp = int(np.sum((y_true == 0) & (preds == 1)))
            fn = int(np.sum((y_true == 1) & (preds == 0)))
            
            acc = float(accuracy_score(y_true, preds))
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))
            
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            
            records.append({
                "threshold": float(t),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "specificity": spec,
                "fpr": fpr,
                "fnr": fnr,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            })
            
        return pd.DataFrame(records)

    def select_optimal_screening_threshold(self, y_true, y_prob, min_recall=0.80):
        """
        Selects optimal screening threshold prioritizing High Recall (>= min_recall) while maximizing F1/Precision.
        """
        df_thresh = self.analyze_thresholds(y_true, y_prob)
        
        # Filter thresholds meeting min_recall
        valid_df = df_thresh[df_thresh["recall"] >= min_recall]
        if len(valid_df) > 0:
            # Pick threshold with highest F1 score among valid candidates
            best_row = valid_df.sort_values(by=["f1_score", "precision"], ascending=False).iloc[0]
        else:
            # Fallback to threshold maximizing F1
            best_row = df_thresh.sort_values(by=["f1_score", "recall"], ascending=False).iloc[0]
            
        self.threshold = float(best_row["threshold"])
        return self.threshold, df_thresh

    def predict_triage(self, sample_ids, raw_probs, calibrated_probs):
        """
        Executes 3-way triage classification across arrays of probabilities.
        Returns DataFrame of formatted triage results.
        """
        results = []
        for sid, rp, cp in zip(sample_ids, raw_probs, calibrated_probs):
            res = TriageResultContract.format_result(
                sample_id=sid,
                raw_prob=rp,
                calibrated_prob=cp,
                threshold=self.threshold,
                margin_delta=self.margin_delta
            )
            results.append(res)
            
        return pd.DataFrame(results)
