# GaitGuard AI — Phase 15 Inter-Rater Reliability Protocol

## 1. Metric Specification
Inter-rater agreement between independent human locomotion evaluators is measured using **Cohen's Kappa ($\kappa$)** for two raters and **Fleiss' Kappa** for multiple raters.

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed raw agreement proportion and $p_e$ is expected chance agreement proportion.

## 2. Landis & Koch Agreement Thresholds
- $\kappa < 0.20$: Poor / Slight Agreement
- $0.21 \le \kappa \le 0.40$: Fair Agreement
- $0.41 \le \kappa \le 0.60$: Moderate Agreement
- $0.61 \le \kappa \le 0.80$: Substantial Agreement
- $0.81 \le \kappa \le 1.00$: Almost Perfect Agreement

## 3. Reporting Rules
Inter-rater agreement measures consistency between reference human annotators, **not** AI model accuracy. Higher inter-rater agreement ensures a more reliable ground-truth reference baseline.
