# Heterogeneous HeterogeneousFedProx Report

## Executive Summary
This report presents the outcomes of the **HeterogeneousFedProx** experiment (heterogeneous_fedprox_secure_dp) under the Heterogeneous-Feature Federated Learning architecture.

---

## Privacy and Security Configuration (Phase 9 Prototype)
- **Simulated Secure Aggregation:** Enabled (Pairwise Additive Masking)
- **Differential Privacy:** Enabled (Client-Side Gaussian Mechanism)
- **Update Clipping Norm (C):** 1.0
- **Noise Multiplier (sigma):** 0.3
- **RDP Target Delta (delta):** 1e-05
- **Accumulated Epsilon (eps):** 148.0259

> [!NOTE]
> **Research Prototype Disclaimers:**
> 1. This system is a research prototype, NOT certified for production clinical privacy, HIPAA compliance, or GDPR compliance.
> 2. Simulated secure aggregation uses pairwise additive masking and explicitly assumes 100% round participation (no dropout resilience).
> 3. Privacy mechanisms apply strictly client-side to shared predictor updates (Z -> 1). Private hospital encoders (D_i -> Z) remain strictly local and are never clipped, noised, masked, or transmitted.

---

## Final Untouched Test Evaluation (Round 5)

| Hospital                 |   Test Samples | Accuracy   | Precision   | Recall   | Specificity   |   F1-Score |   ROC-AUC |   PR-AUC |
|:-------------------------|---------------:|:-----------|:------------|:---------|:--------------|-----------:|----------:|---------:|
| Hospital 1 (Cleveland)   |             46 | 89.13%     | 83.33%      | 95.24%   | 84.00%        |     0.8889 |    0.9505 |   0.9425 |
| Hospital 2 (Hungarian)   |             44 | 81.82%     | 72.22%      | 81.25%   | 82.14%        |     0.7647 |    0.9018 |   0.8056 |
| Hospital 3 (Switzerland) |             19 | 94.74%     | 94.74%      | 100.00%  | 0.00%         |     0.973  |    0.2778 |   0.9205 |

### Overall System Aggregate Performance
- **Macro Test Accuracy:** 88.56%
- **Macro Test Precision:** 83.43%
- **Macro Test Recall (Sensitivity):** 92.16%
- **Macro Test Specificity:** 55.38%
- **Macro Test F1-Score:** 0.8755
- **Macro Test ROC-AUC:** 0.7100
- **Macro Test PR-AUC:** 0.8895
