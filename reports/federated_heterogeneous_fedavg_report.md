# Heterogeneous HeterogeneousFedAvg Report

## Executive Summary
This report presents the outcomes of the **HeterogeneousFedAvg** experiment (heterogeneous_fedavg) under the Heterogeneous-Feature Federated Learning architecture.

---

## Privacy and Security Configuration (Phase 9 Prototype)
- **Simulated Secure Aggregation:** Disabled
- **Differential Privacy:** Disabled
- **Update Clipping Norm (C):** 1.0
- **Noise Multiplier (sigma):** 0.0
- **RDP Target Delta (delta):** 1e-05
- **Accumulated Epsilon (eps):** None / Uncomputed (DP Disabled)

> [!NOTE]
> **Research Prototype Disclaimers:**
> 1. This system is a research prototype, NOT certified for production clinical privacy, HIPAA compliance, or GDPR compliance.
> 2. Simulated secure aggregation uses pairwise additive masking and explicitly assumes 100% round participation (no dropout resilience).
> 3. Privacy mechanisms apply strictly client-side to shared predictor updates (Z -> 1). Private hospital encoders (D_i -> Z) remain strictly local and are never clipped, noised, masked, or transmitted.

---

## Final Untouched Test Evaluation (Round 15)

| Hospital                 |   Test Samples | Accuracy   | Precision   | Recall   | Specificity   |   F1-Score |   ROC-AUC |   PR-AUC |
|:-------------------------|---------------:|:-----------|:------------|:---------|:--------------|-----------:|----------:|---------:|
| Hospital 1 (Cleveland)   |             46 | 84.78%     | 81.82%      | 85.71%   | 84.00%        |     0.8372 |    0.9371 |   0.9432 |
| Hospital 2 (Hungarian)   |             44 | 81.82%     | 72.22%      | 81.25%   | 82.14%        |     0.7647 |    0.8906 |   0.8267 |
| Hospital 3 (Switzerland) |             19 | 89.47%     | 94.44%      | 94.44%   | 0.00%         |     0.9444 |    0.25   |   0.9298 |

### Overall System Aggregate Performance
- **Macro Test Accuracy:** 85.36%
- **Macro Test Precision:** 82.83%
- **Macro Test Recall (Sensitivity):** 87.14%
- **Macro Test Specificity:** 55.38%
- **Macro Test F1-Score:** 0.8488
- **Macro Test ROC-AUC:** 0.6926
- **Macro Test PR-AUC:** 0.8999
