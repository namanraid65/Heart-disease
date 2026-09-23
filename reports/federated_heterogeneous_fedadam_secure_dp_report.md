# Heterogeneous HeterogeneousFedAdam Report

## Executive Summary
This report presents the outcomes of the **HeterogeneousFedAdam** experiment (heterogeneous_fedadam_secure_dp) under the Heterogeneous-Feature Federated Learning architecture.

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

## Final Untouched Test Evaluation (Round 12)

| Hospital                 |   Test Samples | Accuracy   | Precision   | Recall   | Specificity   |   F1-Score |   ROC-AUC |   PR-AUC |
|:-------------------------|---------------:|:-----------|:------------|:---------|:--------------|-----------:|----------:|---------:|
| Hospital 1 (Cleveland)   |             46 | 89.13%     | 86.36%      | 90.48%   | 88.00%        |     0.8837 |    0.9238 |   0.8564 |
| Hospital 2 (Hungarian)   |             44 | 79.55%     | 76.92%      | 62.50%   | 89.29%        |     0.6897 |    0.8717 |   0.6905 |
| Hospital 3 (Switzerland) |             19 | 89.47%     | 94.44%      | 94.44%   | 0.00%         |     0.9444 |    0.4722 |   0.9446 |

### Overall System Aggregate Performance
- **Macro Test Accuracy:** 86.05%
- **Macro Test Precision:** 85.91%
- **Macro Test Recall (Sensitivity):** 82.47%
- **Macro Test Specificity:** 59.10%
- **Macro Test F1-Score:** 0.8393
- **Macro Test ROC-AUC:** 0.7559
- **Macro Test PR-AUC:** 0.8305
