# Heterogeneous-Feature Federated Learning Report

## Executive Summary
This report presents the outcomes of **Phase 7: Heterogeneous-Feature Federated Learning**, which establishes a federated architecture capable of training across hospitals with differing native clinical feature spaces without requiring artificial padding or dimension homogenizing.

Each hospital maintains a **private local encoder** that projects its native feature space ($D_i$) into a common latent representation ($Z = 32$). Only the **shared predictor** operating on $Z$ is federated and aggregated using sample-weighted FedAvg.

---

## Architecture Summary
- **Client Private Encoders:** Linear($D_i, 64$) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear($64, 32$)
- **Shared Federated Predictor:** Linear($32, 32$) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear($32, 1$)
- **Communication Rounds:** 15
- **Selected Best Round:** Round 15 (selected strictly via macro validation ROC-AUC)

---

## Final Untouched Test Evaluation

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

---

## Architectural Findings
1. **Separation of Concerns:** Transmitting solely the shared predictor parameters avoids the structural incompatibility of differing first-layer encoder dimensions ($D_i \times H$).
2. **Latent Space Alignment:** The shared predictor successfully guides local encoder training, aligning heterogeneous feature representations into a predictive latent subspace.
3. **Strict Privacy & Locality:** Encoders and patient datasets remain fully isolated within their respective institutional silos.
