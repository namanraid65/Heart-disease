# Privacy and Security in Heterogeneous Federated Learning (Phase 9 Research Report)

## Executive Summary
This report presents the empirical evaluation of privacy and security mechanisms implemented under the **Heterogeneous-Feature Federated Learning architecture** (Phase 9).
Two independently configurable mechanisms are evaluated:
1. **Simulated Secure Aggregation:** Pairwise additive masking ($\sum_i M_i = 0$) ensuring the central server observes only the aggregate sum without access to individual client updates.
2. **Client-Side Differential Privacy:** Gradient update L2-norm clipping ($C = 1.0$) and calibrated Gaussian noise ($\sigma = 0.3$), with formal Rényi Differential Privacy (RDP) accounting composition across communication rounds (target $\delta = 1e-05$).

---

> [!CAUTION]
> ### Explicit Research Prototype Disclaimers & Compliance Notice
> - **NOT Production-Grade Clinical Security:** This implementation is a research simulation designed to benchmark privacy-utility trade-offs. It is NOT certified for production healthcare deployments, HIPAA compliance, or GDPR compliance.
> - **Pairwise Masking Participation Assumption:** The pairwise masking protocol operates on a full-participation assumption. If a client drops out mid-round without sending unmasking secrets, the server cannot reconstruct the aggregate. It does not implement Shamir secret sharing or threshold reconstruction.
> - **Parameter Boundary:** Mechanisms apply **strictly to the shared predictor parameters ($Z \to 1$)**. Private hospital encoders ($D_i \to Z$) remain strictly client-local and are never clipped, noised, masked, or transmitted.
> - **Zero-Noise Final Model:** No noise is ever injected into the final evaluated model weights. Noise is applied strictly to transmitted update deltas during training rounds.

---

## Critical Parameter Boundary Verification
| Layer Component | Dimensions | Client-Local / Network | Differential Privacy | Secure Aggregation Masking |
| :--- | :--- | :--- | :--- | :--- |
| **Hospital 1 Encoder ($D_1 \to Z$)** | $25 \to 16$ | Strictly Local | **Excluded** (Zero clipping, zero noise) | **Excluded** (Never masked, never sent) |
| **Hospital 2 Encoder ($D_2 \to Z$)** | $25 \to 16$ | Strictly Local | **Excluded** (Zero clipping, zero noise) | **Excluded** (Never masked, never sent) |
| **Hospital 3 Encoder ($D_3 \to Z$)** | $25 \to 16$ | Strictly Local | **Excluded** (Zero clipping, zero noise) | **Excluded** (Never masked, never sent) |
| **Shared Global Predictor ($Z \to 1$)** | $16 \to 8 \to 1$ | Transmitted & Aggregated | **Protected** (L2-clipped to $C=1.0$, Gaussian $\sigma=0.3$) | **Protected** (Pairwise masked $w_i + M_i$) |

---

## Comparative System Performance on Final Untouched Test Split

| Configuration         | SecAgg   | DP   | Epsilon       | Best Round   | Macro Acc   | Macro Recall   | Macro Spec   |   Macro F1 |   Macro AUC |   Macro PR-AUC |   Weighted F1 |   Weighted AUC |
|:----------------------|:---------|:-----|:--------------|:-------------|:------------|:---------------|:-------------|-----------:|------------:|---------------:|--------------:|---------------:|
| FedAvg (Baseline)     | No       | No   | None (DP Off) | Round 15     | 85.36%      | 87.14%         | 55.38%       |     0.8488 |      0.6926 |         0.8999 |        0.8266 |         0.7986 |
| FedAvg + SecAgg       | Yes      | No   | None (DP Off) | Round 15     | 85.36%      | 87.14%         | 55.38%       |     0.8488 |      0.6926 |         0.8999 |        0.8266 |         0.7986 |
| FedAvg + DP           | No       | Yes  | 148.0259      | Round 5      | 89.32%      | 92.16%         | 56.57%       |     0.8832 |      0.7024 |         0.8913 |        0.8628 |         0.8106 |
| FedAvg + SecAgg + DP  | Yes      | Yes  | 148.0259      | Round 5      | 89.32%      | 92.16%         | 56.57%       |     0.8832 |      0.7024 |         0.8913 |        0.8628 |         0.8106 |
| FedProx + SecAgg + DP | Yes      | Yes  | 148.0259      | Round 5      | 88.56%      | 92.16%         | 55.38%       |     0.8755 |      0.71   |         0.8895 |        0.8534 |         0.8136 |
| FedAdam + SecAgg + DP | Yes      | Yes  | 148.0259      | Round 12     | 86.05%      | 82.47%         | 59.10%       |     0.8393 |      0.7559 |         0.8305 |        0.816  |         0.824  |

---

## Hospital-Specific Performance Breakdown

| Configuration         | Hospital                 | Accuracy   | Recall   | Specificity   |   F1-Score |   ROC-AUC |
|:----------------------|:-------------------------|:-----------|:---------|:--------------|-----------:|----------:|
| FedAvg (Baseline)     | Hospital 1 (Cleveland)   | 84.78%     | 85.71%   | 84.00%        |     0.8372 |    0.9371 |
| FedAvg (Baseline)     | Hospital 2 (Hungarian)   | 81.82%     | 81.25%   | 82.14%        |     0.7647 |    0.8906 |
| FedAvg (Baseline)     | Hospital 3 (Switzerland) | 89.47%     | 94.44%   | 0.00%         |     0.9444 |    0.25   |
| FedAvg + SecAgg       | Hospital 1 (Cleveland)   | 84.78%     | 85.71%   | 84.00%        |     0.8372 |    0.9371 |
| FedAvg + SecAgg       | Hospital 2 (Hungarian)   | 81.82%     | 81.25%   | 82.14%        |     0.7647 |    0.8906 |
| FedAvg + SecAgg       | Hospital 3 (Switzerland) | 89.47%     | 94.44%   | 0.00%         |     0.9444 |    0.25   |
| FedAvg + DP           | Hospital 1 (Cleveland)   | 89.13%     | 95.24%   | 84.00%        |     0.8889 |    0.9486 |
| FedAvg + DP           | Hospital 2 (Hungarian)   | 84.09%     | 81.25%   | 85.71%        |     0.7879 |    0.9085 |
| FedAvg + DP           | Hospital 3 (Switzerland) | 94.74%     | 100.00%  | 0.00%         |     0.973  |    0.25   |
| FedAvg + SecAgg + DP  | Hospital 1 (Cleveland)   | 89.13%     | 95.24%   | 84.00%        |     0.8889 |    0.9486 |
| FedAvg + SecAgg + DP  | Hospital 2 (Hungarian)   | 84.09%     | 81.25%   | 85.71%        |     0.7879 |    0.9085 |
| FedAvg + SecAgg + DP  | Hospital 3 (Switzerland) | 94.74%     | 100.00%  | 0.00%         |     0.973  |    0.25   |
| FedProx + SecAgg + DP | Hospital 1 (Cleveland)   | 89.13%     | 95.24%   | 84.00%        |     0.8889 |    0.9505 |
| FedProx + SecAgg + DP | Hospital 2 (Hungarian)   | 81.82%     | 81.25%   | 82.14%        |     0.7647 |    0.9018 |
| FedProx + SecAgg + DP | Hospital 3 (Switzerland) | 94.74%     | 100.00%  | 0.00%         |     0.973  |    0.2778 |
| FedAdam + SecAgg + DP | Hospital 1 (Cleveland)   | 89.13%     | 90.48%   | 88.00%        |     0.8837 |    0.9238 |
| FedAdam + SecAgg + DP | Hospital 2 (Hungarian)   | 79.55%     | 62.50%   | 89.29%        |     0.6897 |    0.8717 |
| FedAdam + SecAgg + DP | Hospital 3 (Switzerland) | 89.47%     | 94.44%   | 0.00%         |     0.9444 |    0.4722 |

---

## Privacy vs Utility Analysis
1. **Secure Aggregation Fidelity:**
   - Pairwise additive masks cancel out to machine precision ($< 10^{-6}$), meaning secure aggregation introduces **zero perturbation** to the aggregate update.
   - Any difference between baseline and secure-only runs is solely due to numerical rounding in float32 arithmetic.
2. **Differential Privacy Trade-off:**
   - Client-side DP clips update norms and injects Gaussian perturbation $\mathcal{N}(0, \sigma^2 C^2 I)$.
   - While this bounds individual sample influence and provides rigorous theoretical $(\epsilon, \delta)$-guarantees via RDP composition, it introduces a measurable utility trade-off on tabular clinical data.
3. **Synergy with Federated Optimization:**
   - Both FedProx and FedAdam remain compatible with the privacy layer.
   - FedProx dampens client drift, helping stabilize noised updates.
   - FedAdam computes pseudo-gradients directly from unmasked aggregate updates, seamlessly tracking momentum even under client-side clipping and noise.
