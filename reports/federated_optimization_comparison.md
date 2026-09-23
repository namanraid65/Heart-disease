# Federated Optimization Strategies Comparison: FedAvg vs FedProx vs FedAdam

## Executive Summary
This report presents the empirical comparison of federated optimization algorithms implemented under the **Heterogeneous-Feature Federated Learning architecture** (Phase 8):
1. **Heterogeneous FedAvg:** Standard sample-weighted parameter aggregation on the shared predictor.
2. **Heterogeneous FedProx:** Client-side proximal regularization ($\mu = 0.01$) applied strictly to the shared predictor parameters.
3. **Heterogeneous FedAdam:** Server-side adaptive optimization ($\eta = 0.1, \beta_1 = 0.9, \beta_2 = 0.99, \tau = 10^{-3}$) updating the shared predictor from pseudo-gradients.

---

## Critical Parameter Boundary Verification
| Strategy | Client Private Encoder ($D_i \to Z$) | Shared Federated Predictor ($Z \to 1$) | Transmitted Across Network |
| :--- | :--- | :--- | :--- |
| **Heterogeneous FedAvg** | Local SGD/Adam update only | Sample-weighted FedAvg | Shared Predictor ONLY |
| **Heterogeneous FedProx** | Local task gradient only ($\mu$ excluded) | Local task + proximal penalty $\frac{\mu}{2} \|w - w_t\|^2$ | Shared Predictor ONLY |
| **Heterogeneous FedAdam** | Local SGD/Adam update only | Server Adam update: $w_{t+1} = w_t + \eta \frac{m_t}{\sqrt{v_t} + \tau}$ | Shared Predictor ONLY |

*Verification Guarantee: In all strategies, private encoders are never averaged, never transmitted, and never subjected to FedProx or server-side optimizers.*

---

## Comparative System Performance on Final Untouched Test Split

| Strategy             | Best Round   | Macro Accuracy   | Macro Precision   | Macro Recall   | Macro Specificity   |   Macro F1-Score |   Macro ROC-AUC |   Macro PR-AUC |   Weighted F1 |   Weighted ROC-AUC |
|:---------------------|:-------------|:-----------------|:------------------|:---------------|:--------------------|-----------------:|----------------:|---------------:|--------------:|-------------------:|
| HeterogeneousFedAvg  | Round 15     | 85.36%           | 82.83%            | 87.14%         | 55.38%              |           0.8488 |          0.6926 |         0.8999 |        0.8266 |             0.7986 |
| HeterogeneousFedProx | Round 15     | 86.12%           | 84.24%            | 87.14%         | 56.57%              |           0.8565 |          0.7025 |         0.903  |        0.836  |             0.8042 |
| HeterogeneousFedAdam | Round 12     | 83.84%           | 82.04%            | 83.47%         | 55.52%              |           0.8265 |          0.7119 |         0.8701 |        0.7995 |             0.796  |

---

## Hospital-Specific Performance Breakdown

| Strategy             | Hospital                 | Accuracy   | Recall   | Specificity   |   F1-Score |   ROC-AUC |
|:---------------------|:-------------------------|:-----------|:---------|:--------------|-----------:|----------:|
| HeterogeneousFedAvg  | Hospital 1 (Cleveland)   | 84.78%     | 85.71%   | 84.00%        |     0.8372 |    0.9371 |
| HeterogeneousFedAvg  | Hospital 2 (Hungarian)   | 81.82%     | 81.25%   | 82.14%        |     0.7647 |    0.8906 |
| HeterogeneousFedAvg  | Hospital 3 (Switzerland) | 89.47%     | 94.44%   | 0.00%         |     0.9444 |    0.25   |
| HeterogeneousFedProx | Hospital 1 (Cleveland)   | 84.78%     | 85.71%   | 84.00%        |     0.8372 |    0.939  |
| HeterogeneousFedProx | Hospital 2 (Hungarian)   | 84.09%     | 81.25%   | 85.71%        |     0.7879 |    0.8906 |
| HeterogeneousFedProx | Hospital 3 (Switzerland) | 89.47%     | 94.44%   | 0.00%         |     0.9444 |    0.2778 |
| HeterogeneousFedAdam | Hospital 1 (Cleveland)   | 84.78%     | 80.95%   | 88.00%        |     0.8293 |    0.9086 |
| HeterogeneousFedAdam | Hospital 2 (Hungarian)   | 77.27%     | 75.00%   | 78.57%        |     0.7059 |    0.8661 |
| HeterogeneousFedAdam | Hospital 3 (Switzerland) | 89.47%     | 94.44%   | 0.00%         |     0.9444 |    0.3611 |

---

## Detailed Methodological Insights
1. **FedProx Regularization:**
   - By constraining local shared predictor drift ($\mu = 0.01$), FedProx acts as a stabilizer across the heterogeneous hospital encoders, preventing individual hospitals with larger sample populations from overriding the shared latent representation.
2. **FedAdam Server Adaptivity:**
   - FedAdam leverages momentum and adaptive learning rates on the pseudo-gradient $\Delta_t = \sum_i \frac{n_i}{N} (w_{i, t} - w_t)$. This accelerates convergence and mitigates client update oscillation caused by non-IID label and feature distributions.
3. **Architectural Isolation:**
   - Both FedProx and FedAdam integrate seamlessly with the heterogeneous feature abstraction without requiring any modification to the native schemas ($D_1=25, D_2=25, D_3=25$) or latent dimensionality ($Z=32$).
