# Model Selection and Clinical Risk Ranking Analysis

This document evaluates the trade-offs among the evaluated models for cardiological risk prediction, emphasizing sensitivity (Recall), minimization of False Negatives, and discrimination (ROC-AUC and PR-AUC).

> [!IMPORTANT]
> In clinical risk screening, False Negatives (undiagnosed coronary disease) present substantial medical risk. Model selection must prioritize Recall and ROC-AUC over raw Accuracy alone.

---

## 1. Multi-Metric Model Comparison Matrix (Sample-Weighted Across All Sites, $N=109$)

| Model   | Paradigm                       | Total False Negatives   | Sample-Weighted Recall   |   Sample-Weighted ROC-AUC |   Sample-Weighted PR-AUC |   Sample-Weighted F1 | Sample-Weighted Accuracy   |
|:--------|:-------------------------------|:------------------------|:-------------------------|--------------------------:|-------------------------:|---------------------:|:---------------------------|
| AlexNet | Federated (FedAvg + FedBN)     | 21 / 55                 | 72.92%                   |                    0.7661 |                   0.8788 |               0.6811 | 68.81%                     |
| ResNet  | Federated (FedAvg + FedBN)     | 14 / 55                 | 78.14%                   |                    0.7745 |                   0.8232 |               0.7571 | 75.23%                     |
| XGBoost | Sample-Weighted Local Ensemble | 13 / 55                 | 79.11%                   |                    0.8708 |                   0.9105 |               0.8048 | 80.73%                     |
| AlexNet | Local Hospital Baseline        | 7 / 55                  | 83.88%                   |                    0.8197 |                   0.8833 |               0.8249 | 85.32%                     |
| ResNet  | Local Hospital Baseline        | 16 / 55                 | 61.69%                   |                    0.7393 |                   0.8362 |               0.6733 | 80.73%                     |
| XGBoost | Local Hospital Baseline        | 5 / 55                  | 90.48%                   |                    0.8665 |                   0.8997 |               0.8366 | 84.40%                     |

---

## 2. Selection Rationale Under Experimental Conditions
- **Neural Collaborative Models (FedBN)**: Federated 1D AlexNet and Federated 1D ResNet provide strong representation learning without sharing patient raw rows, preserving institutional privacy while achieving competitive discrimination.
- **Tabular Ensemble Baseline**: The Sample-Weighted Local XGBoost Ensemble achieves high overall precision and low false negative rates on structured continuous thresholds.
