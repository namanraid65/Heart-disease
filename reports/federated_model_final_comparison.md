# Collaborative Model Final Benchmark Comparison

This report benchmarks the collaborative model architectures across the three medical client silos: the deep federated neural networks (**Federated 1D AlexNet** and **Federated 1D ResNet**, trained via FedAvg with FedBN) and the **Sample-Weighted Local XGBoost Ensemble** baseline.

> [!NOTE]
> **Methodological Clarification**:
> - Federated 1D AlexNet and Federated 1D ResNet utilize true federated learning (FedAvg with client-isolated BatchNorm buffers, FedBN).
> - The XGBoost component is a **Sample-Weighted Local XGBoost Ensemble** combining locally trained client decision tree models weighted by cohort sample size ($P = \sum w_k P_k(x)$), serving as a competitive non-neural tabular baseline.

---

## 1. Hospital-by-Hospital Collaborative Performance Table

| Model   | Strategy                       | Hospital                 | Accuracy   | Precision   | Recall   | Specificity   |   F1 Score |   ROC AUC |   PR AUC |
|:--------|:-------------------------------|:-------------------------|:-----------|:------------|:---------|:--------------|-----------:|----------:|---------:|
| AlexNet | Federated (FedAvg + FedBN)     | Hospital 1 (Cleveland)   | 84.78%     | 79.17%      | 90.48%   | 80.00%        |     0.8444 |    0.9448 |   0.9505 |
| AlexNet | Federated (FedAvg + FedBN)     | Hospital 2 (Hungarian)   | 77.27%     | 65.00%      | 81.25%   | 75.00%        |     0.7222 |    0.8862 |   0.8006 |
| AlexNet | Federated (FedAvg + FedBN)     | Hospital 3 (Switzerland) | 10.53%     | 66.67%      | 11.11%   | 0.00%         |     0.1905 |    0.0556 |   0.8862 |
| ResNet  | Federated (FedAvg + FedBN)     | Hospital 1 (Cleveland)   | 78.26%     | 70.37%      | 90.48%   | 68.00%        |     0.7917 |    0.9029 |   0.8887 |
| ResNet  | Federated (FedAvg + FedBN)     | Hospital 2 (Hungarian)   | 81.82%     | 75.00%      | 75.00%   | 85.71%        |     0.75   |    0.8549 |   0.7048 |
| ResNet  | Federated (FedAvg + FedBN)     | Hospital 3 (Switzerland) | 52.63%     | 90.91%      | 55.56%   | 0.00%         |     0.6897 |    0.2778 |   0.939  |
| XGBoost | Sample-Weighted Local Ensemble | Hospital 1 (Cleveland)   | 86.96%     | 82.61%      | 90.48%   | 84.00%        |     0.8636 |    0.9543 |   0.9524 |
| XGBoost | Sample-Weighted Local Ensemble | Hospital 2 (Hungarian)   | 84.09%     | 80.00%      | 75.00%   | 89.29%        |     0.7742 |    0.9196 |   0.8407 |
| XGBoost | Sample-Weighted Local Ensemble | Hospital 3 (Switzerland) | 57.89%     | 91.67%      | 61.11%   | 0.00%         |     0.7333 |    0.5556 |   0.9707 |

---

## 2. Multi-Center Aggregate Summary Table (Macro vs. Sample-Weighted)

| Model   | Aggregation                | Scope                       | Accuracy   | Precision   | Recall   | Specificity   |   F1 Score |   ROC AUC |   PR AUC |
|:--------|:---------------------------|:----------------------------|:-----------|:------------|:---------|:--------------|-----------:|----------:|---------:|
| AlexNet | Macro Average (Unweighted) | All 3 Medical Centers       | 57.53%     | 70.28%      | 60.95%   | 51.67%        |     0.5857 |    0.6288 |   0.8791 |
| AlexNet | Sample-Weighted Average    | All Medical Centers (N=109) | 68.81%     | 71.27%      | 72.92%   | 64.04%        |     0.6811 |    0.7661 |   0.8788 |
| ResNet  | Macro Average (Unweighted) | All 3 Medical Centers       | 70.90%     | 78.76%      | 73.68%   | 51.24%        |     0.7438 |    0.6785 |   0.8442 |
| ResNet  | Sample-Weighted Average    | All Medical Centers (N=109) | 75.23%     | 75.82%      | 78.14%   | 63.30%        |     0.7571 |    0.7745 |   0.8232 |
| XGBoost | Macro Average (Unweighted) | All 3 Medical Centers       | 76.31%     | 84.76%      | 75.53%   | 57.76%        |     0.7904 |    0.8098 |   0.9213 |
| XGBoost | Sample-Weighted Average    | All Medical Centers (N=109) | 80.73%     | 83.13%      | 79.11%   | 71.49%        |     0.8048 |    0.8708 |   0.9105 |

---

## 3. Global Aggregation Definitions
- **Macro Average (Unweighted)**: Equal arithmetic weighting across the three medical centers ($\frac{1}{3}$ per site). Excludes undefined (`NaN`) values with explicit valid hospital counts.
- **Sample-Weighted Average**: Proportional weighting according to client test set size ($46/109$ Cleveland, $44/109$ Hungarian, $19/109$ Switzerland).
