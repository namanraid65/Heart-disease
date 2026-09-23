# Hospital-Wise Model Performance Comparison

This report details the comparative performance of all six model configurations (Local and Federated 1D AlexNet, 1D ResNet, and XGBoost) evaluated independently on each hospital's held-out test split.

> [!NOTE]
> The explanations and predictions presented in this study represent statistical machine learning model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. This system is designed exclusively for research and educational purposes and is not a substitute for professional clinical diagnosis and medical judgment.

---

## Hospital 1 (Cleveland)
**Cohort Test Distribution:** Total $N=46$ patients ($P=21$ diseased, $N_{\text{neg}}=25$ healthy).
| Model Type   | Strategy                       | Accuracy   | Precision   | Recall   | Sensitivity   | Specificity   |   F1 Score |   ROC AUC |   PR AUC |   TP |   TN |   FP |   FN |
|:-------------|:-------------------------------|:-----------|:------------|:---------|:--------------|:--------------|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|
| AlexNet      | Local                          | 86.96%     | 85.71%      | 85.71%   | 85.71%        | 88.00%        |     0.8571 |    0.9448 |   0.9416 |   18 |   22 |    3 |    3 |
| AlexNet      | Federated (FedAvg + FedBN)     | 84.78%     | 79.17%      | 90.48%   | 90.48%        | 80.00%        |     0.8444 |    0.9448 |   0.9505 |   19 |   20 |    5 |    2 |
| ResNet       | Local                          | 84.78%     | 85.00%      | 80.95%   | 80.95%        | 88.00%        |     0.8293 |    0.8876 |   0.8762 |   17 |   22 |    3 |    4 |
| ResNet       | Federated (FedAvg + FedBN)     | 78.26%     | 70.37%      | 90.48%   | 90.48%        | 68.00%        |     0.7917 |    0.9029 |   0.8887 |   19 |   17 |    8 |    2 |
| XGBoost      | Local                          | 84.78%     | 81.82%      | 85.71%   | 85.71%        | 84.00%        |     0.8372 |    0.939  |   0.9353 |   18 |   21 |    4 |    3 |
| XGBoost      | Sample-Weighted Local Ensemble | 86.96%     | 82.61%      | 90.48%   | 90.48%        | 84.00%        |     0.8636 |    0.9543 |   0.9524 |   19 |   21 |    4 |    2 |

---
## Hospital 2 (Hungarian)
**Cohort Test Distribution:** Total $N=44$ patients ($P=16$ diseased, $N_{\text{neg}}=28$ healthy).
| Model Type   | Strategy                       | Accuracy   | Precision   | Recall   | Sensitivity   | Specificity   |   F1 Score |   ROC AUC |   PR AUC |   TP |   TN |   FP |   FN |
|:-------------|:-------------------------------|:-----------|:------------|:---------|:--------------|:--------------|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|
| AlexNet      | Local                          | 79.55%     | 70.59%      | 75.00%   | 75.00%        | 82.14%        |     0.7273 |    0.875  |   0.792  |   12 |   23 |    5 |    4 |
| AlexNet      | Federated (FedAvg + FedBN)     | 77.27%     | 65.00%      | 81.25%   | 81.25%        | 75.00%        |     0.7222 |    0.8862 |   0.8006 |   13 |   21 |    7 |    3 |
| ResNet       | Local                          | 70.45%     | 80.00%      | 25.00%   | 25.00%        | 96.43%        |     0.381  |    0.8795 |   0.7729 |    4 |   27 |    1 |   12 |
| ResNet       | Federated (FedAvg + FedBN)     | 81.82%     | 75.00%      | 75.00%   | 75.00%        | 85.71%        |     0.75   |    0.8549 |   0.7048 |   12 |   24 |    4 |    4 |
| XGBoost      | Local                          | 81.82%     | 68.18%      | 93.75%   | 93.75%        | 75.00%        |     0.7895 |    0.9129 |   0.8318 |   15 |   21 |    7 |    1 |
| XGBoost      | Sample-Weighted Local Ensemble | 84.09%     | 80.00%      | 75.00%   | 75.00%        | 89.29%        |     0.7742 |    0.9196 |   0.8407 |   12 |   25 |    3 |    4 |

---
## Hospital 3 (Switzerland)
**Cohort Test Distribution:** Total $N=19$ patients ($P=18$ diseased, $N_{\text{neg}}=1$ healthy).
> [!IMPORTANT]
> **Hospital 3 Statistical Limitation Disclosure**:
> The Hospital 3 test partition comprises $18$ positive cases and exactly $1$ negative case ($N=19$, prevalence $94.7\%$). > Consequently, Specificity is evaluated on only one negative test instance: a single false positive prediction yields $0.00\%$ Specificity, > while predicting healthy yields $100.0\%$. ROC-AUC is mathematically defined between the single negative and 18 positive instances, but > reflects rank placement against that single control. All metrics must be interpreted in the context of this extreme prevalence asymmetry.

| Model Type   | Strategy                       | Accuracy   | Precision   | Recall   | Sensitivity   | Specificity   |   F1 Score |   ROC AUC |   PR AUC |   TP |   TN |   FP |   FN |
|:-------------|:-------------------------------|:-----------|:------------|:---------|:--------------|:--------------|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|
| AlexNet      | Local                          | 94.74%     | 94.74%      | 100.00%  | 100.00%       | 0.00%         |     0.973  |    0.3889 |   0.9539 |   18 |    0 |    1 |    0 |
| AlexNet      | Federated (FedAvg + FedBN)     | 10.53%     | 66.67%      | 11.11%   | 11.11%        | 0.00%         |     0.1905 |    0.0556 |   0.8862 |    2 |    0 |    1 |   16 |
| ResNet       | Local                          | 94.74%     | 94.74%      | 100.00%  | 100.00%       | 0.00%         |     0.973  |    0.0556 |   0.8862 |   18 |    0 |    1 |    0 |
| ResNet       | Federated (FedAvg + FedBN)     | 52.63%     | 90.91%      | 55.56%   | 55.56%        | 0.00%         |     0.6897 |    0.2778 |   0.939  |   10 |    0 |    1 |    8 |
| XGBoost      | Local                          | 89.47%     | 94.44%      | 94.44%   | 94.44%        | 0.00%         |     0.9444 |    0.5833 |   0.9707 |   17 |    0 |    1 |    1 |
| XGBoost      | Sample-Weighted Local Ensemble | 57.89%     | 91.67%      | 61.11%   | 61.11%        | 0.00%         |     0.7333 |    0.5556 |   0.9707 |   11 |    0 |    1 |    7 |

---

