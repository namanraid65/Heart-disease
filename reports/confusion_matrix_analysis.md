# Consolidated Confusion Matrix and Error Analysis

This report analyzes True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) across all 18 evaluated experimental configurations.

---

## 1. Master Confusion Matrix Breakdown

| Hospital                 | Model Type   | Strategy                       |   TP |   FP |   TN |   FN |   Total (N) | Recall   | Specificity   | Statistical Disclosure                                                         |
|:-------------------------|:-------------|:-------------------------------|-----:|-----:|-----:|-----:|------------:|:---------|:--------------|:-------------------------------------------------------------------------------|
| Hospital 1 (Cleveland)   | AlexNet      | Local                          |   18 |    3 |   22 |    3 |          46 | 85.7%    | 88.0%         |                                                                                |
| Hospital 2 (Hungarian)   | AlexNet      | Local                          |   12 |    5 |   23 |    4 |          44 | 75.0%    | 82.1%         |                                                                                |
| Hospital 3 (Switzerland) | AlexNet      | Local                          |   18 |    1 |    0 |    0 |          19 | 100.0%   | 0.0%          | Specificity is based on only 1 negative test sample (18 positive, 1 negative). |
| Hospital 1 (Cleveland)   | AlexNet      | Federated (FedAvg + FedBN)     |   19 |    5 |   20 |    2 |          46 | 90.5%    | 80.0%         |                                                                                |
| Hospital 2 (Hungarian)   | AlexNet      | Federated (FedAvg + FedBN)     |   13 |    7 |   21 |    3 |          44 | 81.2%    | 75.0%         |                                                                                |
| Hospital 3 (Switzerland) | AlexNet      | Federated (FedAvg + FedBN)     |    2 |    1 |    0 |   16 |          19 | 11.1%    | 0.0%          | Specificity is based on only 1 negative test sample (18 positive, 1 negative). |
| Hospital 1 (Cleveland)   | ResNet       | Local                          |   17 |    3 |   22 |    4 |          46 | 81.0%    | 88.0%         |                                                                                |
| Hospital 2 (Hungarian)   | ResNet       | Local                          |    4 |    1 |   27 |   12 |          44 | 25.0%    | 96.4%         |                                                                                |
| Hospital 3 (Switzerland) | ResNet       | Local                          |   18 |    1 |    0 |    0 |          19 | 100.0%   | 0.0%          | Specificity is based on only 1 negative test sample (18 positive, 1 negative). |
| Hospital 1 (Cleveland)   | ResNet       | Federated (FedAvg + FedBN)     |   19 |    8 |   17 |    2 |          46 | 90.5%    | 68.0%         |                                                                                |
| Hospital 2 (Hungarian)   | ResNet       | Federated (FedAvg + FedBN)     |   12 |    4 |   24 |    4 |          44 | 75.0%    | 85.7%         |                                                                                |
| Hospital 3 (Switzerland) | ResNet       | Federated (FedAvg + FedBN)     |   10 |    1 |    0 |    8 |          19 | 55.6%    | 0.0%          | Specificity is based on only 1 negative test sample (18 positive, 1 negative). |
| Hospital 1 (Cleveland)   | XGBoost      | Local                          |   18 |    4 |   21 |    3 |          46 | 85.7%    | 84.0%         |                                                                                |
| Hospital 2 (Hungarian)   | XGBoost      | Local                          |   15 |    7 |   21 |    1 |          44 | 93.8%    | 75.0%         |                                                                                |
| Hospital 3 (Switzerland) | XGBoost      | Local                          |   17 |    1 |    0 |    1 |          19 | 94.4%    | 0.0%          | Specificity is based on only 1 negative test sample (18 positive, 1 negative). |
| Hospital 1 (Cleveland)   | XGBoost      | Sample-Weighted Local Ensemble |   19 |    4 |   21 |    2 |          46 | 90.5%    | 84.0%         |                                                                                |
| Hospital 2 (Hungarian)   | XGBoost      | Sample-Weighted Local Ensemble |   12 |    3 |   25 |    4 |          44 | 75.0%    | 89.3%         |                                                                                |
| Hospital 3 (Switzerland) | XGBoost      | Sample-Weighted Local Ensemble |   11 |    1 |    0 |    7 |          19 | 61.1%    | 0.0%          | Specificity is based on only 1 negative test sample (18 positive, 1 negative). |

---

## 2. Mathematical Integrity Assertions
For all evaluated models and clinical sites:
$$\text{TP} + \text{TN} + \text{FP} + \text{FN} = \text{sample\_count}$$
$$\text{positive\_count} = \text{TP} + \text{FN}$$
$$\text{negative\_count} = \text{TN} + \text{FP}$$
These identities were verified automatically during master result generation with zero discrepancy.
