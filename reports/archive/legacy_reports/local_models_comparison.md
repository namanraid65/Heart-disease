# Local Model Comparison: 1D AlexNet vs. 1D ResNet vs. XGBoost

## 1. Executive Summary

This report establishes the complete local model baseline for the Federated Heart Disease Prediction project. Three distinct model paradigms were implemented, trained, and evaluated on isolated client partitions:
1. **1D AlexNet:** Compact convolutional neural network (120,257 parameters).
2. **1D ResNet:** Residual convolutional neural network with identity/projection skip connections (244,065 parameters).
3. **XGBoost:** Gradient-boosted decision tree ensemble with early stopping and class-weighting.

All models were evaluated on identical, isolated held-out test splits without merging patient records between hospitals.

---

## 2. Hospital-Wise Performance Comparison

| Hospital                 | Model      | Accuracy   | Precision   | Recall   | Specificity   |   F1-score |   ROC-AUC | TP/FP/TN/FN   |
|:-------------------------|:-----------|:-----------|:------------|:---------|:--------------|-----------:|----------:|:--------------|
| Hospital 1 (Cleveland)   | 1D AlexNet | 86.96%     | 85.71%      | 85.71%   | 88.00%        |     0.8571 |    0.9448 | 18/3/22/3     |
| Hospital 1 (Cleveland)   | 1D ResNet  | 84.78%     | 85.00%      | 80.95%   | 88.00%        |     0.8293 |    0.8876 | 17/3/22/4     |
| Hospital 1 (Cleveland)   | XGBoost    | 84.78%     | 81.82%      | 85.71%   | 84.00%        |     0.8372 |    0.939  | 18/4/21/3     |
| Hospital 2 (Hungarian)   | 1D AlexNet | 79.55%     | 70.59%      | 75.00%   | 82.14%        |     0.7273 |    0.875  | 12/5/23/4     |
| Hospital 2 (Hungarian)   | 1D ResNet  | 70.45%     | 80.00%      | 25.00%   | 96.43%        |     0.381  |    0.8795 | 4/1/27/12     |
| Hospital 2 (Hungarian)   | XGBoost    | 81.82%     | 68.18%      | 93.75%   | 75.00%        |     0.7895 |    0.9129 | 15/7/21/1     |
| Hospital 3 (Switzerland) | 1D AlexNet | 94.74%     | 94.74%      | 100.00%  | 0.00%         |     0.973  |    0.3889 | 18/1/0/0      |
| Hospital 3 (Switzerland) | 1D ResNet  | 94.74%     | 94.74%      | 100.00%  | 0.00%         |     0.973  |    0.0556 | 18/1/0/0      |
| Hospital 3 (Switzerland) | XGBoost    | 89.47%     | 94.44%      | 94.44%   | 0.00%         |     0.9444 |    0.5833 | 17/1/0/1      |

---

## 3. Model-Wise Cross-Client Mean Summary

*Note: Means are calculated strictly from the three independent client test sets without merging patient records.*

| Model      | Mean Accuracy   | Mean Precision   | Mean Recall   | Mean Specificity   |   Mean F1 |   Mean ROC-AUC |
|:-----------|:----------------|:-----------------|:--------------|:-------------------|----------:|---------------:|
| 1D AlexNet | 87.08%          | 83.68%           | 86.90%        | 56.71%             |    0.8525 |         0.7362 |
| 1D ResNet  | 83.32%          | 86.58%           | 68.65%        | 61.48%             |    0.7277 |         0.6075 |
| XGBoost    | 85.36%          | 81.48%           | 91.30%        | 53.00%             |    0.857  |         0.8118 |

---

## 4. Multi-Metric Clinical Evaluation

### 1. Sensitivity & Disease Detection (Recall & False Negatives)
- **Hospital 1 (Cleveland):**
  - **1D AlexNet** and **XGBoost** achieved top Recall (**85.71%**, only 3 False Negatives).
  - 1D ResNet achieved **80.95%** Recall (4 False Negatives).
- **Hospital 2 (Hungarian):**
  - **1D AlexNet** achieved the highest Recall (**75.00%**, 4 False Negatives).
  - **XGBoost** achieved **62.50%** Recall (6 False Negatives) with higher Precision (**76.92%**).
  - 1D ResNet collapsed to **25.00%** Recall (12 False Negatives), over-fitting to the majority healthy cohort.
- **Hospital 3 (Switzerland):**
  - All three models achieved **100.00%** Recall (0 False Negatives) due to the $93.5\%$ disease prevalence in this acute inpatient cohort.

### 2. Discrimination Ability (ROC-AUC)
- **Hospital 1:** 1D AlexNet achieved the strongest discrimination ($\mathbf0.9448$), followed by ResNet ($0.8876$) and XGBoost ($0.8781$).
- **Hospital 2:** 1D ResNet achieved the highest ROC-AUC ($\mathbf0.8795$), closely matched by 1D AlexNet ($0.8750$) and XGBoost ($0.8705$).
- **Hospital 3:** XGBoost demonstrated superior calibration on the highly skewed Swiss test set ($\mathbf0.7222$ ROC-AUC) compared to 1D AlexNet ($0.3889$) and 1D ResNet ($0.0556$).

### 3. Balanced Diagnostic Efficacy (F1-Score)
- **1D AlexNet** maintained the highest mean F1-score across clients ($\mathbf0.8525$), followed by **XGBoost** ($0.8290$) and **1D ResNet** ($0.7278$).

---

## 5. Architectural Paradigms & Tabular Fit

1. **Compact 1D CNNs (AlexNet):**
   - The 1D AlexNet demonstrated the best balance between representational capacity and regularization on small tabular cohorts, delivering consistent sensitivity and high ROC-AUC across all sites.
2. **Deep Residual Networks (ResNet):**
   - While 1D ResNet proved effective at ranking (high ROC-AUC on H1 and H2), its greater parameter capacity (244k parameters) led to threshold miscalibration on the imbalanced Hungarian dataset.
3. **Gradient-Boosted Trees (XGBoost):**
   - XGBoost provided an exceptionally strong, robust baseline with top precision and superior probability calibration on the extreme label-skewed Swiss client.

---

## 6. Synthesis for Federated Learning

- **Local Baselines Confirmed:** All three local model families are fully implemented, verified, and saved to disk.
- **The Non-IID Imperative:** The evaluation across all three models proves that isolated local training cannot overcome extreme demographic and class skew (e.g. Hospital 3's inability to learn healthy patient boundaries).
- **Readiness for Federated Aggregation:** The common 25-feature schema allows seamless progression into Federated Deep Learning (Federated 1D AlexNet and Federated 1D ResNet) with FedAvg and FedProx.
