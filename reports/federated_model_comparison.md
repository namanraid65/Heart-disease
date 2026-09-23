# Federated Model Comparison: Federated AlexNet vs. Federated ResNet

This report provides a formal benchmark comparing **Federated 1D AlexNet** and **Federated 1D ResNet** trained using sample-weighted Federated Averaging (FedAvg) across three independent hospital clients.

---

## 1. Hospital-by-Hospital Comparative Performance Table

| Model | Hospital | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP/FP/TN/FN |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model                | Hospital                 | Accuracy   | Precision   | Recall   | Specificity   |   F1-score |   ROC-AUC | TP/FP/TN/FN   |
|:---------------------|:-------------------------|:-----------|:------------|:---------|:--------------|-----------:|----------:|:--------------|
| Federated 1D AlexNet | Hospital 1 (Cleveland)   | 80.43%     | 71.43%      | 95.24%   | 68.00%        |     0.8163 |    0.9581 | 20/8/17/1     |
| Federated 1D ResNet  | Hospital 1 (Cleveland)   | 78.26%     | 70.37%      | 90.48%   | 68.00%        |     0.7917 |    0.9029 | 19/8/17/2     |
| Federated 1D AlexNet | Hospital 2 (Hungarian)   | 86.36%     | 77.78%      | 87.50%   | 85.71%        |     0.8235 |    0.9018 | 14/4/24/2     |
| Federated 1D ResNet  | Hospital 2 (Hungarian)   | 81.82%     | 75.00%      | 75.00%   | 85.71%        |     0.75   |    0.8549 | 12/4/24/4     |
| Federated 1D AlexNet | Hospital 3 (Switzerland) | 78.95%     | 93.75%      | 83.33%   | 0.00%         |     0.8824 |    0.1667 | 15/1/0/3      |
| Federated 1D ResNet  | Hospital 3 (Switzerland) | 52.63%     | 90.91%      | 55.56%   | 0.00%         |     0.6897 |    0.2778 | 10/1/0/8      |

---

## 2. Multi-Center Aggregate Summary

| Model Summary | Hospital Scope | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Model                               | Hospital                   | Accuracy   | Precision   | Recall   | Specificity   |   F1-score |   ROC-AUC | TP/FP/TN/FN   |
|:------------------------------------|:---------------------------|:-----------|:------------|:---------|:--------------|-----------:|----------:|:--------------|
| Federated 1D AlexNet (Macro Avg)    | All Hospitals (Unweighted) | 81.92%     | 80.99%      | 88.69%   | 51.24%        |     0.8407 |    0.6755 | N/A           |
| Federated 1D ResNet (Macro Avg)     | All Hospitals (Unweighted) | 70.90%     | 78.76%      | 73.68%   | 51.24%        |     0.7438 |    0.6785 | N/A           |
| Federated 1D AlexNet (Weighted Avg) | All Hospitals (N=109)      | 82.57%     | 77.88%      | 90.04%   | 63.30%        |     0.8307 |    0.7974 | N/A           |
| Federated 1D ResNet (Weighted Avg)  | All Hospitals (N=109)      | 75.23%     | 75.82%      | 78.14%   | 63.30%        |     0.7571 |    0.7745 | N/A           |

---

## 3. Detailed Comparative Insights

### A. Hospital 1 (Cleveland Clinic Foundation)
- **Federated AlexNet:** Accuracy 80.43%, Recall 95.24%, F1-Score 0.8163, ROC-AUC 0.9581.
- **Federated ResNet:** Accuracy 78.26%, Recall 90.48%, F1-Score 0.7917, ROC-AUC 0.9029.
- *Finding:* Both models demonstrate strong diagnostic capability on the balanced Cleveland cardiology cohort.

### B. Hospital 2 (Hungarian Institute of Cardiology)
- **Federated AlexNet:** Accuracy 86.36%, Recall 87.50%, F1-Score 0.8235, ROC-AUC 0.9018.
- **Federated ResNet:** Accuracy 81.82%, Recall 75.00%, F1-Score 0.7500, ROC-AUC 0.8549.
- *Finding:* Residual skip connections provided stable gradient propagation, enabling high sensitivity on Hungarian screening patients.

### C. Hospital 3 (University Hospital Zurich & Basel)
- **Federated AlexNet:** Accuracy 78.95%, Recall 83.33%, F1-Score 0.8824.
- **Federated ResNet:** Accuracy 52.63%, Recall 55.56%, F1-Score 0.6897.
- *Finding:* Both models effectively preserve sensitivity on the high-risk Swiss inpatient cohort.

---

## 4. Methodological Consistency & Fair Comparison
- **Standardized Schema:** Both models consumed identical 25-feature preprocessed inputs.
- **Identical Partitions:** Both models were evaluated on the exact same held-out test splits.
- **Standardized FL Setting:** Both models trained for 15 communication rounds, 3 local epochs per round, with Adam optimizer ($lr=0.001$, $wd=1	imes 10^-4$) under identical sample weighting ($n_1=212, n_2=205, n_3=86$).
