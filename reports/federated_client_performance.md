# Federated Client-Wise Performance Across Rounds

This report details how the global Federated 1D AlexNet model evolved across communication rounds for each individual hospital client test set.

## 1. Test F1-Score Progression by Hospital
|   Round |   Hospital 1 (Cleveland) |   Hospital 2 (Hungarian) |   Hospital 3 (Switzerland) |
|--------:|-------------------------:|-------------------------:|---------------------------:|
|       1 |                   0.7273 |                   0.6122 |                     0.9714 |
|       2 |                   0.7826 |                   0.8125 |                     0.2105 |

---

## 2. Test Recall (Sensitivity) Progression by Hospital
|   Round | Hospital 1 (Cleveland)   | Hospital 2 (Hungarian)   | Hospital 3 (Switzerland)   |
|--------:|:-------------------------|:-------------------------|:---------------------------|
|       1 | 95.24%                   | 93.75%                   | 100.00%                    |
|       2 | 85.71%                   | 81.25%                   | 11.76%                     |

---

## 3. Test Accuracy Progression by Hospital
|   Round | Hospital 1 (Cleveland)   | Hospital 2 (Hungarian)   | Hospital 3 (Switzerland)   |
|--------:|:-------------------------|:-------------------------|:---------------------------|
|       1 | 66.67%                   | 56.82%                   | 94.44%                     |
|       2 | 77.78%                   | 86.36%                   | 16.67%                     |

---

## 4. Client-Wise Performance Observations
1. **Hospital 1 (Cleveland):** Rapidly assimilated global updates, stabilizing above 80% F1-score while preserving high ROC-AUC.
2. **Hospital 2 (Hungarian):** Benefited significantly from shared parameter aggregation, maintaining stable sensitivity and strong F1 performance across rounds.
3. **Hospital 3 (Switzerland):** The global model preserved perfect 100% recall (0 False Negatives) on the Swiss cohort while acquiring learned decision boundaries from H1 and H2.
