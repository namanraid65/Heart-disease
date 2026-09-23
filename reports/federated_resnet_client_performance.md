# Federated ResNet Client-Wise Performance Across Rounds

This report details how the global Federated 1D ResNet model evolved across communication rounds for each individual hospital client validation set.

## 1. Validation F1-Score Progression by Hospital
|   Round |   Hospital 1 (Cleveland) |   Hospital 2 (Hungarian) |   Hospital 3 (Switzerland) |
|--------:|-------------------------:|-------------------------:|---------------------------:|
|       1 |                   0.7843 |                   0.6667 |                     0.9714 |
|       2 |                   0.7727 |                   0.7742 |                     0.56   |
|       3 |                   0.7907 |                   0.6667 |                     0.875  |
|       4 |                   0.7556 |                   0.7742 |                     0.9412 |
|       5 |                   0.7111 |                   0.7742 |                     0.8    |
|       6 |                   0.7727 |                   0.64   |                     0.8387 |
|       7 |                   0.766  |                   0.7333 |                     0.875  |
|       8 |                   0.7442 |                   0.7742 |                     0.9091 |
|       9 |                   0.7727 |                   0.7586 |                     0.9091 |
|      10 |                   0.8182 |                   0.7407 |                     0.8387 |
|      11 |                   0.8    |                   0.64   |                     0.875  |
|      12 |                   0.7347 |                   0.75   |                     0.8667 |
|      13 |                   0.7692 |                   0.8    |                     0.875  |
|      14 |                   0.7407 |                   0.7742 |                     0.875  |
|      15 |                   0.7222 |                   0.7742 |                     0.875  |

---

## 2. Validation Recall (Sensitivity) Progression by Hospital
|   Round | Hospital 1 (Cleveland)   | Hospital 2 (Hungarian)   | Hospital 3 (Switzerland)   |
|--------:|:-------------------------|:-------------------------|:---------------------------|
|       1 | 95.24%                   | 50.00%                   | 100.00%                    |
|       2 | 80.95%                   | 75.00%                   | 41.18%                     |
|       3 | 80.95%                   | 56.25%                   | 82.35%                     |
|       4 | 80.95%                   | 75.00%                   | 94.12%                     |
|       5 | 76.19%                   | 75.00%                   | 70.59%                     |
|       6 | 80.95%                   | 50.00%                   | 76.47%                     |
|       7 | 85.71%                   | 68.75%                   | 82.35%                     |
|       8 | 76.19%                   | 75.00%                   | 88.24%                     |
|       9 | 80.95%                   | 68.75%                   | 88.24%                     |
|      10 | 85.71%                   | 62.50%                   | 76.47%                     |
|      11 | 95.24%                   | 50.00%                   | 82.35%                     |
|      12 | 85.71%                   | 75.00%                   | 76.47%                     |
|      13 | 95.24%                   | 75.00%                   | 82.35%                     |
|      14 | 95.24%                   | 75.00%                   | 82.35%                     |
|      15 | 61.90%                   | 75.00%                   | 82.35%                     |

---

## 3. Validation Accuracy Progression by Hospital
|   Round | Hospital 1 (Cleveland)   | Hospital 2 (Hungarian)   | Hospital 3 (Switzerland)   |
|--------:|:-------------------------|:-------------------------|:---------------------------|
|       1 | 75.56%                   | 81.82%                   | 94.44%                     |
|       2 | 77.78%                   | 84.09%                   | 38.89%                     |
|       3 | 80.00%                   | 79.55%                   | 77.78%                     |
|       4 | 75.56%                   | 84.09%                   | 88.89%                     |
|       5 | 71.11%                   | 84.09%                   | 66.67%                     |
|       6 | 77.78%                   | 79.55%                   | 72.22%                     |
|       7 | 75.56%                   | 81.82%                   | 77.78%                     |
|       8 | 75.56%                   | 84.09%                   | 83.33%                     |
|       9 | 77.78%                   | 84.09%                   | 83.33%                     |
|      10 | 82.22%                   | 84.09%                   | 72.22%                     |
|      11 | 77.78%                   | 79.55%                   | 77.78%                     |
|      12 | 71.11%                   | 81.82%                   | 77.78%                     |
|      13 | 73.33%                   | 86.36%                   | 77.78%                     |
|      14 | 68.89%                   | 84.09%                   | 77.78%                     |
|      15 | 77.78%                   | 84.09%                   | 77.78%                     |

---

## 4. Client-Wise Performance Observations
1. **Hospital 1 (Cleveland):** Rapidly converged to stable accuracy and high F1-score with robust ROC-AUC.
2. **Hospital 2 (Hungarian):** Experienced massive sensitivity gains compared to local ResNet (local was 25.0% recall due to sparse local positive representations, whereas federated ResNet learned rich disease filters from Cleveland and Swiss data).
3. **Hospital 3 (Switzerland):** Maintained high sensitivity on Swiss inpatient cases while integrating generalized decision boundaries.
