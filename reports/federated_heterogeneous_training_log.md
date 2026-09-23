# Heterogeneous Federated Learning Communication Log

This document records the communication rounds and validation telemetry for Heterogeneous-Feature Federated Learning.

## Communication Configuration
- **Federated Strategy:** Heterogeneous Sample-Weighted FedAvg
- **Federated Component:** Shared Latent Predictor ($Z=32 \to 1$)
- **Private Components:** Hospital-Specific Feature Encoders ($D_i \to Z=32$)
- **Communication Rounds:** 15
- **Local Epochs:** 3 per round
- **Participating Hospitals:** 3 (Cleveland, Hungarian, Switzerland)

---

## Round-by-Round Validation Telemetry

|   Round |   Total Samples |   Val Loss | Val Accuracy   | Val Recall   | Val Specificity   |   Val F1 |   Val ROC-AUC |   Val PR-AUC | Elapsed   |
|--------:|----------------:|-----------:|:---------------|:-------------|:------------------|---------:|--------------:|-------------:|:----------|
|       1 |             503 |     0.564  | 81.68%         | 64.29%       | 64.29%            |   0.7289 |        0.6939 |       0.8992 | 1.73s     |
|       2 |             503 |     0.3585 | 85.40%         | 80.06%       | 59.13%            |   0.8235 |        0.699  |       0.9069 | 0.29s     |
|       3 |             503 |     0.3302 | 85.39%         | 85.32%       | 55.36%            |   0.8355 |        0.7589 |       0.9176 | 0.27s     |
|       4 |             503 |     0.3318 | 85.42%         | 82.14%       | 57.74%            |   0.8302 |        0.7577 |       0.9172 | 0.27s     |
|       5 |             503 |     0.3395 | 84.65%         | 85.32%       | 53.97%            |   0.8289 |        0.798  |       0.9203 | 0.28s     |
|       6 |             503 |     0.3531 | 83.91%         | 82.14%       | 55.36%            |   0.8155 |        0.815  |       0.9167 | 0.27s     |
|       7 |             503 |     0.3584 | 83.91%         | 83.73%       | 53.97%            |   0.8193 |        0.8136 |       0.9158 | 0.26s     |
|       8 |             503 |     0.362  | 85.44%         | 82.14%       | 57.54%            |   0.8319 |        0.852  |       0.9193 | 0.27s     |
|       9 |             503 |     0.3828 | 85.44%         | 82.14%       | 57.54%            |   0.8319 |        0.8433 |       0.9109 | 0.28s     |
|      10 |             503 |     0.3883 | 84.70%         | 83.73%       | 54.76%            |   0.8299 |        0.8396 |       0.9082 | 0.29s     |
|      11 |             503 |     0.4169 | 84.70%         | 83.73%       | 54.76%            |   0.8299 |        0.8334 |       0.9022 | 0.31s     |
|      12 |             503 |     0.4446 | 82.44%         | 80.06%       | 53.57%            |   0.7985 |        0.8287 |       0.8977 | 0.31s     |
|      13 |             503 |     0.4623 | 83.20%         | 82.14%       | 53.57%            |   0.8119 |        0.8495 |       0.9023 | 0.26s     |
|      14 |             503 |     0.4909 | 82.44%         | 80.06%       | 53.57%            |   0.7985 |        0.8651 |       0.8989 | 0.26s     |
|      15 |             503 |     0.515  | 84.68%         | 81.65%       | 56.15%            |   0.8222 |        0.8701 |       0.9014 | 0.26s     |

---

## Data Locality & Architectural Boundary Audit
- **Private Encoders:** Kept strictly local to individual hospital silos. Zero encoder weight transmission.
- **Transmitted Payloads:** Audited strictly to contain only shared predictor parameters. Zero patient tabular records or raw feature vectors transmitted.
