# Heterogeneous HeterogeneousFedProx Communication Log

This document records the communication rounds and validation telemetry for HeterogeneousFedProx.

## Communication Configuration
- **Federated Strategy:** HeterogeneousFedProx
- **Experiment:** heterogeneous_fedprox_secure_dp
- **Federated Component:** Shared Latent Predictor ($Z=32 \to 1$)
- **Private Components:** Hospital-Specific Feature Encoders ($D_i \to Z=32$)
- **Communication Rounds:** 15
- **Local Epochs:** 3 per round
- **Participating Hospitals:** 3 (Cleveland, Hungarian, Switzerland)
- **Secure Aggregation (Simulated):** True
- **Differential Privacy:** True

---

## Round-by-Round Validation Telemetry

|   Round |   Total Samples |   Val Loss | Val Accuracy   | Val Recall   | Val Specificity   |   Val F1 |   Val ROC-AUC |   Val PR-AUC |   Epsilon (eps) | Elapsed   |
|--------:|----------------:|-----------:|:---------------|:-------------|:------------------|---------:|--------------:|-------------:|----------------:|:----------|
|       1 |             503 |     0.6719 | 49.73%         | 12.98%       | 66.67%            |   0.2029 |        0.4864 |       0.7336 |         21.5642 | 0.60s     |
|       2 |             503 |     0.4316 | 85.40%         | 86.41%       | 53.57%            |   0.8357 |        0.5994 |       0.8739 |         33.7351 | 0.51s     |
|       3 |             503 |     0.5322 | 83.91%         | 86.90%       | 51.19%            |   0.8258 |        0.6459 |       0.8863 |         44.5172 | 0.71s     |
|       4 |             503 |     0.6085 | 83.18%         | 82.14%       | 53.77%            |   0.8101 |        0.6615 |       0.875  |         54.2395 | 0.55s     |
|       5 |             503 |     0.7656 | 82.44%         | 80.06%       | 53.57%            |   0.7985 |        0.7467 |       0.8906 |         63.9617 | 0.54s     |
|       6 |             503 |     0.9825 | 83.22%         | 78.97%       | 56.15%            |   0.8041 |        0.6435 |       0.8691 |         73.0259 | 0.56s     |
|       7 |             503 |     1.2894 | 80.96%         | 80.56%       | 50.99%            |   0.7885 |        0.7056 |       0.8751 |         81.3592 | 0.45s     |
|       8 |             503 |     1.35   | 81.70%         | 77.98%       | 53.37%            |   0.7863 |        0.7271 |       0.8735 |         89.6925 | 0.35s     |
|       9 |             503 |     1.5439 | 80.96%         | 78.47%       | 52.18%            |   0.7826 |        0.7031 |       0.8655 |         98.0259 | 0.35s     |
|      10 |             503 |     1.8409 | 78.72%         | 76.88%       | 49.60%            |   0.7593 |        0.7067 |       0.8632 |        106.359  | 0.40s     |
|      11 |             503 |     2.6931 | 80.20%         | 80.56%       | 49.80%            |   0.7813 |        0.7194 |       0.8825 |        114.692  | 0.32s     |
|      12 |             503 |     2.4647 | 78.70%         | 75.89%       | 49.40%            |   0.7529 |        0.6841 |       0.855  |        123.026  | 0.36s     |
|      13 |             503 |     2.8104 | 79.46%         | 76.88%       | 50.99%            |   0.7644 |        0.6827 |       0.8478 |        131.359  | 0.35s     |
|      14 |             503 |     3.2286 | 80.98%         | 80.56%       | 50.79%            |   0.7909 |        0.7021 |       0.8751 |        139.692  | 0.29s     |
|      15 |             503 |     3.0682 | 80.98%         | 80.56%       | 50.79%            |   0.7909 |        0.7092 |       0.8774 |        148.026  | 0.36s     |

---

## Data Locality & Architectural Boundary Audit
- **Private Encoders:** Kept strictly local to individual hospital silos. Zero encoder weight transmission.
- **Transmitted Payloads:** Audited strictly to contain only shared predictor parameters. Zero patient tabular records or raw feature vectors transmitted.
