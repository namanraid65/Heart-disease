# Heterogeneous HeterogeneousFedAvg Communication Log

This document records the communication rounds and validation telemetry for HeterogeneousFedAvg.

## Communication Configuration
- **Federated Strategy:** HeterogeneousFedAvg
- **Experiment:** heterogeneous_fedavg_secure_dp
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
|       1 |             503 |     0.6719 | 49.73%         | 12.98%       | 66.67%            |   0.2029 |        0.4861 |       0.7348 |         21.5642 | 0.27s     |
|       2 |             503 |     0.4328 | 85.40%         | 86.41%       | 53.57%            |   0.8357 |        0.5994 |       0.8739 |         33.7351 | 0.28s     |
|       3 |             503 |     0.5321 | 83.91%         | 86.90%       | 51.19%            |   0.8258 |        0.6233 |       0.8786 |         44.5172 | 0.36s     |
|       4 |             503 |     0.6053 | 82.46%         | 82.14%       | 52.18%            |   0.8064 |        0.6631 |       0.8773 |         54.2395 | 0.42s     |
|       5 |             503 |     0.776  | 82.44%         | 80.06%       | 53.57%            |   0.7985 |        0.7266 |       0.8873 |         63.9617 | 0.33s     |
|       6 |             503 |     0.952  | 83.22%         | 78.97%       | 56.15%            |   0.8041 |        0.6634 |       0.8758 |         73.0259 | 0.28s     |
|       7 |             503 |     1.3413 | 79.44%         | 80.56%       | 48.61%            |   0.7746 |        0.7029 |       0.8667 |         81.3592 | 0.42s     |
|       8 |             503 |     1.4521 | 80.94%         | 75.89%       | 53.37%            |   0.7707 |        0.7254 |       0.8661 |         89.6925 | 0.35s     |
|       9 |             503 |     1.7509 | 80.22%         | 78.97%       | 50.99%            |   0.7776 |        0.6874 |       0.8441 |         98.0259 | 0.37s     |
|      10 |             503 |     2.0474 | 79.44%         | 80.56%       | 48.61%            |   0.7746 |        0.6941 |       0.8528 |        106.359  | 0.28s     |
|      11 |             503 |     2.7689 | 82.44%         | 82.64%       | 52.58%            |   0.8048 |        0.7116 |       0.8703 |        114.692  | 0.31s     |
|      12 |             503 |     2.549  | 79.44%         | 80.06%       | 48.41%            |   0.7733 |        0.6883 |       0.8421 |        123.026  | 0.28s     |
|      13 |             503 |     2.9429 | 80.20%         | 76.88%       | 52.38%            |   0.7696 |        0.679  |       0.8443 |        131.359  | 0.29s     |
|      14 |             503 |     2.8609 | 80.96%         | 82.14%       | 49.60%            |   0.7935 |        0.705  |       0.8779 |        139.692  | 0.34s     |
|      15 |             503 |     2.991  | 79.46%         | 76.88%       | 50.99%            |   0.7644 |        0.7017 |       0.8539 |        148.026  | 0.29s     |

---

## Data Locality & Architectural Boundary Audit
- **Private Encoders:** Kept strictly local to individual hospital silos. Zero encoder weight transmission.
- **Transmitted Payloads:** Audited strictly to contain only shared predictor parameters. Zero patient tabular records or raw feature vectors transmitted.
