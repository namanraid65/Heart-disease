# Heterogeneous HeterogeneousFedAdam Communication Log

This document records the communication rounds and validation telemetry for HeterogeneousFedAdam.

## Communication Configuration
- **Federated Strategy:** HeterogeneousFedAdam
- **Experiment:** heterogeneous_fedadam_secure_dp
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
|       1 |             503 |     0.6162 | 73.15%         | 47.39%       | 66.67%            |   0.5985 |        0.5963 |       0.8724 |         21.5642 | 0.24s     |
|       2 |             503 |     0.3647 | 83.89%         | 86.41%       | 51.19%            |   0.8214 |        0.6372 |       0.8899 |         33.7351 | 0.26s     |
|       3 |             503 |     0.415  | 84.65%         | 86.90%       | 52.58%            |   0.8318 |        0.6791 |       0.9027 |         44.5172 | 0.26s     |
|       4 |             503 |     0.5255 | 83.16%         | 83.73%       | 52.58%            |   0.8131 |        0.6763 |       0.8977 |         54.2395 | 0.26s     |
|       5 |             503 |     0.6207 | 84.68%         | 85.32%       | 53.57%            |   0.8314 |        0.668  |       0.89   |         63.9617 | 0.41s     |
|       6 |             503 |     0.817  | 83.94%         | 80.06%       | 56.15%            |   0.8122 |        0.7077 |       0.8711 |         73.0259 | 0.30s     |
|       7 |             503 |     1.0877 | 80.20%         | 80.06%       | 49.60%            |   0.7802 |        0.681  |       0.8651 |         81.3592 | 0.30s     |
|       8 |             503 |     1.226  | 82.44%         | 78.47%       | 54.96%            |   0.7937 |        0.7052 |       0.8772 |         89.6925 | 0.29s     |
|       9 |             503 |     1.546  | 83.20%         | 78.47%       | 56.15%            |   0.8016 |        0.712  |       0.8714 |         98.0259 | 0.26s     |
|      10 |             503 |     1.8304 | 83.92%         | 83.23%       | 53.57%            |   0.8179 |        0.7049 |       0.8512 |        106.359  | 0.28s     |
|      11 |             503 |     2.734  | 81.70%         | 80.56%       | 52.38%            |   0.7939 |        0.7207 |       0.8344 |        114.692  | 0.28s     |
|      12 |             503 |     2.6376 | 81.70%         | 80.06%       | 52.18%            |   0.7929 |        0.7225 |       0.8798 |        123.026  | 0.28s     |
|      13 |             503 |     3.2807 | 81.70%         | 75.30%       | 56.35%            |   0.7766 |        0.698  |       0.8435 |        131.359  | 0.27s     |
|      14 |             503 |     3.6619 | 83.18%         | 76.88%       | 57.74%            |   0.7946 |        0.7186 |       0.843  |        139.692  | 0.27s     |
|      15 |             503 |     3.8392 | 80.96%         | 78.47%       | 52.18%            |   0.7826 |        0.7123 |       0.8358 |        148.026  | 0.27s     |

---

## Data Locality & Architectural Boundary Audit
- **Private Encoders:** Kept strictly local to individual hospital silos. Zero encoder weight transmission.
- **Transmitted Payloads:** Audited strictly to contain only shared predictor parameters. Zero patient tabular records or raw feature vectors transmitted.
