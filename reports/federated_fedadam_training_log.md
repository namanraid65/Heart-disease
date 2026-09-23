# Heterogeneous HeterogeneousFedAdam Communication Log

This document records the communication rounds and validation telemetry for HeterogeneousFedAdam.

## Communication Configuration
- **Federated Strategy:** HeterogeneousFedAdam
- **Federated Component:** Shared Latent Predictor ($Z=32 \to 1$)
- **Private Components:** Hospital-Specific Feature Encoders ($D_i \to Z=32$)
- **Communication Rounds:** 15
- **Local Epochs:** 3 per round
- **Participating Hospitals:** 3 (Cleveland, Hungarian, Switzerland)

---

## Round-by-Round Validation Telemetry

|   Round |   Total Samples |   Val Loss | Val Accuracy   | Val Recall   | Val Specificity   |   Val F1 |   Val ROC-AUC |   Val PR-AUC | Elapsed   |
|--------:|----------------:|-----------:|:---------------|:-------------|:------------------|---------:|--------------:|-------------:|:----------|
|       1 |             503 |     0.4474 | 82.42%         | 70.04%       | 61.90%            |   0.7611 |        0.6764 |       0.8977 | 0.24s     |
|       2 |             503 |     0.4094 | 86.90%         | 86.90%       | 56.35%            |   0.8529 |        0.7223 |       0.9162 | 0.35s     |
|       3 |             503 |     0.4233 | 85.39%         | 86.90%       | 53.97%            |   0.8382 |        0.7398 |       0.9181 | 0.27s     |
|       4 |             503 |     0.3732 | 83.92%         | 83.73%       | 53.77%            |   0.8202 |        0.753  |       0.9152 | 0.24s     |
|       5 |             503 |     0.3606 | 83.94%         | 82.14%       | 54.96%            |   0.8177 |        0.7912 |       0.9137 | 0.25s     |
|       6 |             503 |     0.405  | 83.94%         | 78.97%       | 57.74%            |   0.808  |        0.8051 |       0.9066 | 0.25s     |
|       7 |             503 |     0.4498 | 82.83%         | 77.01%       | 59.13%            |   0.8045 |        0.8065 |       0.9075 | 0.24s     |
|       8 |             503 |     0.5239 | 83.94%         | 78.47%       | 57.54%            |   0.8076 |        0.8144 |       0.8989 | 0.24s     |
|       9 |             503 |     0.5833 | 83.20%         | 76.88%       | 57.54%            |   0.7963 |        0.8092 |       0.8939 | 0.24s     |
|      10 |             503 |     0.6528 | 84.68%         | 82.14%       | 56.35%            |   0.8238 |        0.8084 |       0.8916 | 0.24s     |
|      11 |             503 |     0.8088 | 84.68%         | 78.97%       | 59.13%            |   0.8146 |        0.8191 |       0.8893 | 0.24s     |
|      12 |             503 |     0.831  | 83.18%         | 76.88%       | 57.74%            |   0.7946 |        0.8227 |       0.887  | 0.23s     |
|      13 |             503 |     0.9363 | 81.70%         | 76.88%       | 54.96%            |   0.7826 |        0.7988 |       0.8728 | 0.25s     |
|      14 |             503 |     1.08   | 84.68%         | 80.56%       | 57.74%            |   0.8194 |        0.6881 |       0.8644 | 0.40s     |
|      15 |             503 |     1.1313 | 81.70%         | 78.47%       | 53.57%            |   0.788  |        0.8061 |       0.8696 | 0.27s     |

---

## Data Locality & Architectural Boundary Audit
- **Private Encoders:** Kept strictly local to individual hospital silos. Zero encoder weight transmission.
- **Transmitted Payloads:** Audited strictly to contain only shared predictor parameters. Zero patient tabular records or raw feature vectors transmitted.
