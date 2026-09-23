# Heterogeneous HeterogeneousFedProx Communication Log

This document records the communication rounds and validation telemetry for HeterogeneousFedProx.

## Communication Configuration
- **Federated Strategy:** HeterogeneousFedProx
- **Federated Component:** Shared Latent Predictor ($Z=32 \to 1$)
- **Private Components:** Hospital-Specific Feature Encoders ($D_i \to Z=32$)
- **Communication Rounds:** 15
- **Local Epochs:** 3 per round
- **Participating Hospitals:** 3 (Cleveland, Hungarian, Switzerland)

---

## Round-by-Round Validation Telemetry

|   Round |   Total Samples |   Val Loss | Val Accuracy   | Val Recall   | Val Specificity   |   Val F1 |   Val ROC-AUC |   Val PR-AUC | Elapsed   |
|--------:|----------------:|-----------:|:---------------|:-------------|:------------------|---------:|--------------:|-------------:|:----------|
|       1 |             503 |     0.5642 | 81.68%         | 64.29%       | 64.29%            |   0.7289 |        0.6939 |       0.8992 | 0.34s     |
|       2 |             503 |     0.3588 | 85.40%         | 80.06%       | 59.13%            |   0.8235 |        0.699  |       0.9069 | 0.39s     |
|       3 |             503 |     0.3304 | 85.39%         | 85.32%       | 55.36%            |   0.8355 |        0.7589 |       0.9176 | 0.34s     |
|       4 |             503 |     0.3328 | 85.42%         | 82.14%       | 57.74%            |   0.8302 |        0.7577 |       0.9172 | 0.35s     |
|       5 |             503 |     0.3412 | 85.40%         | 85.32%       | 55.16%            |   0.8361 |        0.798  |       0.9203 | 0.35s     |
|       6 |             503 |     0.3566 | 83.91%         | 82.14%       | 55.36%            |   0.8155 |        0.8136 |       0.915  | 0.33s     |
|       7 |             503 |     0.362  | 83.91%         | 83.73%       | 53.97%            |   0.8193 |        0.8108 |       0.9132 | 0.38s     |
|       8 |             503 |     0.3678 | 85.44%         | 82.14%       | 57.54%            |   0.8319 |        0.8479 |       0.9137 | 0.33s     |
|       9 |             503 |     0.3885 | 84.70%         | 82.14%       | 56.15%            |   0.8258 |        0.8411 |       0.9092 | 0.40s     |
|      10 |             503 |     0.392  | 84.70%         | 83.73%       | 54.76%            |   0.8299 |        0.8382 |       0.9072 | 0.36s     |
|      11 |             503 |     0.4207 | 83.96%         | 82.14%       | 54.76%            |   0.82   |        0.83   |       0.9006 | 0.47s     |
|      12 |             503 |     0.4504 | 82.44%         | 80.06%       | 53.57%            |   0.7985 |        0.8259 |       0.8945 | 0.41s     |
|      13 |             503 |     0.4659 | 83.20%         | 82.14%       | 53.57%            |   0.8119 |        0.8487 |       0.9019 | 0.36s     |
|      14 |             503 |     0.4971 | 82.44%         | 80.06%       | 53.57%            |   0.7985 |        0.8434 |       0.8958 | 0.38s     |
|      15 |             503 |     0.5227 | 84.68%         | 81.65%       | 56.15%            |   0.8222 |        0.8658 |       0.8986 | 0.37s     |

---

## Data Locality & Architectural Boundary Audit
- **Private Encoders:** Kept strictly local to individual hospital silos. Zero encoder weight transmission.
- **Transmitted Payloads:** Audited strictly to contain only shared predictor parameters. Zero patient tabular records or raw feature vectors transmitted.
