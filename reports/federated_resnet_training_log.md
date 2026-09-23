# Federated 1D ResNet Communication Log

This document logs the multi-round communication telemetry for the Federated 1D ResNet (FedAvg) experiment.

## Communication Hyperparameters
- **Model Architecture:** 1D ResNet (Residual blocks, BatchNorm1d, Skip Connections)
- **Federated Strategy:** Weighted Federated Averaging (FedAvg)
- **Communication Rounds:** 15
- **Client Participation Rate:** 100% (3/3 hospitals participating every round)
- **Local Epochs:** 3 per round
- **Local Optimizer:** Adam (LR: 0.001, Weight Decay: 0.0001, Batch Size: 16)
- **Sample Distribution:** Hospital 1 (212, 42.15%), Hospital 2 (205, 40.76%), Hospital 3 (86, 17.10%)

---

## Round-by-Round Telemetry Log (Evaluated on Validation Split)

| Round    | Participating Clients   |   Total Training Samples | Macro Val Acc   | Macro Val Recall   |   Macro Val F1 |   Macro Val ROC-AUC | H1 / H2 / H3 Val F1   | Elapsed   |
|:---------|:------------------------|-------------------------:|:----------------|:-------------------|---------------:|--------------------:|:----------------------|:----------|
| Round 01 | 3/3 (100%)              |                      503 | 83.94%          | 81.75%             |         0.8075 |              0.6456 | 0.784 / 0.667 / 0.971 | 2.40s     |
| Round 02 | 3/3 (100%)              |                      503 | 66.92%          | 65.71%             |         0.7023 |              0.6946 | 0.773 / 0.774 / 0.560 | 2.46s     |
| Round 03 | 3/3 (100%)              |                      503 | 79.11%          | 73.19%             |         0.7775 |              0.7064 | 0.791 / 0.667 / 0.875 | 2.81s     |
| Round 04 | 3/3 (100%)              |                      503 | 82.85%          | 83.36%             |         0.8236 |              0.6936 | 0.756 / 0.774 / 0.941 | 2.70s     |
| Round 05 | 3/3 (100%)              |                      503 | 73.96%          | 73.93%             |         0.7618 |              0.7386 | 0.711 / 0.774 / 0.800 | 2.39s     |
| Round 06 | 3/3 (100%)              |                      503 | 76.52%          | 69.14%             |         0.7505 |              0.7065 | 0.773 / 0.640 / 0.839 | 2.50s     |
| Round 07 | 3/3 (100%)              |                      503 | 78.38%          | 78.94%             |         0.7914 |              0.6684 | 0.766 / 0.733 / 0.875 | 2.35s     |
| Round 08 | 3/3 (100%)              |                      503 | 80.99%          | 79.81%             |         0.8092 |              0.6842 | 0.744 / 0.774 / 0.909 | 2.31s     |
| Round 09 | 3/3 (100%)              |                      503 | 81.73%          | 79.31%             |         0.8135 |              0.785  | 0.773 / 0.759 / 0.909 | 2.26s     |
| Round 10 | 3/3 (100%)              |                      503 | 79.51%          | 74.89%             |         0.7992 |              0.7487 | 0.818 / 0.741 / 0.839 | 2.22s     |
| Round 11 | 3/3 (100%)              |                      503 | 78.37%          | 75.86%             |         0.7717 |              0.7428 | 0.800 / 0.640 / 0.875 | 2.55s     |
| Round 12 | 3/3 (100%)              |                      503 | 76.90%          | 79.06%             |         0.7838 |              0.8588 | 0.735 / 0.750 / 0.867 | 2.80s     |
| Round 13 | 3/3 (100%)              |                      503 | 79.16%          | 84.20%             |         0.8147 |              0.6678 | 0.769 / 0.800 / 0.875 | 2.59s     |
| Round 14 | 3/3 (100%)              |                      503 | 76.92%          | 84.20%             |         0.7966 |              0.6314 | 0.741 / 0.774 / 0.875 | 2.43s     |
| Round 15 | 3/3 (100%)              |                      503 | 79.88%          | 73.09%             |         0.7905 |              0.8141 | 0.722 / 0.774 / 0.875 | 2.39s     |

---

## Data Locality & Privacy Audit
- **Zero Patient Transmission:** Audited each communication step; strictly numeric model parameter tensors were transferred.
- **Client Isolation:** Local datasets remained strictly within local hospital storage partitions.
