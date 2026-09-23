# Federated 1D AlexNet Communication Log

This document logs the multi-round communication telemetry for the Federated 1D AlexNet (FedAvg) experiment.

## Communication Hyperparameters
- **Federated Strategy:** Weighted Federated Averaging (FedAvg)
- **Communication Rounds:** 15
- **Client Participation Rate:** 100% (3/3 hospitals participating every round)
- **Local Epochs:** 3 per round
- **Local Optimizer:** Adam (LR: 0.001, Weight Decay: 1e-4, Batch Size: 16)
- **Sample Distribution:** Hospital 1 (212, 42.1%), Hospital 2 (205, 40.8%), Hospital 3 (86, 17.1%)

---

## Round-by-Round Telemetry Log

| Round    | Participating Clients   |   Total Training Samples | Macro Val Acc   | Macro Val Recall   |   Macro Val F1 |   Macro Val ROC-AUC | H1 / H2 / H3 Val F1   | Elapsed   |
|:---------|:------------------------|-------------------------:|:----------------|:-------------------|---------------:|--------------------:|:----------------------|:----------|
| Round 01 | 3/3 (100%)              |                      503 | 72.64%          | 96.33%             |         0.7703 |              0.6411 | 0.727 / 0.612 / 0.971 | 0.87s     |
| Round 02 | 3/3 (100%)              |                      503 | 60.27%          | 59.58%             |         0.6019 |              0.7336 | 0.783 / 0.812 / 0.211 | 0.48s     |

---

## Data Locality & Privacy Audit
- **Zero Patient Transmission:** Audited each communication step; strictly numeric model parameter tensors were transferred.
- **Client Isolation:** Local datasets remained strictly within local hospital storage partitions.
