# Federated Convergence Analysis Across Communication Rounds

This report evaluates multi-round loss and validation metric trajectories for Federated 1D AlexNet and Federated 1D ResNet trained using FedAvg with FedBN.

---

## 1. Validation-Based Model Selection Protocol
- **Validation Monitoring**: In accordance with strict scientific evaluation integrity, model checkpoint selection is performed strictly on client validation splits (`split="val"`), never on the held-out test split (`split="test"`).
- **Selection Metric**: The best global federated model checkpoint is selected based on maximum macro ROC-AUC (with validation loss tie-breaking) across communication rounds.
- **Frozen Test Evaluation**: The selected best global model is frozen and evaluated exactly once on the held-out test split.
- **Checkpoints**:
  - AlexNet: `models/checkpoints/federated_alexnet/global_alexnet_final.pt`
  - ResNet: `models/checkpoints/federated_resnet/global_resnet_final.pt`
