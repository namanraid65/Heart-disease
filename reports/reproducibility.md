# Experimental Reproducibility Report

This document records the exact runtime environment, dependency versions, dataset partitions, and hyperparameter configurations to guarantee 100% deterministic reproducibility.

---

## 1. Operating Environment & Package Versions
- **Python Version:** `3.11.9`
- **Operating System:** `Windows-10-10.0.19045-SP0`
- **PyTorch:** `2.10.0+cpu`
- **Scikit-Learn:** `1.7.2`
- **XGBoost:** `3.2.0`
- **Flower (flwr):** `1.37.0`
- **LIME:** `0.2.0.1`
- **SHAP:** `0.51.0`
- **Random Seed:** `42`

---

## 2. Dataset Partition Specifications
- **Harmonized Feature Schema:** 25 normalized tabular features ([`preprocessing/feature_schema.py`](../preprocessing/feature_schema.py)).
- **Partition Splits:** Stratified 70% Train, 15% Validation, 15% Held-Out Test.
- **Hospital 1 (Cleveland):** $N=303$ ($212$ Train, $45$ Val, $46$ Test).
- **Hospital 2 (Hungarian):** $N=294$ ($205$ Train, $44$ Val, $44$ Test).
- **Hospital 3 (Switzerland):** $N=123$ ($86$ Train, $18$ Val, $19$ Test).

---

## 3. Training & Federated Hyperparameters
- **Communication Rounds:** 15 rounds
- **Client Participation:** 100% (3/3 hospitals participating every round)
- **Local Epochs ($E$):** 3 epochs per round
- **Local Batch Size ($B$):** 16
- **Optimizer:** Adam (Learning Rate: $0.001$, Weight Decay: $1\times 10^-4$)
- **Aggregation Strategy:** Sample-Weighted FedAvg ($0.4215\cdot\text{H1} + 0.4076\cdot\text{H2} + 0.1710\cdot\text{H3}$) with client-isolated BatchNorm buffers (FedBN).
- **Model Selection:** Validation-based selection on `split="val"` using macro ROC-AUC; single evaluation on frozen held-out test split.
