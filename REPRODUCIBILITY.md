# Comprehensive Experimental Reproducibility Manual

This repository is structured as a certified, research-grade experimental framework suitable for 2026–2027 peer-reviewed academic literature, theses, and comparative clinical benchmarks. This document provides the end-to-end protocol required to reproduce all findings, models, metrics, and figures from scratch.

---

## 1. Computational Environment & Dependencies

### Hardware & Operating System Specifications
* **Operating System:** Tested on Windows 10/11 (PowerShell / Command Prompt) and Ubuntu Linux 22.04 LTS.
* **Python Runtime:** Python `3.11.x` (64-bit).
* **Compute Architecture:** Standard x86_64 CPU (PyTorch CPU build verified; GPU acceleration optional via CUDA).

### Dependency Installation
All external libraries are pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key library versions:
* `torch >= 2.0.0`
* `scikit-learn >= 1.3.0`
* `xgboost >= 2.0.0`
* `pandas >= 2.0.0`
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `pyyaml >= 6.0`
* `lime >= 0.2.0`
* `shap >= 0.42.0`
* `matplotlib >= 3.7.0`
* `seaborn >= 0.12.0`

---

## 2. Deterministic Seed Policy

To ensure bit-level or numerical reproducibility across all local training runs and federated simulations, random seeds are explicitly initialized across all underlying pseudorandom number generators.

### Primary Canonical Seed
* **Default Benchmark Seed:** `42`
* **Sensitivity / Variance Seeds:** `123`, `2026`

### Code-Level Initialization Protocol
Every training script and simulation runner executes the following deterministic setup prior to data loading or model instantiation:
```python
import os
import random
import numpy as np
import torch

def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## 3. Dataset Integrity & Partition Manifest

### Clinical Cohorts
The benchmark evaluates $N = 719$ multi-institutional clinical records across three distinct geographical hospital silos from the UCI Heart Disease Repository:
1. **Hospital 1 (Cleveland Clinic Foundation, USA):** $N = 303$ patients. Balanced research cohort ($45.9\%$ disease prevalence).
2. **Hospital 2 (Hungarian Institute of Cardiology, Budapest):** $N = 293$ patients. Outpatient screening cohort ($36.1\%$ disease prevalence).
3. **Hospital 3 (University Hospital Zurich & Basel, Switzerland):** $N = 123$ patients. High-acuity referral cohort ($93.5\%$ disease prevalence).

### Data Partitioning
Each hospital cohort is partitioned using a stratified $70\% / 15\% / 15\%$ train/validation/test split:
* **Hospital 1:** $212$ Train, $45$ Validation, $46$ Test.
* **Hospital 2:** $205$ Train, $44$ Validation, $44$ Test.
* **Hospital 3:** $86$ Train, $18$ Validation, $19$ Test.
* **Total Global Cohort:** $503$ Train, $107$ Validation, $109$ Test ($N = 719$).

### SHA-256 Hash Verification
The repository includes a tamper-evident cryptographic manifest at `data/split_manifest.json`. You can verify dataset integrity at any time by running:
```bash
python validate_experiments.py
```
If any CSV record has been modified or corrupted, the audit script halts with a detailed hash mismatch warning.

---

## 4. Strict Test-Set Firewall Protocol

To prevent data leakage and optimistic performance bias, the experimental pipeline enforces a strict methodological firewall:

1. **Preprocessing Isolation:** All feature standardizers (`StandardScaler`), median imputers (`SimpleImputer`), and categorical encoders are fitted **strictly on the training split (`split="train"`)**. Validation and test splits are transformed using frozen training statistics.
2. **Model Selection Firewall:** Hyperparameter tuning, early stopping, and best-epoch checkpoint selection are performed **exclusively using validation split metrics (`split="val"`, macro ROC-AUC)**.
3. **Single Test Evaluation:** Held-out test sets (`split="test"`) are evaluated exactly **once** at the conclusion of training using the selected best checkpoint. Test sets are never referenced in gradient computation, aggregation weights, or early stopping decisions.
4. **Swiss Hospital 3 Class Imbalance Handling:** The Swiss test cohort consists of $18$ positive cases and exactly $1$ negative case ($N=19$, prevalence $94.7\%$). Specificity is computed on this single negative instance ($0.0$ or $1.0$). If zero negative predictions are made, specificity evaluates to `0.0` or `NaN` (properly handled). No artificial synthetic oversampling or label rebalancing is applied to the test split.

---

## 5. End-to-End Canonical Execution Workflow

To reproduce the complete experimental matrix from scratch, execute the following commands in sequence from the repository root:

### Step 1: Preprocessing & Data Pipeline
Extracts raw UCI records, handles missingness, standardizes features, and exports stratified train/val/test CSV splits:
```bash
python preprocessing/run_preprocessing.py
```

### Step 2: Isolated Single-Hospital Baselines
Trains local models independently on each hospital's private training split without communication:
```bash
# 1D AlexNet Baselines
python models/train_alexnet.py

# 1D ResNet Baselines
python models/train_resnet.py

# Tabular XGBoost Baselines & Sample-Weighted Ensemble
python models/train_xgboost.py
```

### Step 3: Homogeneous Federated Learning Simulations
Trains 1D AlexNet and 1D ResNet across all three hospitals using sample-weighted FedAvg with client-isolated Batch Normalization buffers (FedBN):
```bash
# Federated 1D AlexNet (15 rounds, 3 local epochs)
python federated/simulation.py

# Federated 1D ResNet (15 rounds, 3 local epochs)
python federated/resnet_simulation.py
```

### Step 4: Heterogeneous Federated Learning Experiments
Trains modular composite architectures ($E_{\phi_k} \circ P_\theta$) where each hospital maintains a private local encoder ($D_k \to Z$) and collaboratively optimizes a shared global predictor ($P_\theta$):
```bash
# Heterogeneous FedAvg Baseline
python federated/heterogeneous_simulation.py

# Heterogeneous Optimization Comparison (FedAvg vs. FedProx vs. FedAdam)
python federated/run_optimization_comparison.py
```

### Step 5: Privacy & Security Evaluations
Evaluates the composition of Pairwise Zero-Sum Secure Aggregation (SecAgg) and client-update-level Differential Privacy (DP-FedAvg: L2 norm clipping + calibrated Gaussian noise on communicated shared-predictor updates, tracked via Rényi DP accounting):
```bash
python federated/run_privacy_security_experiments.py
```

### Step 6: Explainable AI Pipeline
Computes local surrogate attributions (LIME) and Shapley coalition values (KernelSHAP) on held-out test patients:
```bash
python xai/run_xai_pipeline.py
```

### Step 7: Full Reproducibility & Integrity Audit
Audits all 21 experiment declarations, data hashes, test-set firewall isolation, confusion matrix consistency, and compiles the authoritative benchmark tables:
```bash
python validate_experiments.py
```

### Step 8: Comprehensive Test Suite
Executes the automated unit and integration test suite:
```bash
pytest tests/
```

---

## 6. Central Results Store & Experiment Registry

All experimental metadata and empirical results are managed through structured, machine-readable stores:
* **Registry Declaration:** `experiments/experiment_registry.yaml` defines all $21$ planned and completed experiments with complete architectural schemas.
* **Central Results Log:** `results/experiment_results.jsonl` contains the verified JSON records of all completed experimental runs across institutional, macro, and sample-weighted levels.
* **Authoritative Human-Readable Reports:**
  - `reports/authoritative_experiment_matrix_table.md`: Comprehensive model-level benchmark table.
  - `reports/authoritative_hospital_breakdown_table.md`: Per-hospital granular diagnostic metrics.

---

## 7. Model Checkpoint & Artifact Hierarchy

Trained model weights and evaluation artifacts are stored in standardized directories:
```text
models/checkpoints/
├── alexnet_hospital_1_best.pt       # Isolated Local H1 AlexNet
├── alexnet_hospital_2_best.pt       # Isolated Local H2 AlexNet
├── alexnet_hospital_3_best.pt       # Isolated Local H3 AlexNet
├── resnet_hospital_1_best.pt        # Isolated Local H1 ResNet
├── resnet_hospital_2_best.pt        # Isolated Local H2 ResNet
├── resnet_hospital_3_best.pt        # Isolated Local H3 ResNet
├── xgboost_hospital_1.joblib        # Isolated Local H1 XGBoost
├── xgboost_hospital_2.joblib        # Isolated Local H2 XGBoost
├── xgboost_hospital_3.joblib        # Isolated Local H3 XGBoost
├── federated_alexnet_best.pt        # Global Federated 1D AlexNet (FedBN)
├── federated_resnet_best.pt         # Global Federated 1D ResNet (FedBN)
└── heterogeneous/                   # Heterogeneous Composite Checkpoints
    ├── fedavg_shared_predictor.pt   # Federated Shared Predictor (FedAvg)
    ├── fedavg_encoder_h1.pt         # Private H1 Encoder
    ├── fedavg_encoder_h2.pt         # Private H2 Encoder
    ├── fedavg_encoder_h3.pt         # Private H3 Encoder
    ├── fedprox_shared_predictor.pt  # Federated Shared Predictor (FedProx)
    └── fedadam_shared_predictor.pt  # Federated Shared Predictor (FedAdam)
```

---

## 8. Known Limitations & Architectural Boundaries

1. **Simulated Secure Aggregation:** Pairwise additive zero-sum masking ($\sum_{i=1}^K M_i = \mathbf{0}$) mathematically guarantees that the server reconstructs only the sum of updates without observing individual client updates. However, this simulation assumes $100\%$ client round completion. Handling arbitrary client dropout in asynchronous settings requires threshold Shamir secret sharing (e.g., Bonawitz et al.), which is an explicit boundary for future distributed deployment.
2. **Differential Privacy Scope:** Differential privacy (local gradient clipping $C = 1.0$ and Gaussian perturbation $\sigma = 0.05$) is applied strictly to the **shared global predictor**. Private hospital encoders are client-confined and never communicated; hence, they do not expend differential privacy budget.
3. **Statistical Sample Size:** The UCI multi-center repository provides $N=719$ clinical records. While ideal for controlled multi-site benchmarking, real-world hospital federation entails orders of magnitude more patients. Consequently, findings reflect fundamental algorithmic behaviors on tabular medical data rather than exhaustive population-scale epidemiological inferences.
