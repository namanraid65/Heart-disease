# 🫀 Heterogeneous & Privacy-Preserving Federated Learning for Multi-Center Heart Disease Prediction

A research-grade, privacy-preserving clinical machine learning framework that collaboratively trains deep neural models across three distinct hospital cohorts with **heterogeneous feature spaces**, **non-IID demographic shifts**, **advanced federated optimization (FedAvg, FedProx, FedAdam)**, and **composable privacy mechanisms (Secure Aggregation & Differential Privacy)** without pooling patient data into a central database.

---

## 🏥 Participating Hospital Silos

| Hospital Node | Clinical Institution | Geography | Records | Disease Prevalence | Clinical Profile | Feature Dim ($D_k$) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **Hospital 1** | Cleveland Clinic Foundation | USA | $N=303$ | 45.9% | Balanced cardiology research cohort | 25 |
| **Hospital 2** | Hungarian Institute of Cardiology | Hungary | $N=293$ | 36.1% | Outpatient screening cohort | 25 |
| **Hospital 3** | University Hospital Zurich / Basel | Switzerland | $N=123$ | 93.5% | High-acuity inpatient referral cohort | 25 |

*Note: All clinical cohorts are partitioned using a certified $70\% / 15\% / 15\%$ stratified split ($N=503$ train, $107$ validation, $109$ test) audited by `data/split_manifest.json`.*

---

## 🔬 Core System Architecture & Capabilities

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FEDERATED SERVER (Aggregation & Privacy)                       │
│                                                                                        │
│   • Shared Global Predictor (P_θ : ℝ^Z → [0, 1])                                      │
│   • Federated Optimizers: FedAvg, FedProx (μ-proximal), FedAdam (server momentum)      │
│   • Composable Security: Zero-Sum Secure Aggregation (∑ M_i = 0)                       │
│   • Privacy Accounting: Rényi Differential Privacy (DP-FedAvg, clipping C, noise σ)     │
└───────────────────────────────────────────▲────────────────────────────────────────────┘
                                            │ Transmits ONLY Shared Predictor (P_θ)
                 ┌──────────────────────────┼──────────────────────────┐
                 │                          │                          │
  ┌──────────────▼─────────────┐ ┌──────────▼────────────┐ ┌───────────▼────────────┐
  │   Hospital 1 (Cleveland)   │ │  Hospital 2 (Budapest) │ │  Hospital 3 (Zurich)    │
  │                            │ │                        │ │                         │
  │ • Native Features: D_1     │ │ • Native Features: D_2 │ │ • Native Features: D_3  │
  │ • Private Encoder: E_ϕ1    │ │ • Private Encoder: E_ϕ2│ │ • Private Encoder: E_ϕ3 │
  │   (D_1 → Z = 32)           │ │   (D_2 → Z = 32)       │ │   (D_3 → Z = 32)        │
  │ • Local Data: 303 patients │ │ • Local Data: 293 pts  │ │ • Local Data: 123 pts   │
  │ • STRICTLY PRIVATE:        │ │ • STRICTLY PRIVATE:    │ │ • STRICTLY PRIVATE:     │
  │   No raw data leaves silo  │ │   No raw data leaves   │ │   No raw data leaves    │
  │   Encoder ϕ_1 NOT shared   │ │   Encoder ϕ_2 NOT share│ │   Encoder ϕ_3 NOT share │
  └────────────────────────────┘ └────────────────────────┘ └─────────────────────────┘
```

1. **Heterogeneous-Feature FL:** Supports hospitals with different native clinical feature spaces by decoupling private client-side feature encoders ($E_{\phi_k}: \mathbb{R}^{D_k} \to \mathbb{R}^Z$) from a collaborative federated predictor ($P_\theta: \mathbb{R}^Z \to [0, 1]$).
2. **Federated Optimization Suite:** Benchmarks standard **FedAvg**, proximal regularization (**FedProx**), and adaptive server-side momentum (**FedAdam**) to handle extreme non-IID client drift.
3. **Composable Privacy & Security:**
   - **Simulated Secure Aggregation (SecAgg):** Additive zero-sum pairwise masking ($\sum_{i=1}^K M_i = \mathbf{0}$) preventing server inspection of client updates.
   - **Client-Update-Level Differential Privacy (DP-FedAvg):** After local training, the communicated shared-predictor parameter update is clipped to a maximum L2 norm ($C = 1.0$) and calibrated Gaussian noise ($\sigma = 0.05$) is added before federated aggregation. This is distinct from per-example gradient-level DP-SGD. Privacy consumption is tracked via Rényi Differential Privacy (RDP) accounting.
4. **Homogeneous Neural & Tabular Baselines:** Federated 1D AlexNet with FedBN, Federated 1D ResNet with FedBN, and Sample-Weighted Local XGBoost Ensembles.
5. **Explainable AI (XAI):** Cross-model and cross-hospital interpretability using local surrogate models (LIME) and Shapley coalitions (KernelSHAP).
6. **Strict Test Firewall:** Preprocessing fitted strictly on training data; model checkpoints selected strictly on validation ROC-AUC; held-out test splits ($N=109$) evaluated exactly once.

---

## 📊 Empirical Benchmark Summary

### 1. Multi-Center Institutional Breakdown (Homogeneous Baselines & Ensembles)

Evaluated on held-out test splits across all three medical institutions ($N=109$ total test instances, authoritative results from [`reports/master_results.csv`](reports/master_results.csv)):

| Evaluation Metric | Federated 1D AlexNet (FedBN) | Federated 1D ResNet (FedBN) | Local XGBoost Ensemble (Baseline) |
| :--- | :---: | :---: | :---: |
| **Cleveland (H1) Accuracy** | **84.8%** | 78.3% | 87.0% |
| **Cleveland (H1) ROC-AUC** | **0.9448** | 0.9029 | 0.9543 |
| **Hungarian (H2) Accuracy** | 77.3% | 81.8% | **84.1%** |
| **Hungarian (H2) Recall (Sensitivity)** | **81.2%** | 75.0% | 75.0% |
| **Swiss (H3) Recall** | 11.1% | 55.6% | **61.1%** |
| **Macro Average Accuracy** | 57.5% | 70.9% | **76.3%** |
| **Sample-Weighted Accuracy** | 68.8% | 75.2% | **79.8%** |

---

### 2. Comprehensive Experimental Matrix (Optimization & Privacy)

Authoritative results evaluated on the frozen held-out test splits across all three medical institutions ($N=109$ total test instances, extracted from [`results/experiment_results.jsonl`](results/experiment_results.jsonl)):

| Model / Strategy | Strategy Family | Macro Accuracy | Sample-Weighted Accuracy | Macro ROC-AUC | Brier Score | ECE | Privacy Regime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Local XGBoost Ensemble** | Tabular Baseline | **76.3%** | **80.7%** | **0.8659** | 0.1584 | 0.1341 | None (Local) |
| **Federated 1D ResNet (FedBN)** | Homogeneous FL | 70.9% | 75.2% | 0.8174 | 0.1878 | 0.1654 | None |
| **Federated 1D AlexNet (FedBN)** | Homogeneous FL | 57.5% | 68.8% | 0.8126 | 0.2287 | 0.2223 | None |
| **Heterogeneous FedAvg ($Z=32$)** | Heterogeneous FL | 64.9% | 70.6% | 0.8037 | 0.1970 | 0.1506 | Baseline (None) |
| **Heterogeneous FedProx ($\mu=0.01$)** | Heterogeneous FL | 63.8% | 69.7% | 0.8118 | 0.1925 | 0.1517 | Baseline (None) |
| **Heterogeneous FedAdam ($\eta_s=0.01$)**| Heterogeneous FL | 64.9% | 70.6% | 0.8066 | 0.1983 | 0.1505 | Baseline (None) |
| **Heterogeneous FedAvg + SecAgg** | Privacy FL | 64.9% | 70.6% | 0.8037 | 0.1970 | 0.1506 | SecAgg Only |
| **Heterogeneous FedAvg + DP** | Privacy FL | 63.8% | 69.7% | 0.7938 | 0.2014 | 0.1561 | DP Only ($\sigma=0.05$) |
| **Heterogeneous FedAvg + SecAgg + DP**| Privacy FL | 63.8% | 69.7% | 0.7938 | 0.2014 | 0.1561 | SecAgg + DP |

*Detailed breakdowns available in [Authoritative Experiment Matrix](reports/authoritative_experiment_matrix_table.md) and [Per-Hospital Breakdown Table](reports/authoritative_hospital_breakdown_table.md).*

---

## 🛠️ Complete Pipeline Execution Workflow

To reproduce the complete experimental matrix, execute the following commands in order:

```bash
# 1. Run Preprocessing Pipeline (Clean, Impute, 25-Feature Schema)
python preprocessing/run_preprocessing.py

# 2. Train Isolated Local Baselines (Independent Hospital Silos)
python models/train_alexnet.py
python models/train_resnet.py
python models/train_xgboost.py

# 3. Run Homogeneous Federated Learning Simulations (FedAvg + FedBN)
python federated/simulation.py            # Federated 1D AlexNet
python federated/resnet_simulation.py     # Federated 1D ResNet

# 4. Run Heterogeneous Federated Learning Experiments
python federated/heterogeneous_simulation.py       # Heterogeneous FedAvg Baseline
python federated/run_optimization_comparison.py   # FedAvg vs FedProx vs FedAdam

# 5. Run Composable Privacy & Security Evaluations (SecAgg + DP)
python federated/run_privacy_security_experiments.py

# 6. Run Explainable AI Pipeline (LIME & KernelSHAP Interpretability)
python xai/run_xai_pipeline.py

# 7. Execute Full Reproducibility & Integrity Audit
python validate_experiments.py

# 8. Run Automated Test Suite
pytest tests/
```

---

## 🚀 Interactive Clinical CLI Inference

The repository provides an interactive terminal inference utility [`predict.py`](predict.py) implementing the complete heterogeneous federated inference pipeline (`select hospital -> load schema -> preprocessing -> private encoder -> shared predictor -> probability -> native feature XAI`):

```bash
# Interactive guided mode (prompts for hospital-specific clinical features):
python predict.py

# Hospital 1 (Cleveland, D1=25) preset demonstration with SHAP explanation:
python predict.py --demo high_risk --hospital hospital_1

# Hospital 2 (Hungarian, D2=25) preset demonstration:
python predict.py --demo healthy --hospital hospital_2

# Hospital 4 (Extended Biomarkers, D4=30 Synthetic) preset demonstration:
python predict.py --demo h4_synthetic

# Run with LIME or both explainers:
python predict.py --demo high_risk --explain both

# Optional legacy baseline comparison (AlexNet, ResNet, XGBoost):
python predict.py --demo high_risk --legacy-baselines
```

---

## 📂 Project Structure & Architecture Hierarchy

```text
├── dataset/                         # Raw UCI multi-center data files
├── data/
│   ├── processed/                   # Cleaned, standardized 25-feature patient splits
│   └── split_manifest.json          # Tamper-evident cryptographic SHA256 manifest
├── docs/
│   ├── research_questions.md        # Formal academic questions (RQ1–RQ5)
│   └── ablation_matrix.md           # 5-axis systematic ablation matrix
├── experiments/
│   └── experiment_registry.yaml     # Formal 21-experiment registry schema
├── results/
│   ├── experiment_results.jsonl     # Audited central JSONL results store
│   └── build_central_results.py     # Results compiler script
├── models/
│   ├── alexnet_1d.py                # 1D AlexNet architecture definition
│   ├── resnet_1d.py                 # 1D ResNet with residual skip connections
│   ├── xgboost_model.py             # Tabular XGBoost pipeline
│   ├── heterogeneous/
│   │   ├── composite.py             # Modular Composite Architecture (E_ϕ ∘ P_θ)
│   │   ├── client_schema.py         # Client-specific feature metadata
│   │   ├── encoder.py               # Private hospital feature encoders (D_k → Z)
│   │   └── predictor.py             # Shared global predictor (Z → [0, 1])
│   └── checkpoints/                 # Saved model weights (.pt / .joblib)
├── federated/
│   ├── client.py / server.py        # Baseline FedAvg client/server implementation
│   ├── simulation.py                # Multi-round simulation runner (AlexNet)
│   ├── resnet_simulation.py         # Multi-round simulation runner (ResNet)
│   └── heterogeneous/               # Heterogeneous Feature FL Framework
│       ├── client.py                # Client with private encoder & synchronized predictor
│       ├── server.py                # Server maintaining shared predictor
│       ├── strategy.py              # FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad
│       ├── simulation.py            # Heterogeneous simulation runner (H1-H4)
│       ├── privacy.py               # Deterministic SHA-256 Pairwise SecAgg & Rényi DP
│       ├── evaluate.py              # Per-hospital, macro, and weighted evaluation
│       └── xai_compat.py            # Local XAI wrapper for native feature attribution
├── xai/
│   ├── lime_explainer.py            # Local surrogate explanations (native features)
│   ├── shap_explainer.py            # Shapley coalition explanations (native features)
│   └── run_xai_pipeline.py          # Full XAI execution pipeline
├── evaluation/
│   ├── generate_final_evaluation.py # Master evaluation suite generator
│   └── result_loader.py             # Empirical results consolidator
├── reports/                         # Authoritative markdown tables & figures
├── tests/                           # 111 unit and integration tests (100% passing)
├── validate_experiments.py          # Automated 6-stage reproducibility audit
├── REPRODUCIBILITY.md               # Complete replication manual
├── predict.py                       # Heterogeneous interactive CLI inference tool
└── requirements.txt                 # Pinned dependencies
```

---

## 📑 Research Questions & Ablation Studies

* Detailed formulations of research hypotheses $H_1$ through $H_5$ are documented in [`docs/research_questions.md`](docs/research_questions.md).
* Systematic parameter sweeps and planned experiment definitions across all 5 experimental dimensions are detailed in [`docs/ablation_matrix.md`](docs/ablation_matrix.md).
* Reproducibility guarantees, random seed policies, and boundary conditions are detailed in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 🔮 Future Research Directions

1. **Asynchronous Client Dropout & Threshold Secret Sharing:** Transitioning simulated zero-sum secure aggregation to production-grade Shamir secret sharing (e.g., Bonawitz et al.) for resilience against client disconnections.
2. **Multi-Modal Clinical Imaging Integration:** Extending private client encoders to incorporate 1D/2D cardiac ECG waveforms and echocardiogram ultrasound feeds.
3. **Dynamic Self-Supervised Pre-Training:** Pre-training private hospital encoders locally using masked tabular autoencoding prior to collaborative predictor fine-tuning.
