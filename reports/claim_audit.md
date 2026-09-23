# Master Thesis & Academic Paper Claim Audit Matrix

This document provides a formal, evidence-backed audit of all scientific claims intended for publication or thesis submission. Each claim is verified against executed empirical experiments, stored result files, and explicit methodological boundaries.

---

## Audit Classification Taxonomy
- **Directly Supported:** Empirically verified by multiple experimental runs with exact numerical evidence in `reports/final_analysis_dataset.csv`.
- **Supported with Limitations:** Supported by observed data within the specific constraints of the simulated benchmark (e.g., sample size, simulated sites).
- **Not Supported / Prohibited:** Speculative, causal, or clinical claims that cannot be proven with the current codebase and must NOT appear in the manuscript.

---

## Comprehensive Claims Audit Table

| # | Scientific Claim | Relevant Experiments | Evidence File / Metric | Claim Strength | Mandatory Qualification / Limitation |
| :-: | :--- | :--- | :--- | :---: | :--- |
| **C1** | Federated learning achieves diagnostic performance comparable to centralized local training without pooling raw patient records. | `local_*`, `homogeneous_*`, `heterogeneous_fedavg_mlp` | `table_02`, `table_04` (Macro Acc: $84.4\%$ vs $85.3\%$) | **Directly Supported** | Valid within the 3 participating UCI hospital cohorts. |
| **C2** | Private client-side encoders ($D_i 	o Z$) enable federated collaboration across institutions with heterogeneous feature spaces without artificial zero-padding. | `heterogeneous_fedavg_mlp`, `heterogeneous_latent_dim_*` | `table_04`, `figure_04` (Macro ROC-AUC: $0.6926$) | **Directly Supported** | Feature spaces are decoupled via local encoder modules; encoders remain strictly private. |
| **C3** | Proximal regularization (FedProx with $\mu=0.01$) stabilizes global convergence and improves accuracy under non-IID institutional drift. | `heterogeneous_fedprox_mlp_*` | `table_05`, `figure_01` (Macro Acc: $86.1\%$ vs $85.4\%$) | **Supported with Limitations** | Performance is sensitive to $\mu$; $\mu=0.1$ degrades performance relative to $\mu=0.01$. |
| **C4** | Server-side adaptive momentum (FedAdam) accelerates convergence and provides superior robustness to differential privacy noise. | `heterogeneous_fedadam_mlp`, `*_secure_aggregation_dp` | `table_06`, `figure_01` (Peak val checkpoint at Round 2 vs Round 15) | **Supported with Limitations** | Observed on 15 communication rounds; requires tuning server learning rate $\eta=0.1$. |
| **C5** | Simulated pairwise zero-sum masking ($\sum M_i = \mathbf{0}$) achieves server-side update confidentiality with zero degradation of diagnostic accuracy. | `*_secure_aggregation` | `table_09`, `figure_05` (Exact numerical parity: $85.36\%$ Acc) | **Directly Supported** | Mathematical mask cancellation is exact; assumes full client participation ($0\%$ dropout). |
| **C6** | Client-update-level Differential Privacy (DP-FedAvg) incurs an empirical accuracy penalty of $< 2.0\%$ on the shared predictor. | `*_dp`, `*_secure_aggregation_dp` | `table_09`, `figure_05` (Acc: $85.0\%$ vs $85.4\%$) | **Supported with Limitations** | DP-FedAvg: post-training L2 clipping + Gaussian noise on shared-predictor updates (not per-sample gradient clipping). Tested with clipping $C=1.0, \sigma=0.3$; accumulated budget is $\epsilon=148.03$ ($\delta=10^{-5}$). |
| **C7** | Private encoder retention reduces distributed communication payload by $> 96\%$ compared to homogeneous full-network federation. | `homogeneous_fedavg_alexnet`, `heterogeneous_fedavg_mlp` | `table_12`, `figure_06` ($102.8	ext{ KB}$ vs $2.75	ext{ MB/round}$) | **Directly Supported** | Calculated on exact parameter byte counts ($4	ext{ bytes/float32}$). |
| **C8** | Feature attributions (LIME/SHAP) prove that exercise-induced ST depression causes coronary heart disease. | `xai/*` | `xai_results.csv` | **NOT SUPPORTED / PROHIBITED** | **Violation:** Attributions indicate statistical model correlations, NOT medical causality. |
| **C9** | The proposed federated framework is HIPAA-compliant, GDPR-certified, and ready for clinical deployment. | `federated/privacy/*` | Code inspection | **NOT SUPPORTED / PROHIBITED** | **Violation:** This is a research prototype; formal compliance requires production audits. |
| **C10**| Hospital 3 zero specificity proves the federated model completely fails on Swiss patients. | `table_10` | `table_10_hospital_level_results.md` | **NOT SUPPORTED / PROHIBITED** | **Violation:** Hospital 3 test split contains exactly 1 negative sample; 1 error forces Spec to 0. |

---

## Prohibited Statements for Academic Manuscript
The following phrases are explicitly flagged and must NOT appear in the final paper or thesis without formal qualification:
1. *"The system guarantees absolute privacy."* -> Replace with: *"The system implements simulated pairwise masking (SecAgg) and client-update-level differential privacy (DP-FedAvg) as a research prototype. No formal cryptographic or regulatory privacy certification is provided."*
2. *"The model is clinically validated."* -> Replace with: *"The model was evaluated on retrospective multi-center benchmarks."*
3. *"The framework is production-ready."* -> Replace with: *"The framework serves as an open-source experimental research prototype."*
4. *"FedProx is universally superior to FedAvg."* -> Replace with: *"FedProx demonstrated higher observed accuracy under specific proximal hyperparameter settings."*
