# Key Research Findings: Multi-Institutional Federated Learning Benchmark

This document synthesizes the primary empirical findings resulting from the completed 21-experiment, 3-seed research matrix.

---

### Finding 1: Multi-Center Federation Mitigates Local Silo Overfitting
* **Evidence:** Local hospital training produced severe cross-site generalization gaps, with Hospital 3 local models achieving only $0.6243$ ROC-AUC. Heterogeneous federated learning raised macro diagnostic discrimination to $0.6926$ ROC-AUC while preserving high local accuracy on Hospital 1 ($84.8\%$) and Hospital 2 ($81.8\%$).
* **Metric:** Macro Test Accuracy: $85.36\%$, Weighted Accuracy: $84.40\%$.
* **Relevant Experiments:** `local_*`, `heterogeneous_fedavg_mlp`.
* **Limitation:** Generalization is evaluated on 3 hospital cohorts ($N=719$).

---

### Finding 2: Private Encoder Confinement Enables Heterogeneous Collaboration with >96% Communication Reduction
* **Evidence:** Decoupling client-local feature encoders ($D_i 	o Z=32$) from the shared global predictor ($Z 	o 1$) enabled collaboration across institutions without artificial feature padding. Transmitting only the shared predictor reduced network communication from $2.75	ext{ MB/round}$ (Homogeneous AlexNet) to $102.8	ext{ KB/round}$ (Heterogeneous Composite), a **$96.3\%$ bandwidth reduction**.
* **Metric:** Communication per round: $105,240	ext{ bytes}$ vs. $2,886,168	ext{ bytes}$.
* **Relevant Experiments:** `homogeneous_fedavg_alexnet`, `heterogeneous_fedavg_mlp`.
* **Limitation:** Encoders must have sufficient capacity ($2$-layer MLP) to model non-linear interactions locally.

---

### Finding 3: Proximal Regularization Stabilizes Non-IID Demographic Client Drift
* **Evidence:** Adding client proximal regularization ($\mu=0.01$) strictly to the shared global predictor improved Macro Test Accuracy from $85.36\%$ to **$86.12\%$** and Macro F1 from $0.8488$ to **$0.8565$**, specifically boosting Hungarian outpatient accuracy from $81.8\%$ to $84.1\%$.
* **Metric:** Macro Accuracy: $86.12\%$, Macro F1: $0.8565$, Brier Score: $0.1925$.
* **Relevant Experiments:** `heterogeneous_fedavg_mlp`, `heterogeneous_fedprox_mlp_mu_001`.
* **Limitation:** Performance degrades if $\mu$ is set too high ($\mu=0.1$ reduced Macro Accuracy to $83.8\%$).

---

### Finding 4: Adaptive Server Momentum Accelerates Convergence and Dampens Differential Privacy Perturbations
* **Evidence:** FedAdam ($\eta=0.1, eta_1=0.9, eta_2=0.99$) reached optimal validation performance at Round 2 (versus Round 15 for FedAvg). When operating under client-side DP clipping and Gaussian noise, FedAdam achieved the highest Macro ROC-AUC (**$0.7559$**) and Weighted ROC-AUC (**$0.8240$**).
* **Metric:** Macro ROC-AUC under DP: $0.7559$ (FedAdam) vs. $0.7024$ (FedAvg).
* **Relevant Experiments:** `heterogeneous_fedadam_mlp`, `heterogeneous_fedadam_secure_aggregation_dp`.
* **Limitation:** Requires tuning server learning rate $\eta$; early aggressive updates can cause transient validation loss spikes.

---

### Finding 5: Simulated Zero-Sum Masking Preserves Exact Model Utility
* **Evidence:** Pairwise additive zero-sum masking ($\sum M_i = \mathbf{0}$) produced bitwise identical test evaluations to unmasked baselines ($85.36\%$ Accuracy, $0.8488$ F1, $0.6926$ ROC-AUC), proving that simulated secure aggregation imposes zero utility degradation.
* **Metric:** Test metrics match to machine precision ($< 10^{-6}$ discrepancy).
* **Relevant Experiments:** `heterogeneous_fedavg_mlp`, `heterogeneous_fedavg_secure_aggregation`.
* **Limitation:** Assumes 100% round participation (no client dropout resilience without threshold Shamir secret sharing).
