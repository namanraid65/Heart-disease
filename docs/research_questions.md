# Academic Research Questions (RQ1 – RQ5)

This document formalizes the five core scientific research questions driving the multi-institutional federated learning investigation conducted in this repository. All experiments, baseline evaluations, and ablation studies map directly to one or more of these questions.

---

## RQ1: Centralized vs. Federated vs. Local Baselines

> **Question:** *To what extent does multi-institutional federated learning (homogeneous and heterogeneous) preserve or improve diagnostic performance relative to isolated, single-hospital local training, and how closely does it approach centralized data-pooling performance without violating data residency?*

### Motivation & Background
In multi-center clinical diagnostics, individual medical centers frequently lack sufficient training diversity and statistical power. Isolated institutional models run the risk of severe overfitting to local demographic biases and hospital-specific measuring protocols. Conversely, pooling raw patient electronic health records (EHR) into a centralized repository is legally and ethically restricted by patient confidentiality regulations (e.g., HIPAA, GDPR). Federated learning (FL) enables collaborative gradient updates without raw data sharing. However, non-IID demographic shifts and cohort size asymmetries across institutions can degrade federated convergence.

### Formal Hypotheses
* **$H_{1a}$ (Federated Generalization Advantage):** Federated collaborative training yields higher macro-average test ROC-AUC and Sensitivity across all three participating hospital cohorts than the average of models trained exclusively on local institutional data.
* **$H_{1b}$ (Low-Prevalence / Low-Resource Benefit):** The hospital with the lowest sample count or extreme class skew (Hospital 3 / Switzerland, $N=123$, $93.5\%$ prevalence) achieves measurable recall and discrimination gains when participating in federated optimization compared to isolated training.
* **$H_{1c}$ (Centralized Gap):** Federated models retain at least $90\%$ of the diagnostic accuracy and ROC-AUC of an ensemble baseline while eliminating the requirement for raw record transmission.

### Evaluation Protocol & Metrics
* **Metrics:** Accuracy, Balanced Accuracy, Sensitivity (Recall), Specificity, ROC-AUC, PR-AUC, F1-Score, Brier Score, and Expected Calibration Error (ECE).
* **Aggregation Granularity:** Both Macro Hospital Average (unweighted mean across centers) and Sample-Weighted Average (weighted by test cohort size $N_k$).
* **Firewall Compliance:** Evaluated strictly on the frozen held-out test splits ($N=109$) with checkpoint selection determined entirely by validation set ROC-AUC.
* **Registry Experiment Mapping:**
  - `local_alexnet_h1`, `local_alexnet_h2`, `local_alexnet_h3`
  - `local_resnet_h1`, `local_resnet_h2`, `local_resnet_h3`
  - `local_xgboost_ensemble`
  - `homogeneous_fedavg_alexnet`, `homogeneous_fedavg_resnet`
  - `heterogeneous_fedavg`

---

## RQ2: Feature Space Heterogeneity and Representation Alignment

> **Question:** *Can private client-side feature encoders mapping varying hospital-native feature spaces ($D_i \to Z$) into a shared latent space $Z$ enable federated learning without manual feature schema alignment or artificial zero-padding?*

### Motivation & Background
Standard horizontal federated learning strictly assumes a homogeneous feature space across all clients ($\mathcal{X}_1 = \mathcal{X}_2 = \dots = \mathcal{X}_K$). In real-world clinical practice, different hospitals perform varying diagnostic panels (e.g., standard clinical metrics vs. advanced fluoroscopy vs. enzymatic biomarkers). Prior works frequently resort to zero-padding or intersecting features, which discards diagnostic signals or introduces spurious zeros. We investigate a modular composite architecture: each hospital $k$ retains a strictly local, private encoder $E_{\phi_k}: \mathbb{R}^{D_k} \to \mathbb{R}^Z$ that maps hospital-specific input dimensions $D_k$ into an aligned latent manifold $Z \in \mathbb{R}^d$, while sharing a federated global predictor $P_\theta: \mathbb{R}^Z \to [0, 1]$.

### Formal Hypotheses
* **$H_{2a}$ (Architectural Viability):** A heterogeneous composite architecture ($E_{\phi_k} \circ P_\theta$) achieves stable federated convergence across sites with differing feature dimensions without requiring artificial feature padding or manual intersection.
* **$H_{2b}$ (Representational Equivalence):** The latent representation space $Z$ develops cross-hospital diagnostic coherence such that the shared predictor $P_\theta$ can generalize across sites despite private encoder weights $\phi_k$ never being shared or aggregated.
* **$H_{2c}$ (Communication Efficiency):** Restricting communication to the shared predictor parameters $\theta \in \mathbb{R}^{|\theta|}$ reduces per-round communication payload by $25\%\text{--}40\%$ relative to federating the entire network, without sacrificing global accuracy.

### Evaluation Protocol & Metrics
* **Metrics:** Macro test ROC-AUC, validation loss convergence rate across rounds, per-hospital feature attribution stability (via LIME/KernelSHAP on latent and input spaces), and communication bytes per round.
* **Architectural Boundary:** Private encoders $\phi_k$ remain permanently on client devices; zero encoder weights, gradients, or buffers are transmitted to the server.
* **Registry Experiment Mapping:**
  - `heterogeneous_fedavg`
  - `latent_dim_8`, `latent_dim_16`, `latent_dim_32`
  - `encoder_1layer`, `encoder_2layer`, `encoder_3layer`

---

## RQ3: Federated Optimization Dynamics under Non-IID Drift

> **Question:** *How do proximal regularization (FedProx) and server-side adaptive momentum (FedAdam) mitigate client drift, extreme class imbalance (e.g., Switzerland's $94.7\%$ prevalence), and gradient variance compared to standard FedAvg?*

### Motivation & Background
Federated learning across clinical cohorts suffers from severe statistical heterogeneity (non-IID data). In this benchmark, Hospital 1 presents a balanced cohort ($45.9\%$ disease prevalence), Hospital 2 is outpatient-screened ($36.1\%$), and Hospital 3 is an inpatient referral cohort ($93.5\%$ disease prevalence). Under standard FedAvg, local client updates diverge from the global optimum ("client drift"), causing destabilization during server aggregation. We explore whether adding a proximal regularization term $\frac{\mu}{2} \|\theta - \theta^t\|^2$ (FedProx) or server-side adaptive moment estimation (FedAdam with $\beta_1, \beta_2$) improves convergence stability and mitigates catastrophic forgetting in minority classes.

### Formal Hypotheses
* **$H_{3a}$ (Proximal Regularization Stability):** FedProx with optimal proximal coefficient $\mu > 0$ restricts local parameter drift, resulting in lower round-to-round variance in global validation loss compared to FedAvg.
* **$H_{3b}$ (Adaptive Server Momentum):** FedAdam ($\eta_s > 0, \beta_1=0.9, \beta_2=0.99$) accelerates convergence speed in early communication rounds and produces superior calibration (lower Brier score) in the presence of extreme class imbalance.
* **$H_{3c}$ (Tail-Cohort Robustness):** Both FedProx and FedAdam improve recall on the highly skewed Hospital 3 test split relative to unregularized FedAvg without degrading Hospital 1 or 2 accuracy.

### Evaluation Protocol & Metrics
* **Metrics:** Round-by-round validation loss curves, global test ROC-AUC, Brier calibration score, Expected Calibration Error (ECE), and per-hospital sensitivity.
* **Fixed Controls:** Identical random seed ($42$), identical client local epochs ($E=3$), identical local batch size ($B=16$), identical learning rate ($\eta_l = 0.001$), identical latent dimension ($Z=32$).
* **Registry Experiment Mapping:**
  - `heterogeneous_fedavg`
  - `heterogeneous_fedprox` ($\mu = 0.01$)
  - `fedprox_mu_0001` ($\mu = 0.001$), `fedprox_mu_01` ($\mu = 0.1$)
  - `heterogeneous_fedadam` ($\eta_s = 0.01$)

---

## RQ4: Latent Dimension Capacity and Information Bottleneck

> **Question:** *What is the effect of the shared latent embedding dimension $Z \in \{8, 16, 32\}$ on representation expressiveness, diagnostic accuracy, and communication overhead in heterogeneous federated learning?*

### Motivation & Background
The latent dimension $Z$ acts as an information bottleneck between the private hospital encoders $E_{\phi_k}$ and the shared global predictor $P_\theta$. If $Z$ is chosen too small (e.g., $Z=8$), the latent projection may induce underfitting by discarding critical clinical non-linear interactions (e.g., ST depression interacting with age and fluoroscopy vessel counts). Conversely, if $Z$ is excessively large relative to sample sizes ($N_k \approx 86\text{--}212$ training records), the encoder risks overfitting and inflates the parameter count of the shared predictor, increasing communication payload.

### Formal Hypotheses
* **$H_{4a}$ (Bottleneck Threshold):** There exists a critical dimensionality threshold $Z^* \approx 16$ below which diagnostic discrimination (ROC-AUC) degrades significantly due to lossy projection of 25-dimensional clinical data.
* **$H_{4b}$ (Diminishing Returns):** Increasing $Z$ from $16$ to $32$ yields marginal performance improvements ($\le 1.5\%$ ROC-AUC) while doubling the predictor input layer parameter count.
* **$H_{4c}$ (Overfitting Tendency):** Higher latent dimensions without aggressive regularization increase the generalization gap between validation and test performance on smaller cohorts (Hospital 3).

### Evaluation Protocol & Metrics
* **Metrics:** Test ROC-AUC, Macro Accuracy, Validation Loss at convergence, Train-Val Generalization Gap, Parameter Count of Shared Predictor, and Communication Payload per round (KB).
* **Controlled Parameters:** Heterogeneous FedAvg, $E=3, B=16, \eta=0.001$, 2-layer MLP encoder architecture fixed across all runs.
* **Registry Experiment Mapping:**
  - `latent_dim_8` ($Z=8$)
  - `latent_dim_16` ($Z=16$)
  - `heterogeneous_fedavg` / `latent_dim_32` ($Z=32$, canonical baseline)

---

## RQ5: Privacy and Security Overhead Trade-offs

> **Question:** *What are the empirical utility costs (accuracy, ROC-AUC, calibration) and communication penalties incurred by composing pairwise zero-sum simulated secure aggregation ($\sum M_i = 0$) and client-update-level differential privacy (DP-FedAvg: L2 norm clipping $C$ and Gaussian noise $\sigma$ applied to communicated shared-predictor updates, tracked via Rényi DP accounting) on the shared predictor?*

### Motivation & Background
Clinical deployment of federated learning requires provable safeguards against gradient inversion attacks and membership inference attacks. We implement two complementary privacy layers:
1. **Secure Aggregation (SecAgg):** Simulated via pairwise additive zero-sum mask tensors ($M_{i,j} = -M_{j,i}$ with $\sum_{i=1}^K M_i = \mathbf{0}$) to prevent the server from inspecting individual client weight updates.
2. **Differential Privacy (DP-FedAvg):** After local training, the communicated shared-predictor parameter update is clipped to L2 norm ($C = 1.0$) and Gaussian noise ($\sigma = 0.05$) is added before aggregation. This is distinct from per-example gradient-level DP-SGD. Privacy consumption is tracked via Rényi Differential Privacy (RDP) accounting ($\delta=10^{-5}$).

While SecAgg guarantees information-theoretic privacy against an honest-but-curious server with zero utility loss, DP injects stochastic perturbations directly into the shared weights, introducing an explicit privacy-utility trade-off.

### Formal Hypotheses
* **$H_{5a}$ (SecAgg Utility Invariance):** Exact zero-sum pairwise masking preserves the aggregated global parameter vector identically ($\sum_{i} (\Delta \theta_i + M_i) = \sum_i \Delta \theta_i$), yielding exact mathematical parity in model weights, test accuracy, and ROC-AUC compared to unmasked baselines.
* **$H_{5b}$ (DP Utility Degradation):** Injecting Gaussian noise ($\sigma = 0.05, C = 1.0$) induces a modest but measurable decrease in test ROC-AUC ($\approx 1\%\text{--}4\%$) and slightly impairs probability calibration (increasing Brier score).
* **$H_{5c}$ (Optimizer Resilience under DP):** Adaptive optimizers (FedAdam) demonstrate greater resilience to DP noise perturbation than FedAvg due to second-moment gradient smoothing.
* **$H_{5d}$ (Zero Privacy Leakage on Private Encoders):** Because private encoders $E_{\phi_k}$ never transmit gradients or weights, their parameter space inherently enjoys complete institutional confinement without consuming differential privacy budget.

### Evaluation Protocol & Metrics
* **Utility Metrics:** Test Accuracy, ROC-AUC, Brier Calibration Score, ECE.
* **Privacy & Accounting:** Empirical privacy budget $(\epsilon, \delta = 10^{-5})$ computed via RDP, clipping threshold $C$, noise multiplier $\sigma$. For runs where DP is disabled, $\epsilon \equiv \text{None}$.
* **Security Checks:** Verification of zero-sum mask cancellation ($\|\sum_i M_i\|_2 < 10^{-6}$) and confirmation that simulated SecAgg assumes zero client dropout ($100\%$ round completion).
* **Registry Experiment Mapping:**
  - `heterogeneous_fedavg` (No SecAgg, No DP)
  - `heterogeneous_fedavg_secure` (SecAgg=True, DP=False)
  - `heterogeneous_fedavg_dp` (SecAgg=False, DP=True)
  - `heterogeneous_fedavg_secure_dp` (SecAgg=True, DP=True)
  - `heterogeneous_fedprox_secure_dp` (FedProx + SecAgg + DP)
  - `heterogeneous_fedadam_secure_dp` (FedAdam + SecAgg + DP)

---

## Summary Mapping Table

| Research Question | Core Focus | Key Independent Variables | Primary Benchmark Metric | Registry Experiments |
| :--- | :--- | :--- | :--- | :--- |
| **RQ1** | Federated vs Local Baselines | Training paradigm (Local vs FedAvg) | Macro Test ROC-AUC | `local_*`, `homogeneous_*`, `heterogeneous_fedavg` |
| **RQ2** | Feature Space Heterogeneity | Feature dimension handling ($D_k \to Z$) | Test ROC-AUC & Comm Bytes | `heterogeneous_fedavg`, `latent_dim_*`, `encoder_*` |
| **RQ3** | Non-IID Optimization Dynamics | Aggregation strategy (FedAvg, FedProx, FedAdam) | Convergence Loss, Brier Score | `heterogeneous_fedavg`, `heterogeneous_fedprox`, `heterogeneous_fedadam` |
| **RQ4** | Latent Dimension Capacity | Bottleneck size $Z \in \{8, 16, 32\}$ | ROC-AUC vs Predictor Size | `latent_dim_8`, `latent_dim_16`, `latent_dim_32` |
| **RQ5** | Privacy & Security Trade-offs | SecAgg (Masking) & DP ($\sigma, C$) | Accuracy Drop & RDP $\epsilon$ | `*_secure`, `*_dp`, `*_secure_dp` |
