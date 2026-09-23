# Methodology Chapter Draft Material: Multi-Institutional Federated Clinical Machine Learning

## 1. Clinical Cohort Partitioning & Isolation Firewall
The experimental framework evaluates $N=719$ clinical records across three international hospital silos: Cleveland Clinic ($N=303$), Hungarian Institute of Cardiology ($N=293$), and University Hospital Zurich/Basel ($N=123$). Each center is partitioned into stratified $70\%$ training, $15\%$ validation, and $15\%$ test subsets ($N=503, 107, 109$). To strictly prevent data leakage:
- Preprocessing imputers and standardizers are fitted strictly on local institutional training splits.
- Early stopping and model selection operate exclusively on validation split ROC-AUC.
- The frozen test split is evaluated exactly once at the conclusion of training.

## 2. Modular Heterogeneous Composite Architecture
To support hospitals with varying native feature spaces ($D_1, D_2, D_3$), the framework decouples client-specific representation learning from global classification:
- **Private Hospital Encoders ($E_{\phi_k}: \mathbb{R}^{D_k} 	o \mathbb{R}^Z$):** 2-layer MLPs ($D_k 	o 64 	o 32$) with LeakyReLU activations and LayerNorm that remain strictly client-local.
- **Shared Global Predictor ($P_	heta: \mathbb{R}^Z 	o [0, 1]$):** A 2-layer classification head ($32 	o 32 	o 1$) that is collaboratively federated across centers.

## 3. Federated Optimization Algorithms
- **Heterogeneous FedAvg:** Sample-weighted parameter averaging over shared predictor weights.
- **Heterogeneous FedProx:** Local objective modified by a proximal anchor: $\mathcal{L}_{	ext{prox}} = \mathcal{L}_{	ext{task}} + rac{\mu}{2} \|	heta - 	heta^t\|^2$ applied strictly to shared predictor parameters.
- **Heterogeneous FedAdam:** Server updates shared weights using adaptive first and second pseudo-gradient moments: $	heta^{t+1} = 	heta^t + \eta rac{m_t}{\sqrt{v_t} + 	au}$.

## 4. Privacy & Security Mechanisms
- **Pairwise Zero-Sum Masking (SecAgg):** Deterministic pairwise additive masks ($M_{ij} = -M_{ji}$) satisfying $\sum_{i=1}^K M_i = \mathbf{0}$, ensuring the server reconstructs only the sum of updates without observing individual client parameters.
- **Client-Update-Level Differential Privacy (DP-FedAvg):** After local training completes, the communicated shared-predictor parameter update vector is clipped to a maximum L2 norm $C=1.0$ and perturbed with calibrated Gaussian noise $\mathcal{N}(0, (\sigma \cdot C)^2 I)$ with $\sigma=0.3$ before federated aggregation. This is distinct from per-example gradient-level DP-SGD (Abadi et al., 2016). Privacy consumption is tracked via a Rényi Differential Privacy (RDP) accountant ($\delta=10^{-5}$); the accountant does not apply subsampling amplification and is therefore conservative (epsilon overestimated).
