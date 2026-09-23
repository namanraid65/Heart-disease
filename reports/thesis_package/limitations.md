# Comprehensive Methodological and Clinical Limitations

This document provides a rigorous, transparent account of the boundaries, constraints, and limitations of the multi-institutional federated learning framework.

---

## 1. Dataset & Clinical Cohort Boundaries

1. **Sample Size Limitations:** The experimental evidence is derived from the UCI Heart Disease benchmark comprising $N=719$ patient records across three medical institutions (Cleveland: 303, Hungarian: 293, Switzerland: 123). While widely recognized as a foundational clinical machine learning benchmark, this cohort size is modest compared to modern multi-center electronic health record (EHR) registries containing tens of thousands of admissions.
2. **Extreme Class Imbalance in Hospital 3:** The Swiss test split contains $18$ positive cases and exactly $1$ negative case ($N=19$, prevalence $94.7\%$). In this cohort, predicting a positive risk for all 19 patients yields a Sensitivity of $100.0\%$, but forces Specificity to $0.0\%$. Consequently, specificity on Hospital 3 cannot be interpreted as a reliable estimate of true negative classification capability.
3. **Simulated Federated Environment:** Hospital silos are simulated on a single compute host via partitioned data loaders rather than deployed across physically distributed institutional networks with asynchronous internet latency, packet loss, and firewall constraints.

---

## 2. Architectural & Representation Boundaries

1. **Heterogeneous Feature Construction:** While the modular composite architecture ($E_{\phi_k} \circ P_	heta$) successfully demonstrates mathematical alignment from native feature dimensions ($D_k 	o Z=32$), the current 25-feature UCI inputs were derived from common source protocols. Real-world hospital heterogeneity often involves fundamentally different clinical modalities (e.g. EHR tabular records vs. 12-lead ECG waveforms vs. echocardiogram imaging).
2. **Latent Alignment Assumption:** The shared global predictor assumes that local client gradient updates will naturally guide private encoders to project semantically consistent concepts into the shared latent space $Z$. In the absence of contrastive loss regularization or multi-site alignment anchors, latent representations may experience representation drift across communication rounds.

---

## 3. Privacy & Security Assumptions

1. **Pairwise Zero-Sum Masking vs. Production Cryptography:** Secure aggregation is simulated via deterministic additive zero-sum mask tensors ($\sum_{i=1}^K M_i = \mathbf{0}$). This simulation mathematically models honest-but-curious server update confidentiality, but operates under a **$100\%$ client completion assumption**. Real-world deployment requires threshold Shamir secret sharing (e.g., Bonawitz et al.) to recover from asynchronous client dropouts.
2. **Differential Privacy Budget Magnitude:** The client-side DP implementation utilizes Gaussian noise multiplier $\sigma=0.3$ and clipping $C=1.0$, resulting in an accumulated privacy loss of $\epsilon = 148.03$ over 15 rounds ($\delta = 10^{-5}$). While providing measurable utility-privacy trade-off benchmarks, this budget is higher than theoretical sub-unity privacy regimes ($\epsilon \le 1.0$) typically demanded for strict cryptographic indistinguishability.
3. **Parameter Boundary Confinement:** Differential privacy noise is applied strictly to the shared global predictor. Private client encoders remain local and are never noised; this relies on the fundamental guarantee that private encoder weights and gradients never traverse the network.

---

## 4. Explainable AI & Causal Boundaries

1. **Correlation vs. Causation:** Local surrogate attributions (LIME) and Shapley coalition values (KernelSHAP) reflect internal model feature weighting and must NOT be interpreted as establishing clinical etiology or biological causation.
2. **Surrogate Approximation Error:** LIME fits local linear approximations around perturbed samples, which may introduce sampling variance across random seeds. KernelSHAP relies on background reference samples that may not capture complex physiological feature correlations.
