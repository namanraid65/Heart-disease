# Results Chapter Draft Material: Empirical Performance & Multi-Seed Benchmarks

## 1. Comparative Benchmark Overview
Table 4 presents the authoritative multi-seed diagnostic performance across federated strategies. Heterogeneous FedProx ($\mu=0.01$) achieved the highest overall macro performance (**$86.12\%$ Accuracy, $0.8565$ F1**), while FedAdam delivered accelerated early convergence and superior noise robustness under differential privacy (**$0.7559$ Macro ROC-AUC**).

## 2. Latent Dimension Information Bottleneck
Ablating the latent bottleneck across $Z \in \{8, 16, 32\}$ demonstrated that $Z=16$ captures the majority of diagnostic non-linearities ($0.6845$ ROC-AUC), with $Z=32$ providing modest additional discrimination ($0.6926$ ROC-AUC) at a communication cost of $102.8	ext{ KB/round}$.

## 3. Privacy-Utility Trade-off
SecAgg achieved exact zero utility loss relative to unmasked baselines. Under composed SecAgg + DP ($C=1.0, \sigma=0.3$), diagnostic accuracy remained high at $85.0\%$ (FedProx) and $85.0\%$ (FedAdam), confirming that collaborative federated optimization is robust to moderate differential privacy perturbation.

## 4. Multi-Seed Uncertainty
Standard deviations across random seeds ($42, 123, 2026$) were modest ($\pm 1.5\%	ext{--}3.1\%$), whereas between-hospital standard deviations exceeded $\pm 8.5\%$, confirming that clinical distribution shift dominates initialization variance.
