# Discussion Chapter Draft Material: Architectural Implications & Trade-offs

## 1. Feasibility of Heterogeneous Multi-Center Federation
Traditional horizontal federated learning imposes an unrealistic requirement that all medical centers capture identical diagnostic variables. By confining feature encoders to hospital hardware, heterogeneous FL allows institutions with differing diagnostic panels to collaborate without data pooling or lossy feature truncation.

## 2. Proximal Regularization vs. Server Momentum
In clinical cohorts characterized by extreme prevalence differences (e.g. Hospital 3's 93.5% prevalence), FedProx prevents local client updates from over-specializing on local class skews. Conversely, FedAdam smooths out gradient variance, making it the preferred strategy when client updates are perturbed by differential privacy noise.

## 3. Communication Efficiency in Healthcare Networks
Reducing bandwidth from $2.75	ext{ MB}$ to $102.8	ext{ KB}$ per round is of paramount practical significance for hospitals operating over restricted intranet connections or satellite links, facilitating low-latency collaborative training.
