| Experiment                                  | SecAgg   | DP   | Privacy Budget   | Macro Acc   | Weighted Acc   | Macro ROC-AUC   | Weighted ROC-AUC   | Macro F1        |
|:--------------------------------------------|:---------|:-----|:-----------------|:------------|:---------------|:----------------|:-------------------|:----------------|
| heterogeneous_fedavg_mlp                    | Off      | Off  | ε = nan          | 85.7 ± 1.0% | 84.4 ± 1.8%    | 0.6742 ± 0.0498 | 0.7934 ± 0.0258    | 0.8399 ± 0.0309 |
| heterogeneous_fedavg_secure_aggregation     | On       | Off  | ε = nan          | 85.7 ± 1.0% | 84.4 ± 1.8%    | 0.6742 ± 0.0498 | 0.7934 ± 0.0258    | 0.8399 ± 0.0309 |
| heterogeneous_fedavg_dp                     | Off      | On   | ε = 148.03       | 85.2 ± 0.6% | 83.8 ± 0.5%    | 0.6975 ± 0.0175 | 0.8025 ± 0.0067    | 0.8442 ± 0.0143 |
| heterogeneous_fedavg_secure_aggregation_dp  | On       | On   | ε = 148.03       | 85.2 ± 0.6% | 83.8 ± 0.5%    | 0.6975 ± 0.0175 | 0.8025 ± 0.0067    | 0.8442 ± 0.0143 |
| heterogeneous_fedprox_secure_aggregation_dp | On       | On   | ε = 148.03       | 85.1 ± 0.5% | 83.2 ± 0.5%    | 0.6977 ± 0.0169 | 0.8071 ± 0.0081    | 0.8469 ± 0.0017 |
| heterogeneous_fedadam_secure_aggregation_dp | On       | On   | ε = 148.03       | 84.6 ± 1.0% | 82.6 ± 1.8%    | 0.7283 ± 0.0221 | 0.8209 ± 0.0093    | 0.8431 ± 0.0159 |