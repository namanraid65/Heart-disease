# Master Final Experimental Results Report

## 1. Experimental Setup
This multi-center research study investigates collaborative machine learning for heart disease risk prediction across three independent hospital clients without centralizing patient records.

> [!NOTE]
> The explanations and predictions presented in this study represent statistical machine learning model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. This system is designed exclusively for research and educational purposes and is not a substitute for professional clinical diagnosis and medical judgment.

---

## 2. Dataset Distribution Across Hospital Silos
- **Hospital 1 (Cleveland, USA):** $N=303$ total records ($45.9\%$ disease prevalence, balanced research cohort).
- **Hospital 2 (Hungarian, Budapest):** $N=294$ total records ($36.1\%$ disease prevalence, outpatient screening cohort).
- **Hospital 3 (Switzerland, Zurich/Basel):** $N=123$ total records ($93.5\%$ disease prevalence, high-risk referral cohort).
- **Total Population:** $720$ clinical subjects ($503$ collaborative training records, $108$ validation records, $109$ test records).

---

## 3. Authoritative Multi-Center Benchmark Summary

| Model Type   | Training Type   | Aggregation                | Scope                       | Accuracy   | Recall   | Specificity   |   F1 Score |   ROC AUC |   PR AUC |
|:-------------|:----------------|:---------------------------|:----------------------------|:-----------|:---------|:--------------|-----------:|----------:|---------:|
| AlexNet      | Local           | Macro Average (Unweighted) | All 3 Medical Centers       | 87.08%     | 86.90%   | 56.71%        |     0.8525 |    0.7362 |   0.8958 |
| AlexNet      | Local           | Sample-Weighted Average    | All Medical Centers (N=109) | 85.32%     | 83.88%   | 70.30%        |     0.8249 |    0.8197 |   0.8833 |
| AlexNet      | Federated       | Macro Average (Unweighted) | All 3 Medical Centers       | 57.53%     | 60.95%   | 51.67%        |     0.5857 |    0.6288 |   0.8791 |
| AlexNet      | Federated       | Sample-Weighted Average    | All Medical Centers (N=109) | 68.81%     | 72.92%   | 64.04%        |     0.6811 |    0.7661 |   0.8788 |
| ResNet       | Local           | Macro Average (Unweighted) | All 3 Medical Centers       | 83.32%     | 68.65%   | 61.48%        |     0.7277 |    0.6075 |   0.8451 |
| ResNet       | Local           | Sample-Weighted Average    | All Medical Centers (N=109) | 80.73%     | 61.69%   | 76.06%        |     0.6733 |    0.7393 |   0.8362 |
| ResNet       | Federated       | Macro Average (Unweighted) | All 3 Medical Centers       | 70.90%     | 73.68%   | 51.24%        |     0.7438 |    0.6785 |   0.8442 |
| ResNet       | Federated       | Sample-Weighted Average    | All Medical Centers (N=109) | 75.23%     | 78.14%   | 63.30%        |     0.7571 |    0.7745 |   0.8232 |
| XGBoost      | Local           | Macro Average (Unweighted) | All 3 Medical Centers       | 85.36%     | 91.30%   | 53.00%        |     0.857  |    0.8118 |   0.9126 |
| XGBoost      | Local           | Sample-Weighted Average    | All Medical Centers (N=109) | 84.40%     | 90.48%   | 65.72%        |     0.8366 |    0.8665 |   0.8997 |
| XGBoost      | Federated       | Macro Average (Unweighted) | All 3 Medical Centers       | 76.31%     | 75.53%   | 57.76%        |     0.7904 |    0.8098 |   0.9213 |
| XGBoost      | Federated       | Sample-Weighted Average    | All Medical Centers (N=109) | 80.73%     | 79.11%   | 71.49%        |     0.8048 |    0.8708 |   0.9105 |

---

## 4. Hospital 3 Statistical Disclosure
The Hospital 3 test partition comprises $18$ positive cases and $1$ negative case ($N=19$, prevalence $94.7\%$). Specificity is evaluated on only one negative test instance, and performance on Hospital 3 reflects this extreme asymmetry.

---

## 5. Technical Limitations & Future Work
- Evaluated on benchmark retrospective datasets; clinical deployment requires prospective validation.
- Planned extensions: Heterogeneous feature encoders (e.g. 30-feature hospital), FedProx / FedOpt optimization, Differential Privacy (DP), and Secure Aggregation (SecAgg).
