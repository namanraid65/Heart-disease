# Final Master Performance and Statistical Variation Table

This table consolidates the final empirical performance metrics and cross-institution dataset variations across all model configurations, derived directly from `reports/master_results.csv`.

---

## 1. Multi-Center Aggregate Summary (Macro vs. Sample-Weighted)

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

## 2. Institutional Cohort Breakdown
For detailed per-hospital metrics and confusion matrices, consult [`reports/hospital_wise_comparison.md`](hospital_wise_comparison.md) and [`reports/master_results.csv`](master_results.csv).
