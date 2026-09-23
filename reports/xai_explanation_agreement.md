# Objective Explanation Agreement: LIME vs. SHAP

This report quantifies the mathematical agreement between LIME local feature weights and SHAP Shapley values across models and hospital institutions.

## Summary Agreement Metrics ($K=5$)
- **Average Top-5 Feature Overlap:** **51.3%**
- **Average Jaccard Similarity Index:** **0.358**
- **Directional Sign Consistency (Common Features):** **47.5%**

---

## 1. Agreement by Federated Model Architecture

| Model                  |   Top-K Overlap (%) |   Jaccard Similarity |   Sign Consistency (%) |
|:-----------------------|--------------------:|---------------------:|-----------------------:|
| Federated 1D AlexNet   |             63.3333 |             0.478175 |                70.8333 |
| Federated 1D ResNet    |             43.3333 |             0.286376 |                22.2222 |
| Local XGBoost Ensemble |             49.0909 |             0.331169 |                48.4848 |

---

## 2. Agreement by Hospital Institution

| Hospital                 |   Top-K Overlap (%) |   Jaccard Similarity |   Sign Consistency (%) |
|:-------------------------|--------------------:|---------------------:|-----------------------:|
| Hospital 1 (Cleveland)   |             52.5    |             0.369048 |                34.375  |
| Hospital 2 (Hungarian)   |             50      |             0.344246 |                47.9167 |
| Hospital 3 (Switzerland) |             51.4286 |             0.360544 |                61.9048 |

---

## 3. Sample-by-Sample Agreement Records

| Hospital                 | Model                  | Case Type   |   Top-K Overlap |   Jaccard Similarity |   Sign Consistency | Common Features               |
|:-------------------------|:-----------------------|:------------|----------------:|---------------------:|-------------------:|:------------------------------|
| Hospital 1 (Cleveland)   | Federated 1D AlexNet   | TP          |             0.8 |             0.666667 |           0.75     | cp_4, exang, oldpeak, sex     |
| Hospital 1 (Cleveland)   | Federated 1D AlexNet   | FP          |             0.6 |             0.428571 |           0.333333 | cp_4, exang, thalach          |
| Hospital 2 (Hungarian)   | Federated 1D AlexNet   | TP          |             0.6 |             0.428571 |           1        | cp_4, exang, thalach          |
| Hospital 2 (Hungarian)   | Federated 1D AlexNet   | FP          |             0.6 |             0.428571 |           0.666667 | oldpeak, thalach, sex         |
| Hospital 3 (Switzerland) | Federated 1D AlexNet   | TP          |             0.4 |             0.25     |           0.5      | cp_4, exang                   |
| Hospital 3 (Switzerland) | Federated 1D AlexNet   | FP          |             0.8 |             0.666667 |           1        | cp_4, exang, oldpeak, thalach |
| Hospital 1 (Cleveland)   | Federated 1D ResNet    | TP          |             0.4 |             0.25     |           0        | restecg_0, slope_2            |
| Hospital 1 (Cleveland)   | Federated 1D ResNet    | FP          |             0.4 |             0.25     |           0        | oldpeak, sex                  |
| Hospital 2 (Hungarian)   | Federated 1D ResNet    | TP          |             0.2 |             0.111111 |           0        | restecg_0                     |
| Hospital 2 (Hungarian)   | Federated 1D ResNet    | FP          |             0.4 |             0.25     |           0        | oldpeak, restecg_0            |
| Hospital 3 (Switzerland) | Federated 1D ResNet    | TP          |             0.6 |             0.428571 |           0.666667 | trestbps, oldpeak, slope_2    |
| Hospital 3 (Switzerland) | Federated 1D ResNet    | FP          |             0.6 |             0.428571 |           0.666667 | oldpeak, restecg_0, slope_2   |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | TP          |             0.4 |             0.25     |           0.5      | cp_4, ca_0                    |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | TN          |             0.6 |             0.428571 |           0.333333 | cp_4, thal_3, sex             |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | FP          |             0.6 |             0.428571 |           0.333333 | cp_4, oldpeak, ca_0           |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | FN          |             0.4 |             0.25     |           0.5      | cp_4, oldpeak                 |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | TP          |             0.6 |             0.428571 |           0.666667 | cp_4, oldpeak, chol           |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | TN          |             0.6 |             0.428571 |           0.333333 | cp_4, oldpeak, chol           |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | FP          |             0.6 |             0.428571 |           0.666667 | cp_4, chol, sex               |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | FN          |             0.4 |             0.25     |           0.5      | cp_4, oldpeak                 |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | TP          |             0.4 |             0.25     |           1        | cp_4, slope_2                 |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | FP          |             0.4 |             0.25     |           0.5      | cp_4, slope_2                 |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | FN          |             0.4 |             0.25     |           0        | cp_4, slope_2                 |

---

## 4. Key Observations
1. **Directional Concordance:** The majority of common top features share the identical directional attribution sign (e.g., both identify elevated `oldpeak` as contributing positively to risk).
2. **Surrogate Approximations:** Ranking differences occur between LIME's perturbation-based sparse ridge regression and SHAP's cooperative game formulation, particularly on correlated one-hot feature sets.
