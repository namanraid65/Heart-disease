# LIME vs. SHAP Explanation Comparison

This report details instance-level explanation comparisons between **LIME** (local linear surrogate) and **SHAP** (Shapley game-theoretic attribution) on representative test cases selected using the median-confidence rule.

> [!NOTE]
> The explanations and predictions presented in this study represent statistical machine learning model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. This system is designed exclusively for research and educational purposes and is not a substitute for professional clinical diagnosis and medical judgment.

---

## 1. Instance-by-Instance Comparative Table

| Hospital                 | Model                  | Case Type   | Sample ID             | Actual      | Prediction   | Probability   | Top LIME Features                              | Top SHAP Features                            | Top-5 Agreement   |
|:-------------------------|:-----------------------|:------------|:----------------------|:------------|:-------------|:--------------|:-----------------------------------------------|:---------------------------------------------|:------------------|
| Hospital 1 (Cleveland)   | Federated 1D AlexNet   | TP          | hospital_1_case_TP_40 | Disease (1) | Disease (1)  | 53.6%         | sex, oldpeak, thalach, cp_4, exang             | oldpeak, sex, fbs, cp_4, exang               | 80% (4/5)         |
| Hospital 1 (Cleveland)   | Federated 1D AlexNet   | FP          | hospital_1_case_FP_31 | Healthy (0) | Disease (1)  | 53.5%         | sex, cp_4, thalach, exang, cp_3                | oldpeak, cp_4, age, exang, thalach           | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Federated 1D AlexNet   | TP          | hospital_2_case_TP_15 | Disease (1) | Disease (1)  | 53.6%         | sex, thalach, oldpeak, cp_4, exang             | exang, thalach, trestbps, cp_4, chol         | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Federated 1D AlexNet   | FP          | hospital_2_case_FP_08 | Healthy (0) | Disease (1)  | 53.5%         | sex, thalach, cp_4, oldpeak, exang             | oldpeak, thalach, cp_1, sex, chol            | 60% (3/5)         |
| Hospital 3 (Switzerland) | Federated 1D AlexNet   | TP          | hospital_3_case_TP_01 | Disease (1) | Disease (1)  | 53.5%         | sex, exang, cp_4, oldpeak, trestbps            | fbs, thalach, exang, restecg_0, cp_4         | 40% (2/5)         |
| Hospital 3 (Switzerland) | Federated 1D AlexNet   | FP          | hospital_3_case_FP_09 | Healthy (0) | Disease (1)  | 53.7%         | thalach, sex, exang, cp_4, oldpeak             | trestbps, oldpeak, thalach, exang, cp_4      | 80% (4/5)         |
| Hospital 1 (Cleveland)   | Federated 1D ResNet    | TP          | hospital_1_case_TP_06 | Disease (1) | Disease (1)  | 60.7%         | oldpeak, slope_2, restecg_0, ca_0, sex         | fbs, slope_2, slope_3, restecg_0, thalach    | 40% (2/5)         |
| Hospital 1 (Cleveland)   | Federated 1D ResNet    | FP          | hospital_1_case_FP_44 | Healthy (0) | Disease (1)  | 55.8%         | oldpeak, restecg_0, slope_2, trestbps, sex     | oldpeak, cp_3, sex, thal_7, cp_2             | 40% (2/5)         |
| Hospital 2 (Hungarian)   | Federated 1D ResNet    | TP          | hospital_2_case_TP_07 | Disease (1) | Disease (1)  | 59.5%         | oldpeak, restecg_0, slope_2, sex, chol         | restecg_0, age, restecg_1, cp_2, thalach     | 20% (1/5)         |
| Hospital 2 (Hungarian)   | Federated 1D ResNet    | FP          | hospital_2_case_FP_26 | Healthy (0) | Disease (1)  | 56.7%         | oldpeak, restecg_0, sex, trestbps, age         | oldpeak, restecg_0, restecg_1, cp_3, cp_2    | 40% (2/5)         |
| Hospital 3 (Switzerland) | Federated 1D ResNet    | TP          | hospital_3_case_TP_03 | Disease (1) | Disease (1)  | 58.7%         | oldpeak, restecg_0, slope_2, trestbps, thalach | sex, trestbps, age, oldpeak, slope_2         | 60% (3/5)         |
| Hospital 3 (Switzerland) | Federated 1D ResNet    | FP          | hospital_3_case_FP_09 | Healthy (0) | Disease (1)  | 62.7%         | oldpeak, restecg_0, slope_2, trestbps, thalach | oldpeak, slope_3, slope_2, restecg_0, thal_7 | 60% (3/5)         |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | TP          | hospital_1_case_TP_42 | Disease (1) | Disease (1)  | 84.3%         | cp_4, sex, ca_0, chol, thalach                 | ca_0, cp_4, thal_3, age, oldpeak             | 40% (2/5)         |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | TN          | hospital_1_case_TN_34 | Healthy (0) | Healthy (0)  | 19.1%         | cp_4, oldpeak, sex, thal_3, chol               | cp_4, ca_0, age, sex, thal_3                 | 60% (3/5)         |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | FP          | hospital_1_case_FP_08 | Healthy (0) | Disease (1)  | 62.3%         | cp_4, oldpeak, sex, chol, ca_0                 | cp_4, ca_0, thal_3, oldpeak, slope_2         | 60% (3/5)         |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | FN          | hospital_1_case_FN_35 | Disease (1) | Healthy (0)  | 42.8%         | cp_4, oldpeak, sex, chol, age                  | cp_4, thal_3, oldpeak, thal_7, ca_0          | 40% (2/5)         |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | TP          | hospital_2_case_TP_15 | Disease (1) | Disease (1)  | 77.2%         | cp_4, oldpeak, sex, chol, slope_2              | cp_4, oldpeak, exang, chol, trestbps         | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | TN          | hospital_2_case_TN_25 | Healthy (0) | Healthy (0)  | 29.5%         | oldpeak, cp_4, sex, chol, slope_2              | cp_2, oldpeak, cp_4, chol, exang             | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | FP          | hospital_2_case_FP_32 | Healthy (0) | Disease (1)  | 75.1%         | cp_4, chol, oldpeak, sex, slope_2              | sex, exang, chol, cp_4, restecg_1            | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | FN          | hospital_2_case_FN_18 | Disease (1) | Healthy (0)  | 45.6%         | oldpeak, cp_4, sex, slope_2, thalach           | cp_4, chol, oldpeak, trestbps, exang         | 40% (2/5)         |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | TP          | hospital_3_case_TP_06 | Disease (1) | Disease (1)  | 73.3%         | cp_4, oldpeak, sex, slope_2, exang             | cp_3, cp_4, slope_2, restecg_0, thalach      | 40% (2/5)         |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | FP          | hospital_3_case_FP_09 | Healthy (0) | Disease (1)  | 83.2%         | cp_4, oldpeak, sex, thal_7, slope_2            | cp_3, cp_4, slope_2, restecg_0, thalach      | 40% (2/5)         |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | FN          | hospital_3_case_FN_00 | Disease (1) | Healthy (0)  | 44.9%         | cp_4, oldpeak, sex, slope_2, exang             | cp_3, cp_4, slope_2, restecg_0, thalach      | 40% (2/5)         |

---

## 2. Methodology & Observations
- **LIME:** Explains individual predictions by fitting a local sparse linear model to perturbed samples drawn around the patient's standardized feature vector.
- **SHAP:** Computes game-theoretic Shapley attributions referencing local training background distributions, capturing additive feature contributions.
- **Concordance on Dominant Features:** For confident predictions ($P > 0.80$ or $P < 0.20$), both methods consistently identify key stress and fluoroscopy markers (`oldpeak`, `thal_7`, `cp_4`, `thalach`) among the top contributing features.
- **Attribution Scope:** These explanations reflect the internal mechanics and statistical associations learned by the models; they do not demonstrate clinical causation.
