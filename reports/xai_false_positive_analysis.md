# Explainable AI False Positive Analysis

This report investigates representative **False Positive (FP)** predictions (cases where the patient was clinically classified as healthy, but the model produced an elevated disease probability).

> [!WARNING]
> These explanations describe machine learning model mechanics and feature attributions; they do not constitute medical justification or clinical fault analysis.

---

## 1. Representative False Positive Cases

| Hospital                 | Model                  | Case Type   | Sample ID             | Actual      | Prediction   | Probability   | Top LIME Features                              | Top SHAP Features                            | Top-5 Agreement   |
|:-------------------------|:-----------------------|:------------|:----------------------|:------------|:-------------|:--------------|:-----------------------------------------------|:---------------------------------------------|:------------------|
| Hospital 1 (Cleveland)   | Federated 1D AlexNet   | FP          | hospital_1_case_FP_31 | Healthy (0) | Disease (1)  | 53.5%         | sex, cp_4, thalach, exang, cp_3                | oldpeak, cp_4, age, exang, thalach           | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Federated 1D AlexNet   | FP          | hospital_2_case_FP_08 | Healthy (0) | Disease (1)  | 53.5%         | sex, thalach, cp_4, oldpeak, exang             | oldpeak, thalach, cp_1, sex, chol            | 60% (3/5)         |
| Hospital 3 (Switzerland) | Federated 1D AlexNet   | FP          | hospital_3_case_FP_09 | Healthy (0) | Disease (1)  | 53.7%         | thalach, sex, exang, cp_4, oldpeak             | trestbps, oldpeak, thalach, exang, cp_4      | 80% (4/5)         |
| Hospital 1 (Cleveland)   | Federated 1D ResNet    | FP          | hospital_1_case_FP_44 | Healthy (0) | Disease (1)  | 55.8%         | oldpeak, restecg_0, slope_2, trestbps, sex     | oldpeak, cp_3, sex, thal_7, cp_2             | 40% (2/5)         |
| Hospital 2 (Hungarian)   | Federated 1D ResNet    | FP          | hospital_2_case_FP_26 | Healthy (0) | Disease (1)  | 56.7%         | oldpeak, restecg_0, sex, trestbps, age         | oldpeak, restecg_0, restecg_1, cp_3, cp_2    | 40% (2/5)         |
| Hospital 3 (Switzerland) | Federated 1D ResNet    | FP          | hospital_3_case_FP_09 | Healthy (0) | Disease (1)  | 62.7%         | oldpeak, restecg_0, slope_2, trestbps, thalach | oldpeak, slope_3, slope_2, restecg_0, thal_7 | 60% (3/5)         |
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | FP          | hospital_1_case_FP_08 | Healthy (0) | Disease (1)  | 62.3%         | cp_4, oldpeak, sex, chol, ca_0                 | cp_4, ca_0, thal_3, oldpeak, slope_2         | 60% (3/5)         |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | FP          | hospital_2_case_FP_32 | Healthy (0) | Disease (1)  | 75.1%         | cp_4, chol, oldpeak, sex, slope_2              | sex, exang, chol, cp_4, restecg_1            | 60% (3/5)         |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | FP          | hospital_3_case_FP_09 | Healthy (0) | Disease (1)  | 83.2%         | cp_4, oldpeak, sex, thal_7, slope_2            | cp_3, cp_4, slope_2, restecg_0, thalach      | 40% (2/5)         |

---

## 2. Model Feature Contributions in False Positive Cases
Across the investigated false positive instances, features contributing strongly toward positive model predictions include:
- **Elevated ST Depression (`oldpeak > 1.5`):** Exercise ST depression contributed high positive weight to the risk score even in the absence of fluoroscopy findings.
- **Advanced Age (`age > 60`):** Continuous age weighting shifted baseline model scores toward positive classification.
- **Chest Pain Classification (`cp_4`):** Presence of asymptomatic designation contributed positively to predicted risk in screening cohorts.
