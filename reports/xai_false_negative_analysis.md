# Explainable AI False Negative Analysis

This report analyzes representative **False Negative (FN)** predictions (cases where the patient had confirmed coronary disease, but the model produced a low disease probability).

> [!CAUTION]
> False negative predictions represent critical performance discrepancies. This analysis examines feature attributions that contributed toward the model underestimating disease risk relative to ground truth.

---

## 1. Representative False Negative Cases

| Hospital                 | Model                  | Case Type   | Sample ID             | Actual      | Prediction   | Probability   | Top LIME Features                    | Top SHAP Features                       | Top-5 Agreement   |
|:-------------------------|:-----------------------|:------------|:----------------------|:------------|:-------------|:--------------|:-------------------------------------|:----------------------------------------|:------------------|
| Hospital 1 (Cleveland)   | Local XGBoost Ensemble | FN          | hospital_1_case_FN_35 | Disease (1) | Healthy (0)  | 42.8%         | cp_4, oldpeak, sex, chol, age        | cp_4, thal_3, oldpeak, thal_7, ca_0     | 40% (2/5)         |
| Hospital 2 (Hungarian)   | Local XGBoost Ensemble | FN          | hospital_2_case_FN_18 | Disease (1) | Healthy (0)  | 45.6%         | oldpeak, cp_4, sex, slope_2, thalach | cp_4, chol, oldpeak, trestbps, exang    | 40% (2/5)         |
| Hospital 3 (Switzerland) | Local XGBoost Ensemble | FN          | hospital_3_case_FN_00 | Disease (1) | Healthy (0)  | 44.9%         | cp_4, oldpeak, sex, slope_2, exang   | cp_3, cp_4, slope_2, restecg_0, thalach | 40% (2/5)         |

---

## 2. Feature Contributions in Underestimated Risk Predictions
- **High Peak Heart Rate (`thalach > 160`):** High exercise chronotropic response produced negative (risk-reducing) attributions in the model, counterbalancing subtle ST changes.
- **Absence of Exercise Angina (`exang = 0`):** Absence of recorded exercise angina contributed toward lower predicted model probabilities.
- **Zero Fluoroscopy Vessels (`ca_0 = 1`):** In records where fluoroscopy was negative or unrecorded, the model relied heavily on `ca_0` as a negative contributor to predicted risk.
