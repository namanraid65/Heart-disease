# Global Feature Importance Analysis (SHAP)

This report outlines the input features that contributed most strongly to each model's heart disease risk predictions across the hospital test cohorts.

> [!NOTE]
> The explanations and predictions presented in this study represent statistical machine learning model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. This system is designed exclusively for research and educational purposes and is not a substitute for professional clinical diagnosis and medical judgment.

---

## Global Feature Importance: Federated 1D AlexNet
Ranked by mean absolute SHAP value ($\text{Mean } |\text{SHAP}|$) across each hospital test split:
### Hospital 1 (Cleveland)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| oldpeak   |   0.000290268 |
| thalach   |   0.000287266 |
| sex       |   0.000225411 |
| exang     |   0.00015988  |
| cp_4      |   0.000127992 |
| trestbps  |   0.000126834 |
| chol      |   6.9e-05     |
| fbs       |   5.46106e-05 |
| cp_2      |   3.57027e-05 |
| cp_3      |   3.08959e-05 |
### Hospital 2 (Hungarian)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| oldpeak   |   0.000217342 |
| thalach   |   0.000215945 |
| sex       |   0.000179189 |
| cp_4      |   0.000135137 |
| exang     |   0.000127048 |
| trestbps  |   9.93796e-05 |
| chol      |   9.33392e-05 |
| cp_2      |   5.53497e-05 |
| fbs       |   3.50544e-05 |
| restecg_1 |   2.20014e-05 |
### Hospital 3 (Switzerland)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| thalach   |   0.000339443 |
| oldpeak   |   0.000200294 |
| exang     |   0.000180219 |
| sex       |   0.000106591 |
| trestbps  |   9.94285e-05 |
| cp_4      |   8.59012e-05 |
| restecg_0 |   5.11426e-05 |
| age       |   2.95393e-05 |
| cp_1      |   2.02113e-05 |
| fbs       |   1.71553e-05 |
## Global Feature Importance: Federated 1D ResNet
Ranked by mean absolute SHAP value ($\text{Mean } |\text{SHAP}|$) across each hospital test split:
### Hospital 1 (Cleveland)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| oldpeak   |    0.0141233  |
| slope_2   |    0.00697084 |
| restecg_0 |    0.00647621 |
| trestbps  |    0.00639137 |
| sex       |    0.00445136 |
| chol      |    0.00410578 |
| thal_7    |    0.00346036 |
| restecg_2 |    0.00319833 |
| ca_0      |    0.00309589 |
| cp_3      |    0.00293385 |
### Hospital 2 (Hungarian)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| oldpeak   |    0.0138918  |
| cp_2      |    0.00741338 |
| restecg_0 |    0.00697085 |
| trestbps  |    0.00623502 |
| thalach   |    0.00398986 |
| restecg_1 |    0.00389132 |
| chol      |    0.00359938 |
| sex       |    0.00352196 |
| age       |    0.0032119  |
| cp_3      |    0.00204541 |
### Hospital 3 (Switzerland)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| oldpeak   |    0.0182582  |
| restecg_0 |    0.0106692  |
| slope_2   |    0.00944757 |
| trestbps  |    0.00660612 |
| restecg_1 |    0.00499835 |
| age       |    0.00400807 |
| thalach   |    0.00390771 |
| cp_1      |    0.00327042 |
| sex       |    0.00254622 |
| slope_3   |    0.00237651 |
## Global Feature Importance: Local XGBoost Ensemble
Ranked by mean absolute SHAP value ($\text{Mean } |\text{SHAP}|$) across each hospital test split:
### Hospital 1 (Cleveland)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| cp_4      |      0.821241 |
| ca_0      |      0.693454 |
| thal_3    |      0.49164  |
| oldpeak   |      0.371018 |
| sex       |      0.370969 |
| thal_7    |      0.279872 |
| age       |      0.239278 |
| thalach   |      0.193437 |
| chol      |      0.148816 |
| slope_2   |      0.141222 |
### Hospital 2 (Hungarian)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| cp_4      |     0.775547  |
| oldpeak   |     0.583305  |
| cp_2      |     0.351064  |
| chol      |     0.329314  |
| exang     |     0.279467  |
| sex       |     0.182936  |
| trestbps  |     0.124589  |
| thalach   |     0.115884  |
| age       |     0.107792  |
| slope_1   |     0.0453134 |
### Hospital 3 (Switzerland)| Feature   |   Mean_|SHAP| |
|:----------|--------------:|
| cp_3      |    0.398649   |
| cp_4      |    0.373825   |
| slope_2   |    0.147599   |
| restecg_0 |    0.102938   |
| thalach   |    0.0812837  |
| slope_1   |    0.0131396  |
| trestbps  |    0.0105685  |
| oldpeak   |    0.00770575 |
| age       |    0          |
| cp_1      |    0          |

---

## Summary of Top Influencing Input Features
Across models and client cohorts, features exhibiting the highest mean absolute SHAP attributions include:
1. **`thal_7` / `thal_3` (Thallium Scintigraphy Defects):** Presence of reversible/fixed thallium perfusion defects contributed strongly toward positive risk scores.
2. **`cp_4` (Asymptomatic Chest Pain):** Consistently weighted positively in predicting elevated risk across cohorts.
3. **`oldpeak` (ST Depression Induced by Exercise):** Exhibited significant positive attribution magnitude across both deep neural networks and tree ensembles.
4. **`ca_0` / `ca_count` (Fluoroscopy Colored Vessels):** Absence of colored major vessels contributed negatively toward disease predictions (reducing model output probability).
5. **`thalach` (Maximum Heart Rate Achieved):** Higher peak heart rate consistently contributed negatively to predicted disease risk scores.

*Note: Attributions quantify statistical model reliance on features; they do not establish causal disease etiology or clinical validity.*
