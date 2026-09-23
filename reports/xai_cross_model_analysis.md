# Cross-Model Explanation Consistency Analysis

This report compares how **Federated 1D AlexNet**, **Federated 1D ResNet**, and **Local XGBoost Ensemble** distribute feature importance when predicting heart disease risk.

---

## 1. Comparative Feature Attribution Profiles

| Feature Category | Federated 1D AlexNet | Federated 1D ResNet | Local XGBoost Ensemble |
|:---|:---|:---|:---|
| **Perfusion Markers (`thal_7`, `thal_3`)** | High Impact | High Impact | High Impact |
| **Vessel Fluoroscopy (`ca_0`, `ca_count`)** | Moderate-High | Moderate-High | Very High (Top Tree Split) |
| **Exercise ST Depression (`oldpeak`)** | Very High | Very High | Very High |
| **Chest Pain Type (`cp_4`, `cp_3`)** | High Impact | High Impact | High Impact |
| **Max Heart Rate (`thalach`)** | Moderate (Linear/Conv) | Moderate (Residual) | High (Non-linear Threshold) |

---

## 2. Key Model Behavior Distinctions
1. **Tree Ensembles vs. Deep Neural Networks:** Local XGBoost trees evaluate non-linear split points on continuous variables (`oldpeak`, `thalach`), whereas 1D AlexNet and 1D ResNet produce smooth continuous attribution slopes.
2. **Residual Skip Connections in ResNet:** 1D ResNet exhibits more distributed attributions across secondary clinical markers (`slope_2`, `sex`, `trestbps`), reflecting deeper multi-layer representations.
3. **Consistency on Primary Drivers:** All three model paradigms independently place high attribution on ST depression, thallium defects, and chest pain type as dominant predictive features.
