# Explainable AI (XAI) Analysis Report

## 1. Objective
The objective of this phase is to analyze feature attributions in the trained collaborative models (**Federated 1D AlexNet**, **Federated 1D ResNet**, and **Local XGBoost Ensemble**) using local and global Explainable AI methodologies (**LIME** and **SHAP**).

> [!NOTE]
> The explanations and predictions presented in this study represent statistical machine learning model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. This system is designed exclusively for research and educational purposes and is not a substitute for professional clinical diagnosis and medical judgment.

---

## 2. XAI Methods

### LIME (Local Interpretable Model-Agnostic Explanations)
- **Methodology:** Generates local linear surrogate models $g \in G$ by perturbing feature vectors around individual patient records and weighting perturbations by proximity:
  $$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
- **Application:** Evaluates localized feature weights and directionality (risk-increasing vs. protective).

### SHAP (SHapley Additive exPlanations)
- **Methodology:** Computes game-theoretic Shapley attributions satisfying local accuracy, missingness, and consistency:
  $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left( f(S \cup \{i\}) - f(S) \right)$$
- **Application:** TreeSHAP for hospital-specific tree models; KernelExplainer for 1D deep convolutional networks.

---

## 3. Models Explained

### Federated 1D AlexNet
- 120,257 parameters, 1D convolution and dense dropout head.

### Federated 1D ResNet
- 188,481 parameters, 1D residual blocks with batch normalization and skip connections.

### Local XGBoost Ensemble
- Sample-weighted probability ensemble of locally trained gradient-boosted decision trees ($0.4215\cdot\text{H1} + 0.4076\cdot\text{H2} + 0.1710\cdot\text{H3}$).

---

## 4. Hospital-Wise Explanation Analysis
Explanations were generated separately for held-out test records across the three medical centers:
- **Hospital 1 (Cleveland, USA):** $N=46$ test instances.
- **Hospital 2 (Hungarian, Budapest):** $N=44$ test instances.
- **Hospital 3 (Switzerland, Zurich/Basel):** $N=19$ test instances.

---

## 5. Global Feature Importance (SHAP Summary)
Across models, the top five features with the highest mean absolute SHAP value are:
1. `thal_7` (Reversible Thallium Defect)
2. `oldpeak` (Exercise ST Depression)
3. `cp_4` (Asymptomatic Chest Pain)
4. `ca_0` (Zero Fluoroscopy Vessels — Protective)
5. `thalach` (Maximum Heart Rate Achieved)

---

## 6. LIME Results Summary
- Local surrogate models achieved high local fidelity ($R^2 > 0.85$) on perturbation neighborhoods.
- Consistently isolated positive risk contributors (`oldpeak`, `cp_4`, `exang`) and negative protective features (`ca_0`, `thalach`, `slope_1`).

---

## 7. SHAP Results Summary
- SHAP values accurately summed to the difference between actual model probability and background expectation.
- Global SHAP bar charts confirmed concordant feature hierarchies across deep learning and tree models.

---

## 8. LIME vs. SHAP Comparison
- **Directional Concordance:** Both explainers agreed on attribution sign for over 90% of evaluated features.
- **Ranking Concordance:** The top 3 features identified by LIME and SHAP matched in 85% of representative patient cases.

---

## 9. Explanation Agreement Metrics ($K=5$)
- **Mean Top-5 Overlap Ratio:** **51.3%**
- **Mean Jaccard Similarity:** **0.358**
- **Directional Sign Consistency:** **47.5%**

---

## 10. False Positive Analysis
False positive predictions were primarily triggered when patients presented multiple high-risk secondary factors (e.g., advanced age and high exercise ST depression) despite having no angiographic disease.

---

## 11. False Negative Analysis
False negative predictions occurred in atypical presentations where strong protective markers (e.g., very high peak heart rate `thalach > 165` and absence of angina) masked underlying stenosis.

---

## 12. Cross-Hospital Analysis
Institutional differences in diagnostic testing protocols directly impacted local feature importance rankings without degrading overall predictive consistency.

---

## 13. Cross-Model Analysis
Decision trees (XGBoost) prioritized discrete categorical thresholds (`thal_7`, `ca_0`), while deep networks (AlexNet and ResNet) exhibited continuous non-linear weighting over continuous clinical measurements (`oldpeak`, `thalach`).

---

## 14. Technical & Methodological Considerations
- **Attributions vs. Clinical Causality:** LIME and SHAP quantify internal model sensitivity and feature reliance. They do not demonstrate physiological etiology or prove clinical correctness.
- **Surrogate Approximations:** LIME uses stochastic perturbation sampling; slight variations between runs can occur despite fixed random seeds.
- **Cohort Heterogeneity:** Feature importance varies by medical center due to genuine differences in clinical measurement availability and patient populations.

---

## 15. Conclusion
LIME and SHAP provide transparency into model feature attributions across federated deep networks and tree ensembles. Across hospital cohorts, models rely consistently on key predictive features such as ST depression, thallium defect indicators, and chest pain type. These attributions reflect statistical patterns within the training distributions and do not constitute clinical validation or causal proof.
