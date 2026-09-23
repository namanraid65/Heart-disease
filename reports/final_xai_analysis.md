# Consolidated Explainable AI (XAI) Synthesis

This report provides a consolidated synthesis of LIME and SHAP feature attributions across models and hospital cohorts.

> [!NOTE]
> The explanations and predictions presented in this study represent statistical machine learning model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. This system is designed exclusively for research and educational purposes and is not a substitute for professional clinical diagnosis and medical judgment.

---

## 1. Top Feature Attributions by Model and Cohort
- **ST Depression (`oldpeak`):** Consistently identified as having the largest magnitude continuous attribution across hospital cohorts and model families.
- **Chest Pain Classification (`cp_4`):** Asymptomatic designation consistently contributed positively to risk predictions across algorithms.
- **Fluoroscopy Vessels (`ca_0`):** Contributed negatively toward disease risk predictions (protective direction) in clinical centers with angiographic testing.

*Note: Attributions quantify statistical model reliance on features; they do not establish causal disease etiology or clinical validity.*
