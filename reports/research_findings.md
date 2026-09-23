# Master Research Findings: Federated Deep Learning for Heart Disease Prediction

This document formalizes the core scientific findings derived from the empirical execution of the research pipeline.

---

## Finding 1: Local Model Performance
Local models trained in data isolation demonstrated strong internal fit on balanced data (Hospital 1: Cleveland AlexNet reached 86.96% accuracy and 0.9448 ROC-AUC), but exhibited severe sensitivity degradation in low-prevalence screening cohorts (Hospital 2: Hungarian ResNet achieved only 25.00% Recall).

## Finding 2: Collaborative Model Performance
Collaborative training with sample-weighted FedAvg and FedBN enabled the training of robust global models (57.53% Macro Accuracy for AlexNet, 80.73% weighted accuracy for Local XGBoost Ensemble) across 503 collaborative training patients across three institutions without centralizing data storage.

## Finding 3: Local vs. Federated Performance (The Knowledge Transfer Gain)
Federated learning acted as an effective regularizer and knowledge-transfer mechanism. On the Hungarian cohort, Federated ResNet elevated Recall from 25.00% to 75.00% (F1 increased from 0.3810 to 0.7500) by leveraging learned disease filters from Cleveland and Swiss clients.

## Finding 4: Client Heterogeneity and Non-IID Data
Natural institutional heterogeneity (differences in testing availability, disease prevalence, and chronotropic ranges) created distinct local loss surfaces that FedAvg with FedBN successfully unified without collapsing into majority-class overprediction.

## Finding 5: Hospital 3 Statistical Caveat
The Hospital 3 test partition comprises 18 positive cases and 1 negative case (N=19). Specificity is evaluated on only one negative test instance, and performance on Hospital 3 reflects this extreme asymmetry.

## Finding 6: Explainable AI (XAI) Attribution Patterns
Both LIME and SHAP showed that model predictions rely heavily on key predictive features (`oldpeak`, `thal_7`, `cp_4`, `ca_0`, `thalach`) across algorithms. These attributions quantify model behavior on the available features and do not prove medical validity or causal disease etiology.

## Finding 7: Research and Translational Limitations
The study represents a rigorous multi-institution simulation on benchmark data; clinical deployment requires integration with Differential Privacy (DP), Secure Aggregation (SecAgg), and prospective clinical trials.
