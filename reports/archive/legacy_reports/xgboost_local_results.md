# Local XGBoost Results

## 1. Model Configuration

An independent Extreme Gradient Boosting (XGBoost) classifier was implemented for each hospital client partition to establish a decision tree ensemble benchmark.

### Hyperparameters & Settings
| Parameter | Configured Value | Description |
| :--- | :--- | :--- |
| **Algorithm** | `XGBClassifier` | Gradient-boosted decision trees |
| **Number of Estimators ($M$)** | 100 | Maximum boosting iterations |
| **Max Tree Depth** | 4 | Controls model complexity & interaction depth |
| **Learning Rate ($\eta$)** | $0.05$ | Shrinkage step size |
| **Subsample Ratio** | $0.80$ | Row subsampling per tree |
| **Column Subsample (`colsample_bytree`)**| $0.80$ | Feature subsampling per tree |
| **Regularization ($\alpha, \lambda$)** | $\alpha = 0.01, \lambda = 1.0$ | $L_1$ and $L_2$ leaf weight regularization |
| **Objective / Loss** | `binary:logistic` (`logloss`) | Binary cross-entropy |
| **Early Stopping** | 15 rounds | Evaluated strictly on validation split |
| **Class Balancing (`scale_pos_weight`)**| Client-specific ($\frac{N_{\text{neg}}}{N_{\text{pos}}}$) | H1: $1.1856$, H2: $1.7703$, H3: $0.0750$ |
| **Random Seed** | 42 | Full reproducibility |

---

## 2. Hospital 1 Results (Cleveland Clinic Foundation)

### Evaluation on Held-Out Test Set ($N=46$)
- **Best Boosting Iteration:** 97
- **Accuracy:** **$84.78\%$** ($39 / 46$ correct predictions)
- **Precision:** **$81.82\%$**
- **Recall (Sensitivity):** **$85.71\%$** ($18 / 21$ positive disease cases detected, only **3 False Negatives**)
- **Specificity:** **$84.00\%$** ($21 / 25$ healthy patients correctly ruled out)
- **F1-Score:** **$0.8372$**
- **ROC-AUC:** **$0.9390$** (Excellent ranking & probability calibration)
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 18
  - False Positives (FP): 4
  - True Negatives (TN): 21
  - False Negatives (FN): 3

---

## 3. Hospital 2 Results (Hungarian Institute of Cardiology)

### Evaluation on Held-Out Test Set ($N=44$)
- **Best Boosting Iteration:** 63
- **Accuracy:** **$81.82\%$** ($36 / 44$ correct predictions)
- **Precision:** **$68.18\%$**
- **Recall (Sensitivity):** **$93.75\%$** ($15 / 16$ positive disease cases detected, only **1 False Negative**)
- **Specificity:** **$75.00\%$** ($21 / 28$ healthy patients correctly ruled out)
- **F1-Score:** **$0.7895$**
- **ROC-AUC:** **$0.9129$**
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 15
  - False Positives (FP): 7
  - True Negatives (TN): 21
  - False Negatives (FN): 1

---

## 4. Hospital 3 Results (Switzerland University Hospital)

### Evaluation on Held-Out Test Set ($N=19$)
- **Best Boosting Iteration:** 77
- **Accuracy:** **$89.47\%$** ($17 / 19$ correct predictions)
- **Precision:** **$94.44\%$**
- **Recall (Sensitivity):** **$94.44\%$** ($17 / 18$ positive disease cases detected)
- **Specificity:** **$0.00\%$** ($0 / 1$ healthy patient correctly classified; 1 False Positive)
- **F1-Score:** **$0.9444$**
- **ROC-AUC:** **$0.5833$**
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 17
  - False Positives (FP): 1
  - True Negatives (TN): 0
  - False Negatives (FN): 1

---

## 5. Performance Comparison

All metrics evaluated on held-out test splits:

| Hospital Client | Test Samples ($N$) | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score | ROC-AUC | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital 1 (Cleveland)** | **46** | **84.78%** | **81.82%** | **85.71%** | **84.00%** | **0.8372** | **0.9390** | 18 / 4 / 21 / 3 |
| **Hospital 2 (Hungarian)** | **44** | **81.82%** | **68.18%** | **93.75%** | **75.00%** | **0.7895** | **0.9129** | 15 / 7 / 21 / 1 |
| **Hospital 3 (Switzerland)** | **19** | **89.47%** | **94.44%** | **94.44%** | **0.00%** | **0.9444** | **0.5833** | 17 / 1 / 0 / 1 |

---

## 6. Confusion Matrix Analysis

1. **High Diagnostic Sensitivity:**
   - **Hospital 1:** 3 False Negatives out of 21 true positives ($14.3\%$ miss rate).
   - **Hospital 2:** Only 1 False Negative out of 16 true positives ($6.25\%$ miss rate) — XGBoost's class weighting (`scale_pos_weight=1.77`) prevented the sensitivity collapse observed in 1D ResNet.
   - **Hospital 3:** 1 False Negative out of 18 true positives ($5.56\%$ miss rate).
2. **False Positives & Specificity:**
   - **Hospital 1:** 4 False Positives ($84.00\%$ Specificity).
   - **Hospital 2:** 7 False Positives ($75.00\%$ Specificity).
   - **Hospital 3:** 1 False Positive on the single negative test instance.

---

## 7. Feature Importance Analysis

Feature importances derived from XGBoost's tree gain splits highlight the primary diagnostic drivers across clients:

### Top Feature Drivers
- **Hospital 1 (Cleveland):** `thal_7` (Reversible thallium defect, $0.231$), `ca_0` (Zero major vessels, $0.147$), `cp_4` (Asymptomatic chest pain, $0.118$), `thalach` (Max heart rate, $0.076$), `oldpeak` (ST depression, $0.063$).
- **Hospital 2 (Hungarian):** `oldpeak` ($0.198$), `exang` ($0.154$), `cp_4` ($0.129$), `sex` ($0.087$), `thalach` ($0.078$).
- **Hospital 3 (Switzerland):** `oldpeak` ($0.264$), `exang` ($0.182$), `cp_4` ($0.141$), `trestbps` ($0.103$), `thalach` ($0.095$).

*Observation:* When invasive procedural tests (`ca`, `thal`) are unavailable (as in European centers H2 & H3), XGBoost automatically shifts its split reliance to exercise stress markers (`oldpeak`, `exang`, `cp_4`, `thalach`).

---

## 8. Observations

1. **Robust Handling of Class Asymmetry:**
   - By weighting positive gradients with `scale_pos_weight = 1.77`, local XGBoost on Hospital 2 achieved **$93.75\%$** Recall and **$0.9129$** ROC-AUC, demonstrating superior tabular decision-boundary handling on small imbalanced datasets.
2. **Superior Tree Calibration on Client 3:**
   - On Hospital 3, XGBoost yielded an ROC-AUC of **$0.5833$**, outperforming both 1D AlexNet ($0.3889$) and 1D ResNet ($0.0556$).

---

## 9. Limitations

1. **Local Data Isolation:**
   - Similar to neural network counterparts, local XGBoost cannot generate healthy-patient decision trees on Hospital 3 because only 6 healthy training cases were present locally.
2. **Decentralized Ensembling Constraints:**
   - While decision trees excel locally, horizontal federated gradient aggregation across non-IID clients requires specialized tree architectures (e.g. Federated Gradient Boosted Decision Trees / Secure Tree Ensembles) compared to the direct tensor averaging available in PyTorch neural networks.
