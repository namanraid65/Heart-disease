# Preprocessing Report: Federated Heart Disease Prediction

## 1. Common Feature Schema

A unified, mathematically consistent schema was constructed to align all three hospital clients into an identical feature space without losing clinical semantics or merging patient records.

### Unified 25-Feature Layout (Fixed Schema Order)
| Index | Feature Column Name | Feature Group | Description & Range / Values |
| :---: | :--- | :--- | :--- |
| 1 | `age` | Continuous (Standardized) | Patient age in years ($Z$-score standardized) |
| 2 | `trestbps` | Continuous (Standardized) | Resting blood pressure in mm Hg ($Z$-score standardized) |
| 3 | `chol` | Continuous (Standardized) | Serum cholesterol in mg/dl ($Z$-score standardized) |
| 4 | `thalach` | Continuous (Standardized) | Maximum heart rate achieved ($Z$-score standardized) |
| 5 | `oldpeak` | Continuous (Standardized) | ST depression induced by exercise ($Z$-score standardized) |
| 6 | `sex` | Binary Indicator | Biological sex (`1.0` = Male, `0.0` = Female) |
| 7 | `fbs` | Binary Indicator | Fasting blood sugar > 120 mg/dl (`1.0` = True, `0.0` = False) |
| 8 | `exang` | Binary Indicator | Exercise induced angina (`1.0` = Yes, `0.0` = No) |
| 9 | `cp_1` | One-Hot (Chest Pain) | Typical angina (`1.0` if active, else `0.0`) |
| 10 | `cp_2` | One-Hot (Chest Pain) | Atypical angina (`1.0` if active, else `0.0`) |
| 11 | `cp_3` | One-Hot (Chest Pain) | Non-anginal pain (`1.0` if active, else `0.0`) |
| 12 | `cp_4` | One-Hot (Chest Pain) | Asymptomatic (`1.0` if active, else `0.0`) |
| 13 | `restecg_0` | One-Hot (Resting ECG) | Normal resting ECG (`1.0` if active, else `0.0`) |
| 14 | `restecg_1` | One-Hot (Resting ECG) | ST-T wave abnormality (`1.0` if active, else `0.0`) |
| 15 | `restecg_2` | One-Hot (Resting ECG) | Left ventricular hypertrophy (`1.0` if active, else `0.0`) |
| 16 | `slope_1` | One-Hot (ST Slope) | Upsloping peak exercise ST segment (`1.0` if active, else `0.0`) |
| 17 | `slope_2` | One-Hot (ST Slope) | Flat peak exercise ST segment (`1.0` if active, else `0.0`) |
| 18 | `slope_3` | One-Hot (ST Slope) | Downsloping peak exercise ST segment (`1.0` if active, else `0.0`) |
| 19 | `ca_0` | One-Hot (Fluoroscopy) | 0 major vessels colored (`1.0` if active, else `0.0`) |
| 20 | `ca_1` | One-Hot (Fluoroscopy) | 1 major vessel colored (`1.0` if active, else `0.0`) |
| 21 | `ca_2` | One-Hot (Fluoroscopy) | 2 major vessels colored (`1.0` if active, else `0.0`) |
| 22 | `ca_3` | One-Hot (Fluoroscopy) | 3 major vessels colored (`1.0` if active, else `0.0`) |
| 23 | `thal_3` | One-Hot (Thallium Scan) | Normal thallium scintigraphy (`1.0` if active, else `0.0`) |
| 24 | `thal_6` | One-Hot (Thallium Scan) | Fixed defect (`1.0` if active, else `0.0`) |
| 25 | `thal_7` | One-Hot (Thallium Scan) | Reversible defect (`1.0` if active, else `0.0`) |

---

## 2. Hospital-wise Data Cleaning

Each client's dataset is cleaned locally prior to partitioning:

### Hospital 1 (Cleveland)
- **Raw Count:** 303 records.
- **Cleaning Applied:** Replaced missing marker `?` with `NaN`. Verified values within physiological ranges.
- **Post-Cleaning Count:** 303 records.

### Hospital 2 (Hungarian)
- **Raw Count:** 294 records.
- **Cleaning Applied:** Replaced missing marker `?` with `NaN`. Identified and removed 1 exact duplicate record pair (row index 101 and 102).
- **Post-Cleaning Count:** 293 unique records.

### Hospital 3 (Switzerland)
- **Raw Count:** 123 records.
- **Cleaning Applied:** 
  1. Converted `chol == 0` to `NaN` (all 123 records had 0 mg/dl because serum cholesterol was unrecorded).
  2. Rectified 11 negative `oldpeak` values by clipping lower bound to `0.0` (as ST depression is mathematically non-negative; negative values reflect ST segment elevation).
- **Post-Cleaning Count:** 123 records.

---

## 3. Missing Value Handling

To strictly prevent data leakage across client folds and between train/validation/test splits:
1. **Training Partition Fitting:** Imputation statistics are computed strictly on the local training split (`X_train`) of each client.
   - **Continuous Features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):** Imputed with the median of `X_train`.
   - **Categorical & Binary Features (`sex`, `fbs`, `exang`, `cp`, `restecg`, `slope`, `ca`, `thal`):** Imputed with the mode (most frequent value) of `X_train`.
2. **Clinical Reference Fallback:** In cases where a client training split contains 100% missing values for a diagnostic test (e.g. unrecorded `chol` in Switzerland or sparse `ca` in Hungarian), the imputer falls back to established medical reference medians (e.g., standard baseline `chol = 240.0`, `ca = 0.0`, `thal = 3.0`, `slope = 2.0`).
3. **Validation and Test Application:** The exact numerical values learned during the training step are applied directly to transform `X_val` and `X_test`.

---

## 4. Categorical Feature Encoding

Categorical features are encoded to avoid imposing artificial ordinal relationships on nominal variables:
- **Binary Features (`sex`, `fbs`, `exang`):** Retained as binary indicator floats (`0.0` or `1.0`).
- **Multi-Class Nominal Features (`cp`, `restecg`, `slope`, `ca`, `thal`):** Encoded using One-Hot Encoding over the global domain categories specified in [`preprocessing/feature_schema.py`](../preprocessing/feature_schema.py).
  - This ensures that even if a small fold (e.g., Hospital 3 validation set) lacks instances of a rare category (such as `restecg = 2` or `ca = 3`), the resulting transformed matrix retains all 25 columns in identical order.

---

## 5. Numerical Feature Scaling

- **Continuous Features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):** Scaled using standard Z-score normalization:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- The mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are computed exclusively on the imputed `X_train` of each client.
- The fitted `StandardScaler` is applied to transform `X_val` and `X_test`.

---

## 6. Target Transformation

The raw datasets contained heterogeneous target representations:
- **Hospital 1 & Hospital 3:** Multiclass integer labels $0, 1, 2, 3, 4$ representing angiographic narrowing severity.
- **Hospital 2:** Pre-binarized labels $0, 1$.

### Standardized Target Mapping
The target `num` is mapped to a binary classification task:
$$y = \begin{cases} 0 & \text{if } \text{num} = 0 \quad (\text{No coronary artery disease, } <50\% \text{ stenosis}) \\ 1 & \text{if } \text{num} \ge 1 \quad (\text{Coronary artery disease present, } >50\% \text{ stenosis}) \end{cases}$$

This harmonizes the objective across all three hospital nodes into an identical supervised task.

---

## 7. Train / Validation / Test Split

A stratified 70% / 15% / 15% train / validation / test partitioning was performed independently for each hospital using a fixed random seed (`random_state = 42`).

### Partition Counts and Target Distributions
| Client | Split Partition | Sample Count | Healthy ($y=0$) | Disease ($y=1$) | Disease Prevalence (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Hospital 1 (Cleveland)** | **Train (70%)** | **212** | 115 | 97 | 45.8% |
| | **Validation (15%)** | **45** | 24 | 21 | 46.7% |
| | **Test (15%)** | **46** | 25 | 21 | 45.7% |
| | *Total* | *303* | *164* | *139* | *45.9%* |
| **Hospital 2 (Hungarian)** | **Train (70%)** | **205** | 131 | 74 | 36.1% |
| | **Validation (15%)** | **44** | 28 | 16 | 36.4% |
| | **Test (15%)** | **44** | 28 | 16 | 36.4% |
| | *Total (Deduplicated)* | *293* | *188* | *105* | *35.8%* |
| **Hospital 3 (Switzerland)** | **Train (70%)** | **86** | 6 | 80 | 93.0% |
| | **Validation (15%)** | **18** | 1 | 17 | 94.4% |
| | **Test (15%)** | **19** | 1 | 18 | 94.7% |
| | *Total* | *123* | *8* | *115* | *93.5%* |

---

## 8. Data Leakage Prevention

To ensure strict compliance with federated learning standards and prevent data leakage:
1. **Temporal Isolation:** Partitioning was executed *before* computing any summary statistics (medians, modes, scaling means/standard deviations).
2. **Train-Only Parameter Estimation:** The `ClientPreprocessor` computes imputation values and `StandardScaler` parameters strictly on `X_train`.
3. **Immutable Validation & Test:** Validation and test sets are exclusively transformed using the parameters previously frozen from the training split.
4. **Isolated Client Environments:** No statistics or records are shared between Hospital 1, Hospital 2, and Hospital 3 during preprocessing.
5. **Bitwise Immutability of Raw Data:** Verified via SHA-256 cryptographic hashing that original `.data` files remained 100% untouched.

---

## 9. Final Feature Compatibility

All three clients produce arrays and DataFrames with identical mathematical shapes:
- **Input Dimension:** Exactly **25 numeric features** ($X \in \mathbb{R}^{N \times 25}$).
- **Target Dimension:** Exactly **1 binary label** ($y \in \{0, 1\}^N$).
- **Data Types:** 32-bit floating-point tensors ready for PyTorch / TensorFlow federated model architectures.
- **Completeness:** 0 missing values (`NaN`) and 0 infinite values (`Inf`) across all splits.

---

## 10. Preprocessing Limitations & Notes

1. **Hospital 3 Cholesterol Sparsity:** Because serum cholesterol was completely unrecorded in the Swiss cohort, its values are imputed using clinical median fallback. In future federated modeling, model attention or ablation experiments should account for this feature's zero-variance local state on Client 3.
2. **Hospital 2 & 3 Procedural Sparsity:** Fluoroscopy vessel count (`ca`) and thallium scintigraphy (`thal`) were rarely administered at European clinical sites, resulting in predominant modal baseline imputation (`ca = 0`, `thal = 3`) for those nodes.
3. **Extreme Class Asymmetry on Hospital 3:** With only 8 healthy patients out of 123 in the Swiss cohort, the stratified test set contains 1 healthy instance and 18 disease instances. Evaluation metrics on Client 3 should emphasize ROC-AUC, Precision-Recall AUC (PR-AUC), and F1-score alongside accuracy.
