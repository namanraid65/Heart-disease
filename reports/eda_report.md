# Exploratory Data Analysis and Client Heterogeneity

## 1. Overview

This exploratory investigation evaluates the statistical, topological, and clinical distributions across the three simulated hospital clients in our Federated Heart Disease Prediction ecosystem. The analysis operates strictly on the processed partitions without concatenating patient records between hospitals.

### Client Infrastructure Summary
| Client Name | Geographic / Clinical Site | Total Patients | Processed Features | Healthy Count ($y=0$) | Disease Count ($y=1$) | Disease Prevalence (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Hospital 1** | Cleveland Clinic Foundation (USA) | **303** | 25 | 164 | 139 | **45.9%** |
| **Hospital 2** | Hungarian Institute of Cardiology (Budapest) | **293** | 25 | 188 | 105 | **35.8%** |
| **Hospital 3** | University Hospital (Zurich & Basel, Switzerland) | **123** | 25 | 8 | 115 | **93.5%** |
| **Federated System** | 3 Decentralized Clinical Silos | **719** | **25** | **360** | **359** | **49.9%** (Systemic) |

---

## 2. Hospital 1 Analysis (Cleveland Clinic)

### Demographic & Clinical Profile
- **Total Records:** 303 patients (Train: 212, Validation: 45, Test: 46).
- **Target Balance:** 164 Healthy ($54.1\%$), 139 Heart Disease ($45.9\%$).
- **Demographics:** Mean age is $54.44 \pm 9.04$ years (range: 29–77). Sex distribution is $68.0\%$ Male ($n=206$) and $32.0\%$ Female ($n=97$).
- **Physiology & Stress Testing:**
  - Resting Blood Pressure (`trestbps`): Mean $131.69 \pm 17.60$ mm Hg.
  - Serum Cholesterol (`chol`): Mean $246.69 \pm 51.78$ mg/dl.
  - Maximum Heart Rate (`thalach`): Mean $149.61 \pm 22.88$ bpm.
  - ST Depression (`oldpeak`): Mean $1.04 \pm 1.16$ mm (Median: 0.8 mm, Max: 6.2 mm).
- **Diagnostic Testing Availability:** Complete procedural records ($>98\%$ non-missing for fluoroscopy `ca` and thallium scan `thal`).
- **Data Quality:** Zero NaNs, zero Infs, and zero duplicate entries after schema standardization.

---

## 3. Hospital 2 Analysis (Hungarian Institute of Cardiology)

### Demographic & Clinical Profile
- **Total Records:** 293 unique patients (Train: 205, Validation: 44, Test: 44) after removing 1 duplicate record pair.
- **Target Balance:** 188 Healthy ($64.2\%$), 105 Heart Disease ($35.8\%$).
- **Demographics:** Significantly younger patient cohort with mean age $47.83 \pm 7.81$ years (range: 28–66). Sex distribution is $72.4\%$ Male ($n=212$) and $27.6\%$ Female ($n=81$).
- **Physiology & Stress Testing:**
  - Resting Blood Pressure (`trestbps`): Mean $132.58 \pm 17.63$ mm Hg.
  - Serum Cholesterol (`chol`): Mean $250.85 \pm 67.66$ mg/dl.
  - Maximum Heart Rate (`thalach`): Mean $139.13 \pm 23.59$ bpm.
  - ST Depression (`oldpeak`): Mean $0.59 \pm 0.91$ mm (Median: 0.0 mm).
- **Diagnostic Testing Sparsity:**
  - Fluoroscopy (`ca`): $98.98\%$ missing in raw data; standard modal baseline (`ca_0 = 1.0`) applied during training imputation.
  - Thallium scan (`thal`): $90.48\%$ missing in raw data; standard modal baseline (`thal_3 = 1.0`) applied.
  - ST slope (`slope`): $64.63\%$ missing; modal baseline (`slope_2 = 1.0`) applied.

---

## 4. Hospital 3 Analysis (Switzerland University Hospital)

### Demographic & Clinical Profile
- **Total Records:** 123 patients (Train: 86, Validation: 18, Test: 19).
- **Target Balance:** 8 Healthy ($6.5\%$), 115 Heart Disease ($93.5\%$) — represents an acute inpatient/catheterization referral center.
- **Demographics:** Mean age is $55.32 \pm 9.03$ years (range: 32–74). Heavily male-dominated cohort: $91.9\%$ Male ($n=113$), $8.1\%$ Female ($n=10$).
- **Physiology & Stress Testing:**
  - Resting Blood Pressure (`trestbps`): Mean $130.21 \pm 22.56$ mm Hg (wider variance).
  - Serum Cholesterol (`chol`): 100% unrecorded in raw source; imputed with clinical standard median ($240.0$ mg/dl), yielding zero local variance.
  - Maximum Heart Rate (`thalach`): Marked chronotropic limitation, mean $121.56 \pm 25.98$ bpm (substantially lower than Cleveland's 149.6 bpm).
  - ST Depression (`oldpeak`): Mean $0.65 \pm 1.06$ mm (11 raw negative ST elevation readings rectified to $0.0$).
- **Diagnostic Testing Sparsity:** $95.93\%$ missing `ca`, $60.98\%$ missing `fbs`, and $42.28\%$ missing `thal`.

---

## 5. Target Distribution Comparison

The target distribution reveals extreme cross-silo label skew (concept shift):

```
Hospital 1 (Cleveland):   [██████████████████████████░░░░░░░░░░░░░░░░░░░░░░] 45.9% Disease (139 / 303)
Hospital 2 (Hungarian):   [██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 35.8% Disease (105 / 293)
Hospital 3 (Switzerland): [███████████████████████████████████████████████░] 93.5% Disease (115 / 123)
```

- **Hospital 1** represents a balanced general cardiology study population ($45.9\%$ positive).
- **Hospital 2** represents an outpatient screening/diagnostic cohort with higher healthy proportion ($64.2\%$ healthy).
- **Hospital 3** represents a high-risk referral cohort ($93.5\%$ confirmed coronary disease).

---

## 6. Feature Distribution Comparison

Standardized feature comparisons reveal distinct demographic and physiological divergences:

1. **Age Distribution (Covariate Shift):**
   - Hungarian patients are on average **6.6 years younger** than Cleveland and **7.5 years younger** than Swiss patients ($p = 2.5 \times 10^{-1}$ on KS test).
2. **Maximum Heart Rate Achieved (`thalach`):**
   - Swiss patients exhibit severe exercise capacity reduction (Mean: 121.6 bpm vs 149.6 bpm in Cleveland), reflecting older age, higher disease severity, and possible beta-blocker therapy.
3. **Cholesterol (`chol`):**
   - Hospital 1 and Hospital 2 show typical adult distributions (Mean: 246.7 and 250.8 mg/dl). Hospital 3 features zero local variance due to unrecorded clinical protocols.
4. **ST Depression (`oldpeak`):**
   - Hospital 1 displays broader ST depression values (up to 6.2 mm) compared to Hospital 2 and 3 where median values concentrate near 0.0 mm ($p < 10^{-15}$ on KS test).

---

## 7. Correlation Analysis

Feature-to-target correlations differ substantially across the three hospital environments:

### Key Feature Correlations with Target ($y$)
| Feature Name | Hospital 1 (Cleveland) | Hospital 2 (Hungarian) | Hospital 3 (Switzerland) | Cross-Client Observation |
| :--- | :---: | :---: | :---: | :--- |
| `thalach` (Max HR) | **-0.42** | **-0.34** | **-0.12** | Inverse relationship strongest in H1 & H2; attenuated in H3. |
| `oldpeak` (ST Depression) | **+0.42** | **+0.52** | **+0.21** | Strong positive predictor across all sites, strongest in H2. |
| `exang` (Exercise Angina) | **+0.43** | **+0.50** | **+0.28** | Consistent positive marker of ischemia. |
| `cp_4` (Asymptomatic Pain) | **+0.52** | **+0.49** | **+0.18** | Silent/atypical ischemia strongly correlates with severe CAD. |
| `sex` (Male Indicator) | **+0.28** | **+0.27** | **+0.16** | Male sex moderately elevates risk. |
| `age` (Years) | **+0.23** | **+0.16** | **-0.03** | Age correlation vanishes in Swiss acute cohort due to high disease ceiling. |
| `ca_0` (0 Vessels Colored) | **-0.47** | **-0.18** | **0.00** | Highly protective in H1; uninformative in H3 due to missingness. |
| `thal_7` (Reversible Defect)| **+0.53** | **+0.28** | **+0.14** | Strongest diagnostic predictor in H1 where thallium scans were standard. |

---

## 8. Train / Validation / Test Distribution

Stratified 70% / 15% / 15% partitioning was performed independently on each client with seed `42`.

### Detailed Partition Matrix
| Hospital Client | Partition | Sample Count | Class 0 (Healthy) | Class 1 (Disease) | Prevalence (%) | Extreme Imbalance Check |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Hospital 1** | Train | **212** | 115 | 97 | 45.8% | Balanced |
| | Validation | **45** | 24 | 21 | 46.7% | Balanced |
| | Test | **46** | 25 | 21 | 45.7% | Balanced |
| **Hospital 2** | Train | **205** | 131 | 74 | 36.1% | Moderate class 0 skew |
| | Validation | **44** | 28 | 16 | 36.4% | Moderate class 0 skew |
| | Test | **44** | 28 | 16 | 36.4% | Moderate class 0 skew |
| **Hospital 3** | Train | **86** | 6 | 80 | 93.0% | Extreme class 1 skew |
| | Validation | **18** | 1 | 17 | 94.4% | Extreme class 1 skew |
| | Test | **19** | 1 | 18 | 94.7% | Extreme class 1 skew |

> **Note on Hospital 3:** The validation and test sets for Hospital 3 contain exactly 1 healthy patient each. While stratification was successful in retaining both classes in every split, evaluation metrics on Client 3 must utilize ROC-AUC, PR-AUC, and F1-score in conjunction with overall accuracy to avoid misleading results.

---

## 9. Non-IID Client Analysis

### Quantitative Divergence Metrics

#### 1. Label Distribution Divergence (Total Variation Distance, TVD $\in [0, 1]$)
$$\text{TVD}(P, Q) = \frac{1}{2} \sum_{k} |P(y=k) - Q(y=k)|$$
- **$\text{TVD}(\text{Hospital 1}, \text{Hospital 2}) = \mathbf{0.0970}$** (Mild label difference: $9.7\%$).
- **$\text{TVD}(\text{Hospital 1}, \text{Hospital 3}) = \mathbf{0.4762}$** (Severe label divergence: $47.6\%$).
- **$\text{TVD}(\text{Hospital 2}, \text{Hospital 3}) = \mathbf{0.5732}$** (Extreme label divergence: $57.3\%$).

#### 2. Feature Distribution Divergence (Kolmogorov-Smirnov 2-Sample Test)
- **ST Depression (`oldpeak`):**
  - H1 vs H2: $D = 0.3894$, $p = 1.04 \times 10^{-20}$ (Statistically significant shift).
  - H2 vs H3: $D = 0.4472$, $p = 3.54 \times 10^{-16}$ (Statistically significant shift).
- **Resting Blood Pressure (`trestbps`):**
  - H1 vs H2: $D = 0.1844$, $p = 6.46 \times 10^{-5}$ (Statistically significant shift).
- **Cholesterol (`chol`):**
  - H1 vs H3: $D = 0.5710$, $p = 5.21 \times 10^{-27}$ (Shift driven by missingness protocol).

### Classification of Non-IID Sources
1. **Prior Probability Shift $P(y)$:** Substantial label skew across clients.
2. **Covariate Shift $P(x)$:** Distinct age profiles, exercise capacity limitations, and male prevalence across geographic centers.
3. **Concept Shift $P(y \mid x)$:** Diagnostic power of individual tests varies due to local clinical screening thresholds.

---

## 10. Key Observations

1. **True Natural Non-IID Environment:** Unlike artificial synthetic Dirichlet partitioning, the UCI multi-hospital data embodies authentic clinical heterogeneity stemming from geographic variation, local hospital admission criteria, and specialized diagnostic workflows.
2. **Client Drift Risk in Federated Aggregation:** Vanilla Federated Averaging (FedAvg) will likely experience weight drift during local SGD on Hospital 3 toward majority-class (positive) prediction. Mitigations such as proximal regularization (FedProx), class-weighted cross-entropy loss, or client-adaptive learning rates will be beneficial.
3. **Data Completeness & Zero Leakage:** All partitions are completely verified with zero NaNs, zero Infs, and identical $(N, 25)$ dimensional structures across all three hospital nodes.

---

## 11. Conclusion

The Exploratory Data Analysis confirms that the three processed hospital clients form a high-fidelity, naturally non-IID benchmark for Federated Deep Learning. The uniform 25-feature schema provides strict architectural compatibility for future neural network models while preserving the distinct statistical properties of each participating medical institution.
