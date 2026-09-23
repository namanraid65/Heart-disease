# Dataset Analysis: Federated Learning for Heart Disease Prediction

## 1. Dataset Overview

This research project builds a simulated horizontal Federated Learning (FL) system for Heart Disease prediction across three distinct hospital institutions. The datasets correspond to the classical UCI Heart Disease repository collected from three clinical sites:

1. **Hospital 1 (Client 1):** Cleveland Clinic Foundation, USA (`processed.cleveland.data`)
2. **Hospital 2 (Client 2):** Hungarian Institute of Cardiology, Budapest (`processed.hungarian.data`)
3. **Hospital 3 (Client 3):** University Hospital, Zurich & Basel, Switzerland (`processed.switzerland.data`)

Each dataset is structured with 14 clinical attributes per patient record (13 diagnostic features + 1 target outcome).

| Client Identifier | Source Institution | File Name | Total Samples | Feature Count | Target Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hospital 1** | Cleveland Clinic Foundation | `processed.cleveland.data` | 303 | 14 | Multiclass (0–4) |
| **Hospital 2** | Hungarian Institute of Cardiology | `processed.hungarian.data` | 294 | 14 | Binary (0, 1) |
| **Hospital 3** | University Hospital, Switzerland | `processed.switzerland.data` | 123 | 14 | Multiclass (0–4) |
| **Total Ecosystem** | 3 Simulated Hospital Clients | — | **720** | **14** | Harmonized Binary |

---

## 2. Hospital 1 - Cleveland

### Summary
- **Records:** 303
- **Features:** 14 columns
- **Duplicate Records:** 0
- **Data Quality:** Very high completeness. Only 6 values are missing across the entire dataset.

### Attribute-Level Details
| Feature | Description | Inferred Type | Missing Count (%) | Observed Unique Values / Distribution |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Patient age (years) | Float / Int | 0 (0.00%) | Range: 29.0 – 77.0 (Mean: 54.44, Std: 9.04) |
| `sex` | Biological sex | Float / Int | 0 (0.00%) | `0.0`: 97 (32.0%), `1.0`: 206 (68.0%) |
| `cp` | Chest pain type | Float / Int | 0 (0.00%) | `1.0`: 23, `2.0`: 50, `3.0`: 86, `4.0`: 144 |
| `trestbps` | Resting blood pressure (mm Hg) | Float | 0 (0.00%) | Range: 94.0 – 200.0 (Mean: 131.69, Std: 17.60) |
| `chol` | Serum cholesterol (mg/dl) | Float | 0 (0.00%) | Range: 126.0 – 564.0 (Mean: 246.69, Std: 51.78) |
| `fbs` | Fasting blood sugar > 120 mg/dl | Float / Int | 0 (0.00%) | `0.0`: 258 (85.1%), `1.0`: 45 (14.9%) |
| `restecg` | Resting ECG results | Float / Int | 0 (0.00%) | `0.0`: 151 (49.8%), `1.0`: 4 (1.3%), `2.0`: 148 (48.8%) |
| `thalach` | Maximum heart rate achieved | Float | 0 (0.00%) | Range: 71.0 – 202.0 (Mean: 149.61, Std: 22.88) |
| `exang` | Exercise induced angina | Float / Int | 0 (0.00%) | `0.0`: 204 (67.3%), `1.0`: 99 (32.7%) |
| `oldpeak` | ST depression (exercise vs rest) | Float | 0 (0.00%) | Range: 0.0 – 6.2 (Mean: 1.04, Std: 1.16, Median: 0.8) |
| `slope` | ST segment slope | Float / Int | 0 (0.00%) | `1.0`: 142, `2.0`: 140, `3.0`: 21 |
| `ca` | Major vessels via fluoroscopy | Float / Int | 4 (1.32%) | `0.0`: 176, `1.0`: 65, `2.0`: 38, `3.0`: 20 |
| `thal` | Thalassemia status | Float / Int | 2 (0.66%) | `3.0` (normal): 166, `6.0` (fixed): 18, `7.0` (reversable): 117 |
| `num` | Angiographic disease status (Target) | Integer | 0 (0.00%) | Class 0: 164 (54.1%), Class 1: 55, Class 2: 36, Class 3: 35, Class 4: 13 |

---

## 3. Hospital 2 - Hungarian

### Summary
- **Records:** 294
- **Features:** 14 columns
- **Duplicate Records:** 1 pair of exact duplicate records (Row indices 101 and 102).
- **Data Quality:** Significant missingness in procedural tests (`ca`, `thal`, `slope`).

### Attribute-Level Details
| Feature | Description | Inferred Type | Missing Count (%) | Observed Unique Values / Distribution |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Patient age (years) | Integer | 0 (0.00%) | Range: 28.0 – 66.0 (Mean: 47.83, Std: 7.81) |
| `sex` | Biological sex | Integer | 0 (0.00%) | `0`: 81 (27.6%), `1`: 213 (72.4%) |
| `cp` | Chest pain type | Integer | 0 (0.00%) | `1`: 11, `2`: 106, `3`: 54, `4`: 123 |
| `trestbps` | Resting blood pressure (mm Hg) | Float | 1 (0.34%) | Range: 92.0 – 200.0 (Mean: 132.58, Std: 17.63) |
| `chol` | Serum cholesterol (mg/dl) | Float | 23 (7.82%) | Range: 85.0 – 603.0 (Mean: 250.85, Std: 67.66) |
| `fbs` | Fasting blood sugar > 120 mg/dl | Float | 8 (2.72%) | `0.0`: 266, `1.0`: 20 |
| `restecg` | Resting ECG results | Float | 1 (0.34%) | `0.0`: 235 (80.2%), `1.0`: 52 (17.7%), `2.0`: 6 (2.0%) |
| `thalach` | Maximum heart rate achieved | Float | 1 (0.34%) | Range: 82.0 – 190.0 (Mean: 139.13, Std: 23.59) |
| `exang` | Exercise induced angina | Float | 1 (0.34%) | `0.0`: 204 (69.6%), `1.0`: 89 (30.4%) |
| `oldpeak` | ST depression (exercise vs rest) | Float | 0 (0.00%) | Range: 0.0 – 5.0 (Mean: 0.59, Std: 0.91, Median: 0.0) |
| `slope` | ST segment slope | Float | 190 (64.63%) | `1.0`: 12, `2.0`: 91, `3.0`: 1 |
| `ca` | Major vessels via fluoroscopy | Float | 291 (98.98%) | `0.0`: 3 non-missing instances |
| `thal` | Thalassemia status | Float | 266 (90.48%) | `3.0`: 7, `6.0`: 10, `7.0`: 11 |
| `num` | Disease status (Target) | Integer | 0 (0.00%) | Class 0: 188 (63.9%), Class 1: 106 (36.1%) |

---

## 4. Hospital 3 - Switzerland

### Summary
- **Records:** 123
- **Features:** 14 columns
- **Duplicate Records:** 0
- **Data Quality:** Severe missingness in `chol` (all 123 rows are 0), `ca` (95.93%), `fbs` (60.98%), and `thal` (42.28%). Includes negative values for `oldpeak`.

### Attribute-Level Details
| Feature | Description | Inferred Type | Missing Count (%) | Observed Unique Values / Distribution |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Patient age (years) | Integer | 0 (0.00%) | Range: 32.0 – 74.0 (Mean: 55.32, Std: 9.03) |
| `sex` | Biological sex | Integer | 0 (0.00%) | `0`: 10 (8.1%), `1`: 113 (91.9%) |
| `cp` | Chest pain type | Integer | 0 (0.00%) | `1`: 4, `2`: 4, `3`: 17, `4`: 98 |
| `trestbps` | Resting blood pressure (mm Hg) | Float | 2 (1.63%) | Range: 80.0 – 200.0 (Mean: 130.21, Std: 22.56) |
| `chol` | Serum cholesterol (mg/dl) | Integer / Float | 0 explicit (100% 0s) | All 123 records are recorded as `0` (unmeasured) |
| `fbs` | Fasting blood sugar > 120 mg/dl | Float | 75 (60.98%) | `0.0`: 43, `1.0`: 5 |
| `restecg` | Resting ECG results | Float | 1 (0.81%) | `0.0`: 85 (69.7%), `1.0`: 30 (24.6%), `2.0`: 7 (5.7%) |
| `thalach` | Maximum heart rate achieved | Float | 1 (0.81%) | Range: 60.0 – 182.0 (Mean: 121.56, Std: 25.98) |
| `exang` | Exercise induced angina | Float | 1 (0.81%) | `0.0`: 68 (55.7%), `1.0`: 54 (44.3%) |
| `oldpeak` | ST depression (exercise vs rest) | Float | 6 (4.88%) | Range: -2.6 – 3.7 (Mean: 0.65, Std: 1.06, 11 negative values) |
| `slope` | ST segment slope | Float | 17 (13.82%) | `1.0`: 33, `2.0`: 61, `3.0`: 12 |
| `ca` | Major vessels via fluoroscopy | Float | 118 (95.93%) | `1.0`: 2, `2.0`: 3 |
| `thal` | Thalassemia status | Float | 52 (42.28%) | `3.0`: 19, `6.0`: 10, `7.0`: 42 |
| `num` | Disease status (Target) | Integer | 0 (0.00%) | Class 0: 8 (6.5%), Class 1: 48 (39.0%), Class 2: 32 (26.0%), Class 3: 30 (24.4%), Class 4: 5 (4.1%) |

---

## 5. Feature Comparison

All three raw files contain exactly 14 columns aligned to the standard UCI attribute order.

| Attribute Index | Feature Name | Clinical Meaning | Cleveland (H1) | Hungarian (H2) | Switzerland (H3) | Status in Schema |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | `age` | Age in years | Continuous | Continuous | Continuous | Common |
| 2 | `sex` | 1=Male, 0=Female | Binary (0,1) | Binary (0,1) | Binary (0,1) | Common |
| 3 | `cp` | Chest Pain Type (1-4) | Categorical | Categorical | Categorical | Common |
| 4 | `trestbps` | Resting Blood Pressure | Continuous | Continuous | Continuous | Common (Low missing H2/H3) |
| 5 | `chol` | Serum Cholesterol | Continuous | Continuous | All zeros (Missing) | Problematic on H3 |
| 6 | `fbs` | Fasting Blood Sugar > 120 | Binary (0,1) | Binary (0,1) | Binary (0,1) | High missing on H3 |
| 7 | `restecg` | Resting ECG (0,1,2) | Categorical | Categorical | Categorical | Common |
| 8 | `thalach` | Max Heart Rate Achieved | Continuous | Continuous | Continuous | Common |
| 9 | `exang` | Exercise Induced Angina | Binary (0,1) | Binary (0,1) | Binary (0,1) | Common |
| 10 | `oldpeak` | ST Depression | Continuous | Continuous | Continuous (Negative vals) | Common |
| 11 | `slope` | ST Slope (1,2,3) | Categorical | Categorical | Categorical | High missing on H2 |
| 12 | `ca` | Major fluoroscopy vessels | Count (0-3) | Count (0-3) | Count (0-3) | Sparse on H2 & H3 |
| 13 | `thal` | Thalassemia (3,6,7) | Categorical | Categorical | Categorical | Sparse on H2 & H3 |
| 14 | `num` | Heart Disease Target | Multiclass (0-4) | Binary (0,1) | Multiclass (0-4) | Harmonizable to Binary |

---

## 6. Missing Value Analysis

The datasets exhibit distinct, non-random missingness patterns reflecting differences in clinical protocols across hospitals.

### Missing Rate Comparison Table (%)
| Feature | Hospital 1 (Cleveland) | Hospital 2 (Hungarian) | Hospital 3 (Switzerland) | Aggregate Impact |
| :--- | :---: | :---: | :---: | :--- |
| `age` | 0.00% (0) | 0.00% (0) | 0.00% (0) | Complete across all clients |
| `sex` | 0.00% (0) | 0.00% (0) | 0.00% (0) | Complete across all clients |
| `cp` | 0.00% (0) | 0.00% (0) | 0.00% (0) | Complete across all clients |
| `trestbps` | 0.00% (0) | 0.34% (1) | 1.63% (2) | Negligible missingness (3 total) |
| `chol` | 0.00% (0) | 7.82% (23) | **100.00%** (123 zeros) | 0 mg/dl in H3 denotes missing |
| `fbs` | 0.00% (0) | 2.72% (8) | **60.98%** (75) | High missingness in H3 |
| `restecg` | 0.00% (0) | 0.34% (1) | 0.81% (1) | Negligible missingness (2 total) |
| `thalach` | 0.00% (0) | 0.34% (1) | 0.81% (1) | Negligible missingness (2 total) |
| `exang` | 0.00% (0) | 0.34% (1) | 0.81% (1) | Negligible missingness (2 total) |
| `oldpeak` | 0.00% (0) | 0.00% (0) | 4.88% (6) | Low missingness (6 in H3) |
| `slope` | 0.00% (0) | **64.63%** (190) | 13.82% (17) | High missingness in H2 |
| `ca` | 1.32% (4) | **98.98%** (291) | **95.93%** (118) | Invasive test, omitted in H2/H3 |
| `thal` | 0.66% (2) | **90.48%** (266) | **42.28%** (52) | Nuclear test, omitted in H2/H3 |
| `num` | 0.00% (0) | 0.00% (0) | 0.00% (0) | 100% complete labels |

---

## 7. Target Distribution

The target variable `num` measures the presence and angiographic severity of coronary artery stenosis:
- `0`: `< 50%` diameter narrowing (Absence of significant heart disease)
- `1, 2, 3, 4`: `> 50%` diameter narrowing (Presence of heart disease with increasing severity)

### Raw Multiclass Distribution
| Class Label | Hospital 1 (Cleveland) | Hospital 2 (Hungarian) | Hospital 3 (Switzerland) | Total Cohort |
| :---: | :---: | :---: | :---: | :---: |
| **0** (No Disease) | 164 (54.1%) | 188 (63.9%) | 8 (6.5%) | 360 (50.0%) |
| **1** (Mild/Moderate) | 55 (18.2%) | 106 (36.1%) | 48 (39.0%) | 209 (29.0%) |
| **2** (Moderate) | 36 (11.9%) | 0 (0.0%) | 32 (26.0%) | 68 (9.4%) |
| **3** (Severe) | 35 (11.6%) | 0 (0.0%) | 30 (24.4%) | 65 (9.0%) |
| **4** (Critical) | 13 (4.3%) | 0 (0.0%) | 5 (4.1%) | 18 (2.5%) |

### Harmonized Binary Target Distribution (0 = Absence, 1 = Presence)
| Target Class | Hospital 1 (Cleveland) | Hospital 2 (Hungarian) | Hospital 3 (Switzerland) |
| :--- | :---: | :---: | :---: |
| **No Disease (`0`)** | 164 (54.1%) | 188 (63.9%) | 8 (6.5%) |
| **Disease Present (`1`)** | 139 (45.9%) | 106 (36.1%) | 115 (93.5%) |
| **Total Samples** | **303** | **294** | **123** |

---

## 8. Client Heterogeneity / Non-IID Analysis

The three hospital clients represent an authentic non-IID (non-Independent and Identically Distributed) federated setting along three major dimensions:

### 1. Label Distribution Skew (Concept Drift / Class Imbalance)
- **Hospital 1 (Cleveland):** Balanced population (54.1% Healthy, 45.9% Disease).
- **Hospital 2 (Hungarian):** Mild healthy skew (63.9% Healthy, 36.1% Disease).
- **Hospital 3 (Switzerland):** Severe pathological skew (6.5% Healthy, 93.5% Disease). Swiss patients admitted were almost exclusively high-risk/confirmed cases.

### 2. Feature Distribution Skew (Covariate Shift)
- **Demographics:**
  - Hungarian patients are significantly younger (Mean age: 47.8 ± 7.8 years vs 54.4 in Cleveland and 55.3 in Switzerland).
  - Switzerland client is heavily male-dominated (91.9% Male vs 68.0% in Cleveland and 72.4% in Hungarian).
- **Hemodynamics & Stress Response:**
  - Maximum Heart Rate (`thalach`) is markedly lower in Switzerland (Mean: 121.6 ± 26.0 bpm) compared to Cleveland (149.6 ± 22.9 bpm) and Hungarian (139.1 ± 23.6 bpm).
  - Resting Blood Pressure (`trestbps`) variance is wider in Switzerland (Std: 22.56 vs 17.6 in H1/H2).

### 3. Missingness & Operational Heterogeneity (Clinical Protocol Shift)
- Invasive fluoroscopy (`ca`) and radioactive thallium scintigraphy (`thal`) were routine in the US hospital (Cleveland), but rarely performed at European centers (Hungarian & Swiss), causing 90–99% missingness in those features for Clients 2 and 3.
- Cholesterol measurements (`chol`) were completely unrecorded in the Switzerland hospital.

---

## 9. Data Compatibility

### Feasibility as Separate FL Clients
Yes, the three datasets can successfully operate as isolated hospital clients within a Federated Learning framework, provided standard harmonization is performed.

### Compatibility Strengths
1. **Identical Column Indexing:** All three `.data` files follow the standard 14-attribute UCI ordering.
2. **Standardized Value Codes:** Categorical encodings (`sex`, `cp`, `restecg`, `exang`, `slope`) use identical numeric mappings.
3. **Harmonizable Target:** Target binarization (`num > 0 -> 1`) produces a uniform supervised task across all nodes.

### Incompatibilities & Anomalies Requiring Resolution
1. **Unrecorded Cholesterol in Hospital 3:** `chol == 0` for all 123 rows must be treated as missing values (NaN) rather than numeric zero.
2. **Negative `oldpeak` in Hospital 3:** 11 instances have negative ST depression (e.g. -2.6, -1.5). ST depression is mathematically ≥ 0; negative values indicate ST segment elevation. These must be handled consistently (e.g., rectified or clipped).
3. **Extreme Missing Rates in `ca` and `thal`:** Features missing >90% on Clients 2 and 3 cannot rely on simple global mean imputation during decentralized execution without leaking information or distorting local distributions.
4. **Duplicate Record in Hospital 2:** One duplicate record pair exists at rows 101–102.

---

## 10. Recommended Preprocessing Plan

To prepare the clients for decentralized Federated Learning without violating privacy or merging patient records:

1. **Client-Level Isolation:**
   - Execute all preprocessing pipelines independently per hospital client.
   - Maintain client-specific transformation state (e.g. local imputation values or federated global parameters).

2. **Target Harmonization:**
   - Map `num` to binary classification: `y = 1 if num > 0 else 0`.

3. **Feature Selection / Dimensionality Handling:**
   - **Option A (Core Common Subset):** Select the 10 consistently available features (`age`, `sex`, `cp`, `trestbps`, `restecg`, `thalach`, `exang`, `oldpeak`, `fbs`, `slope`) and exclude `ca`, `thal`, and `chol` from models requiring full feature density.
   - **Option B (Robust Imputation):** Apply client-specific median/mode imputation or iterative imputation to preserve the 13-feature input vector across all clients.

4. **Anomaly Remediation:**
   - Convert `chol = 0` to NaN.
   - Rectify or clip negative `oldpeak` values (`oldpeak = max(0.0, oldpeak)` or encode ST elevation).
   - Deduplicate Hospital 2 records locally.

5. **Local Feature Scaling:**
   - Standardize continuous variables (`StandardScaler` or `MinMaxScaler`) fitted on local client data (or scaled via privacy-preserving federated normalization).
