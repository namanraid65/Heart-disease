# Cross-Hospital Feature Consistency Analysis

This report evaluates how client heterogeneity and non-IID covariate distributions across Cleveland, Hungarian, and Swiss cohorts affect model feature attributions.

---

## 1. Hospital-Specific Feature Reliance Patterns

### A. Hospital 1 (Cleveland Clinic Foundation)
- **Primary Contributing Features:** `thal_7`, `ca_count` (`ca_0`, `ca_1`), `oldpeak`, `cp_4`.
- **Cohort Context:** Recorded invasive diagnostic testing (fluoroscopy and thallium scans available) allows the models to rely heavily on structural and perfusion markers.

### B. Hospital 2 (Hungarian Institute of Cardiology)
- **Primary Contributing Features:** `oldpeak`, `exang`, `thalach`, `cp_4`, `age`.
- **Cohort Context:** Outpatient screening cohort where invasive tests were frequently unrecorded. Models adapted by shifting attribution mass to physiological stress test variables.

### C. Hospital 3 (University Hospital Zurich & Basel)
- **Primary Contributing Features:** `oldpeak`, `trestbps`, `age`, `cp_4`.
- **Cohort Context:** Referral population with high disease prevalence. Baseline resting blood pressure and ST changes dominate local model attributions.

---

## 2. Client Heterogeneity Impact on Explainability
Institutional differences in available clinical measurements and cohort risk profiles lead models to weight different feature subsets across hospital test sets. Federated aggregation enabled models to learn shared representations while accommodating localized covariate distributions without collapsing to a single site's profile.
