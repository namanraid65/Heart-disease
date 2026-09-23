# Client Heterogeneity and Non-IID Impact Analysis

This report synthesizes the structural and statistical differences among the three hospital cohorts and evaluates how client heterogeneity shaped federated training.

---

## 1. Multi-Center Client Distribution Summary

| Hospital Node | Clinical Institution | Geography | Total Patients | Disease Prevalence | Clinical Cohort Profile |
|:---|:---|:---:|:---:|:---:|:---|
| **Hospital 1** | Cleveland Clinic Foundation | USA | 303 | 45.87% (Balanced) | Comprehensive cardiology research center |
| **Hospital 2** | Hungarian Inst. of Cardiology | Europe | 294 | 36.05% (Negative Skew) | Outpatient clinical screening cohort |
| **Hospital 3** | University Hospital Zurich/Basel | Europe | 123 | 93.50% (Extreme Positive) | High-acuity inpatient referral center |

---

## 2. Manifestations of Non-IID Skew on Model Training
1. **Label Distribution Shift:** Hospital 3's extreme 93.5% positive prevalence caused isolated local models to collapse into predicting positive for all inputs. Federated learning counteracted this by injecting balanced decision priors from Cleveland and Hungarian nodes.
2. **Covariate Feature Shift:** European centers (H2 and H3) routinely omit invasive fluoroscopy (`ca`) and thallium (`thal`) tests. The common 25-feature schema with clinical reference fallbacks enabled global models to seamlessly bridge data availability gaps without leaking patient rows.
