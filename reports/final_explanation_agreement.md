# Final Explanation Agreement Summary: LIME vs. SHAP

This report summarizes the quantitative concordance between LIME and SHAP across representative patient cases ($K=5$).

---

## 1. Overall Summary Agreement Observations
- High directional concordance: LIME and SHAP agree on the attribution sign (risk-increasing vs. protective) for the vast majority of dominant features (`oldpeak`, `cp_4`, `thal_7`).
- Ranking variations occur on collinear one-hot categories due to differences between LIME's perturbation-based sparse ridge regression and SHAP's cooperative game formulation.
- Both explainers explain internal model mechanics and statistical dependencies; they do not establish clinical causality.
