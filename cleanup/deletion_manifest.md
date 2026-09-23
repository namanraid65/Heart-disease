# REPOSITORY CLEANUP — DELETION & ARCHIVE MANIFEST

**Repository:** `e:\Heart-disease-main`  
**Execution Date:** 2026-09-23  
**Audit & Cleanup Phase:** Final Repository Cleanup  

---

## 1. Safety Classification Framework

Every candidate file is evaluated under the following strict criteria:
- **SAFE TO DELETE:** Zero references in active code, zero references in tests, zero references in documentation, not required for reproducibility, not an official result/checkpoint, and already superseded or temporary.
- **SAFE TO ARCHIVE:** Historical or provenance artifact that should be preserved for archival completeness but removed from the active top-level working tree.
- **KEEP:** Actively required by the runtime, testing, evaluation, or reporting pipeline.
- **REQUIRES REVIEW:** Any file with ambiguous usage.

---

## 2. Deletion Manifest

| File / Pattern | Category | Reason | Reference Search Result | Why Safe to Delete | Replacement | Safety Classification |
|---|---|---|---|---|---|---|
| `Federated XGBoost.png` (root) | STALE GENERATED ARTIFACT | Misleading title; actual model is a Local XGBoost Ensemble, not federated boosting | 0 references across all `.py`, `.md`, `.json`, `.yaml` files | Stale redundant copy; accurately titled replacement already present | `Local XGBoost Ensemble.png` | **SAFE TO DELETE** |
| `__pycache__` (11 directories) | TEMPORARY | Compiled Python bytecode (`.pyc`) files | Ignored in `.gitignore` | Dynamically recompiled by Python interpreter on-demand; should not be tracked | None needed | **SAFE TO DELETE** |
| `*.pyc` (68 files) | TEMPORARY | Cached bytecode modules | Ignored in `.gitignore` | Dynamically regenerated on import; non-source binary files | None needed | **SAFE TO DELETE** |
| `cleanup/inventory_analysis.py` | TEMPORARY | One-time analysis script used to generate repository inventory | Created during cleanup phase | Temporary scratch analysis tool; findings formalized in this manifest | `cleanup/deletion_manifest.md` | **SAFE TO DELETE** |

---

## 3. Archive Manifest

| File | Category | Original Location | New Archive Location | Reason | Safety Classification |
|---|---|---|---|---|---|
| `Heart-disease-main.zip` | REQUIRED REPRODUCIBILITY ARTIFACT | `e:\Heart-disease-main\` | `e:\Heart-disease-main\archive\` | Full repository snapshot (79.78 MB). Preserved for provenance while keeping root directory uncluttered. | **SAFE TO ARCHIVE** |
| `thesis_package.zip` | REQUIRED REPRODUCIBILITY ARTIFACT | `e:\Heart-disease-main\` | `e:\Heart-disease-main\archive\` | Packaged thesis materials snapshot (1.11 MB). Preserved for distribution while keeping root directory clean. | **SAFE TO ARCHIVE** |

---

## 4. Retained & Protected Assets (Sample Proof)

All active code, checkpoints, datasets, and documentation are strictly preserved:

1. **Official Results:** `results/experiment_results.jsonl` (315 verified records across 21 experiments and 3 seeds) — **KEPT**.
2. **Official Checkpoints:** All 475 checkpoints in `models/checkpoints/` referenced by `checkpoint_metadata.json` — **KEPT**.
3. **Data & Manifests:** `data/split_manifest.json` (18 verified SHA-256 hashes) and raw `.data` files in `dataset/` — **KEPT**.
4. **Test Suite:** All 7 test modules containing 84 unit tests — **KEPT**.
5. **Legacy Quarantined Reports:** `reports/archive/legacy_reports/` — **KEPT** (explicitly verified by `validate_experiments.py`).
6. **Active Pipelines:** All 68 core Python modules across `models/`, `federated/`, `preprocessing/`, `evaluation/`, and `xai/` — **KEPT**.
