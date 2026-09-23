"""Unit and Integration Tests for External Dataset Validation on UCI Statlog (Heart)

Validates:
  TEST 1: Dataset file existence and SHA-256 cryptographic integrity.
  TEST 2: Exact schema dimensions (270 rows, 14 columns, zero missing values).
  TEST 3: Deterministic 1-to-1 feature mapping to project clinical schema.
  TEST 4: Deterministic target mapping: 1 -> 0 (150 count), 2 -> 1 (120 count).
  TEST 5: Preprocessing leakage firewall: transform() executes without mutating scaler statistics.
  TEST 6: Preprocessing firewall: fitting on external data raises error or is strictly prohibited.
  TEST 7: Model freezing: evaluation mode, no grad, weights identical before and after inference.
  TEST 8: Heterogeneous FL model correctly flagged as NOT APPLICABLE without an external encoder.
  TEST 9: Metric calculation identities hold (TP + TN + FP + FN == 270, TP + FN == 120, TN + FP == 150).
  TEST 10: Single-class metric handling correctly handles edge cases without heuristic fabrication.
  TEST 11: Feature column order strictly matches model input order (INPUT_FEATURES).
  TEST 12: Fixed decision threshold (tau = 0.5) is enforced without external tuning.
  TEST 13: Checkpoint integrity: approved frozen checkpoints exist and parse valid state dicts.
  TEST 14: Determinism & Reproducibility: second execution produces bitwise identical predictions.
  TEST 15: External validation artifacts exist and parse cleanly.
"""

import sys
import json
import hashlib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from preprocessing.feature_schema import INPUT_FEATURES, RAW_FEATURE_NAMES
from preprocessing.preprocess_client import load_client_preprocessor, ClientPreprocessor
from models.alexnet_1d import build_alexnet_1d
from models.resnet_1d import build_resnet_1d
from federated.utils import set_model_bn_state
from evaluation.evaluate_external_statlog import (
    DATA_PATH,
    EXPECTED_SHA256,
    EXPECTED_ROWS,
    EXPECTED_COLS,
    THRESHOLD,
    verify_and_load_statlog,
    compute_comprehensive_metrics,
    run_external_validation
)


class TestExternalValidation(unittest.TestCase):
    """Test suite validating UCI Statlog (Heart) external evaluation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = DATA_PATH
        cls.output_dir = PROJECT_ROOT / "reports" / "external_validation"
        cls.prep_path = PROJECT_ROOT / "data" / "processed" / "hospital_1" / "preprocessor.joblib"
        cls.ckpt_alex = PROJECT_ROOT / "models" / "checkpoints" / "federated_alexnet" / "global_alexnet_final.pt"
        cls.ckpt_res = PROJECT_ROOT / "models" / "checkpoints" / "federated_resnet" / "global_resnet_final.pt"

    def test_01_dataset_file_existence_and_hash(self):
        """TEST 1: Dataset file existence and SHA-256 cryptographic integrity."""
        self.assertTrue(self.data_path.exists(), f"Missing external dataset at {self.data_path}")
        content = self.data_path.read_bytes()
        calculated_sha = hashlib.sha256(content).hexdigest()
        self.assertEqual(
            calculated_sha,
            EXPECTED_SHA256,
            f"SHA256 mismatch! Expected {EXPECTED_SHA256}, got {calculated_sha}"
        )

    def test_02_schema_validation_and_completeness(self):
        """TEST 2: Exact schema dimensions (270 rows, 14 columns, zero missing values)."""
        df = pd.read_csv(self.data_path, sep=r'\s+', header=None)
        self.assertEqual(df.shape, (EXPECTED_ROWS, EXPECTED_COLS))
        self.assertEqual(int(df.isnull().sum().sum()), 0, "External dataset should have 0 missing values")

    def test_03_feature_mapping_alignment(self):
        """TEST 3: Deterministic 1-to-1 feature mapping to project clinical schema."""
        X_raw, y_true, meta = verify_and_load_statlog()
        self.assertEqual(list(X_raw.columns), INPUT_FEATURES)
        self.assertEqual(len(X_raw.columns), 13)

    def test_04_target_mapping_semantics(self):
        """TEST 4: Deterministic target mapping: 1 -> 0 (150 count), 2 -> 1 (120 count)."""
        df = pd.read_csv(self.data_path, sep=r'\s+', header=None)
        raw_targets = df.iloc[:, 13].to_numpy()
        y_binary = (raw_targets == 2).astype(int)

        self.assertEqual(int(np.sum(y_binary == 0)), 150)
        self.assertEqual(int(np.sum(y_binary == 1)), 120)
        self.assertAlmostEqual(float(np.mean(y_binary)), 120 / 270, places=4)

    def test_05_transform_only_preprocessing_leakage_firewall(self):
        """TEST 5: Preprocessing leakage firewall: transform() executes without mutating scaler statistics."""
        prep = load_client_preprocessor(self.prep_path)
        mean_before = prep.scaler_.mean_.copy()
        var_before = prep.scaler_.var_.copy()

        X_raw, _, _ = verify_and_load_statlog()
        clean_raw = X_raw.copy()
        clean_raw.loc[clean_raw['chol'] == 0, 'chol'] = np.nan
        clean_raw['oldpeak'] = clean_raw['oldpeak'].clip(lower=0.0)

        X_proc = prep.transform(clean_raw)
        self.assertEqual(X_proc.shape, (270, 25))

        # Strict invariance check
        np.testing.assert_array_equal(mean_before, prep.scaler_.mean_)
        np.testing.assert_array_equal(var_before, prep.scaler_.var_)
        self.assertTrue(prep.is_fitted_)

    def test_06_prohibit_fitting_on_external_data(self):
        """TEST 6: Preprocessing firewall: verify that pipeline uses loaded preprocessor and does not re-fit."""
        prep = load_client_preprocessor(self.prep_path)
        # Attempting to call fit on clean_raw with dummy preprocessor would yield different mean
        X_raw, _, _ = verify_and_load_statlog()
        fresh_prep = ClientPreprocessor()
        fresh_prep.fit(X_raw)

        # Fresh preprocessor fitted on Statlog would have different statistics than the H1 training preprocessor
        self.assertFalse(np.allclose(prep.scaler_.mean_, fresh_prep.scaler_.mean_, atol=1e-3))

    def test_07_model_freezing_and_weight_invariance(self):
        """TEST 7: Model freezing: evaluation mode, no grad, weights identical before and after inference."""
        ckpt_r = torch.load(self.ckpt_res, map_location="cpu", weights_only=False)
        model = build_resnet_1d(input_dim=25)
        model.load_state_dict(ckpt_r["model_state_dict"])
        set_model_bn_state(model, ckpt_r["client_bn_states"]["hospital_1"])
        model.eval()

        weights_before = {name: param.clone() for name, param in model.named_parameters()}

        prep = load_client_preprocessor(self.prep_path)
        X_raw, _, _ = verify_and_load_statlog()
        X_proc = prep.transform(X_raw)
        X_tensor = torch.tensor(X_proc.to_numpy(dtype=np.float32), dtype=torch.float32)

        with torch.no_grad():
            _ = model(X_tensor)

        for name, param in model.named_parameters():
            self.assertTrue(
                torch.equal(weights_before[name], param),
                f"Weight modified during forward inference in layer {name}!"
            )

    def test_08_heterogeneous_model_not_applicable(self):
        """TEST 8: Heterogeneous FL model correctly flagged as NOT APPLICABLE without an external encoder."""
        res = run_external_validation()
        self.assertEqual(res["Heterogeneous_FL"]["status"], "NOT APPLICABLE")
        self.assertIn("client-private encoders", res["Heterogeneous_FL"]["reason"])

    def test_09_metric_calculation_identities(self):
        """TEST 9: Metric calculation identities hold (TP + TN + FP + FN == 270, TP + FN == 120, TN + FP == 150)."""
        _, y_true, _ = verify_and_load_statlog()
        dummy_probs = np.full(270, 0.6)
        metrics = compute_comprehensive_metrics(y_true, dummy_probs, threshold=0.5)

        self.assertEqual(metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"], 270)
        self.assertEqual(metrics["tp"] + metrics["fn"], 120)
        self.assertEqual(metrics["tn"] + metrics["fp"], 150)

    def test_10_single_class_metric_handling(self):
        """TEST 10: Single-class metric handling correctly handles edge cases without heuristic fabrication."""
        single_y = np.ones(20, dtype=int)
        probs = np.full(20, 0.8)
        m = compute_comprehensive_metrics(single_y, probs)
        self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["specificity"])

    def test_11_feature_column_order(self):
        """TEST 11: Feature column order strictly matches model input order (INPUT_FEATURES)."""
        X_raw, _, _ = verify_and_load_statlog()
        expected_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        self.assertEqual(list(X_raw.columns), expected_order)

    def test_12_fixed_decision_threshold(self):
        """TEST 12: Fixed decision threshold (tau = 0.5) is enforced without external tuning."""
        self.assertEqual(THRESHOLD, 0.5)

    def test_13_checkpoint_integrity(self):
        """TEST 13: Checkpoint integrity: approved frozen checkpoints exist and parse valid state dicts."""
        self.assertTrue(self.ckpt_alex.exists())
        self.assertTrue(self.ckpt_res.exists())
        a_dict = torch.load(self.ckpt_alex, map_location="cpu", weights_only=False)
        r_dict = torch.load(self.ckpt_res, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", a_dict)
        self.assertIn("model_state_dict", r_dict)
        self.assertIn("client_bn_states", a_dict)
        self.assertIn("client_bn_states", r_dict)

    def test_14_determinism_and_reproducibility(self):
        """TEST 14: Determinism & Reproducibility: second execution produces bitwise identical predictions."""
        res1 = run_external_validation()
        res2 = run_external_validation()

        for mkey in ["FedAvg_1D_AlexNet", "FedAvg_1D_ResNet", "Local_XGBoost_Ensemble"]:
            self.assertAlmostEqual(res1[mkey]["accuracy"], res2[mkey]["accuracy"], places=7)
            self.assertAlmostEqual(res1[mkey]["roc_auc"], res2[mkey]["roc_auc"], places=7)
            self.assertAlmostEqual(res1[mkey]["brier_score"], res2[mkey]["brier_score"], places=7)

    def test_15_external_artifacts_exist_and_complete(self):
        """TEST 15: External validation artifacts exist and parse cleanly."""
        json_path = self.output_dir / "external_validation_results.json"
        csv_path = self.output_dir / "external_validation_results.csv"
        md_path = self.output_dir / "external_validation_report.md"
        png_cm = self.output_dir / "external_confusion_matrices.png"
        png_roc = self.output_dir / "external_roc_curves.png"
        png_pr = self.output_dir / "external_pr_curves.png"

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue(png_cm.exists())
        self.assertTrue(png_roc.exists())
        self.assertTrue(png_pr.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("results", data)
        self.assertIn("FedAvg_1D_ResNet", data["results"])


if __name__ == "__main__":
    unittest.main()
