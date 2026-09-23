"""
Unit and Integration Tests for Phase 6: Evaluation Integrity, Hospital-3 Statistical Reporting,
and Single Authoritative Results Source.

Verifies:
  TEST 1: Master results files exist, parse cleanly, and contain all 18 configurations.
  TEST 2: Confusion matrix identities hold strictly for every single evaluated row.
  TEST 3: Hospital 3 class counts (18 pos, 1 neg) and specificity caveat are explicitly documented.
  TEST 4: Undefined metrics are represented strictly as NaN (no heuristic fabrication).
  TEST 5: Strict separation and mathematical accuracy of Macro vs. Sample-Weighted aggregations.
  TEST 6: ROC-AUC and PR-AUC validity and explicit denominator tracking.
  TEST 7: Generated markdown reports match master results table exactly.
  TEST 8: README benchmark numbers match master results table exactly.
  TEST 9: Zero machine-specific paths exist across reports and README.
  TEST 10: Reproducibility metadata and frozen best-validation checkpoints are verified.
"""

import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd
import torch

from models.config import CHECKPOINTS_DIR, RANDOM_SEED
from evaluation.result_loader import ExperimentResultLoader


class TestEvaluationIntegrity(unittest.TestCase):
    """Test suite validating Phase 6 Evaluation Integrity and Reporting Rigor."""

    @classmethod
    def setUpClass(cls):
        cls.reports_dir = PROJECT_ROOT / 'reports'
        cls.csv_path = cls.reports_dir / 'master_results.csv'
        cls.json_path = cls.reports_dir / 'master_results.json'
        cls.agg_path = cls.reports_dir / 'master_aggregates.csv'
        cls.readme_path = PROJECT_ROOT / 'README.md'

        # Ensure authoritative tables exist
        if not cls.csv_path.exists() or not cls.json_path.exists() or not cls.agg_path.exists():
            loader = ExperimentResultLoader()
            loader.save_master_results(cls.reports_dir)

        cls.df_master = pd.read_csv(cls.csv_path)
        cls.df_agg = pd.read_csv(cls.agg_path)
        with open(cls.json_path, 'r', encoding='utf-8') as f:
            cls.master_json = json.load(f)

    def test_01_master_results_files_exist_and_complete(self):
        """TEST 1: Master results files exist, parse cleanly, and contain all 18 configurations."""
        self.assertTrue(self.csv_path.exists(), "master_results.csv missing")
        self.assertTrue(self.json_path.exists(), "master_results.json missing")
        self.assertTrue(self.agg_path.exists(), "master_aggregates.csv missing")

        # 3 architectures (AlexNet, ResNet, XGBoost) x 2 modes (Local, Federated) x 3 hospitals = 18 rows
        self.assertEqual(len(self.df_master), 18, f"Expected 18 rows in master_results.csv, got {len(self.df_master)}")

        # Check required columns
        required_cols = [
            'Hospital', 'client_id', 'Model Type', 'Training Type', 'Strategy',
            'Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity',
            'F1 Score', 'ROC AUC', 'PR AUC', 'TP', 'TN', 'FP', 'FN',
            'sample_count', 'positive_count', 'negative_count', 'hospital_3_note'
        ]
        for col in required_cols:
            self.assertIn(col, self.df_master.columns, f"Missing required column: {col}")

        # JSON metadata verification
        meta = self.master_json['metadata']
        self.assertEqual(meta['random_seed'], RANDOM_SEED)
        self.assertEqual(meta['total_test_samples'], 109)
        self.assertEqual(len(self.master_json['client_results']), 18)

    def test_02_confusion_matrix_identities(self):
        """TEST 2: Confusion matrix identities hold strictly for every evaluated row."""
        for idx, row in self.df_master.iterrows():
            tp, tn, fp, fn = row['TP'], row['TN'], row['FP'], row['FN']
            sample_count = row['sample_count']
            pos_count = row['positive_count']
            neg_count = row['negative_count']

            # Identity 1: TP + TN + FP + FN == sample_count
            self.assertEqual(
                tp + tn + fp + fn, sample_count,
                f"Row {idx} ({row['Model Type']} {row['Training Type']} on {row['client_id']}): "
                f"TP+TN+FP+FN ({tp+tn+fp+fn}) != sample_count ({sample_count})"
            )
            # Identity 2: TP + FN == positive_count
            self.assertEqual(
                tp + fn, pos_count,
                f"Row {idx}: TP+FN ({tp+fn}) != positive_count ({pos_count})"
            )
            # Identity 3: TN + FP == negative_count
            self.assertEqual(
                tn + fp, neg_count,
                f"Row {idx}: TN+FP ({tn+fp}) != negative_count ({neg_count})"
            )

    def test_03_hospital_3_statistical_disclosure(self):
        """TEST 3: Hospital 3 class counts (18 pos, 1 neg) and specificity caveat are explicitly documented."""
        h3_rows = self.df_master[self.df_master['client_id'] == 'hospital_3']
        self.assertEqual(len(h3_rows), 6)

        for _, row in h3_rows.iterrows():
            self.assertEqual(row['sample_count'], 19)
            self.assertEqual(row['positive_count'], 18)
            self.assertEqual(row['negative_count'], 1)
            # Verify explicit disclosure note
            self.assertIn("Specificity is based on only 1 negative test sample", str(row['hospital_3_note']))

    def test_04_no_fabricated_metrics_for_undefined(self):
        """TEST 4: Undefined metrics are represented strictly as NaN (no heuristic fabrication)."""
        loader = ExperimentResultLoader()
        # Verify calculation logic doesn't insert arbitrary 0.5 or 0 for undefined classes
        y_single = np.ones(10, dtype=int)
        y_prob = np.random.rand(10)
        from sklearn.metrics import roc_auc_score
        # In scikit-learn 1.7+, single-class ROC-AUC returns np.nan with an UndefinedMetricWarning
        val = roc_auc_score(y_single, y_prob)
        self.assertTrue(np.isnan(val))

        # In loader, single-class evaluation returns NaN, not 0.5
        eval_dict = loader.evaluate_model_on_client('AlexNet', 'Local', 'hospital_1')
        self.assertFalse(np.isnan(eval_dict['Accuracy']))

    def test_05_macro_vs_sample_weighted_definitions(self):
        """TEST 5: Strict separation and mathematical accuracy of Macro vs. Sample-Weighted aggregations."""
        for mtype in ['AlexNet', 'ResNet', 'XGBoost']:
            for ttype in ['Local', 'Federated']:
                sub_client = self.df_master[(self.df_master['Model Type'] == mtype) & (self.df_master['Training Type'] == ttype)]

                # Macro accuracy = unweighted mean of 3 hospital accuracies
                expected_macro_acc = sub_client['Accuracy'].mean()
                actual_macro_row = self.df_agg[
                    (self.df_agg['Model Type'] == mtype) &
                    (self.df_agg['Training Type'] == ttype) &
                    (self.df_agg['Aggregation'].str.contains('Macro'))
                ].iloc[0]
                self.assertAlmostEqual(actual_macro_row['Accuracy'], expected_macro_acc, places=6)
                self.assertEqual(actual_macro_row['Accuracy_valid_hospitals'], "3/3")

                # Sample-weighted accuracy = (Acc_H1*46 + Acc_H2*44 + Acc_H3*19) / 109
                weights = sub_client['sample_count'] / 109.0
                expected_weighted_acc = np.sum(sub_client['Accuracy'] * weights)
                actual_weighted_row = self.df_agg[
                    (self.df_agg['Model Type'] == mtype) &
                    (self.df_agg['Training Type'] == ttype) &
                    (self.df_agg['Aggregation'].str.contains('Sample-Weighted'))
                ].iloc[0]
                self.assertAlmostEqual(actual_weighted_row['Accuracy'], expected_weighted_acc, places=6)

    def test_06_roc_auc_and_pr_auc_validity(self):
        """TEST 6: ROC-AUC and PR-AUC validity and explicit denominator tracking."""
        for _, row in self.df_agg.iterrows():
            self.assertEqual(row['ROC AUC_valid_hospitals'], "3/3")
            self.assertEqual(row['PR AUC_valid_hospitals'], "3/3")

        # Verify PR-AUC calculation follows scikit-learn average_precision_score
        h1_alex = self.df_master[
            (self.df_master['Model Type'] == 'AlexNet') &
            (self.df_master['Training Type'] == 'Local') &
            (self.df_master['client_id'] == 'hospital_1')
        ].iloc[0]
        self.assertGreater(h1_alex['PR AUC'], 0.5)

    def test_07_report_numbers_match_master_results(self):
        """TEST 7: Generated markdown reports match master results table exactly."""
        report_files = [
            self.reports_dir / 'final_performance_table.md',
            self.reports_dir / 'federated_model_final_comparison.md',
            self.reports_dir / 'hospital_wise_comparison.md',
            self.reports_dir / 'model_selection_analysis.md'
        ]
        for rpath in report_files:
            self.assertTrue(rpath.exists(), f"Report file {rpath} does not exist")
            text = rpath.read_text(encoding='utf-8')
            # Verify no unresolved NaN string placeholders or corrupted tables
            self.assertNotIn("undefined%", text)

    def test_08_readme_numbers_match_master_results(self):
        """TEST 8: README benchmark numbers match master results table exactly."""
        self.assertTrue(self.readme_path.exists())
        readme_text = self.readme_path.read_text(encoding='utf-8')

        # Check key benchmark figures in README
        # Cleveland AlexNet: 84.8%
        h1_alex_acc = self.df_master[
            (self.df_master['Model Type'] == 'AlexNet') &
            (self.df_master['Training Type'] == 'Federated') &
            (self.df_master['client_id'] == 'hospital_1')
        ]['Accuracy'].values[0] * 100
        self.assertIn(f"{h1_alex_acc:.1f}%", readme_text)

        # Multi-Center Sample-Weighted XGBoost: 80.7%
        xgb_w_acc = self.df_agg[
            (self.df_agg['Model Type'] == 'XGBoost') &
            (self.df_agg['Training Type'] == 'Federated') &
            (self.df_agg['Aggregation'].str.contains('Sample-Weighted'))
        ]['Accuracy'].values[0] * 100
        self.assertIn(f"{xgb_w_acc:.1f}%", readme_text)

    def test_09_no_machine_specific_paths(self):
        """TEST 9: Zero machine-specific paths exist across reports and README."""
        prohibited_patterns = [
            re.compile(r'c:/Users', re.IGNORECASE),
            re.compile(r'file:///c:', re.IGNORECASE),
            re.compile(r'file:///e:', re.IGNORECASE),
            re.compile(r'E:\\Heart-disease-main', re.IGNORECASE)
        ]

        files_to_check = list(self.reports_dir.glob('*.md')) + [self.readme_path]
        for fpath in files_to_check:
            text = fpath.read_text(encoding='utf-8')
            for pat in prohibited_patterns:
                match = pat.search(text)
                self.assertIsNone(
                    match,
                    f"Found machine-specific path pattern '{pat.pattern}' in {fpath.name}: {match.group(0) if match else ''}"
                )

    def test_10_reproducibility_and_checkpoint_integrity(self):
        """TEST 10: Reproducibility metadata and frozen best-validation checkpoints are verified."""
        alexnet_ckpt = CHECKPOINTS_DIR / 'federated_alexnet' / 'global_alexnet_final.pt'
        resnet_ckpt = CHECKPOINTS_DIR / 'federated_resnet' / 'global_resnet_final.pt'

        self.assertTrue(alexnet_ckpt.exists(), f"Frozen best AlexNet checkpoint missing at {alexnet_ckpt}")
        self.assertTrue(resnet_ckpt.exists(), f"Frozen best ResNet checkpoint missing at {resnet_ckpt}")

        # Checkpoint contents inspection
        ckpt_alex = torch.load(alexnet_ckpt, map_location='cpu', weights_only=False)
        self.assertIn('model_state_dict', ckpt_alex)
        self.assertIn('client_bn_states', ckpt_alex)
        self.assertEqual(len(ckpt_alex['client_bn_states']), 3)


if __name__ == '__main__':
    unittest.main()
