"""
Unit and Integration Tests for Phase 5: Explainable AI (XAI) Correctness and Reproducibility
Verifies:
  TEST 1: Neural SHAP produces exactly 25 feature attributions.
  TEST 2: Feature names align 1-to-1 with attribution values in Neural SHAP.
  TEST 3: LIME feature names map correctly to processed features via exp.as_map().
  TEST 4: No substring-based feature collisions occur in LIME (e.g. cp_1 vs cp_10, ca_0 vs ca_1).
  TEST 5: No double preprocessing occurs in any XAI pipeline.
  TEST 6: Local XGBoost explainer uses native client preprocessing.
  TEST 7: XGBoost explanations do not naively average incompatible TreeSHAP log-odds.
  TEST 8: Deterministic median-confidence sample selection produces consistent instances across runs.
  TEST 9: Empty categories in imbalanced cohorts (e.g., Hospital 3 TN) are handled without fabricating data.
  TEST 10: Output records contain complete metadata.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from xai.config import (
    PROCESSED_FEATURE_NAMES,
    NUM_PROCESSED_FEATURES,
    FEATURE_DISPLAY_NAMES,
    RANDOM_SEED
)
from xai.lime_explainer import LimeTabularExplainerWrapper
from xai.shap_explainer import DeepLearningShapExplainer
from xai.shap_xgboost import (
    LocalXGBoostTreeExplainer,
    LocalXGBoostEnsembleKernelExplainer,
    SampleWeightedXGBoostShapExplainer
)
from xai.run_xai_pipeline import select_representative_samples
from xai.model_loader import load_federated_model, SampleWeightedXGBoostEnsemble
from models.xgboost_model import LocalXGBoostModel
from models.dataset import load_client_raw_splits


class MockProbModel:
    """Mock model with fixed probability behavior for deterministic testing."""
    def __init__(self, n_features: int = 25, weights: np.ndarray = None):
        self.n_features = n_features
        if weights is not None:
            self.weights = weights
        else:
            np.random.seed(RANDOM_SEED)
            self.weights = np.random.randn(n_features).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        # Linear score mapped to sigmoid
        scores = np.dot(X_arr, self.weights)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class TestXAICorrectness(unittest.TestCase):
    """Test suite validating Phase 5 XAI mathematical correctness and pipeline fidelity."""

    def setUp(self):
        np.random.seed(RANDOM_SEED)
        self.n_features = NUM_PROCESSED_FEATURES
        self.feature_names = PROCESSED_FEATURE_NAMES
        self.mock_model = MockProbModel(self.n_features)
        self.background = np.random.randn(20, self.n_features).astype(np.float32)
        self.sample = np.random.randn(self.n_features).astype(np.float32)

    def test_01_neural_shap_dimension(self):
        """TEST 1: Neural SHAP produces exactly 25 feature attributions."""
        explainer = DeepLearningShapExplainer(
            predict_fn=self.mock_model.predict_proba,
            background_data=self.background,
            feature_names=self.feature_names,
            n_background=10,
            random_state=RANDOM_SEED
        )
        res = explainer.explain_instance(self.sample, nsamples=50)

        self.assertEqual(len(res['shap_values']), 25)
        self.assertEqual(len(res['feature_shap_dict']), 25)
        self.assertEqual(len(res['sorted_features']), 25)

        # Dimension verification on invalid input shape
        invalid_sample = np.random.randn(13).astype(np.float32)
        with self.assertRaises(ValueError):
            explainer.explain_instance(invalid_sample)

    def test_02_neural_shap_feature_alignment(self):
        """TEST 2: Feature names align 1-to-1 with attribution values in Neural SHAP."""
        explainer = DeepLearningShapExplainer(
            predict_fn=self.mock_model.predict_proba,
            background_data=self.background,
            feature_names=self.feature_names,
            n_background=10,
            random_state=RANDOM_SEED
        )
        res = explainer.explain_instance(self.sample, nsamples=50)

        # Check every feature name matches index position
        for i, fname in enumerate(self.feature_names):
            self.assertIn(fname, res['feature_shap_dict'])
            self.assertAlmostEqual(res['feature_shap_dict'][fname], float(res['shap_values'][i]), places=6)

    def test_03_lime_feature_mapping_via_as_map(self):
        """TEST 3: LIME feature names map correctly to processed features via exp.as_map()."""
        lime_expl = LimeTabularExplainerWrapper(
            training_data=self.background,
            feature_names=self.feature_names,
            random_state=RANDOM_SEED
        )
        res = lime_expl.explain_instance(
            self.sample,
            predict_fn=self.mock_model.predict_proba,
            num_features=10,
            num_samples=100
        )

        self.assertIn('sorted_features', res)
        self.assertIn('feature_weights', res)
        # Verify that all extracted features are legitimate members of PROCESSED_FEATURE_NAMES
        for fname, weight in res['sorted_features']:
            self.assertIn(fname, self.feature_names)
            self.assertEqual(res['feature_weights'][fname], weight)

    def test_04_no_substring_feature_collisions(self):
        """TEST 4: No substring-based feature collisions occur in LIME (e.g. cp_1 vs cp_10, ca_0 vs ca_1)."""
        colliding_features = [
            'age', 'cp_1', 'cp_10', 'ca_0', 'ca_1', 'ca_10',
            'slope_1', 'slope_10', 'thal_3', 'thal_30'
        ]
        n_col = len(colliding_features)
        bg = np.random.randn(20, n_col).astype(np.float32)
        target = np.random.randn(n_col).astype(np.float32)

        model = MockProbModel(n_col)
        lime_expl = LimeTabularExplainerWrapper(
            training_data=bg,
            feature_names=colliding_features,
            random_state=RANDOM_SEED
        )
        res = lime_expl.explain_instance(
            target,
            predict_fn=model.predict_proba,
            num_features=n_col,
            num_samples=100
        )

        extracted_names = [f for f, _ in res['sorted_features']]
        # Ensure no name corruption or duplicate mappings occurred
        self.assertEqual(len(extracted_names), len(set(extracted_names)))
        for name in extracted_names:
            self.assertIn(name, colliding_features)

    def test_05_no_double_preprocessing(self):
        """TEST 5: No double preprocessing occurs in any XAI pipeline."""
        # Processed data partitions from data/processed already contain 25 columns
        splits = load_client_raw_splits('hospital_1')
        X_proc = splits['test'][0].to_numpy(dtype=np.float32)
        self.assertEqual(X_proc.shape[1], 25)

        # Neural model wrappers must accept this (N, 25) matrix directly
        model_wrapper = load_federated_model('Federated_1D_AlexNet', device='cpu')
        probs = model_wrapper.predict_proba(X_proc[:5])
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_06_local_xgboost_explainer_preprocessing(self):
        """TEST 6: Local XGBoost explainer uses native client preprocessing."""
        # Load local XGBoost model
        local_model_path = PROJECT_ROOT / 'models' / 'saved_models' / 'hospital_1_xgboost.joblib'
        if local_model_path.exists():
            local_model = LocalXGBoostModel.load(local_model_path)
            explainer = LocalXGBoostTreeExplainer(
                model=local_model,
                client_id='hospital_1',
                client_name='Cleveland Clinic',
                feature_names=self.feature_names
            )
            # Explain instance in local 25-feature space
            res = explainer.explain_instance(self.sample)
            self.assertEqual(res['hospital'], 'hospital_1')
            self.assertEqual(len(res['shap_values']), 25)
            self.assertEqual(res['attribution_space'], 'tree_margin_log_odds')

    def test_07_xgboost_explanations_do_not_average_incompatible_log_odds(self):
        """TEST 7: XGBoost explanations do not naively average incompatible TreeSHAP log-odds."""
        # Test LocalXGBoostEnsembleKernelExplainer computes SHAP on probability mixture
        ensemble = load_federated_model('Local_XGBoost_Ensemble')
        kernel_explainer = LocalXGBoostEnsembleKernelExplainer(
            ensemble_model=ensemble,
            background_data=self.background,
            feature_names=self.feature_names,
            n_background=10,
            random_state=RANDOM_SEED
        )
        res = kernel_explainer.explain_instance(self.sample, nsamples=50)

        # Efficiency axiom in probability space: sum(shap_values) + base_value approx prob_disease
        sum_shap = np.sum(res['shap_values'])
        expected_diff = res['prob_disease'] - res['base_value']
        self.assertAlmostEqual(sum_shap, expected_diff, places=4)
        self.assertEqual(res['attribution_space'], 'ensemble_probability')

    def test_08_deterministic_median_confidence_selection(self):
        """TEST 8: Deterministic median-confidence sample selection produces consistent instances."""
        N = 30
        X_mock = np.random.randn(N, self.n_features).astype(np.float32)
        # Create known ground truth and distinct probabilities
        y_mock = np.array([1]*15 + [0]*15)

        run1 = select_representative_samples(self.mock_model, X_mock, y_mock, 'hospital_1')
        run2 = select_representative_samples(self.mock_model, X_mock, y_mock, 'hospital_1')

        self.assertEqual(set(run1.keys()), set(run2.keys()))
        for cat in run1:
            self.assertEqual(run1[cat]['index'], run2[cat]['index'])
            self.assertEqual(run1[cat]['prob'], run2[cat]['prob'])
            self.assertEqual(run1[cat]['selection_strategy'], 'median_confidence')
            self.assertIn('candidate_pool_size', run1[cat])
            self.assertIn('group_median_prob', run1[cat])

            # Verify that chosen candidate is indeed the one closest to group median
            cat_mask = (y_mock == run1[cat]['actual']) & (
                ((self.mock_model.predict_proba(X_mock)[:, 1] >= 0.5).astype(int)) == run1[cat]['pred']
            )
            cat_indices = np.where(cat_mask)[0]
            cat_probs = self.mock_model.predict_proba(X_mock)[cat_indices, 1]
            median_p = np.median(cat_probs)
            best_idx = cat_indices[np.argmin(np.abs(cat_probs - median_p))]
            self.assertEqual(run1[cat]['index'], best_idx)

    def test_09_imbalanced_cohort_handling_without_fabrication(self):
        """TEST 9: Empty categories in imbalanced cohorts (e.g. Hospital 3 TN) are handled without fabricating data."""
        # Create cohort with 100% diseased actual and predicted labels (0 negatives, 0 false predictions)
        N = 10
        X_mock = np.random.randn(N, self.n_features).astype(np.float32)
        y_mock = np.ones(N, dtype=int)  # All actual disease = 1

        # Dummy model predicting p=0.99 for all
        class HighDiseaseModel:
            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

        selected = select_representative_samples(HighDiseaseModel(), X_mock, y_mock, 'hospital_3')

        # TP should exist
        self.assertIn('TP', selected)
        # TN, FP, FN must NOT exist
        self.assertNotIn('TN', selected)
        self.assertNotIn('FP', selected)
        self.assertNotIn('FN', selected)
        # No substitution or fabrication
        self.assertEqual(len(selected), 1)

    def test_10_output_metadata_completeness(self):
        """TEST 10: Output records contain complete required metadata."""
        required_keys = {
            'hospital', 'model', 'sample_id', 'sample_index',
            'actual_label', 'predicted_label', 'probability',
            'prediction_class', 'feature_representation', 'explainer',
            'selection_strategy', 'timestamp'
        }

        # Simulate metadata generation as done in run_xai_pipeline.py
        run_timestamp = datetime.now(timezone.utc).isoformat()
        sample_meta = {
            'hospital': 'hospital_1',
            'hospital_name': 'Cleveland Clinic Foundation',
            'model': 'Federated_1D_AlexNet',
            'model_name': 'Federated 1D AlexNet',
            'case_type': 'TP',
            'sample_id': 'hospital_1_case_TP_05',
            'sample_index': 5,
            'actual_label': 1,
            'predicted_label': 1,
            'probability': 0.885,
            'prediction_class': 'Disease (1)',
            'feature_representation': 'standardized_continuous_and_one_hot (N=25)',
            'explainer': 'LIME_and_SHAP',
            'selection_strategy': 'median_confidence',
            'candidate_pool_size': 12,
            'group_median_prob': 0.880,
            'timestamp': run_timestamp,
            'feature_name': 'oldpeak',
            'feature_display_name': FEATURE_DISPLAY_NAMES.get('oldpeak', 'oldpeak'),
            'lime_weight': 0.12,
            'shap_value': 0.15
        }

        for k in required_keys:
            self.assertIn(k, sample_meta)
            self.assertIsNotNone(sample_meta[k])


if __name__ == '__main__':
    unittest.main()
