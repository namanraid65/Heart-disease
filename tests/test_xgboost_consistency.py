"""
Unit and Integration Tests for Phase 4:
XGBoost Terminology Correction and Preprocessing Consistency
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from preprocessing.feature_schema import INPUT_FEATURES, PROCESSED_FEATURE_NAMES, NUM_PROCESSED_FEATURES
from xai.model_loader import (
    SampleWeightedXGBoostEnsemble,
    LocalXGBoostEnsemble,
    FederatedXGBoostModel,
    load_federated_model
)
from xai.shap_xgboost import (
    FederatedXGBoostShapExplainer,
    LocalXGBoostEnsembleShapExplainer,
    SampleWeightedXGBoostShapExplainer
)
from evaluation.result_loader import ExperimentResultLoader


class TestXGBoostConsistency(unittest.TestCase):
    """Test suite validating XGBoost terminology and preprocessing consistency."""

    def test_01_class_aliases(self):
        """Verify that SampleWeightedXGBoostEnsemble, LocalXGBoostEnsemble, and FederatedXGBoostModel are aliases."""
        self.assertIs(LocalXGBoostEnsemble, SampleWeightedXGBoostEnsemble)
        self.assertIs(FederatedXGBoostModel, SampleWeightedXGBoostEnsemble)

    def test_02_factory_loader_aliases(self):
        """Verify factory function load_federated_model accepts all accepted alias names."""
        for alias in ['Local_XGBoost_Ensemble', 'Sample_Weighted_XGBoost_Ensemble', 'Federated_XGBoost']:
            model = load_federated_model(alias)
            self.assertIsInstance(model, SampleWeightedXGBoostEnsemble)

    def test_03_preprocessors_loaded(self):
        """Verify that all client preprocessors (hospital 1, 2, 3) are loaded and marked fitted."""
        ensemble = SampleWeightedXGBoostEnsemble()
        for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
            prep = ensemble.get_client_preprocessor(cid)
            self.assertIsNotNone(prep, f"Preprocessor for {cid} was not loaded.")
            self.assertTrue(getattr(prep, 'is_fitted_', False), f"Preprocessor for {cid} is not marked fitted.")

    def test_04_raw_patient_prediction(self):
        """Verify that raw patient DataFrame (13 features) is processed through per-client preprocessors."""
        ensemble = SampleWeightedXGBoostEnsemble()
        raw_patient = pd.DataFrame([{
            'age': 55.0,
            'sex': 1.0,
            'cp': 4.0,
            'trestbps': 130.0,
            'chol': 240.0,
            'fbs': 0.0,
            'restecg': 0.0,
            'thalach': 145.0,
            'exang': 1.0,
            'oldpeak': 1.5,
            'slope': 2.0,
            'ca': 0.0,
            'thal': 3.0
        }])

        probs = ensemble.predict_proba(raw_patient)
        self.assertEqual(probs.shape, (1, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

        preds = ensemble.predict(raw_patient)
        self.assertEqual(preds.shape, (1,))
        self.assertIn(int(preds[0]), [0, 1])

    def test_05_missing_feature_handling_in_raw(self):
        """Verify that missing raw features (e.g. chol=0 or NaN) are handled by client imputation values."""
        ensemble = SampleWeightedXGBoostEnsemble()
        raw_patient = pd.DataFrame([{
            'age': 60.0,
            'sex': 0.0,
            'cp': 2.0,
            'trestbps': 120.0,
            'chol': 0.0,  # 0 indicates unrecorded, to be cleaned to NaN and imputed
            'fbs': 0.0,
            'restecg': 1.0,
            'thalach': 160.0,
            'exang': 0.0,
            'oldpeak': 0.0,
            'slope': 1.0,
            'ca': 0.0,
            'thal': 3.0
        }])

        probs = ensemble.predict_proba(raw_patient)
        self.assertEqual(probs.shape, (1, 2))
        self.assertFalse(np.isnan(probs).any(), "Prediction output contains NaN.")

    def test_06_preprocessed_backward_compatibility(self):
        """Verify that passing 25-feature processed input directly still functions properly."""
        ensemble = SampleWeightedXGBoostEnsemble()
        processed_input = np.random.randn(4, NUM_PROCESSED_FEATURES).astype(np.float32)

        probs = ensemble.predict_proba(processed_input)
        self.assertEqual(probs.shape, (4, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_07_constituent_client_models(self):
        """Verify access to underlying client XGBoost models."""
        ensemble = SampleWeightedXGBoostEnsemble()
        for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
            clf = ensemble.get_client_model(cid)
            self.assertTrue(clf.is_fitted)

    def test_08_xai_shap_aliases(self):
        """Verify SHAP explainer class aliases."""
        self.assertIs(LocalXGBoostEnsembleShapExplainer, FederatedXGBoostShapExplainer)
        self.assertIs(SampleWeightedXGBoostShapExplainer, FederatedXGBoostShapExplainer)

    def test_09_result_loader_evaluation(self):
        """Verify ExperimentResultLoader evaluates XGBoost with per-client preprocessing."""
        loader = ExperimentResultLoader()
        res = loader.evaluate_model_on_client('XGBoost', 'Federated', 'hospital_1')
        self.assertEqual(res['client_id'], 'hospital_1')
        self.assertIn('Accuracy', res)
        self.assertIn('ROC AUC', res)
        self.assertGreater(res['Accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
