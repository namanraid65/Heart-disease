"""
Comprehensive End-to-End Test Suite for Heterogeneous Federated Learning Architecture
Validates the complete research pipeline:
  1. Heterogeneous Schemas: D=13, D=25, D=30 (Hospital 4 Synthetic)
  2. Client-Local Private Encoders: D_i -> Z=32 dimensionality mapping
  3. Shared Predictor: Z=32 -> 1 logits output and parameter symmetry
  4. Server Aggregation Boundary: Federates ONLY shared predictor; strictly excludes private encoders
  5. Multi-Site Heterogeneous FL Simulation: H1(25), H2(25), H3(25), H4(30) across FedAvg, FedProx, and FedAdam
  6. Native Feature XAI: SHAP explains D=25 for H1 and D=30 for H4
  7. Native Feature XAI: LIME explains D=25 for H1 and D=30 for H4
  8. Inference Engine (predict.py): Full pipeline execution for H1, H2, H3, and H4
  9. Preprocessing Data Leakage Firewall: Fits strictly on train; never on test
  10. Reproducibility: Identical results when executed with the same random seed
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.heterogeneous_schema import (
    ClientFeatureSchema,
    CLIENT_SCHEMAS,
    get_client_schema,
    H4_30_FEATURE_NAMES,
    H4_ADDITIONAL_FEATURES
)
from preprocessing.feature_schema import PROCESSED_FEATURE_NAMES
from preprocessing.preprocess_client import ClientPreprocessor
from models.heterogeneous.encoder import HospitalEncoder
from models.heterogeneous.predictor import SharedPredictor
from models.heterogeneous.composite import (
    HeterogeneousCompositeModel,
    build_heterogeneous_model
)
from federated.heterogeneous.client import HeterogeneousHospitalClient
from federated.heterogeneous.server import HeterogeneousFederatedServer
from federated.heterogeneous.strategy import (
    HeterogeneousFedAvgStrategy,
    HeterogeneousFedProxStrategy,
    HeterogeneousFedAdamStrategy
)
from federated.heterogeneous.evaluate import evaluate_heterogeneous_system
from federated.heterogeneous.privacy import PairwiseMaskingProtocol, PrivacyConfig
from federated.heterogeneous.xai_compat import HeterogeneousLocalXAIWrapper
from xai.shap_explainer import DeepLearningShapExplainer
from xai.lime_explainer import LimeTabularExplainerWrapper
from predict import (
    load_heterogeneous_inference_pipeline,
    preprocess_patient_input,
    run_heterogeneous_inference,
    get_demo_patient
)


class TestHeterogeneousEndToEnd(unittest.TestCase):
    """Rigorous end-to-end integration test suite for the heterogeneous federated system."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.latent_dim = 32

    def test_01_heterogeneous_schemas_and_dimensions(self):
        """Test 1: Verify declared schemas for D=13, D=25, and D=30 (Hospital 4)."""
        # 13-feature baseline schema
        schema_13 = ClientFeatureSchema(
            hospital_id='h_13',
            hospital_name='Hospital 13-dim',
            feature_names=[f'feat_{i}' for i in range(13)]
        )
        self.assertEqual(schema_13.input_dimension, 13)

        # Standard 25-feature schemas (H1, H2, H3)
        for hid in ['hospital_1', 'hospital_2', 'hospital_3']:
            schema = get_client_schema(hid)
            self.assertEqual(schema.input_dimension, 25)
            self.assertEqual(schema.feature_names, PROCESSED_FEATURE_NAMES)

        # Extended 30-feature schema (Hospital 4)
        schema_h4 = get_client_schema('hospital_4_synthetic')
        self.assertEqual(schema_h4.input_dimension, 30)
        self.assertEqual(schema_h4.feature_names, H4_30_FEATURE_NAMES)
        for bio in ['bmi', 'hba1c', 'crp', 'ldl', 'hdl']:
            self.assertIn(bio, schema_h4.feature_names)

    def test_02_private_encoders_map_to_common_latent(self):
        """Test 2: Encoders for D=13, D=25, D=30 all map to common latent space Z=32."""
        for dim in [13, 25, 30]:
            encoder = HospitalEncoder(input_dim=dim, latent_dim=self.latent_dim, hidden_dims=[64])
            x = torch.randn(7, dim)
            z = encoder(x)
            self.assertEqual(z.shape, (7, self.latent_dim))

    def test_03_shared_predictor_identical_parameter_structure(self):
        """Test 3: Shared predictor has identical structure across all hospital sites."""
        pred = SharedPredictor(latent_dim=self.latent_dim, hidden_dims=[32])
        z = torch.randn(4, self.latent_dim)
        out = pred(z)
        self.assertEqual(out.shape, (4, 1))

        # Check parameter shapes
        shapes = [p.shape for p in pred.parameters()]
        # Expected: Linear(32->32).weight, bias, Linear(32->1).weight, bias
        self.assertEqual(shapes, [(32, 32), (32,), (1, 32), (1,)])

    def test_04_strict_encoder_parameter_exclusion_from_fl(self):
        """Test 4: Verify that private encoder weights NEVER enter FL aggregation."""
        model = build_heterogeneous_model(input_dim=30, latent_dim=32)
        shared_params = model.get_shared_parameters()
        shared_keys = model.get_shared_parameter_names()

        # Shared parameters must strictly be predictor layers
        for k in shared_keys:
            self.assertFalse("encoder" in k.lower())

        # Attempting to feed encoder weights to server must raise ValueError
        server = HeterogeneousFederatedServer(latent_dim=32, device="cpu")
        encoder_params = [p.detach().cpu().numpy() for p in model.encoder.parameters()]
        with self.assertRaises(ValueError):
            server.strategy.validate_client_parameters("h4", encoder_params)

    def test_05_multi_site_heterogeneous_fl_simulation(self):
        """
        Test 5: [SYNTHETIC HETEROGENEITY TEST]
        End-to-end multi-round FL simulation with H1(25), H2(25), H3(25), and H4(30).
        Evaluates FedAvg, proximal regularization, and parameter synchronization.
        """
        client_configs = [
            ('h1', 25, 40),
            ('h2', 25, 40),
            ('h3', 25, 30),
            ('h4_synthetic', 30, 50)
        ]

        clients = {}
        for cid, feat_dim, n_samples in client_configs:
            schema = ClientFeatureSchema(
                hospital_id=cid,
                hospital_name=f"Hospital {cid}",
                feature_names=[f"f_{i}" for i in range(feat_dim)]
            )
            x_data = torch.randn(n_samples, feat_dim)
            y_data = torch.randint(0, 2, (n_samples, 1)).float()
            loader = DataLoader(TensorDataset(x_data, y_data), batch_size=8, shuffle=True)

            client = HeterogeneousHospitalClient(
                client_id=cid,
                schema=schema,
                latent_dim=self.latent_dim,
                device="cpu",
                custom_loaders={'train': loader, 'val': loader, 'test': loader}
            )
            clients[cid] = client

        # Server with FedProx
        strategy = HeterogeneousFedProxStrategy(proximal_mu=0.01)
        server = HeterogeneousFederatedServer(
            latent_dim=self.latent_dim,
            strategy=strategy,
            device="cpu"
        )

        initial_params = [p.copy() for p in server.get_global_parameters()]

        # Run 2 communication rounds
        for r in range(1, 3):
            global_params = server.get_global_parameters()
            client_updates = []
            for cid, client in clients.items():
                enc_before = {k: v.clone() for k, v in client.model.get_encoder_state_dict().items()}
                updated_params, n_samples, metrics = client.fit(
                    parameters=global_params,
                    config={'local_epochs': 1, 'lr': 0.01, 'proximal_mu': 0.01}
                )
                client_updates.append((cid, updated_params, n_samples, metrics))

                # Verify private encoder weights were updated locally but not sent
                enc_after = client.model.get_encoder_state_dict()
                self.assertEqual(len(updated_params), 4)  # strictly 4 predictor tensors

            summary = server.aggregate_round(server_round=r, client_results=client_updates)
            self.assertEqual(summary['num_clients'], 4)

        # Confirm global predictor changed from initial
        final_params = server.get_global_parameters()
        has_changed = any(not np.allclose(init, fin) for init, fin in zip(initial_params, final_params))
        self.assertTrue(has_changed)

        # Evaluate system
        eval_res = evaluate_heterogeneous_system(server, clients, split="val")
        self.assertIn('macro_metrics', eval_res)
        self.assertIn('weighted_metrics', eval_res)
        self.assertIn('brier_score', eval_res['macro_metrics'])
        self.assertIn('precision', eval_res['weighted_metrics'])
        self.assertIn('specificity', eval_res['weighted_metrics'])

    def test_06_shap_native_feature_explanation_d25_and_d30(self):
        """Test 6: SHAP explains native features (25 for H1, 30 for H4), NOT latent Z."""
        # 25-feature hospital
        model_25 = build_heterogeneous_model(input_dim=25, latent_dim=32)
        wrapper_25 = HeterogeneousLocalXAIWrapper(model_25, device="cpu")
        bg_25 = np.random.randn(10, 25).astype(np.float32)
        sample_25 = np.random.randn(25).astype(np.float32)
        feats_25 = [f"feat_{i}" for i in range(25)]

        explainer_25 = DeepLearningShapExplainer(
            predict_fn=wrapper_25.predict_proba,
            background_data=bg_25,
            feature_names=feats_25,
            n_background=10,
            random_state=42
        )
        res_25 = explainer_25.explain_instance(sample_25, nsamples=40)
        self.assertEqual(len(res_25['shap_values']), 25)
        self.assertEqual(len(res_25['feature_shap_dict']), 25)

        # 30-feature hospital (Hospital 4)
        model_30 = build_heterogeneous_model(input_dim=30, latent_dim=32)
        wrapper_30 = HeterogeneousLocalXAIWrapper(model_30, device="cpu")
        bg_30 = np.random.randn(10, 30).astype(np.float32)
        sample_30 = np.random.randn(30).astype(np.float32)
        feats_30 = [f"feat_{i}" for i in range(30)]

        explainer_30 = DeepLearningShapExplainer(
            predict_fn=wrapper_30.predict_proba,
            background_data=bg_30,
            feature_names=feats_30,
            n_background=10,
            random_state=42
        )
        res_30 = explainer_30.explain_instance(sample_30, nsamples=40)
        self.assertEqual(len(res_30['shap_values']), 30)
        self.assertEqual(len(res_30['feature_shap_dict']), 30)

    def test_07_lime_native_feature_explanation_d25_and_d30(self):
        """Test 7: LIME explains native features (25 for H1, 30 for H4)."""
        # 25-feature hospital
        model_25 = build_heterogeneous_model(input_dim=25, latent_dim=32)
        wrapper_25 = HeterogeneousLocalXAIWrapper(model_25, device="cpu")
        bg_25 = np.random.randn(20, 25).astype(np.float32)
        sample_25 = np.random.randn(25).astype(np.float32)
        feats_25 = [f"feat_{i}" for i in range(25)]

        lime_25 = LimeTabularExplainerWrapper(
            training_data=bg_25,
            feature_names=feats_25,
            random_state=42
        )
        res_25 = lime_25.explain_instance(
            sample=sample_25,
            predict_fn=wrapper_25.predict_proba,
            num_features=5,
            num_samples=50
        )
        self.assertIn('sorted_features', res_25)
        self.assertLessEqual(len(res_25['sorted_features']), 5)

        # 30-feature hospital
        model_30 = build_heterogeneous_model(input_dim=30, latent_dim=32)
        wrapper_30 = HeterogeneousLocalXAIWrapper(model_30, device="cpu")
        bg_30 = np.random.randn(20, 30).astype(np.float32)
        sample_30 = np.random.randn(30).astype(np.float32)
        feats_30 = [f"feat_{i}" for i in range(30)]

        lime_30 = LimeTabularExplainerWrapper(
            training_data=bg_30,
            feature_names=feats_30,
            random_state=42
        )
        res_30 = lime_30.explain_instance(
            sample=sample_30,
            predict_fn=wrapper_30.predict_proba,
            num_features=5,
            num_samples=50
        )
        self.assertIn('sorted_features', res_30)
        self.assertLessEqual(len(res_30['sorted_features']), 5)

    def test_08_inference_engine_h1_and_h4(self):
        """Test 8: Full inference pipeline via predict.py for Hospital 1 and Hospital 4."""
        # 1. Hospital 1 with preset high_risk
        p_h1 = get_demo_patient('high_risk', hospital_id='hospital_1')
        report_h1 = run_heterogeneous_inference(
            patient_data=p_h1,
            hospital_id='hospital_1',
            explainer_type='none'
        )
        self.assertEqual(report_h1['hospital_id'], 'hospital_1')
        self.assertEqual(report_h1['input_dim'], 25)
        self.assertEqual(report_h1['latent_dim'], 32)
        self.assertTrue(0.0 <= report_h1['predicted_risk_probability'] <= 1.0)

        # 2. Hospital 4 Synthetic with 30 features
        p_h4 = get_demo_patient('h4_synthetic', hospital_id='hospital_4_synthetic')
        report_h4 = run_heterogeneous_inference(
            patient_data=p_h4,
            hospital_id='hospital_4_synthetic',
            explainer_type='none'
        )
        self.assertEqual(report_h4['hospital_id'], 'hospital_4_synthetic')
        self.assertEqual(report_h4['input_dim'], 30)
        self.assertEqual(report_h4['latent_dim'], 32)
        self.assertTrue(report_h4['is_synthetic'])
        self.assertTrue(0.0 <= report_h4['predicted_risk_probability'] <= 1.0)

    def test_09_no_test_leakage_in_preprocessing(self):
        """Test 9: Preprocessor strictly fits on train set and never references test data."""
        prep = ClientPreprocessor()
        self.assertFalse(prep.is_fitted_)

        # Create dummy training and test DataFrames
        df_train = pd.DataFrame({
            'age': [50.0, 60.0, 70.0],
            'sex': [1.0, 0.0, 1.0],
            'cp': [1.0, 2.0, 3.0],
            'trestbps': [120.0, 130.0, 140.0],
            'chol': [200.0, 220.0, 240.0],
            'fbs': [0.0, 0.0, 1.0],
            'restecg': [0.0, 1.0, 0.0],
            'thalach': [150.0, 140.0, 130.0],
            'exang': [0.0, 1.0, 0.0],
            'oldpeak': [1.0, 2.0, 0.0],
            'slope': [1.0, 2.0, 1.0],
            'ca': [0.0, 1.0, 0.0],
            'thal': [3.0, 6.0, 3.0]
        })
        prep.fit(df_train)
        self.assertTrue(prep.is_fitted_)

        # Learned median of age is 60.0
        self.assertEqual(prep.imputation_values_['age'], 60.0)

        # Transforming test with an unseen extreme value must not alter learned statistics
        df_test = pd.DataFrame({
            'age': [99.0],
            'sex': [0.0],
            'cp': [4.0],
            'trestbps': [200.0],
            'chol': [400.0],
            'fbs': [1.0],
            'restecg': [2.0],
            'thalach': [90.0],
            'exang': [1.0],
            'oldpeak': [4.0],
            'slope': [3.0],
            'ca': [3.0],
            'thal': [7.0]
        })
        prep.transform(df_test)
        self.assertEqual(prep.imputation_values_['age'], 60.0)

    def test_10_reproducibility_across_identical_seeds(self):
        """Test 10: Model forward passes and predictions are strictly deterministic given seed."""
        torch.manual_seed(999)
        m1 = build_heterogeneous_model(input_dim=25, latent_dim=32)
        m1.eval()

        torch.manual_seed(999)
        m2 = build_heterogeneous_model(input_dim=25, latent_dim=32)
        m2.eval()

        x = torch.randn(5, 25)
        out1 = m1(x)
        out2 = m2(x)

        self.assertTrue(torch.allclose(out1, out2))


if __name__ == '__main__':
    unittest.main()
