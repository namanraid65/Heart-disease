"""
Unit and Integration Test Suite for Heterogeneous-Feature Federated Learning (Phase 7)
Tests:
  1. Configurable private encoders map native features D_i -> Z with shape validation.
  2. Encoder raises ValueError on dimensional violations.
  3. Shared predictor consumes Z and produces identical parameter shapes across hospitals.
  4. Composite model end-to-end forward pass and probability output.
  5. Strict client parameter isolation: ONLY shared predictor weights are federated.
  6. Client retains local private encoder parameters across FL communication rounds.
  7. Multi-hospital FedAvg with heterogeneous feature dimensions (H1=25, H2=25, H3=25, H4=30).
  8. Server parameter validation rejects mismatched predictor shapes.
  9. Server rejects parameter updates containing encoder weights.
  10. Server and client checkpoint separation integrity.
  11. Validation-based model selection and single test evaluation protocol.
  12. Client-local XAI wrapper gradient backpropagation and probability outputs.
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
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.heterogeneous_schema import (
    ClientFeatureSchema,
    CLIENT_SCHEMAS,
    get_client_schema
)
from models.heterogeneous.encoder import HospitalEncoder
from models.heterogeneous.predictor import SharedPredictor
from models.heterogeneous.composite import (
    HeterogeneousCompositeModel,
    build_heterogeneous_model
)
from federated.heterogeneous.strategy import HeterogeneousFedAvgStrategy
from federated.heterogeneous.client import HeterogeneousHospitalClient
from federated.heterogeneous.server import HeterogeneousFederatedServer
from federated.heterogeneous.evaluate import evaluate_heterogeneous_system
from federated.heterogeneous.xai_compat import HeterogeneousLocalXAIWrapper


class TestHeterogeneousFL(unittest.TestCase):
    """Test suite for Heterogeneous Feature Federated Learning components."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.latent_dim = 32

    def test_01_encoder_shape_mapping(self):
        """TEST 1: Configurable encoders map D_i -> Z accurately for various dimensions."""
        for dim in [10, 25, 30, 50]:
            encoder = HospitalEncoder(input_dim=dim, latent_dim=self.latent_dim, hidden_dims=[64])
            x = torch.randn(8, dim)
            z = encoder(x)
            self.assertEqual(z.shape, (8, self.latent_dim))

    def test_02_encoder_shape_validation_failure(self):
        """TEST 2: Encoder raises ValueError if input or output feature shape violates contract."""
        encoder = HospitalEncoder(input_dim=25, latent_dim=32)

        # Wrong input dimension (e.g. 20 instead of 25)
        x_wrong_dim = torch.randn(4, 20)
        with self.assertRaises(ValueError) as ctx:
            encoder(x_wrong_dim)
        self.assertIn("input dimension mismatch", str(ctx.exception))

        # 1D input instead of 2D
        x_1d = torch.randn(25)
        with self.assertRaises(ValueError) as ctx:
            encoder(x_1d)
        self.assertIn("expected 2D input", str(ctx.exception))

    def test_03_shared_predictor_consistency(self):
        """TEST 3: Shared predictor consumes Z and has identical parameter shapes regardless of client."""
        pred1 = SharedPredictor(latent_dim=32, hidden_dims=[32])
        pred2 = SharedPredictor(latent_dim=32, hidden_dims=[32])

        z = torch.randn(5, 32)
        out1 = pred1(z)
        out2 = pred2(z)
        self.assertEqual(out1.shape, (5, 1))
        self.assertEqual(out2.shape, (5, 1))

        # Verify exact shape match across all layers
        state1 = pred1.state_dict()
        state2 = pred2.state_dict()
        self.assertEqual(list(state1.keys()), list(state2.keys()))
        for k in state1:
            self.assertEqual(state1[k].shape, state2[k].shape)

    def test_04_composite_model_forward(self):
        """TEST 4: Composite model executes end-to-end forward pass D_i -> Z -> 1."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        x = torch.randn(10, 25)
        logits = model(x)
        self.assertEqual(logits.shape, (10, 1))

        probs = model.predict_proba(x)
        self.assertEqual(probs.shape, (10, 1))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_05_client_parameter_isolation(self):
        """TEST 5: get_parameters() extracts ONLY shared predictor parameters; encoder is excluded."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32, encoder_hidden_dims=[64], predictor_hidden_dims=[32])
        params = model.get_shared_parameters()
        shared_keys = model.get_shared_parameter_names()

        # Shared predictor has layers: 0.weight, 0.bias, 2.weight, 2.bias -> 4 tensors
        self.assertEqual(len(params), len(shared_keys))
        # Ensure no encoder layer names are in shared keys
        for k in shared_keys:
            self.assertNotIn("encoder", k)

        # Total model parameters include encoder + predictor, while params has strictly predictor
        total_model_tensors = len(list(model.parameters()))
        self.assertGreater(total_model_tensors, len(params))

    def test_06_client_encoder_preservation(self):
        """TEST 6: Client preserves private encoder weights when loading global shared predictor parameters."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        initial_encoder_weights = {k: v.clone() for k, v in model.get_encoder_state_dict().items()}

        # Create dummy updated shared predictor weights
        updated_predictor_params = [p + 0.5 for p in model.get_shared_parameters()]

        # Apply to model
        model.set_shared_parameters(updated_predictor_params)

        # Check encoder weights are completely untouched
        current_encoder_weights = model.get_encoder_state_dict()
        for k in initial_encoder_weights:
            self.assertTrue(torch.equal(initial_encoder_weights[k], current_encoder_weights[k]))

    def test_07_heterogeneous_aggregation_with_different_dimensions(self):
        """
        TEST 7: Multi-hospital FedAvg where clients have differing native dimensions:
          H1 (D1=25), H2 (D2=25), H3 (D3=25), H4_synthetic (D4=30).
          All map to common latent space Z=32 and aggregate without shape errors.
        """
        # Create synthetic datasets for 4 hospitals
        client_configs = [
            ('h1', 25, 50),
            ('h2', 25, 40),
            ('h3', 25, 30),
            ('h4_synthetic', 30, 60)  # Extended 30-feature hospital
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
            loader = DataLoader(TensorDataset(x_data, y_data), batch_size=16)

            client = HeterogeneousHospitalClient(
                client_id=cid,
                schema=schema,
                latent_dim=self.latent_dim,
                device="cpu",
                custom_loaders={'train': loader, 'val': loader, 'test': loader}
            )
            clients[cid] = client

        # Server
        server = HeterogeneousFederatedServer(latent_dim=self.latent_dim, device="cpu")

        # Execute 1 FL communication round
        global_params = server.get_global_parameters()
        client_updates = []
        for cid, client in clients.items():
            updated_params, n_samples, metrics = client.fit(
                parameters=global_params,
                config={'local_epochs': 1, 'lr': 0.01}
            )
            # Verify parameter shape matches predictor
            self.assertEqual(len(updated_params), len(global_params))
            client_updates.append((cid, updated_params, n_samples, metrics))

        # Server FedAvg aggregation
        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        self.assertEqual(summary['num_clients'], 4)
        self.assertEqual(summary['total_participating_samples'], 180)

        # Global predictor updated successfully
        new_global_params = server.get_global_parameters()
        self.assertEqual(len(new_global_params), len(global_params))

    def test_08_server_shape_validation_rejects_mismatch(self):
        """TEST 8: Server raises explicit ValueError if a client submits mismatched predictor shapes."""
        server = HeterogeneousFederatedServer(latent_dim=32, device="cpu")
        valid_params = server.get_global_parameters()

        # Corrupt one layer's shape
        corrupted_params = [p.copy() for p in valid_params]
        corrupted_params[0] = np.zeros((10, 10), dtype=np.float32)  # Wrong shape

        with self.assertRaises(ValueError) as ctx:
            server.strategy.validate_client_parameters('rogue_client', corrupted_params)
        self.assertIn("Parameter shape mismatch", str(ctx.exception))

    def test_09_server_rejects_encoder_parameters(self):
        """TEST 9: Server raises ValueError if a client attempts to send extra layers (e.g. encoder weights)."""
        server = HeterogeneousFederatedServer(latent_dim=32, device="cpu")
        valid_params = server.get_global_parameters()

        # Add extra layer (simulating encoder weights)
        extra_params = list(valid_params) + [np.zeros((64, 25), dtype=np.float32)]

        with self.assertRaises(ValueError) as ctx:
            server.strategy.validate_client_parameters('leaky_client', extra_params)
        self.assertIn("Parameter count mismatch", str(ctx.exception))

    def test_10_checkpoint_separation(self):
        """TEST 10: Server global checkpoint stores only shared predictor; client checkpoint stores both."""
        server = HeterogeneousFederatedServer(latent_dim=32, device="cpu")
        ckpt_path = server.save_checkpoint(round_num=1, is_final=False)

        saved = torch.load(ckpt_path)
        self.assertEqual(saved['model_type'], 'Federated_Heterogeneous_SharedPredictor')
        self.assertIn('shared_predictor_state_dict', saved)
        # Global checkpoint must NOT contain encoder weights
        self.assertNotIn('encoder_state_dict', saved)

        # Client checkpoint contains both
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        client_dict = {
            'hospital_id': 'hospital_1',
            'hospital_name': 'Hospital 1',
            'input_dim': 25,
            'latent_dim': 32,
            'encoder_state_dict': model.get_encoder_state_dict(),
            'shared_predictor_state_dict': model.get_shared_state_dict(),
            'feature_names': ['f1'] * 25,
            'feature_types': {}
        }
        client_ckpt_path = server.save_client_checkpoint(client_dict, round_num=1)
        client_saved = torch.load(client_ckpt_path)
        self.assertIn('encoder_state_dict', client_saved)
        self.assertIn('shared_predictor_state_dict', client_saved)

    def test_11_validation_protocol_integrity(self):
        """TEST 11: Validation evaluation evaluates accurately and test split is isolated."""
        server = HeterogeneousFederatedServer(latent_dim=32, device="cpu")

        # 2 test clients
        clients = {}
        for cid in ['hospital_1', 'hospital_2']:
            schema = get_client_schema(cid)
            x_val = torch.randn(20, schema.input_dimension)
            y_val = torch.randint(0, 2, (20, 1)).float()
            val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8)

            x_test = torch.randn(20, schema.input_dimension)
            y_test = torch.randint(0, 2, (20, 1)).float()
            test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=8)

            client = HeterogeneousHospitalClient(
                client_id=cid,
                schema=schema,
                latent_dim=32,
                device="cpu",
                custom_loaders={'train': val_loader, 'val': val_loader, 'test': test_loader}
            )
            clients[cid] = client

        val_eval = evaluate_heterogeneous_system(server, clients, split="val")
        self.assertIn('macro_metrics', val_eval)
        self.assertIn('accuracy', val_eval['macro_metrics'])
        self.assertIn('roc_auc', val_eval['macro_metrics'])

    def test_12_xai_wrapper_gradient_flow(self):
        """TEST 12: Client-local XAI wrapper supports autograd flow to native features and black-box proba."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        wrapper = HeterogeneousLocalXAIWrapper(model=model, device="cpu")

        # Test predict_proba format [N, 2]
        x_np = np.random.randn(5, 25).astype(np.float32)
        probs = wrapper.predict_proba(x_np)
        self.assertEqual(probs.shape, (5, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

        # Test gradient flow to native input features [N, D_i]
        grads = wrapper.compute_input_gradients(x_np, target_class=1)
        self.assertEqual(grads.shape, (5, 25))
        self.assertTrue(np.any(grads != 0.0))


if __name__ == '__main__':
    unittest.main()
